"""Training-free road-corridor estimation using classical computer vision.

This module deliberately has no dependency on YOLO, Ultralytics, tracking, or
any learned lane detector. It turns one BGR frame into a confidence-qualified
corridor estimate using grayscale -> Gaussian blur -> Canny -> a perspective
mask -> probabilistic Hough segments -> robust quadratic fits.

It is a heuristic for research/dashboard visualization, not a lane-level
semantic model. When the evidence is weak, it reuses a bounded number of recent
valid estimates and then returns the caller-provided fallback polygon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DynamicROIConfig:
    """Tunable parameters for the independent classical-CV ROI engine.

    Ratios are fractions of the input frame. ``search_*_width_ratio`` are full
    widths, not half-widths. ``smoothing`` is the fraction of a new valid
    measurement used in the temporal EMA (1.0 disables smoothing).
    """

    enabled: bool = True
    search_top_ratio: float = 0.50
    search_top_width_ratio: float = 0.30
    search_bottom_width_ratio: float = 0.88
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 24
    min_confidence: float = 0.30
    reliable_confidence: float = 0.56
    smoothing: float = 0.35
    max_reuse_frames: int = 8
    confidence_decay: float = 0.82
    expected_lane_width_ratio: float = 0.44
    center_tolerance_ratio: float = 0.30
    min_lane_width_ratio: float = 0.12
    max_lane_width_ratio: float = 0.92
    sample_count: int = 24
    ransac_iterations: int = 48

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "DynamicROIConfig":
        """Create a safe config from a partial API/runtime dictionary."""
        defaults = cls()
        data = values or {}

        def number(name: str, lo: float, hi: float) -> float:
            try:
                value = float(data.get(name, getattr(defaults, name)))
            except (TypeError, ValueError):
                value = float(getattr(defaults, name))
            return float(np.clip(value, lo, hi))

        def integer(name: str, lo: int, hi: int) -> int:
            try:
                value = int(data.get(name, getattr(defaults, name)))
            except (TypeError, ValueError):
                value = int(getattr(defaults, name))
            return int(np.clip(value, lo, hi))

        enabled = data.get("dynamic_enabled", data.get("enabled", defaults.enabled))
        if isinstance(enabled, str):
            enabled = enabled.lower() not in {"0", "false", "off", "no"}

        low = integer("canny_low", 5, 240)
        high = integer("canny_high", 10, 255)
        if high <= low:
            high = min(255, low + 20)

        min_width = number("min_lane_width_ratio", 0.08, 0.7)
        max_width = number("max_lane_width_ratio", 0.2, 0.98)
        if max_width <= min_width:
            max_width = min(0.98, min_width + 0.1)

        return cls(
            enabled=bool(enabled),
            search_top_ratio=number("search_top_ratio", 0.25, 0.78),
            search_top_width_ratio=number("search_top_width_ratio", 0.08, 0.85),
            search_bottom_width_ratio=number("search_bottom_width_ratio", 0.20, 1.0),
            canny_low=low,
            canny_high=high,
            hough_threshold=integer("hough_threshold", 5, 160),
            min_confidence=number("min_confidence", 0.05, 0.9),
            reliable_confidence=number("reliable_confidence", 0.2, 0.98),
            smoothing=number("smoothing", 0.05, 1.0),
            max_reuse_frames=integer("max_reuse_frames", 0, 60),
            confidence_decay=number("confidence_decay", 0.3, 0.99),
            expected_lane_width_ratio=number("expected_lane_width_ratio", 0.12, 0.92),
            center_tolerance_ratio=number("center_tolerance_ratio", 0.08, 0.48),
            min_lane_width_ratio=min_width,
            max_lane_width_ratio=max_width,
            sample_count=integer("sample_count", 8, 64),
            ransac_iterations=integer("ransac_iterations", 8, 200),
        )


@dataclass
class ROIFrameState:
    """Geometry and honesty/uncertainty metadata for one frame."""

    polygon: list[tuple[int, int]]
    state: str
    confidence: float
    uncertainty: bool
    reason: str
    width: int
    height: int
    left_boundary: list[tuple[int, int]] = field(default_factory=list)
    right_boundary: list[tuple[int, int]] = field(default_factory=list)
    corners: list[tuple[int, int]] = field(default_factory=list)
    search_region: list[tuple[int, int]] = field(default_factory=list)
    candidate_count: int = 0
    frames_since_detection: int = 0

    @property
    def source(self) -> str:
        if self.state in {"detected", "low_confidence"}:
            return "classical_cv"
        if self.state == "reused":
            return "temporal_reuse"
        return "configured_fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [[int(x), int(y)] for x, y in self.polygon],
            "corners": [[int(x), int(y)] for x, y in self.corners],
            "left_boundary": [[int(x), int(y)] for x, y in self.left_boundary],
            "right_boundary": [[int(x), int(y)] for x, y in self.right_boundary],
            "search_region": [[int(x), int(y)] for x, y in self.search_region],
            "width": int(self.width),
            "height": int(self.height),
            "state": self.state,
            "confidence": round(float(self.confidence), 3),
            "uncertainty": bool(self.uncertainty),
            "reason": self.reason,
            "source": self.source,
            "candidate_count": int(self.candidate_count),
            "frames_since_detection": int(self.frames_since_detection),
        }


@dataclass
class _Segment:
    points: np.ndarray
    anchor_x: float
    length: float
    y_span: float


@dataclass
class _BoundaryFit:
    x_samples: np.ndarray
    y_samples: np.ndarray
    confidence: float
    anchor_x: float
    support_segments: int
    coverage: float
    residual_px: float


class DynamicROIEngine:
    """Stateful, YOLO-independent dynamic corridor estimator.

    Call :meth:`update` once per frame. Pass the static/fallback ROI polygon
    generated by the application's existing ``DrivingROI`` so its calibration
    sliders remain meaningful. The history is normalized to image dimensions;
    same-aspect resolution changes therefore scale cleanly, while large aspect
    changes invalidate stale geometry.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = DynamicROIConfig.from_mapping(config)
        self._last_valid: Optional[dict[str, np.ndarray]] = None
        self._last_confidence = 0.0
        self._last_aspect: Optional[float] = None
        self._misses = 0
        self._last_state: Optional[ROIFrameState] = None

    def reset(self) -> None:
        """Forget temporal geometry (new camera/video, calibration reset)."""
        self._last_valid = None
        self._last_confidence = 0.0
        self._last_aspect = None
        self._misses = 0
        self._last_state = None

    def update(
        self,
        frame: np.ndarray,
        fallback_polygon: Optional[Sequence[Sequence[float]]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> ROIFrameState:
        """Estimate the corridor for ``frame`` and return the current state.

        Invalid/empty frames are treated as failed evidence and resolve to
        bounded reuse or the supplied fallback polygon. No learned model is
        called from this method.
        """
        if config is not None:
            self.config = DynamicROIConfig.from_mapping(config)
        cfg = self.config

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            if self._last_state is not None:
                return self._last_state
            return ROIFrameState([], "fallback", 0.0, True,
                                 "No valid image frame is available", 0, 0)

        height, width = frame.shape[:2]
        aspect = width / max(1.0, float(height))
        fallback = _normalise_polygon(fallback_polygon, width, height)
        if fallback is None:
            fallback = _default_fallback(width, height)

        if self._last_aspect is not None and abs(aspect / self._last_aspect - 1.0) > 0.18:
            # A sharp aspect-ratio change commonly means a camera/source switch.
            # Do not project old lane geometry into a materially different view.
            self.reset()
        self._last_aspect = aspect

        if not cfg.enabled:
            self._last_valid = None
            self._last_confidence = 0.0
            self._misses = 0
            state = self._fallback_state(
                fallback, width, height, "Dynamic ROI disabled; configured fallback is active",
                state_name="disabled", candidate_count=0,
            )
            self._last_state = state
            return state

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        if gray.ndim != 2:
            return self._on_failure(fallback, width, height, 0, "Image is not grayscale/BGR")

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

        center_x = _fallback_center_x(fallback, width)
        y_top = int(np.clip(cfg.search_top_ratio * height, 0, height - 2))
        y_bottom = int(np.clip(max(y_top + 1, height - 0.02 * height), y_top + 1, height - 1))
        search_region = _perspective_polygon(
            width, height, center_x,
            y_top / max(1, height), y_bottom / max(1, height),
            cfg.search_top_width_ratio, cfg.search_bottom_width_ratio,
        )
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [np.asarray(search_region, dtype=np.int32)], 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        min_line_length = max(10, int(height * 0.035))
        max_line_gap = max(6, int(height * 0.035))
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=cfg.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        segments, transverse_count = self._collect_segments(
            lines, width, height, y_bottom,
        )
        fits = self._fit_boundary_candidates(
            segments, width, height, y_top, y_bottom, center_x, cfg,
        )
        pair = self._choose_pair(fits, width, height, y_top, y_bottom, center_x, cfg)
        candidate_count = len(segments)

        if pair is None:
            reason = "Not enough consistent left/right lane-edge evidence"
            if transverse_count >= 2:
                reason = "Intersection-like transverse edges; lane geometry is ambiguous"
            return self._on_failure(fallback, width, height, candidate_count, reason,
                                    search_region=search_region)

        left_fit, right_fit, confidence = pair
        if transverse_count >= 2:
            # Crosswalks and intersections can produce strong Hough edges that
            # resemble lane geometry. Penalize instead of claiming certainty.
            confidence *= 0.48

        if confidence < cfg.min_confidence:
            reason = ("Intersection-like transverse edges; lane geometry is ambiguous"
                      if transverse_count >= 2 else
                      f"Lane evidence below the configured confidence threshold ({confidence:.2f})")
            return self._on_failure(fallback, width, height, candidate_count, reason,
                                    search_region=search_region)

        y_values = np.linspace(y_top, y_bottom, cfg.sample_count)
        left_px = np.column_stack((left_fit.x_samples, y_values))
        right_px = np.column_stack((right_fit.x_samples, y_values))
        # At the estimated vanishing point, noisy fits may cross by a few pixels.
        # Collapse only those top samples to their midpoint rather than creating
        # a self-intersecting ROI polygon.
        crossed = left_px[:, 0] > right_px[:, 0]
        if np.any(crossed):
            midpoint = 0.5 * (left_px[crossed, 0] + right_px[crossed, 0])
            left_px[crossed, 0] = midpoint
            right_px[crossed, 0] = midpoint
        candidate = {
            "left": left_px / np.array([width, height], dtype=np.float64),
            "right": right_px / np.array([width, height], dtype=np.float64),
        }

        if self._last_valid is not None:
            alpha = cfg.smoothing
            candidate = {
                side: (1.0 - alpha) * self._last_valid[side] + alpha * candidate[side]
                for side in ("left", "right")
            }
            # Guard against a stale/smoothed boundary crossing its partner.
            if not _normalized_pair_valid(candidate["left"], candidate["right"], cfg):
                candidate = {
                    "left": left_px / np.array([width, height], dtype=np.float64),
                    "right": right_px / np.array([width, height], dtype=np.float64),
                }

        if not _normalized_pair_valid(candidate["left"], candidate["right"], cfg):
            return self._on_failure(fallback, width, height, candidate_count,
                                    "Fitted boundaries do not form a valid corridor",
                                    search_region=search_region)

        self._last_valid = {side: values.copy() for side, values in candidate.items()}
        self._last_confidence = float(np.clip(confidence, 0.0, 1.0))
        self._misses = 0

        left_points = _denormalize_points(candidate["left"], width, height)
        right_points = _denormalize_points(candidate["right"], width, height)
        polygon = left_points + list(reversed(right_points))
        corners = [left_points[0], right_points[0], right_points[-1], left_points[-1]]
        uncertain = confidence < cfg.reliable_confidence
        state_name = "low_confidence" if uncertain else "detected"
        reason = (
            "Weak classical-CV evidence; treat this corridor as provisional"
            if uncertain else "Both boundaries supported by Canny/Hough geometry"
        )
        if transverse_count >= 2 and not uncertain:
            # This branch is defensive; the penalty normally makes it low confidence.
            reason = "Transverse edges present; confidence is reduced"
            uncertain = True
            state_name = "low_confidence"

        state = ROIFrameState(
            polygon=polygon,
            state=state_name,
            confidence=self._last_confidence,
            uncertainty=uncertain,
            reason=reason,
            width=width,
            height=height,
            left_boundary=left_points,
            right_boundary=right_points,
            corners=corners,
            search_region=search_region,
            candidate_count=candidate_count,
            frames_since_detection=0,
        )
        self._last_state = state
        return state

    @staticmethod
    def _collect_segments(lines, width: int, height: int, y_bottom: int):
        if lines is None:
            return [], 0
        raw = np.asarray(lines).reshape(-1, 4)
        # Longest-first cap prevents pathological Hough output from making
        # candidate fitting expensive on highly textured frames.
        lengths = np.hypot(raw[:, 2] - raw[:, 0], raw[:, 3] - raw[:, 1])
        order = np.argsort(lengths)[::-1][:320]
        segments: list[_Segment] = []
        transverse_count = 0
        for index in order:
            x1, y1, x2, y2 = (float(v) for v in raw[index])
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < max(10.0, 0.025 * height):
                continue
            if abs(dx) >= 2.0 * max(1.0, abs(dy)) and length >= 0.16 * width:
                transverse_count += 1
            # Ignore mostly horizontal edges. A lane boundary must have useful
            # vertical extent in image coordinates.
            if abs(dy) < max(5.0, 0.35 * abs(dx)):
                continue
            if abs(dy) < 7.0:
                continue
            slope = dx / dy
            anchor_x = x1 + slope * (y_bottom - y1)
            if not (-0.12 * width <= anchor_x <= 1.12 * width):
                continue

            sample_n = max(5, min(15, int(abs(dy) / max(1.0, height * 0.025))))
            ts = np.linspace(0.0, 1.0, sample_n)
            points = np.column_stack((x1 + ts * dx, y1 + ts * dy))
            segments.append(_Segment(points, float(anchor_x), length, abs(dy)))
        return segments, transverse_count

    @staticmethod
    def _fit_boundary_candidates(
        segments: list[_Segment], width: int, height: int, y_top: int,
        y_bottom: int, center_x: float, cfg: DynamicROIConfig,
    ) -> list[tuple[str, _BoundaryFit]]:
        sides: dict[str, list[_Segment]] = {"left": [], "right": []}
        for segment in segments:
            sides["left" if segment.anchor_x < center_x else "right"].append(segment)

        fits: list[tuple[str, _BoundaryFit]] = []
        for side, side_segments in sides.items():
            for group in _cluster_segments(side_segments, width):
                fit = _robust_curve_fit(group, width, height, y_top, y_bottom, cfg)
                if fit is not None:
                    fits.append((side, fit))
        return fits

    @staticmethod
    def _choose_pair(
        fits: list[tuple[str, _BoundaryFit]], width: int, height: int,
        y_top: int, y_bottom: int, center_x: float, cfg: DynamicROIConfig,
    ) -> Optional[tuple[_BoundaryFit, _BoundaryFit, float]]:
        left_candidates = [fit for side, fit in fits if side == "left" and fit.anchor_x < center_x + 0.015 * width]
        right_candidates = [fit for side, fit in fits if side == "right" and fit.anchor_x > center_x - 0.015 * width]
        if not left_candidates or not right_candidates:
            return None

        best = None
        ys = np.linspace(y_top, y_bottom, 7)
        for left in left_candidates:
            for right in right_candidates:
                lx = _eval_fit(left, ys, height)
                rx = _eval_fit(right, ys, height)
                widths = (rx - lx) / max(1.0, width)
                # Perspective boundaries may converge near the horizon. Require
                # a plausible lower-frame lane width, positive separation along
                # the complete fit, and reject corridors that grow implausibly wide.
                if (widths[0] < -0.025
                        or np.any(widths[1:] <= 0.0)
                        or widths[-1] < cfg.min_lane_width_ratio
                        or widths[-1] > cfg.max_lane_width_ratio
                        or np.any(widths > cfg.max_lane_width_ratio)):
                    continue
                if np.any(lx < -0.04 * width) or np.any(rx > 1.04 * width):
                    continue

                bottom_center = (lx[-1] + rx[-1]) / (2.0 * width)
                width_ratio = float(widths[-1])
                center_error = abs(bottom_center - center_x / width)
                if center_error > cfg.center_tolerance_ratio:
                    continue

                center_score = max(0.0, 1.0 - center_error / cfg.center_tolerance_ratio)
                width_error = abs(width_ratio - cfg.expected_lane_width_ratio)
                width_score = max(0.0, 1.0 - width_error / max(0.28, cfg.expected_lane_width_ratio))
                pair_score = (
                    0.43 * left.confidence + 0.43 * right.confidence
                    + 0.92 * center_score + 0.38 * width_score
                )
                pair_confidence = min(left.confidence, right.confidence)
                pair_confidence *= 0.72 + 0.28 * center_score
                pair_confidence *= 0.82 + 0.18 * width_score
                # Give a small preference to the more strongly-supported pair
                # when two adjacent lanes are equally close to the center prior.
                pair_confidence = float(np.clip(pair_confidence, 0.0, 1.0))
                candidate = (pair_score, left, right, pair_confidence)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _on_failure(
        self,
        fallback: list[tuple[int, int]],
        width: int,
        height: int,
        candidate_count: int,
        reason: str,
        search_region: Optional[list[tuple[int, int]]] = None,
    ) -> ROIFrameState:
        cfg = self.config
        if self._last_valid is not None and self._misses < cfg.max_reuse_frames:
            self._misses += 1
            confidence = self._last_confidence * (cfg.confidence_decay ** self._misses)
            left_points = _denormalize_points(self._last_valid["left"], width, height)
            right_points = _denormalize_points(self._last_valid["right"], width, height)
            polygon = left_points + list(reversed(right_points))
            state = ROIFrameState(
                polygon=polygon,
                state="reused",
                confidence=confidence,
                uncertainty=True,
                reason=f"{reason}; reusing recent geometry ({self._misses}/{cfg.max_reuse_frames})",
                width=width,
                height=height,
                left_boundary=left_points,
                right_boundary=right_points,
                corners=[left_points[0], right_points[0], right_points[-1], left_points[-1]],
                search_region=search_region or [],
                candidate_count=candidate_count,
                frames_since_detection=self._misses,
            )
            self._last_state = state
            return state

        self._last_valid = None
        self._last_confidence = 0.0
        self._misses = 0
        state = self._fallback_state(
            fallback, width, height, reason,
            state_name="fallback", candidate_count=candidate_count,
            search_region=search_region,
        )
        self._last_state = state
        return state

    @staticmethod
    def _fallback_state(
        fallback: list[tuple[int, int]], width: int, height: int,
        reason: str, state_name: str = "fallback", candidate_count: int = 0,
        search_region: Optional[list[tuple[int, int]]] = None,
    ) -> ROIFrameState:
        polygon = fallback
        corners = fallback[:4] if len(fallback) >= 4 else fallback
        return ROIFrameState(
            polygon=polygon,
            state=state_name,
            confidence=0.0,
            uncertainty=True,
            reason=reason,
            width=width,
            height=height,
            corners=corners,
            search_region=search_region or [],
            candidate_count=candidate_count,
            frames_since_detection=0,
        )


def _cluster_segments(segments: list[_Segment], width: int) -> list[list[_Segment]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: item.anchor_x)
    # Curved-lane tangents extrapolated to the bottom can fan out modestly;
    # this tolerance groups a single curved stripe but keeps neighboring lanes
    # separate in ordinary multi-lane scenes.
    gap = max(20.0, 0.085 * width)
    groups: list[list[_Segment]] = [[ordered[0]]]
    for item in ordered[1:]:
        group_anchor = float(np.median([segment.anchor_x for segment in groups[-1]]))
        if item.anchor_x - group_anchor <= gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _robust_curve_fit(
    segments: list[_Segment], width: int, height: int,
    y_top: int, y_bottom: int, cfg: DynamicROIConfig,
) -> Optional[_BoundaryFit]:
    points = np.concatenate([segment.points for segment in segments], axis=0)
    if len(points) < 6:
        return None

    x_px = points[:, 0]
    y_norm = points[:, 1] / max(1.0, float(height))
    x_norm = x_px / max(1.0, float(width))
    design = np.column_stack((y_norm * y_norm, y_norm, np.ones_like(y_norm)))
    threshold = max(3.5, 0.0055 * width)  # reprojection tolerance in pixels
    rng_seed = int((width * 73856093 + height * 19349663 + len(points) * 83492791) % (2**32))
    rng = np.random.default_rng(rng_seed)

    best_inliers = None
    best_key = (-1, float("inf"))
    unique_y = np.unique(np.round(y_norm, 3))
    if len(unique_y) < 3:
        return None
    trials = min(cfg.ransac_iterations, max(12, len(points) * 2))
    for _ in range(trials):
        sample_idx = rng.choice(len(points), size=3, replace=False)
        # Reject nearly identical y samples which make a quadratic ill-conditioned.
        if np.ptp(y_norm[sample_idx]) < 0.06:
            continue
        try:
            coeff = np.linalg.lstsq(design[sample_idx], x_norm[sample_idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs((design @ coeff - x_norm) * width)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        if count < 5:
            continue
        med = float(np.median(residuals[inliers]))
        key = (count, -med)
        if key > best_key:
            best_key = key
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < 5:
        return None

    # Refit on RANSAC consensus, compare a linear model to a quadratic. The
    # quadratic is retained only when broad vertical evidence materially
    # improves the residual, avoiding curvature from isolated noisy segments.
    try:
        quadratic = np.linalg.lstsq(design[best_inliers], x_norm[best_inliers], rcond=None)[0]
        linear_design = np.column_stack((y_norm[best_inliers], np.ones(best_inliers.sum())))
        linear = np.linalg.lstsq(linear_design, x_norm[best_inliers], rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    q_pred = design[best_inliers] @ quadratic
    l_pred = linear_design @ linear
    q_rmse = float(np.sqrt(np.mean(((q_pred - x_norm[best_inliers]) * width) ** 2)))
    l_rmse = float(np.sqrt(np.mean(((l_pred - x_norm[best_inliers]) * width) ** 2)))
    inlier_y = points[best_inliers, 1]
    coverage = float(np.ptp(inlier_y) / max(1.0, y_bottom - y_top))
    if coverage < 0.20:
        return None
    if l_rmse <= 1e-3 or q_rmse >= 0.86 * l_rmse:
        # Linear fit represented as a quadratic with zero curvature.
        coeff = np.array([0.0, linear[0], linear[1]], dtype=np.float64)
    else:
        coeff = quadratic

    # Robustly refine once using a Huber-like weight on the consensus set.
    pred = design @ coeff
    residual_px = np.abs((pred - x_norm) * width)
    refine = residual_px <= max(threshold * 1.5, 5.0)
    if int(refine.sum()) >= 5:
        weights = 1.0 / np.maximum(1.0, residual_px[refine] / max(1.0, threshold * 0.65))
        weighted_design = design[refine] * weights[:, None]
        weighted_x = x_norm[refine] * weights
        try:
            refined = np.linalg.lstsq(weighted_design, weighted_x, rcond=None)[0]
            if l_rmse <= 1e-3 or q_rmse >= 0.86 * l_rmse:
                refined[0] = 0.0
            coeff = refined
        except np.linalg.LinAlgError:
            pass

    final_residual = np.abs((design @ coeff - x_norm) * width)
    final_inliers = final_residual <= max(threshold * 1.5, 5.0)
    inlier_count = int(np.count_nonzero(final_inliers))
    if inlier_count < 5:
        return None

    supported_segments = 0
    for segment in segments:
        seg_design = np.column_stack((
            (segment.points[:, 1] / max(1.0, height)) ** 2,
            segment.points[:, 1] / max(1.0, height),
            np.ones(len(segment.points)),
        ))
        seg_resid = np.abs((seg_design @ coeff - segment.points[:, 0] / max(1.0, width)) * width)
        if float(np.mean(seg_resid <= max(threshold * 1.5, 5.0))) >= 0.55:
            supported_segments += 1

    consensus = inlier_count / len(points)
    segment_support = min(1.0, supported_segments / 2.0)
    point_support = min(1.0, inlier_count / max(8.0, 0.25 * len(points)))
    fit_confidence = 0.39 * min(1.0, coverage) + 0.30 * consensus + 0.18 * segment_support + 0.13 * point_support
    if supported_segments == 1 and coverage < 0.48:
        fit_confidence *= 0.82

    y_eval = np.linspace(y_top, y_bottom, cfg.sample_count) / max(1.0, float(height))
    x_eval = (coeff[0] * y_eval**2 + coeff[1] * y_eval + coeff[2]) * width
    if not np.all(np.isfinite(x_eval)):
        return None
    if np.any(x_eval < -0.10 * width) or np.any(x_eval > 1.10 * width):
        return None

    # Extrapolate to the corridor's bottom for pair grouping/selection.
    yb = y_bottom / max(1.0, float(height))
    anchor = float((coeff[0] * yb**2 + coeff[1] * yb + coeff[2]) * width)
    residual_mean = float(np.mean(final_residual[final_inliers]))
    return _BoundaryFit(
        x_samples=x_eval,
        y_samples=y_eval,
        confidence=float(np.clip(fit_confidence, 0.0, 1.0)),
        anchor_x=anchor,
        support_segments=supported_segments,
        coverage=coverage,
        residual_px=residual_mean,
    )


def _eval_fit(fit: _BoundaryFit, ys: np.ndarray, height: int) -> np.ndarray:
    # x_samples and y_samples share the same uniform fitted-y grid.
    target = ys / max(1.0, float(height))
    return np.interp(target, fit.y_samples, fit.x_samples)


def _normalized_pair_valid(left: np.ndarray, right: np.ndarray, cfg: DynamicROIConfig) -> bool:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        return False
    widths = right[:, 0] - left[:, 0]
    if np.any(~np.isfinite(widths)) or np.any(widths < -0.004):
        return False
    if widths[-1] < cfg.min_lane_width_ratio * 0.70:
        return False
    if np.any(left[:, 0] < -0.06) or np.any(right[:, 0] > 1.06):
        return False
    return True


def _denormalize_points(points: np.ndarray, width: int, height: int) -> list[tuple[int, int]]:
    scaled = points * np.array([width, height], dtype=np.float64)
    return [(int(round(x)), int(round(y))) for x, y in scaled]


def _normalise_polygon(
    polygon: Optional[Sequence[Sequence[float]]], width: int, height: int,
) -> Optional[list[tuple[int, int]]]:
    if polygon is None:
        return None
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return None
    points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
    return [(int(round(x)), int(round(y))) for x, y in points]


def _default_fallback(width: int, height: int) -> list[tuple[int, int]]:
    return [
        (int(0.35 * width), int(0.62 * height)),
        (int(0.75 * width), int(0.62 * height)),
        (int(0.94 * width), height - 1),
        (int(0.16 * width), height - 1),
    ]


def _fallback_center_x(fallback: Sequence[tuple[int, int]], width: int) -> float:
    if len(fallback) >= 4:
        # Existing DrivingROI order is TL, TR, BR, BL.
        return 0.5 * (fallback[2][0] + fallback[3][0])
    return width * 0.5


def _perspective_polygon(
    width: int, height: int, center_x: float,
    top_ratio: float, bottom_ratio: float,
    top_width_ratio: float, bottom_width_ratio: float,
) -> list[tuple[int, int]]:
    top_y = int(np.clip(top_ratio * height, 0, height - 1))
    bottom_y = int(np.clip(bottom_ratio * height, top_y + 1, height - 1))
    top_half = 0.5 * top_width_ratio * width
    bottom_half = 0.5 * bottom_width_ratio * width
    return [
        (int(np.clip(center_x - top_half, 0, width - 1)), top_y),
        (int(np.clip(center_x + top_half, 0, width - 1)), top_y),
        (int(np.clip(center_x + bottom_half, 0, width - 1)), bottom_y),
        (int(np.clip(center_x - bottom_half, 0, width - 1)), bottom_y),
    ]
