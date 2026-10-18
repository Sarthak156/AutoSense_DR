"""YOLO-agnostic object-to-corridor association helpers.

The caller supplies plain bounding boxes, class names, confidences, optional
track IDs, and the polygon estimated by :mod:`dynamic_roi`. This module never
runs inference and never filters/deletes detections.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Optional, Sequence

import cv2
import numpy as np


ROAD_USER_CLASSES = {
    "person", "pedestrian", "people", "rider", "cyclist", "bicycle",
    "bike", "motorcycle", "motorbike", "scooter", "moped", "car",
    "truck", "bus", "van", "vehicle", "train", "trailer", "taxi",
    "ambulance", "fire engine", "police car", "auto", "autorickshaw",
}
TRAFFIC_CONTROL_CLASSES = {
    "traffic light", "traffic sign", "stop sign", "speed limit", "road sign",
    "signal", "traffic signal", "sign",
}


class ROIObjectAssociator:
    """Compute explainable ROI relevance while retaining every input object.

    The score combines bottom-center inclusion, clipped bounding-box/polygon
    overlap, detector confidence and a light class-context prior. If a track ID
    exists, a short EMA reduces frame-to-frame boundary flicker. Geometric
    uncertainty is kept separate from the score and is never hidden.
    """

    def __init__(
        self,
        history_size: int = 5,
        smoothing: float = 0.35,
        in_threshold: float = 0.30,
        near_threshold: float = 0.08,
        overlap_threshold: float = 0.12,
        max_tracked_objects: int = 512,
    ):
        self.history_size = max(1, int(history_size))
        self.smoothing = float(np.clip(smoothing, 0.05, 1.0))
        self.in_threshold = float(np.clip(in_threshold, 0.05, 0.9))
        self.near_threshold = float(np.clip(near_threshold, 0.0, self.in_threshold))
        self.overlap_threshold = float(np.clip(overlap_threshold, 0.01, 0.8))
        self.max_tracked_objects = max(1, int(max_tracked_objects))
        self._track_scores: OrderedDict[int, deque[float]] = OrderedDict()

    def reset(self) -> None:
        self._track_scores.clear()

    def associate(
        self,
        bbox: Sequence[float],
        polygon: Sequence[Sequence[float]],
        image_shape: Sequence[int],
        class_name: str,
        confidence: float,
        track_id: Optional[int] = None,
        roi_state: str = "detected",
        roi_confidence: float = 1.0,
        roi_uncertainty: Optional[bool] = None,
    ) -> dict:
        """Return a relevance record; never changes the supplied detection.

        ``image_shape`` follows NumPy order (height, width[, channels]).
        ``in_path`` remains a geometry-based boolean for compatibility with the
        existing safety logic. ``roi_label`` states when that geometry is only
        a reused/fallback estimate.
        """
        height, width = int(image_shape[0]), int(image_shape[1])
        x1, y1, x2, y2 = (float(v) for v in bbox)
        poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(poly) < 3 or width <= 0 or height <= 0:
            return self._unknown_result(roi_state, roi_confidence)

        clipped_box = (
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        )
        bx1, by1, bx2, by2 = clipped_box
        if bx2 <= bx1 or by2 <= by1:
            return self._unknown_result(roi_state, roi_confidence)

        center_x = 0.5 * (x1 + x2)
        bottom_y = y2
        center_inside = cv2.pointPolygonTest(
            poly, (float(center_x), float(bottom_y)), False
        ) >= 0
        signed_distance = float(cv2.pointPolygonTest(
            poly, (float(center_x), float(bottom_y)), True
        ))
        overlap = _bbox_polygon_overlap((bx1, by1, bx2, by2), poly, width, height)

        det_conf = float(np.clip(confidence, 0.0, 1.0))
        class_factor = _class_factor(class_name)
        geometric = min(
            1.0,
            0.60 * float(center_inside) + 0.65 * min(1.0, overlap / 0.30),
        )
        confidence_factor = 0.62 + 0.38 * det_conf
        raw_score = float(np.clip(geometric * confidence_factor * class_factor, 0.0, 1.0))

        score = raw_score
        has_history = track_id is not None
        if track_id is not None:
            track_key = int(track_id)
            history = self._track_scores.pop(track_key, None)
            if history is None:
                history = deque(maxlen=self.history_size)
            self._track_scores[track_key] = history
            while len(self._track_scores) > self.max_tracked_objects:
                self._track_scores.popitem(last=False)
            if history:
                previous = float(np.mean(history))
                score = (1.0 - self.smoothing) * previous + self.smoothing * raw_score
            history.append(raw_score)

        has_spatial_support = center_inside or overlap >= self.overlap_threshold
        in_path = bool(has_spatial_support and score >= self.in_threshold)
        near = bool(
            not in_path
            and (overlap > 0.01 or signed_distance >= -0.035 * min(width, height)
                 or score >= self.near_threshold)
        )

        if roi_uncertainty is None:
            roi_uncertainty = roi_state != "detected"
        roi_uncertainty = bool(roi_uncertainty or roi_state in {
            "low_confidence", "reused", "fallback", "disabled"
        })
        if roi_uncertainty:
            relevance = "uncertain"
            position = "inside" if in_path else "near" if near else "outside"
            if roi_state in {"fallback", "disabled"}:
                label = f"UNCERTAIN · {position} by fallback"
            elif roi_state == "reused":
                label = f"UNCERTAIN · {position} by reused ROI"
            else:
                label = f"UNCERTAIN · {position} by weak ROI"
        elif in_path:
            relevance, label = "in_corridor", "IN CORRIDOR"
        elif near:
            relevance, label = "near_corridor", "NEAR CORRIDOR"
        else:
            relevance, label = "outside_corridor", "OUTSIDE CORRIDOR"

        return {
            "in_path": in_path,
            "roi_relevance": relevance,
            "roi_label": label,
            "roi_score": round(score, 3),
            "roi_raw_score": round(raw_score, 3),
            "roi_overlap": round(overlap, 3),
            "bottom_center_in_roi": bool(center_inside),
            "roi_signed_distance_px": round(signed_distance, 1),
            "roi_class_factor": round(class_factor, 3),
            "roi_confidence": round(float(np.clip(roi_confidence, 0.0, 1.0)), 3),
            "roi_state": str(roi_state),
            "roi_uncertainty": roi_uncertainty,
            "roi_track_history": has_history,
        }

    @staticmethod
    def _unknown_result(roi_state: str, roi_confidence: float) -> dict:
        return {
            "in_path": False,
            "roi_relevance": "uncertain",
            "roi_label": "UNCERTAIN · no usable ROI geometry",
            "roi_score": 0.0,
            "roi_raw_score": 0.0,
            "roi_overlap": 0.0,
            "bottom_center_in_roi": False,
            "roi_signed_distance_px": None,
            "roi_class_factor": 0.0,
            "roi_confidence": round(float(np.clip(roi_confidence, 0.0, 1.0)), 3),
            "roi_state": str(roi_state),
            "roi_uncertainty": True,
            "roi_track_history": False,
        }


def _class_factor(class_name: str) -> float:
    name = str(class_name).strip().lower().replace("_", " ")
    if name in ROAD_USER_CLASSES:
        return 1.0
    if name in TRAFFIC_CONTROL_CLASSES:
        return 0.72
    # Unknown model-specific classes remain visible and associateable; class
    # uncertainty mildly lowers the score instead of excluding them.
    return 0.84


def _bbox_polygon_overlap(
    bbox: tuple[float, float, float, float],
    polygon: np.ndarray,
    width: int,
    height: int,
) -> float:
    """Raster-area overlap / visible bbox area for a potentially curved polygon."""
    x1, y1, x2, y2 = bbox
    ix1, iy1 = int(np.floor(x1)), int(np.floor(y1))
    ix2, iy2 = int(np.ceil(x2)), int(np.ceil(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    # Work in the detection's local crop to keep this inexpensive.
    local_polygon = polygon.copy()
    local_polygon[:, 0] -= ix1
    local_polygon[:, 1] -= iy1
    # OpenCV clips scan conversion against this small local canvas; retaining
    # off-crop vertices preserves correct edge intersections.
    mask = np.zeros((iy2 - iy1, ix2 - ix1), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(local_polygon).astype(np.int32)], 1)
    bbox_area = max(1, mask.shape[0] * mask.shape[1])
    return float(np.count_nonzero(mask) / bbox_area)
