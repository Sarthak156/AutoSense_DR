"""Synthetic regression tests for the training-free dynamic ROI engine."""

import cv2
import numpy as np
import pytest

from backend.perception.dynamic_roi import DynamicROIEngine
from backend.perception.roi import DrivingROI


def make_road(
    width=640,
    height=360,
    *,
    curve=0.0,
    lane_bottoms=(0.32, 0.68),
    markings=True,
    shadows=False,
    intersection=False,
):
    """Draw deterministic, non-photorealistic road-marking test frames."""
    frame = np.full((height, width, 3), 65, dtype=np.uint8)
    road = np.array([
        [int(0.12 * width), height - 1],
        [int(0.38 * width), int(0.43 * height)],
        [int(0.72 * width), int(0.43 * height)],
        [int(0.95 * width), height - 1],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [road], (85, 85, 85))

    if markings:
        ys = np.arange(int(0.48 * height), height)
        for bottom_ratio in lane_bottoms:
            side = -1 if bottom_ratio <= 0.5 else 1
            points = []
            for y in ys:
                progress = (y / height - 0.48) / 0.52
                x_ratio = 0.5 + (bottom_ratio - 0.5) * progress
                # Opposing curvature bends both lane boundaries while they
                # still share a perspective vanishing point.
                x_ratio += side * curve * progress * (1.0 - progress)
                points.append([int(round(x_ratio * width)), int(y)])
            cv2.polylines(
                frame, [np.asarray(points, dtype=np.int32)], False,
                (235, 235, 235), max(3, int(width / 128)), cv2.LINE_AA,
            )

    if shadows:
        # A broad hard-edged shadow produces Canny edges but no lane semantics.
        shadow = np.array([
            [int(0.10 * width), int(0.66 * height)],
            [int(0.90 * width), int(0.66 * height)],
            [int(0.95 * width), int(0.79 * height)],
            [int(0.12 * width), int(0.79 * height)],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [shadow], (30, 30, 30))

    if intersection:
        # Crosswalk-like transverse bars deliberately resemble strong Hough
        # edges, but should not be presented as reliable lane boundaries.
        for ratio in (0.60, 0.66, 0.72, 0.78):
            y = int(ratio * height)
            cv2.rectangle(
                frame, (int(0.15 * width), y),
                (int(0.85 * width), y + max(4, height // 80)),
                (235, 235, 235), -1,
            )
    return frame


def fallback_for(frame, center=0.5):
    height, width = frame.shape[:2]
    return DrivingROI(
        height_start=0.70,
        top_width_ratio=0.30,
        bottom_width_ratio=0.75,
        ego_center_ratio=center,
        bottom_margin=0.02,
    ).get_polygon(width, height)


def test_curved_road_fits_smooth_non_linear_boundaries():
    frame = make_road(curve=0.22)
    result = DynamicROIEngine().update(frame, fallback_for(frame))

    assert result.state == "detected"
    assert result.confidence >= 0.56
    assert len(result.left_boundary) >= 12
    assert len(result.right_boundary) >= 12

    for boundary in (result.left_boundary, result.right_boundary):
        x = np.asarray([point[0] for point in boundary], dtype=float)
        straight_chord = np.linspace(x[0], x[-1], len(x))
        # A curved road should not be forced to a single straight Hough line.
        assert np.max(np.abs(x - straight_chord)) > 12.0


def test_multiple_lanes_select_pair_near_configured_ego_center():
    # Three visible lane markings: the ego-center prior at 0.62 selects the
    # adjacent corridor around the center lane, not the far-left lane.
    frame = make_road(lane_bottoms=(0.28, 0.50, 0.72))
    result = DynamicROIEngine().update(frame, fallback_for(frame, center=0.62))

    assert result.state == "detected"
    left_bottom = result.left_boundary[-1][0] / frame.shape[1]
    right_bottom = result.right_boundary[-1][0] / frame.shape[1]
    corridor_center = (left_bottom + right_bottom) / 2
    assert 0.45 <= left_bottom < right_bottom <= 0.78
    assert abs(corridor_center - 0.62) < 0.09


def test_missing_or_single_sided_markings_use_configured_fallback():
    blank = make_road(markings=False)
    fallback = fallback_for(blank, center=0.5)
    result = DynamicROIEngine().update(blank, fallback)
    assert result.state == "fallback"
    assert result.uncertainty is True
    assert result.confidence == 0.0
    assert result.polygon == fallback

    one_side = make_road(markings=False)
    h, w = one_side.shape[:2]
    points = []
    for y in range(int(0.48 * h), h):
        progress = (y / h - 0.48) / 0.52
        points.append([int((0.5 - 0.18 * progress) * w), y])
    cv2.polylines(one_side, [np.asarray(points, dtype=np.int32)], False,
                  (235, 235, 235), 5, cv2.LINE_AA)
    one_side_result = DynamicROIEngine().update(one_side, fallback_for(one_side))
    assert one_side_result.state == "fallback"
    assert one_side_result.uncertainty is True


def test_shadow_edges_do_not_get_reported_as_reliable_lanes():
    frame = make_road(markings=False, shadows=True)
    result = DynamicROIEngine().update(frame, fallback_for(frame))
    assert result.state in {"fallback", "low_confidence"}
    assert result.uncertainty is True
    assert "evidence" in result.reason.lower() or "ambiguous" in result.reason.lower()


def test_intersection_crosswalk_edges_are_marked_uncertain():
    frame = make_road(markings=False, intersection=True)
    result = DynamicROIEngine().update(frame, fallback_for(frame))
    assert result.state == "fallback"
    assert result.uncertainty is True
    assert "intersection" in result.reason.lower()


def test_resolution_changes_reproject_geometry_and_aspect_changes_reset_it():
    engine = DynamicROIEngine()
    for width, height in ((640, 360), (1280, 720), (800, 600)):
        frame = make_road(width, height, curve=0.12)
        result = engine.update(frame, fallback_for(frame))
        assert result.state in {"detected", "low_confidence"}
        assert result.width == width and result.height == height
        assert result.polygon
        xs = [x for x, _ in result.polygon]
        ys = [y for _, y in result.polygon]
        assert min(xs) >= 0 and max(xs) < width
        assert min(ys) >= 0 and max(ys) < height


def test_temporal_reuse_is_bounded_and_confidence_decays():
    engine = DynamicROIEngine({
        "max_reuse_frames": 2,
        "confidence_decay": 0.75,
        "smoothing": 0.4,
    })
    road = make_road()
    detected = engine.update(road, fallback_for(road))
    assert detected.state == "detected"

    blank = make_road(markings=False)
    first_miss = engine.update(blank, fallback_for(blank))
    second_miss = engine.update(blank, fallback_for(blank))
    expired = engine.update(blank, fallback_for(blank))

    assert first_miss.state == "reused"
    assert second_miss.state == "reused"
    assert first_miss.uncertainty and second_miss.uncertainty
    assert second_miss.confidence < first_miss.confidence < detected.confidence
    assert first_miss.frames_since_detection == 1
    assert second_miss.frames_since_detection == 2
    assert expired.state == "fallback"
    assert expired.polygon == fallback_for(blank)


def test_dynamic_switch_uses_fallback_without_lane_claim():
    frame = make_road()
    fallback = fallback_for(frame)
    result = DynamicROIEngine({"dynamic_enabled": False}).update(frame, fallback)
    assert result.state == "disabled"
    assert result.polygon == fallback
    assert result.uncertainty is True
