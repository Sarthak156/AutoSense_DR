"""Regression tests for independent fallback/search-guide visibility."""

from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pipeline import PerceptionEngine  # noqa: E402
from perception.dynamic_roi import ROIFrameState  # noqa: E402
from perception.roi import DrivingROI  # noqa: E402


class _DrawCapture:
    current_roi = None
    _fps_ema = None

    def __init__(self):
        self.curves = []

    def _draw_roi_curve(self, frame, points, color, dashed=False, closed=False):
        self.curves.append((list(points), dashed, closed))


def _state(state_name="detected"):
    left = [(140, 180), (90, 360)]
    right = [(500, 180), (550, 360)]
    search = [(220, 150), (420, 150), (600, 460), (40, 460)]
    return ROIFrameState(
        polygon=[left[0], right[0], right[-1], left[-1]],
        state=state_name,
        confidence=0.9 if state_name == "detected" else 0.0,
        uncertainty=state_name != "detected",
        reason="test state",
        width=640,
        height=480,
        left_boundary=left if state_name == "detected" else [],
        right_boundary=right if state_name == "detected" else [],
        corners=[left[0], right[0], right[-1], left[-1]],
        search_region=search,
    )


def _draw(state, overlays, static_roi_polygon=None):
    capture = _DrawCapture()
    config = {"overlays": {
        "roi": True,
        "lane": True,
        "static_roi": False,
        "boxes": False,
        "risk": False,
        **overlays,
    }}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = PerceptionEngine._draw(
        capture, frame, config, 640, 480, [], "SAFE", "CONTINUE", None,
        roi_state=state, static_roi_polygon=static_roi_polygon,
    )
    return capture, result


def test_hiding_static_layer_keeps_dynamic_corridor_and_boundaries(monkeypatch):
    fill_calls = []
    original_fill = cv2.fillPoly

    def record_fill(image, polygons, color, *args, **kwargs):
        fill_calls.append(color)
        return original_fill(image, polygons, color, *args, **kwargs)

    monkeypatch.setattr(cv2, "fillPoly", record_fill)
    state = _state("detected")
    capture, _ = _draw(state, {"static_roi": False})

    # Dynamic polygon shading remains, while the separate static ROI is hidden.
    assert len(fill_calls) == 1
    assert [curve[0] for curve in capture.curves] == [
        state.left_boundary, state.right_boundary,
    ]


def test_static_trapezoid_is_drawn_independently_over_dynamic_roi(monkeypatch):
    fill_polygons = []
    original_fill = cv2.fillPoly

    def record_fill(image, polygons, color, *args, **kwargs):
        fill_polygons.append(np.asarray(polygons[0]).reshape(-1, 2).copy())
        return original_fill(image, polygons, color, *args, **kwargs)

    monkeypatch.setattr(cv2, "fillPoly", record_fill)
    static_points = [(10, 100), (630, 100), (630, 470), (10, 470)]
    state = _state("detected")
    _draw(state, {"static_roi": True}, static_roi_polygon=static_points)

    # Static blue fallback is drawn from the configurable DrivingROI polygon;
    # the dynamic corridor is drawn as a separate second polygon.
    assert len(fill_polygons) == 2
    assert [tuple(point) for point in fill_polygons[0]] == static_points
    assert [tuple(point) for point in fill_polygons[1]] == state.polygon


def test_static_slider_geometry_changes_without_moving_dynamic_corridor(monkeypatch):
    fill_polygons = []
    original_fill = cv2.fillPoly

    def record_fill(image, polygons, color, *args, **kwargs):
        fill_polygons.append(np.asarray(polygons[0]).reshape(-1, 2).copy())
        return original_fill(image, polygons, color, *args, **kwargs)

    monkeypatch.setattr(cv2, "fillPoly", record_fill)
    engine = PerceptionEngine.__new__(PerceptionEngine)
    engine.distance_estimator = SimpleNamespace(focal_length=700, smoothing_window=5)
    engine.roi = DrivingROI(
        height_start=0.70, top_width_ratio=0.30,
        bottom_width_ratio=0.75, ego_center_ratio=0.62, bottom_margin=0.02,
    )
    state = _state("detected")

    def apply_static_controls(height_start, top_width, bottom_width, center):
        engine._apply_calibration({
            "focal_length": 700,
            "smoothing": 5,
            "roi": {
                "height_start": height_start,
                "top_width_ratio": top_width,
                "bottom_width_ratio": bottom_width,
                "ego_center_ratio": center,
                "bottom_margin": 0.02,
            },
        })
        return engine.roi.get_polygon(640, 480)

    first_static = apply_static_controls(0.70, 0.30, 0.75, 0.62)
    _draw(state, {"static_roi": True}, static_roi_polygon=first_static)
    first_dynamic = [tuple(point) for point in fill_polygons[-1]]

    fill_polygons.clear()
    second_static = apply_static_controls(0.56, 0.46, 0.92, 0.52)
    _draw(state, {"static_roi": True}, static_roi_polygon=second_static)
    second_dynamic = [tuple(point) for point in fill_polygons[-1]]

    assert first_static != second_static
    assert [tuple(point) for point in fill_polygons[0]] == second_static
    assert first_dynamic == second_dynamic == state.polygon
    assert state.state == "detected"  # static controls do not switch dynamic ROI off


def test_pipeline_defaults_dynamic_search_center_separately_from_static_center(monkeypatch):
    # Avoid starting the background engine thread; this only checks its defaults.
    monkeypatch.setattr(threading.Thread, "start", lambda _thread: None)
    engine = PerceptionEngine("unused-models-dir")

    assert engine.cfg["roi"]["search_center_ratio"] == 0.62
    assert engine.dynamic_roi.config.search_center_ratio == 0.62
    assert engine.cfg["roi"]["ego_center_ratio"] == 0.62


def test_hiding_static_layer_also_hides_fallback_polygon(monkeypatch):
    fill_calls = []
    original_fill = cv2.fillPoly

    def record_fill(image, polygons, color, *args, **kwargs):
        fill_calls.append(color)
        return original_fill(image, polygons, color, *args, **kwargs)

    monkeypatch.setattr(cv2, "fillPoly", record_fill)
    state = _state("fallback")
    capture, _ = _draw(state, {"static_roi": False})

    # The fallback geometry remains in ROIFrameState for calculations, but is
    # omitted from rendering; no search guide or lane boundaries are drawn.
    assert all(not points for points, _dashed, _closed in capture.curves)
    assert not fill_calls
