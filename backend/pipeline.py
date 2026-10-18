"""
AutoSense Perception Engine
============================

Threaded wrapper around the existing Vehicle_perception pipeline modules:

    perception/detector.py   -> YOLO inference (model.predict)
    tracking/tracker.py      -> YOLO + ByteTrack  (model.track)
    perception/distance.py   -> Monocular distance estimation  (DistanceEstimator)
    perception/roi.py        -> Ego-lane driving corridor      (DrivingROI)
    safety/ttc.py            -> Time-to-collision              (TTCEstimator)
    safety/collision.py      -> Rule-based risk levels         (CollisionRisk)
    safety/decision.py       -> CONTINUE / WARNING / BRAKE     (DecisionEngine)

Every metric exposed to the web layer (FPS, inference time, object counts,
distances, TTC, risk, decision) is produced by the real pipeline above.
Nothing is fabricated: if a module is unavailable, the payload says so.
"""

import base64
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from perception.distance import DistanceEstimator
from perception.roi import DrivingROI
from perception.dynamic_roi import DynamicROIEngine, ROIFrameState
from perception.roi_association import ROIObjectAssociator
from safety.ttc import TTCEstimator
from safety.collision import CollisionRisk
from safety.decision import DecisionEngine

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except Exception:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_OK = False


# ---------------------------------------------------------------------------
# Class grouping used by the Live Statistics panel
# ---------------------------------------------------------------------------

GROUP_DEFS = {
    "vehicles": {
        "car", "truck", "bus", "van", "lorry", "trailer", "vehicle",
        "ambulance", "police", "police car", "fire engine", "taxi",
        "auto", "autorickshaw", "auto-rickshaw", "minibus", "pickup",
    },
    "pedestrians": {
        "person", "pedestrian", "people", "man", "woman", "child",
        "pedestrians", "human",
    },
    "two_wheelers": {
        "motorcycle", "motorbike", "bicycle", "bike", "scooter",
        "rider", "moped", "cyclist", "motorcyclist", "two wheeler",
        "two-wheeler",
    },
    "traffic_signs": {
        "traffic sign", "traffic light", "stop sign", "sign",
        "traffic signal", "signal", "speed limit", "speed sign",
        "traffic_sign", "road sign",
    },
}


def classify_group(class_name: str) -> str:
    name = class_name.lower().strip()
    for group, names in GROUP_DEFS.items():
        if name in names:
            return group
    return "other"


RISK_PRIORITY = {"SAFE": 0, "WARNING": 1, "DANGER": 2, "CRITICAL": 3}

# BGR colors (OpenCV) aligned with the AutoSense accent palette
COLOR_ACCENT = (248, 189, 56)      # #38BDF8
COLOR_SAFE = (128, 222, 74)        # #4ADE80
COLOR_WARNING = (36, 191, 251)     # #FBBF24
COLOR_DANGER = (60, 147, 251)      # #FB923C
COLOR_CRITICAL = (113, 113, 248)   # #F87171
COLOR_OUTSIDE = (150, 150, 150)    # gray


def _placeholder_frame(text, sub=""):
    """Dark engineering placeholder published when there is no input."""
    w, h = 960, 540
    frame = np.full((h, w, 3), (32, 22, 12), dtype=np.uint8)  # deep navy
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (60, 45, 25), 1)
    cv2.putText(frame, "AUTOSENSE", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, text, ((w - tw) // 2, h // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 230), 2, cv2.LINE_AA)
    if sub:
        (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, sub, ((w - sw) // 2, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 150, 165), 1, cv2.LINE_AA)
    return frame


class PerceptionEngine:
    """Owns the capture thread, the pipeline modules and the metrics."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir

        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()

        self.latest_jpeg = None
        self.jpeg_seq = 0

        # ---------------- runtime (only touched by engine thread) ----------
        self.model = None
        self.model_error = None
        self._loaded_model_name = None
        self._loaded_device = None
        self.cap = None
        self.source_open_error = None

        self.distance_estimator = DistanceEstimator(focal_length=700, smoothing_window=5)
        self.roi = DrivingROI(height_start=0.7, top_width_ratio=0.3,
                              bottom_width_ratio=0.75, ego_center_ratio=0.62,
                              bottom_margin=0.02)
        self.roi_associator = ROIObjectAssociator()
        self.current_roi = None
        self.ttc_estimator = TTCEstimator(history_size=8)
        self.risk_engine = CollisionRisk()
        self.decision_engine = DecisionEngine()

        self.track_registry = {}          # id -> {first_seen, last_seen, seen_frames}
        self.recent = deque(maxlen=12)    # recent detections w/ thumbnails
        self.fps_history = deque(maxlen=180)
        self.risk_history = deque(maxlen=180)

        self.frames_processed = 0
        self.started_at = time.time()
        self.session_detections = 0

        # ---------------- public config (guarded by lock) ------------------
        self.cfg = {
            "source": None,          # {"type": "video"|"camera", "path"|"index", "name"}
            "playing": False,
            "paused": False,
            "loop": True,
            "conf": 0.35,
            "iou": 0.45,
            "tracking": True,
            "tracker": "bytetrack.yaml",
            "imgsz": 640,
            "device": "auto",        # "auto" | "gpu" | "cpu"  (gpu needs the CUDA torch build)
            "model_name": "yolo11n.pt",
            "focal_length": 700,
            "smoothing": 5,
            "roi": {
                # Existing static trapezoid remains the user-configurable fallback.
                "height_start": 0.70,
                "top_width_ratio": 0.30,
                "bottom_width_ratio": 0.75,
                "ego_center_ratio": 0.62,
                "bottom_margin": 0.02,
                # Training-free classical-CV search / temporal controls.
                "dynamic_enabled": True,
                "search_center_ratio": 0.62,
                "search_top_ratio": 0.50,
                "search_top_width_ratio": 0.30,
                "search_bottom_width_ratio": 0.88,
                "canny_low": 50,
                "canny_high": 150,
                "hough_threshold": 24,
                "min_confidence": 0.30,
                "reliable_confidence": 0.56,
                "smoothing": 0.35,
                "max_reuse_frames": 8,
                "confidence_decay": 0.82,
                "expected_lane_width_ratio": 0.44,
                "center_tolerance_ratio": 0.30,
            },
            "overlays": {
                "boxes": True,
                "confidence": True,
                "ids": True,
                "distance": True,
                "roi": True,
                "lane": True,
                "static_roi": True,
                "search_region": False,
                "risk": False,
            },
        }

        self.dynamic_roi = DynamicROIEngine(self.cfg["roi"])
        self.state = "starting"      # idle | running | paused | error
        self.state_detail = ""

        self.stats = self._base_stats()

        self._fps_ema = None
        self._last_frame_completed = None
        self._last_inference_ms = None
        self._last_processing_ms = None
        self._last_roi_ms = None
        self._last_detections = []
        self._last_frame_shape = None

        self.thread = threading.Thread(target=self._loop, daemon=True, name="engine")
        self.thread.start()

    # ------------------------------------------------------------------
    # Config / control API (called from FastAPI handlers)
    # ------------------------------------------------------------------

    def update_config(self, patch: dict):
        """Deep-merge a partial config dict."""
        with self.lock:
            self._deep_merge(self.cfg, patch)
        return self.get_config()

    @staticmethod
    def _deep_merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                PerceptionEngine._deep_merge(dst[k], v)
            else:
                dst[k] = v

    def get_config(self):
        with self.lock:
            import copy
            return copy.deepcopy(self.cfg)

    def _clear_histories(self):
        self.distance_estimator.clear_all()
        self.ttc_estimator.clear()
        self.track_registry.clear()
        self.roi_associator.reset()
        self.dynamic_roi.reset()
        self.current_roi = None
        self.session_detections = 0

    def set_video_source(self, path, name=None, start=True):
        with self.lock:
            self.cfg["source"] = {"type": "video", "path": path,
                                  "name": name or os.path.basename(path)}
            self.cfg["playing"] = start
            self.cfg["paused"] = False
        self._reset_capture()

    def set_camera_source(self, index: int, start=True):
        with self.lock:
            self.cfg["source"] = {"type": "camera", "index": int(index),
                                  "name": f"Camera {index}"}
            self.cfg["playing"] = start
            self.cfg["paused"] = False
        self._reset_capture()

    def _reset_capture(self):
        """Called with new source; engine thread re-opens capture itself."""
        with self.lock:
            self._want_reopen = True
            self._clear_histories()
            self.frames_processed = 0
            self._fps_ema = None
            self._last_frame_completed = None

    def pause(self):
        with self.lock:
            self.cfg["paused"] = True

    def resume(self):
        with self.lock:
            self.cfg["paused"] = False
            self._last_frame_completed = None

    def start(self):
        with self.lock:
            self.cfg["playing"] = True
            self.cfg["paused"] = False
            self._last_frame_completed = None

    def stop(self):
        with self.lock:
            self.cfg["playing"] = False
            self.cfg["source"] = None
            self._want_reopen = True
        self._clear_histories()

    def restart(self):
        self._want_reopen = True
        self._clear_histories()
        self._fps_ema = None
        self._last_frame_completed = None

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def model_path(self, name):
        return os.path.join(self.models_dir, name)

    def available_models(self):
        out = []
        if os.path.isdir(self.models_dir):
            for f in sorted(os.listdir(self.models_dir)):
                if f.endswith(".pt"):
                    p = os.path.join(self.models_dir, f)
                    out.append({
                        "name": f,
                        "size_mb": round(os.path.getsize(p) / 1e6, 2),
                        "active": f == self.get_config()["model_name"],
                    })
        return out

    # ------------------------------------------------------------------
    # Engine thread
    # ------------------------------------------------------------------

    def _loop(self):
        self._want_reopen = True
        while True:
            cfg = self.get_config()

            # ---------- make sure model is loaded ----------
            if not ULTRALYTICS_OK:
                self.state = "error"
                self.state_detail = "ultralytics not installed"
                self._publish_placeholder("MODEL RUNTIME UNAVAILABLE",
                                          "pip install ultralytics")
                self._push_stats(cfg, [], None)
                time.sleep(1.0)
                continue

            if (self.model is None or self._loaded_model_name != cfg["model_name"]
                    or self._loaded_device != cfg["device"]):
                try:
                    path = self.model_path(cfg["model_name"])
                    self.model = YOLO(path)
                    self._loaded_model_name = cfg["model_name"]
                    self._loaded_device = cfg["device"]
                    self.model_error = None
                    try:
                        import torch
                        print(f"[AutoSense] {cfg['model_name']} loaded — torch {torch.__version__}, "
                              f"CUDA {'available' if torch.cuda.is_available() else 'NOT available (CPU-only torch build?)'}",
                              flush=True)
                    except Exception:
                        pass
                    self._clear_histories()
                    # reset ByteTrack state when model changes
                    try:
                        self.model.predictor = None
                    except Exception:
                        pass
                except Exception as exc:  # noqa
                    self.model = None
                    self.model_error = str(exc)
                    self.state = "error"
                    self.state_detail = f"Model load failed: {exc}"
                    self._publish_placeholder("MODEL NOT LOADED", cfg["model_name"])
                    self._push_stats(cfg, [], None)
                    time.sleep(1.0)
                    continue

            # ---------- need a source ----------
            if not cfg["playing"] or cfg["source"] is None:
                self.state = "idle"
                self.state_detail = ""
                self._close_cap()
                self._publish_placeholder("NO INPUT SOURCE",
                                          "Load a video or start the camera")
                self._push_stats(cfg, [], None)
                time.sleep(0.25)
                continue

            # ---------- open / reopen capture ----------
            if self.cap is None or self._want_reopen:
                self._want_reopen = False
                self._close_cap()
                src = cfg["source"]
                target = src["path"] if src["type"] == "video" else int(src["index"])
                cap = cv2.VideoCapture(target)
                if not cap.isOpened():
                    cap.release()
                    self.cap = None
                    self.state = "error"
                    self.state_detail = f"Could not open source: {src['name']}"
                    self._publish_placeholder("SOURCE NOT AVAILABLE", src["name"])
                    self._push_stats(cfg, [], None)
                    time.sleep(1.0)
                    continue
                self.cap = cap
                self.source_open_error = None
                self._clear_histories()
                self._last_frame_completed = None

            # ---------- paused ----------
            if cfg["paused"]:
                self.state = "paused"
                self._push_stats(
                    cfg, self._last_detections, self._last_frame_shape,
                    infer_ms=self._last_inference_ms,
                    processing_ms=self._last_processing_ms,
                    roi_ms=self._last_roi_ms,
                )
                time.sleep(0.15)
                continue

            # ---------- read frame ----------
            frame_start = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if cfg["source"]["type"] == "video" and cfg["loop"]:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._clear_histories()
                    continue
                self.state = "error" if cfg["source"]["type"] == "camera" else "idle"
                self.state_detail = ("Camera frame grab failed"
                                     if cfg["source"]["type"] == "camera"
                                     else "Video finished")
                self._close_cap()
                self._publish_placeholder(
                    "STREAM ENDED" if self.state == "idle" else "CAMERA ERROR",
                    self.state_detail)
                self._push_stats(cfg, [], None)
                time.sleep(0.3)
                continue

            height, width = frame.shape[:2]

            # ---------- rebuild fallback geometry / classical ROI -----------
            self._apply_calibration(cfg)
            fallback_polygon = self.roi.get_polygon(width, height)
            roi_t0 = time.perf_counter()
            roi_state = self.dynamic_roi.update(
                frame,
                fallback_polygon=fallback_polygon,
                config=cfg["roi"],
            )
            roi_ms = (time.perf_counter() - roi_t0) * 1000.0
            self.current_roi = roi_state

            # ---------- YOLO11 inference (detection or detection+tracking) ---
            t0 = time.perf_counter()
            try:
                dev = self._device_arg(cfg["device"])
                if cfg["tracking"]:
                    results = self.model.track(
                        frame, persist=True, tracker=cfg["tracker"],
                        conf=cfg["conf"], iou=cfg["iou"], imgsz=cfg["imgsz"],
                        device=dev, verbose=False)
                else:
                    results = self.model.predict(
                        frame, conf=cfg["conf"], iou=cfg["iou"],
                        imgsz=cfg["imgsz"], device=dev, verbose=False)
            except Exception as exc:  # noqa
                self.state = "error"
                self.state_detail = f"Inference failed: {exc}"
                self._last_inference_ms = None
                self._last_processing_ms = None
                self._last_roi_ms = roi_ms
                self._push_stats(
                    cfg, [], (width, height),
                    processing_ms=(time.perf_counter() - frame_start) * 1000.0,
                    roi_ms=roi_ms,
                )
                time.sleep(0.4)
                continue
            infer_ms = (time.perf_counter() - t0) * 1000.0
            result = results[0]

            # ---------- per-detection pipeline ----------
            timestamp = time.time()
            detections = []
            counts = {"vehicles": 0, "pedestrians": 0, "two_wheelers": 0,
                      "traffic_signs": 0, "other": 0}
            distribution = {}
            closest_distance = None
            overall_risk = "SAFE"
            min_ttc = None

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
                names = result.names if hasattr(result, "names") else self.model.names

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                    bbox = [x1, y1, x2, y2]
                    class_id = int(clss[i])
                    class_name = str(names[class_id])
                    conf = float(confs[i])
                    track_id = int(ids[i]) if ids is not None else None

                    group = classify_group(class_name)
                    counts[group] += 1
                    distribution[class_name] = distribution.get(class_name, 0) + 1

                    distance = self.distance_estimator.estimate(bbox, class_name, track_id)
                    roi_association = self.roi_associator.associate(
                        bbox=bbox,
                        polygon=roi_state.polygon,
                        image_shape=frame.shape,
                        class_name=class_name,
                        confidence=conf,
                        track_id=track_id,
                        roi_state=roi_state.state,
                        roi_confidence=roi_state.confidence,
                        roi_uncertainty=roi_state.uncertainty,
                    )
                    in_path = roi_association["in_path"]

                    if in_path and distance is not None:
                        if closest_distance is None or distance < closest_distance:
                            closest_distance = distance

                    ttc_data = self.ttc_estimator.update(track_id, distance, timestamp)
                    ttc, closing_speed = None, 0.0
                    if ttc_data is not None:
                        ttc = ttc_data.get("ttc")
                        closing_speed = ttc_data.get("closing_speed", 0.0) or 0.0

                    if in_path:
                        risk = self.risk_engine.evaluate(
                            distance=distance, ttc=ttc,
                            closing_speed=closing_speed, in_path=True,
                            class_name=class_name)
                    else:
                        risk = "SAFE"

                    if RISK_PRIORITY[risk] > RISK_PRIORITY[overall_risk]:
                        overall_risk = risk
                    if ttc is not None:
                        if min_ttc is None or ttc < min_ttc:
                            min_ttc = ttc

                    if track_id is not None:
                        reg = self.track_registry.setdefault(
                            track_id, {"first_seen": timestamp, "seen_frames": 0})
                        reg["last_seen"] = timestamp
                        reg["seen_frames"] += 1
                        reg["class_name"] = class_name
                        reg["distance"] = distance
                        reg["risk"] = risk
                        reg["in_path"] = bool(in_path)
                        reg["roi_relevance"] = roi_association["roi_relevance"]
                        reg["roi_label"] = roi_association["roi_label"]

                    detections.append({
                        "track_id": track_id,
                        "class_name": class_name,
                        "group": group,
                        "confidence": round(conf, 3),
                        "bbox": bbox,
                        "distance": distance,
                        "ttc": ttc,
                        "closing_speed": round(float(closing_speed), 2),
                        "risk": risk,
                        "in_path": bool(in_path),
                        **{key: value for key, value in roi_association.items()
                           if key != "in_path"},
                    })

            self.session_detections += len(detections)
            decision = self.decision_engine.decide(overall_risk)

            # ---------- thumbnails for top detections (real frame crops) ----
            ranked = sorted(detections, key=lambda d: d["confidence"], reverse=True)
            for det in ranked[:5]:
                det["thumb"] = self._crop_thumb(frame, det["bbox"])
            self._update_recent(frame, detections, timestamp)

            # ---------- draw overlays ----------
            annotated = self._draw(frame.copy(), cfg, width, height, detections,
                                   overall_risk, decision, closest_distance,
                                   roi_state=roi_state,
                                   static_roi_polygon=fallback_polygon)

            # ---------- encode + publish frame ----------
            ok_enc, buf = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                with self.frame_cond:
                    self.latest_jpeg = buf.tobytes()
                    self.jpeg_seq += 1
                    self.frame_cond.notify_all()

            # ---------- measured latency, output FPS, source pacing ----------
            # Processing latency covers capture read through inference, ROI,
            # drawing and JPEG encode. It excludes intentional video pacing.
            processing_ms = (time.perf_counter() - frame_start) * 1000.0
            self._last_inference_ms = infer_ms
            self._last_processing_ms = processing_ms
            self._last_roi_ms = roi_ms

            try:
                raw_source_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap is not None else 0.0
            except (TypeError, ValueError):
                raw_source_fps = 0.0
            src_fps = raw_source_fps if 1.0 <= raw_source_fps <= 240.0 else None
            if cfg["source"]["type"] == "video" and src_fps is not None:
                frame_budget = 1.0 / src_fps
                elapsed = time.perf_counter() - frame_start
                if elapsed < frame_budget:
                    time.sleep(frame_budget - elapsed)

            # Throughput is measured between completed frames, so video pacing
            # and real camera cadence are represented. No assumed 25/30 FPS.
            completed_at = time.perf_counter()
            if self._last_frame_completed is not None:
                interval = completed_at - self._last_frame_completed
                if interval > 1e-6:
                    inst_fps = 1.0 / interval
                    self._fps_ema = inst_fps if self._fps_ema is None else (
                        0.75 * self._fps_ema + 0.25 * inst_fps
                    )
                    self.fps_history.append(round(self._fps_ema, 2))
            self._last_frame_completed = completed_at

            self.frames_processed += 1
            self.risk_history.append(overall_risk)

            self.state = "running"
            self.state_detail = ""
            self._last_detections = detections
            self._last_frame_shape = (width, height)
            self._push_stats(cfg, detections, (width, height), infer_ms=infer_ms,
                             processing_ms=processing_ms, roi_ms=roi_ms,
                             overall_risk=overall_risk, decision=decision,
                             closest_distance=closest_distance, min_ttc=min_ttc,
                             counts=counts, distribution=distribution)

    # ------------------------------------------------------------------

    def _apply_calibration(self, cfg):
        if self.distance_estimator.focal_length != cfg["focal_length"]:
            self.distance_estimator.focal_length = cfg["focal_length"]
        if self.distance_estimator.smoothing_window != cfg["smoothing"]:
            self.distance_estimator.smoothing_window = cfg["smoothing"]
            self.distance_estimator.clear_all()
            self.distance_estimator.history = __import__(
                "collections").defaultdict(
                    lambda: __import__("collections").deque(maxlen=cfg["smoothing"]))
        r = cfg["roi"]
        roi = self.roi
        changed = (
            roi.height_start != r["height_start"] or
            roi.top_width_ratio != r["top_width_ratio"] or
            roi.bottom_width_ratio != r["bottom_width_ratio"] or
            roi.ego_center_ratio != r["ego_center_ratio"] or
            roi.bottom_margin != r["bottom_margin"]
        )
        if changed:
            self.roi = DrivingROI(
                height_start=r["height_start"],
                top_width_ratio=r["top_width_ratio"],
                bottom_width_ratio=r["bottom_width_ratio"],
                ego_center_ratio=r["ego_center_ratio"],
                bottom_margin=r["bottom_margin"])

    def _close_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self._last_detections = []
        self._last_frame_shape = None

    _last_detections = []
    _last_frame_shape = None

    def _publish_placeholder(self, text, sub=""):
        frame = _placeholder_frame(text, sub)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self.frame_cond:
                self.latest_jpeg = buf.tobytes()
                self.jpeg_seq += 1
                self.frame_cond.notify_all()

    # ------------------------------------------------------------------
    # Drawing (adapts visualization/overlay.py logic to toggleable layers)
    # ------------------------------------------------------------------

    def _draw(self, frame, cfg, width, height, detections,
              overall_risk, decision, closest_distance, roi_state=None,
              static_roi_polygon=None):
        ov = cfg["overlays"]
        roi_state = roi_state or self.current_roi
        if roi_state is None:
            polygon = self.roi.get_polygon(width, height)
            roi_state = ROIFrameState(
                polygon=polygon, state="fallback", confidence=0.0,
                uncertainty=True, reason="No measured lane geometry yet",
                width=width, height=height, corners=polygon,
            )
        if static_roi_polygon is None:
            if hasattr(self, "roi"):
                static_roi_polygon = self.roi.get_polygon(width, height)
            elif roi_state.state in {"fallback", "disabled"}:
                static_roi_polygon = roi_state.polygon
            else:
                static_roi_polygon = []

        # The fixed blue trapezoid is always generated from DrivingROI config;
        # it is drawn independently of the dynamic corridor/search mask.
        static_poly = np.asarray(static_roi_polygon, dtype=np.int32).reshape(-1, 1, 2)
        if ov.get("static_roi", True) and len(static_poly) >= 3:
            shade = frame.copy()
            cv2.fillPoly(shade, [static_poly], COLOR_ACCENT)
            frame = cv2.addWeighted(shade, 0.035, frame, 0.965, 0)
            cv2.polylines(frame, [static_poly], True, COLOR_ACCENT, 2, cv2.LINE_AA)

        # The dynamic corridor is a separate layer. The fallback state is not
        # drawn here because its geometry is already represented by static_poly.
        roi_poly = np.asarray(roi_state.polygon, dtype=np.int32).reshape(-1, 1, 2)
        uncertain_color = COLOR_WARNING
        roi_color = (COLOR_ACCENT if not roi_state.uncertainty else uncertain_color)
        is_static_fallback = roi_state.state in {"fallback", "disabled"}
        if ov.get("roi") and not is_static_fallback and len(roi_poly) >= 3:
            shade = frame.copy()
            cv2.fillPoly(shade, [roi_poly], roi_color)
            frame = cv2.addWeighted(shade, 0.07, frame, 0.93, 0)
            cv2.polylines(frame, [roi_poly], True, roi_color, 2, cv2.LINE_AA)

        if ov.get("lane"):
            line_color = COLOR_SAFE if not roi_state.uncertainty else uncertain_color
            dashed = roi_state.state in {"reused", "fallback", "disabled", "low_confidence"}
            self._draw_roi_curve(frame, roi_state.left_boundary, line_color, dashed)
            self._draw_roi_curve(frame, roi_state.right_boundary, line_color, dashed)

            if roi_state.state == "detected":
                roi_text = f"ROI DETECTED  {roi_state.confidence:.2f}"
            elif roi_state.state == "low_confidence":
                roi_text = f"ROI LOW CONFIDENCE  {roi_state.confidence:.2f}"
            elif roi_state.state == "reused":
                roi_text = f"ROI REUSED {roi_state.frames_since_detection}  {roi_state.confidence:.2f}"
            elif roi_state.state == "disabled":
                roi_text = "ROI DISABLED - FALLBACK ACTIVE"
            else:
                roi_text = "ROI UNCERTAIN - FALLBACK ACTIVE"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(roi_text, font, 0.48, 1)
            bx, by = 12, 28
            cv2.rectangle(frame, (bx, by - th - 8), (bx + tw + 18, by + 6), (14, 20, 28), -1)
            cv2.rectangle(frame, (bx, by - th - 8), (bx + tw + 18, by + 6), line_color, 1)
            cv2.putText(frame, roi_text, (bx + 8, by), font, 0.48,
                        (240, 244, 248), 1, cv2.LINE_AA)

        # Optional diagnostic outline of the estimator's perspective search mask.
        if ov.get("search_region", False) and roi_state.search_region:
            self._draw_roi_curve(
                frame, roi_state.search_region, COLOR_ACCENT, True, closed=True
            )

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            in_path = det["in_path"]
            risk = det["risk"]
            is_uncertain = det.get("roi_uncertainty", False)

            if ov.get("risk"):
                if is_uncertain:
                    color = COLOR_WARNING
                elif not in_path:
                    color = COLOR_OUTSIDE
                elif risk == "CRITICAL":
                    color = COLOR_CRITICAL
                elif risk == "DANGER":
                    color = COLOR_DANGER
                elif risk == "WARNING":
                    color = COLOR_WARNING
                else:
                    color = COLOR_SAFE
            else:
                color = COLOR_WARNING if is_uncertain else (
                    COLOR_ACCENT if in_path or ov.get("roi") else COLOR_SAFE
                )

            if ov.get("boxes"):
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Keep the detector's class/confidence and explicitly label every
            # detection's corridor relevance, including objects outside ROI.
            parts = [det["class_name"].upper()]
            if ov.get("ids") and det["track_id"] is not None:
                parts[0] += f" #{det['track_id']}"
            if ov.get("confidence"):
                parts.append(f"{det['confidence']:.2f}")
            if ov.get("distance") and det["distance"] is not None:
                parts.append(f"{det['distance']:.1f}m")
            if ov.get("risk") and in_path and det["ttc"] is not None:
                parts.append(f"TTC {det['ttc']:.1f}s")
            relevance = det.get("roi_relevance", "uncertain")
            parts.append({
                "in_corridor": "ROI IN",
                "near_corridor": "ROI NEAR",
                "outside_corridor": "ROI OUT",
                "uncertain": "ROI UNCERTAIN",
            }.get(relevance, "ROI UNCERTAIN"))
            label = " | ".join(parts)

            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            ly = y1 - 8 if y1 - th - 14 > 0 else y2 + th + 10
            lx = max(2, min(x1, width - tw - 10))
            cv2.rectangle(frame, (lx - 2, ly - th - 6), (lx + tw + 6, ly + 4),
                          (20, 26, 18), -1)
            cv2.rectangle(frame, (lx - 2, ly - th - 6), (lx + tw + 6, ly + 4),
                          color, 1)
            cv2.putText(frame, label, (lx + 2, ly), font, scale,
                        (235, 240, 245), thick, cv2.LINE_AA)

        if ov.get("risk"):
            # compact HUD (adapted from SafetyOverlay.draw_dashboard)
            x, y, w, h = 14, 48 if ov.get("lane") else 14, 300, 96
            hud = frame.copy()
            cv2.rectangle(hud, (x, y), (x + w, y + h), (30, 24, 12), -1)
            frame = cv2.addWeighted(hud, 0.75, frame, 0.25, 0)
            rc = {"SAFE": COLOR_SAFE, "WARNING": COLOR_WARNING,
                  "DANGER": COLOR_DANGER}.get(overall_risk, COLOR_CRITICAL)
            cv2.putText(frame, f"RISK  {overall_risk}", (x + 12, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc, 2, cv2.LINE_AA)
            cv2.putText(frame, f"ACTION  {decision}", (x + 12, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
                        cv2.LINE_AA)
            cd = f"{closest_distance:.1f}m" if closest_distance is not None else "--"
            fps = f"{self._fps_ema:.1f}" if self._fps_ema is not None else "--"
            cv2.putText(
                frame,
                f"FPS {fps}  OBJ {len(detections)}  CLOSE {cd}",
                (x + 12, y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (170, 180, 190), 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def _draw_roi_curve(frame, points, color, dashed=False, closed=False):
        if not points or len(points) < 2:
            return
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
        pairs = list(zip(pts[:-1], pts[1:]))
        if closed:
            pairs.append((pts[-1], pts[0]))
        for index, (start, end) in enumerate(pairs):
            if dashed and index % 3 == 2:
                continue
            cv2.line(frame, tuple(start), tuple(end), color, 2, cv2.LINE_AA)


    # ------------------------------------------------------------------
    # Thumbnails & recent list
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_thumb(frame, bbox, size=96, pad=0.25):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        sq = np.full((side, side, 3), (32, 22, 12), dtype=np.uint8)
        oy, ox = (side - ch) // 2, (side - cw) // 2
        sq[oy:oy + ch, ox:ox + cw] = crop
        sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", sq, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    def _update_recent(self, frame, detections, ts):
        """Recent detections panel — deduped by track, newest first."""
        now_new = []
        for det in detections:
            key = det["track_id"]
            if key is None:
                continue
            entry = {
                "track_id": key,
                "class_name": det["class_name"],
                "group": det["group"],
                "confidence": det["confidence"],
                "distance": det["distance"],
                "risk": det["risk"],
                "ts": ts,
            }
            thumb = det.get("thumb") or self._crop_thumb(frame, det["bbox"])
            if thumb:
                entry["thumb"] = thumb
                now_new.append(entry)

        # update existing entries in place, move touched to front
        fresh = []
        remainder = list(self.recent)
        for entry in now_new:
            remainder = [e for e in remainder if e["track_id"] != entry["track_id"]]
            fresh.append(entry)
        self.recent.clear()
        for e in fresh + remainder:
            self.recent.append(e)

    # ------------------------------------------------------------------
    # Stats snapshot
    # ------------------------------------------------------------------

    def _device_arg(self, want):
        """Map the device setting ("auto" | "gpu" | "cpu") to an ultralytics
        `device=` override. None = ultralytics default (GPU when available).
        Requesting the GPU without a working CUDA torch falls back to auto."""
        if want == "cpu":
            return "cpu"
        if want == "gpu":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
            except Exception:
                pass
        return None

    def _device_label(self):
        """Actual compute device running inference: 'GPU (name)' or 'CPU'."""
        if self.model is not None:
            try:
                import torch
                pred = getattr(self.model, "predictor", None)
                dev = getattr(pred, "device", None) if pred is not None else None
                if dev is None:
                    dev = next(self.model.model.parameters()).device
                if getattr(dev, "type", "") == "cuda":
                    name = torch.cuda.get_device_name(dev.index or 0)
                    return f"GPU ({name.replace('NVIDIA GeForce ', '')})"
            except Exception:
                pass
        return "CPU"

    def _base_stats(self):
        return {}

    def _push_stats(self, cfg, detections, frame_shape, infer_ms=None,
                    processing_ms=None, roi_ms=None,
                    overall_risk="SAFE", decision="CONTINUE",
                    closest_distance=None, min_ttc=None,
                    counts=None, distribution=None):
        cfg = cfg or self.get_config()
        now = time.time()

        if frame_shape:
            width, height = frame_shape
        elif self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        else:
            width = height = 0

        src_fps, pos_s, dur_s, total_frames = None, 0.0, 0.0, 0
        if self.cap is not None and cfg["source"]:
            try:
                reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            except (TypeError, ValueError):
                reported_fps = 0.0
            if 1.0 <= reported_fps <= 240.0:
                src_fps = reported_fps
            if cfg["source"]["type"] == "video":
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if src_fps is not None:
                    pos_s = (self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) / src_fps
                    dur_s = total_frames / src_fps

        counts = counts or {"vehicles": 0, "pedestrians": 0, "two_wheelers": 0,
                            "traffic_signs": 0, "other": 0}
        distribution = distribution or {}

        recent_list = list(self.recent)[:8]
        if not recent_list and detections:
            recent_list = [
                {"track_id": d["track_id"], "class_name": d["class_name"],
                 "group": d["group"], "confidence": d["confidence"],
                 "distance": d["distance"], "risk": d["risk"], "ts": now,
                 "thumb": d.get("thumb")}
                for d in detections[:8]
            ]

        tracks = []
        for tid, reg in sorted(self.track_registry.items()):
            if now - reg.get("last_seen", 0) > 2.0:
                continue
            tracks.append({
                "id": tid,
                "class_name": reg.get("class_name", "?"),
                "distance": reg.get("distance"),
                "risk": reg.get("risk", "SAFE"),
                "in_path": reg.get("in_path", False),
                "roi_relevance": reg.get("roi_relevance", "uncertain"),
                "roi_label": reg.get("roi_label", "UNCERTAIN"),
                "age_s": round(now - reg["first_seen"], 1),
                "seen_frames": reg["seen_frames"],
            })

        model_classes = []
        if self.model is not None:
            try:
                model_classes = [str(v) for _, v in sorted(self.model.names.items())]
            except Exception:
                model_classes = []

        roi_state = self.current_roi
        if width and height and (
            roi_state is None or roi_state.width != width or roi_state.height != height
        ):
            fallback = self.roi.get_polygon(width, height)
            roi_state = ROIFrameState(
                polygon=fallback, state="fallback", confidence=0.0,
                uncertainty=True, reason="Waiting for a processed frame; fallback ROI shown",
                width=width, height=height, corners=fallback,
            )
        roi_poly = roi_state.to_dict() if roi_state is not None and width and height else None
        static_points = self.roi.get_polygon(width, height) if width and height else []
        static_roi_poly = ({
            "points": [[int(x), int(y)] for x, y in static_points],
            "corners": [[int(x), int(y)] for x, y in static_points],
            "width": int(width),
            "height": int(height),
        } if static_points else None)
        roi_info = ({
            "state": roi_state.state,
            "confidence": round(float(roi_state.confidence), 3),
            "uncertainty": bool(roi_state.uncertainty),
            "reason": roi_state.reason,
            "source": roi_state.source,
            "candidate_count": int(roi_state.candidate_count),
            "frames_since_detection": int(roi_state.frames_since_detection),
        } if roi_state is not None else {
            "state": "unavailable", "confidence": 0.0, "uncertainty": True,
            "reason": "No video frame is available", "source": "fallback_or_reuse",
            "candidate_count": 0, "frames_since_detection": 0,
        })

        roi_state_name = roi_info["state"]
        if roi_state_name == "detected":
            roi_module_sub = f"Confidence {roi_info['confidence']:.2f}"
        elif roi_state_name == "low_confidence":
            roi_module_sub = f"Weak evidence · {roi_info['confidence']:.2f}"
        elif roi_state_name == "reused":
            roi_module_sub = (
                f"Reusing geometry · {roi_info['frames_since_detection']} frame(s)"
            )
        elif roi_state_name == "fallback":
            reason = str(roi_info.get("reason") or "Lane evidence unavailable")
            if "waiting for a processed frame" in reason.lower():
                roi_module_sub = "Waiting for a processed frame"
            elif "intersection" in reason.lower():
                roi_module_sub = "Intersection-like edges · fallback"
            else:
                roi_module_sub = "Fallback active · uncertain geometry"
        elif roi_state_name == "disabled":
            roi_module_sub = "Dynamic ROI off · fallback active"
        else:
            roi_module_sub = "Waiting for a processed frame"

        self.stats = {
            "state": self.state,
            "detail": self.state_detail,
            "mode": {"running": "live", "paused": "paused",
                     "idle": "idle", "error": "error",
                     "starting": "idle"}.get(self.state, "idle"),
            "time": now,
            "source": {
                **(cfg["source"] or {"type": None, "name": None}),
                "width": width, "height": height,
                "fps": round(src_fps, 1) if src_fps is not None else None,
                "position_s": round(pos_s, 1),
                "duration_s": round(dur_s, 1),
                "total_frames": total_frames,
            },
            "model": {
                "name": cfg["model_name"],
                "loaded": self.model is not None,
                "error": self.model_error,
                "classes": model_classes,
                "device": self._device_label(),
            },
            "config": cfg,
            "perf": {
                "fps": round(self._fps_ema, 1) if self._fps_ema is not None else None,
                "inference_ms": round(infer_ms, 1) if infer_ms is not None else None,
                "processing_ms": round(processing_ms, 1) if processing_ms is not None else None,
                "roi_ms": round(roi_ms, 1) if roi_ms is not None else None,
                "input_w": cfg["imgsz"], "input_h": cfg["imgsz"],
                "frames_processed": self.frames_processed,
                "uptime_s": round(now - self.started_at, 0),
                "session_detections": self.session_detections,
            },
            "counts": {**counts, "total": sum(counts.values())},
            "distribution": distribution,
            # Never discard detections because they fall outside the ROI.
            "detections": detections,
            "detection_total": len(detections),
            "tracks": tracks,
            "safety": {
                "overall_risk": overall_risk,
                "decision": decision,
                "closest_distance": closest_distance,
                "min_ttc": min_ttc,
                "thresholds": {
                    "distance": {
                        "critical": self.risk_engine.distance_critical,
                        "danger": self.risk_engine.distance_danger,
                        "warning": self.risk_engine.distance_warning,
                    },
                    "ttc": {
                        "critical": self.risk_engine.ttc_critical,
                        "danger": self.risk_engine.ttc_danger,
                        "warning": self.risk_engine.ttc_warning,
                    },
                },
            },
            "recent": recent_list,
            "roi": roi_info,
            "roi_polygon": roi_poly,
            "static_roi_polygon": static_roi_poly,
            "fps_history": list(self.fps_history)[-120:],
            "risk_history": list(self.risk_history)[-120:],
            "modules": {
                "model": {
                    "label": "Model",
                    "value": cfg["model_name"].replace(".pt", ""),
                    "status": ("ok" if self.model is not None else "err"),
                    "sub": "Loaded" if self.model is not None else "Not loaded",
                },
                "roi": {
                    "label": "Dynamic ROI",
                    "value": roi_info["state"].replace("_", " ").title(),
                    "status": ("ok" if roi_info["state"] == "detected" and not roi_info["uncertainty"]
                              else "off" if roi_info["state"] == "disabled" else "warn"),
                    "sub": roi_module_sub,
                    "title": roi_info.get("reason", "Classical-CV ROI state"),
                },
                "tracking": {
                    "label": "Tracking",
                    "value": ("ByteTrack" if "bytetrack" in cfg["tracker"]
                              else "BoT-SORT") if cfg["tracking"] else "Disabled",
                    "status": "ok" if cfg["tracking"] else "off",
                    "sub": ("Active" if self.state == "running"
                            else ("Paused" if self.state == "paused" else "Standby"))
                    if cfg["tracking"] else "Inactive",
                },
                "source": {
                    "label": "Video Source",
                    "value": (cfg["source"] or {}).get("name") or "None",
                    "status": ("ok" if self.state == "running"
                               else "warn" if self.state == "paused"
                               else "err" if self.state == "error" else "off"),
                    "sub": {"running": "Running", "paused": "Paused",
                            "error": "Error", "idle": "Idle",
                            "starting": "Starting"}.get(self.state, "Idle"),
                },
                "safety": {
                    "label": "Safety Module",
                    "value": "TTC + Collision",
                    "status": ("ok" if self.state == "running"
                               else "warn" if self.state == "paused" else "off"),
                    "sub": "Active" if self.state == "running"
                           else ("Paused" if self.state == "paused" else "Standby"),
                },
            },
            "ego_speed": None,  # no speed sensor connected — UI shows "not available"
        }

    def get_stats(self):
        return self.stats

    # ------------------------------------------------------------------

    def get_report(self):
        s = self.get_stats()
        cfg = self.get_config()
        return {
            "report": "AutoSense Perception Session Report",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session": {
                "uptime_s": s["perf"]["uptime_s"],
                "frames_processed": s["perf"]["frames_processed"],
                "session_detections": s["perf"]["session_detections"],
                # Legacy key retained; this is a measured rolling output-FPS EMA,
                # not a source-FPS claim or a fabricated session average.
                "avg_fps": s["perf"]["fps"],
                "measured_fps_ema": s["perf"]["fps"],
                "last_inference_ms": s["perf"]["inference_ms"],
                "last_processing_ms": s["perf"]["processing_ms"],
                "last_roi_ms": s["perf"]["roi_ms"],
            },
            "configuration": cfg,
            "model": {"name": s["model"]["name"], "classes": s["model"]["classes"]},
            "current_counts": s["counts"],
            "active_tracks": s["tracks"],
            "safety": s["safety"],
        }

    # MJPEG ------------------------------------------------------------

    def mjpeg_frames(self):
        """Sync generator consumed by FastAPI StreamingResponse."""
        last = -1
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with self.frame_cond:
                if self.jpeg_seq == last:
                    self.frame_cond.wait(timeout=1.5)
                if self.latest_jpeg is None:
                    continue
                data, last = self.latest_jpeg, self.jpeg_seq
            yield boundary + data + b"\r\n"
