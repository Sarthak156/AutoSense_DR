# AutoSense — Technical Documentation

Complete implementation reference for every file in the repository: classes,
functions, formulas, configuration keys, the telemetry payload and the HTTP API.
Companion documents: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) (how it all
fits together), [RESULTS.md](RESULTS.md) (measured behavior),
[FUTURE_SCOPE.md](FUTURE_SCOPE.md) (roadmap).

Paths below are relative to the repository root (`d:\Autosense\autosense`).

```
autosense/
├── backend/
│   ├── server.py                FastAPI app: REST + WebSocket + static frontend
│   ├── pipeline.py              PerceptionEngine: threaded pipeline + metrics
│   ├── measure_fps.py           standalone FPS measurement client
│   ├── perception/  detector.py · distance.py · roi.py · dynamic_roi.py · roi_association.py
│   ├── tracking/    tracker.py
│   ├── safety/      ttc.py · collision.py · decision.py
│   ├── visualization/ overlay.py
│   ├── models/      yolo11n.pt (+ any uploaded .pt weights)
│   ├── samples/     dashcam_city.mp4 · tunnel_drive.mp4 · pedestrians.mp4
│   └── uploads/     user-uploaded videos (runtime)
├── frontend/
│   ├── index.html               single-page dashboard (9 pages)
│   ├── css/style.css            dark/light design system
│   └── js/app.js                all UI logic (no build step)
├── requirements.txt             runtime dependencies
├── requirements-dev.txt         pytest + runtime dependencies
└── tests/                        synthetic dynamic ROI / association tests
```

---

# 1. Perception Modules

## 1.1 `backend/perception/detector.py` — `ObjectDetector`

Ultralytics wrapper for **detection only** (no track IDs).

```python
class ObjectDetector:
    def __init__(self, model_path, confidence=0.35)
    def detect(self, frame) -> ultralytics.engine.results.Results
```

- Loads `YOLO(model_path)` once; each `detect()` call runs
  `model.predict(source=frame, conf=self.confidence, verbose=False)` and returns
  `results[0]`.
- Kept as the minimal building block; the live dashboard uses `model.track()`
  inside `PerceptionEngine` because tracking is enabled by default.

## 1.2 `backend/perception/distance.py` — `DistanceEstimator`

Monocular distance from bounding-box height — the pinhole relationship:

```
      D = (H · F) / h        D = distance [m]
                             H = assumed real height of the class [m]
                             F = focal length in pixels (cfg "focal_length")
                             h = bounding-box height in pixels
```

```python
class DistanceEstimator:
    def __init__(self, focal_length=700, smoothing_window=5)
    def estimate(self, bbox, class_name, track_id=None) -> float | None
    def clear_track(self, track_id)   # drop one ID's history
    def clear_all(self)               # drop every history (source change, reset)
```

Behavior details:

- `pixel_height <= 0` → `None` (degenerate box, distance unknown).
- Unknown class → fallback real height **1.50 m**.
- With a `track_id`, raw estimates are averaged over a per-ID sliding window
  (`deque(maxlen=smoothing_window)`), which stabilizes the number across frames;
  without an ID the raw estimate is returned.
- Result rounded to 2 decimals.
- Assumed real heights: person / pedestrian / rider 1.70 · car 1.50 ·
  truck / bus 3.20 · motorcycle 1.40 · bicycle 1.50 · train 3.50 ·
  traffic light 0.60 · traffic sign 0.75.
- `pipeline.py` rebuilds estimator histories whenever `focal_length` or
  `smoothing` changes at runtime (`_apply_calibration`).

## 1.3 `backend/perception/roi.py` — `DrivingROI`

Geometric ego-corridor used for the "is this object in my driving path?" decision.

```python
class DrivingROI:
    def __init__(self, width_ratio=0.48, height_start=0.615,
                 top_width_ratio=0.206, bottom_width_ratio=0.48,
                 ego_center_ratio=0.67, bottom_margin=0.0,
                 top_left_ratio=None, top_right_ratio=None,
                 bottom_left_ratio=None, bottom_right_ratio=None)
    def get_polygon(self, frame_width, frame_height) -> [(x, y) × 4]
    def is_in_path(self, bbox, frame_width, frame_height) -> bool
    def draw(self, frame) -> frame
```

- The polygon is a perspective trapezoid: top edge at `height_start · H` between
  `ego_center ± top_width/2`; bottom edge at `H · (1 − bottom_margin)` between
  `ego_center ± bottom_width/2` (all ratios of frame size). Independent corner
  ratios override the symmetric defaults when provided.
- `is_in_path()` tests the **bottom-center point of the bbox** (the object's
  contact point with the road surface) against the polygon using
  `cv2.pointPolygonTest(...) >= 0`.
- `draw()` shades the polygon (α ≈ 0.025) and strokes it (BGR `(255, 255, 0)`).
- `PerceptionEngine` re-instantiates `DrivingROI` whenever a fallback-geometry
  slider changes. This original trapezoid remains the operator-configurable
  fallback and ego-center prior for the dynamic engine.

### `backend/perception/dynamic_roi.py` — `DynamicROIEngine`

Inference-independent, training-free OpenCV pipeline. It accepts a BGR frame,
a fallback polygon, and a config mapping; it has no YOLO/Ultralytics import.
Each call executes grayscale → Gaussian blur → Canny → configurable perspective
mask → `HoughLinesP` → deterministic RANSAC consensus → robust line/quadratic
`x(y)` fits → center/width pair selection. Curved fits are kept only when the
quadratic materially improves on a line and has sufficient vertical evidence.
The two sampled boundaries form the forward corridor polygon.

The engine smooths normalized geometry over frames, validates confidence and
geometry, reuses recent valid estimates for at most `max_reuse_frames` with
confidence decay, and then emits the configured `DrivingROI` fallback. State is
reported as `detected`, `low_confidence`, `reused`, `fallback`, or `disabled`;
all but a sufficiently confident `detected` state are explicitly uncertain.
Aspect-ratio changes clear stale history. Full configuration and limitations are
in [DYNAMIC_ROI.md](DYNAMIC_ROI.md).

### `backend/perception/roi_association.py` — `ROIObjectAssociator`

Consumes plain bounding boxes, class names, confidences, optional tracking IDs,
and an ROI polygon. It combines bottom-center inclusion, rasterized box/polygon
overlap, a light class prior, detection confidence, and a short score history
when a track ID exists. It returns `in_path`, overlap, score, relevance and
uncertainty metadata. It never deletes objects; outside-ROI detections stay in
the full detection list and receive an outside/uncertain label.

## 1.4 `backend/tracking/tracker.py` — `ObjectTracker`

```python
class ObjectTracker:
    def __init__(self, model_path, confidence=0.35)
    def track(self, frame) -> Results     # model.track(persist=True, tracker="bytetrack.yaml")
    def get_objects(self, result) -> [{track_id, class_id, confidence, bbox:[x1,y1,x2,y2]}]
```

- `track()` uses Ultralytics persistent tracking so IDs stay stable across
  frames; `get_objects()` flattens a `Results` object into plain dicts with
  integer boxes.
- The dashboard engine performs the same `model.track(...)` call inline so the
  tracker YAML, confidence, IoU, input size and device are all runtime-switchable;
  this class remains the reusable, framework-free extraction layer.

# 2. Safety Modules

## 2.1 `backend/safety/ttc.py` — `TTCEstimator`

Turns the distance time-series of each track into closing speed and
time-to-collision.

```python
class TTCEstimator:
    def __init__(self, history_size=8, min_closing_speed=0.1)
    def update(self, track_id, distance, timestamp) -> dict | None
    def remove_track(self, track_id)
    def clear(self)
```

- Per track ID it keeps a `deque(maxlen=history_size)` of distances and one of
  timestamps (the engine constructs it with `history_size=8`).
- Fewer than 2 samples, non-positive `Δt`, or `track_id / distance` missing
  → `None`.
- **Closing speed** (m/s, positive = approaching):

  ```
  v_close = (d_oldest − d_newest) / (t_newest − t_oldest)
  ```

- If `v_close <= min_closing_speed (0.1 m/s)` → returns
  `{"ttc": None, "closing_speed": v, "approaching": False}` — a receding or
  static object is reported as *not approaching* rather than a bogus TTC.
- Otherwise `ttc = d_newest / v_close`; values above **60 s** are reported as
  `None`; both numbers rounded to 2 decimals.

## 2.2 `backend/safety/collision.py` — `CollisionRisk`

Rule table producing `SAFE / WARNING / DANGER / CRITICAL` (documented in the
source as a research prototype — it does not control a vehicle).

```python
class CollisionRisk:
    RISK_PRIORITY = {"SAFE": 0, "WARNING": 1, "DANGER": 2, "CRITICAL": 3}
    ttc_critical = 1.0   # s        distance_critical = 3.0   # m
    ttc_danger   = 2.0   # s        distance_danger   = 6.0   # m
    ttc_warning  = 4.0   # s        distance_warning  = 12.0  # m
    def evaluate(self, distance, ttc, closing_speed, in_path, class_name) -> str
    def get_priority(self, risk) -> int
```

Evaluation order (first match wins):

1. Not `in_path` → **SAFE** (objects outside the ego corridor are never hazards).
2. `distance is None` → **SAFE**.
3. `distance ≤ 3.0 m` or `ttc ≤ 1.0 s` → **CRITICAL**.
4. `distance ≤ 6.0 m` or `ttc ≤ 2.0 s` → **DANGER**.
5. `distance ≤ 12.0 m` or `ttc ≤ 4.0 s` → **WARNING**.
6. Otherwise → **SAFE**.

`class_name` is accepted (used by the caller for display/grouping) but the
shipped rule set treats all classes equally; `closing_speed` is informational.

## 2.3 `backend/safety/decision.py` — `DecisionEngine`

```python
class DecisionEngine:
    def decide(self, risk) -> "CONTINUE" | "WARNING" | "BRAKE"
```

Mapping: `CRITICAL → BRAKE`, `DANGER → BRAKE`, `WARNING → WARNING`,
anything else → `CONTINUE`. This is the action shown on the HUD and in the
Safety page hero panel.

## 2.4 `backend/visualization/overlay.py` — `SafetyOverlay`

Reference rendering style used by the dashboard's overlay design:

- `draw_object(frame, bbox, class_name, track_id, distance, ttc, risk, in_path)`
  — status-colored box (`CRITICAL` red, `DANGER` orange, `WARNING` amber,
  `SAFE` green, `OUTSIDE LANE` gray) plus a label such as
  `CLASS #id | D 12.4m | TTC 1.2s` (or `OUTSIDE LANE`).
- `draw_roi(frame, polygon)` — shaded corridor + yellow boundary.
- `draw_dashboard(frame, risk, decision, fps, object_count, closest_distance)` —
  the 380×135 HUD (`RISK`, `ACTION`, `FPS / Objects / Closest`).

`PerceptionEngine._draw()` mirrors this visual language while honoring the
runtime overlay toggles (see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
§6.1); this module remains the standalone, import-anywhere version of that design.

# 3. Backend Engine — `backend/pipeline.py`

The largest backend file: a threaded `PerceptionEngine` that owns the capture,
the model, the pipeline modules and the telemetry. It is deliberately the *only*
place where those pieces meet.

## 3.1 Engine Anatomy

| Attribute | Type | Purpose |
|---|---|---|
| `lock` / `frame_cond` | `threading.Lock` / `Condition` | config guards; frame publication signaling |
| `latest_jpeg`, `jpeg_seq` | bytes / int | newest encoded frame + monotonic sequence |
| `model`, `_loaded_model_name`, `_loaded_device` | YOLO / str | active weights + reload triggers |
| `cap`, `_want_reopen` | VideoCapture / bool | capture handle and reopen flag |
| `distance_estimator`, `roi`, `dynamic_roi`, `roi_associator`, `ttc_estimator`, `risk_engine`, `decision_engine` | module instances | distance, classical corridor, association and safety stack |
| `track_registry` | dict | per-ID first/last seen, class/distance/risk and ROI relevance history |
| `recent` | deque(12) | recent detections with thumbnails, deduped per track |
| `fps_history`, `risk_history` | deque(180) | chart buffers published as `[-120:]` |
| `frames_processed`, `session_detections`, `started_at` | counters | session metrics |
| `cfg`, `state`, `state_detail`, `stats` | dict / str | config snapshot, state machine, telemetry snapshot |
| `_fps_ema`, `_last_frame_completed` | float / monotonic time | Measured completed-frame throughput EMA; no assumed FPS |
| `_last_inference_ms`, `_last_processing_ms`, `_last_roi_ms` | float / `None` | Last measured inference, total frame-processing and independent ROI-update latencies |
| `thread` | daemon `Thread(name="engine")` | the pipeline loop |

Public control surface (all thread-safe):

```python
update_config(patch) · get_config() · get_stats() · get_report()
set_video_source(path, name, start=True) · set_camera_source(index, start=True)
pause() · resume() · start() · stop() · restart()
available_models() · mjpeg_frames()
```

## 3.2 Config Schema (defaults as shipped)

| Key | Default | Meaning |
|---|---|---|
| `source` | `null` | `{"type": "video", "path", "name"}` or `{"type": "camera", "index", "name"}` |
| `playing` | `false` | Master start flag; set by source loading |
| `paused` | `false` | Pause flag (engine keeps republishing last stats) |
| `loop` | `true` | Loop video files on EOF |
| `conf` | `0.35` | YOLO confidence threshold |
| `iou` | `0.45` | YOLO NMS IoU threshold |
| `tracking` | `true` | `model.track()` vs `model.predict()` |
| `tracker` | `"bytetrack.yaml"` | Ultralytics tracker config (or `botsort.yaml`) |
| `imgsz` | `640` | YOLO input size (320 = Turbo FPS) |
| `device` | `"auto"` | `"auto"` \| `"gpu"` \| `"cpu"` — mapped to ultralytics `device=` |
| `model_name` | `"yolo11n.pt"` | File inside `backend/models/` |
| `focal_length` | `700` | Pixels — distance calibration |
| `smoothing` | `5` | Per-track distance-smoothing window |
| `roi.height_start` | `0.70` | Top edge of the configured fallback corridor |
| `roi.top_width_ratio` | `0.30` | Fallback corridor width at the top |
| `roi.bottom_width_ratio` | `0.75` | Fallback corridor width at the bottom |
| `roi.ego_center_ratio` | `0.62` | Horizontal ego-center/search prior |
| `roi.bottom_margin` | `0.02` | Gap between fallback bottom and frame bottom |
| `roi.dynamic_enabled` | `true` | Enable the training-free OpenCV estimator |
| `roi.search_top_ratio` | `0.50` | Perspective search horizon |
| `roi.search_top_width_ratio` | `0.30` | Full search-mask width at the horizon |
| `roi.search_bottom_width_ratio` | `0.88` | Full search-mask width near the frame bottom |
| `roi.canny_low/high` | `50 / 150` | Canny edge thresholds |
| `roi.hough_threshold` | `24` | Hough line-segment vote threshold |
| `roi.min_confidence` | `0.30` | Minimum fit confidence to use a new estimate |
| `roi.reliable_confidence` | `0.56` | Below this, accepted geometry is marked low-confidence |
| `roi.smoothing` | `0.35` | Weight of the current valid estimate |
| `roi.max_reuse_frames` | `8` | Bounded holdover before fallback |
| `roi.confidence_decay` | `0.82` | Per-missed-frame confidence multiplier during reuse |
| `roi.expected_lane_width_ratio` | `0.44` | Perspective lane-width prior used to choose candidate pairs |
| `roi.center_tolerance_ratio` | `0.30` | Allowed corridor-center offset from the configured ego prior |
| `overlays.boxes` | `true` | Bounding boxes |
| `overlays.confidence` | `true` | Confidence text |
| `overlays.ids` | `true` | Track IDs |
| `overlays.distance` | `true` | Distance text |
| `overlays.roi` | `true` | Corridor shading / fallback status |
| `overlays.lane` | `true` | Estimated boundaries and confidence/uncertainty |
| `overlays.risk` | `false` | Risk colors, TTC text, ROI relevance, HUD |

Patch semantics: `update_config()` deep-merges, so
`{"roi": {"top_width_ratio": 0.4}}` changes one number without touching the rest.

## 3.3 The Engine Loop (`_loop`)

Infinite daemon loop; each iteration works on one config snapshot:

1. **Guard:** ultralytics import missing → `error` state + "MODEL RUNTIME UNAVAILABLE"
   placeholder + stats push + 1 s sleep.
2. **Model reload check:** `model is None or model_name changed or device changed`
   → `YOLO(path)`; on success resets the ByteTrack predictor
   (`model.predictor = None`) and clears histories; on failure publishes
   "MODEL NOT LOADED".
3. **No source:** `not playing or source is None` → `idle` + "NO INPUT SOURCE"
   placeholder; 250 ms sleep.
4. **Capture:** reopens when `cap is None or _want_reopen`; failure →
   "SOURCE NOT AVAILABLE" (`error`); success clears histories and resets pacing.
5. **Paused:** republishes last stats, 150 ms sleep.
6. **Frame read:** on EOF, loops (`loop=True`) or publishes "STREAM ENDED"
   (`idle`) / "CAMERA ERROR" (`error`).
7. **Classical ROI update:** runs grayscale/blur/Canny/Hough geometry before the
   detector call; it can update even when YOLO inference fails.
8. **YOLO11 inference:** existing `model.track(...)` or `model.predict(...)`
   inside try/except → on exception `error` + message, 400 ms sleep.
9. **Processing** (§3.4), **drawing** (§3.5), **JPEG publish** (quality 80),
   measured pacing/FPS/latencies, **stats push** (§3.9).

### 3.4 Per-Detection Processing

For each box in `result.boxes`:

```python
bbox = [x1, y1, x2, y2]                         # ints from YOLO11
name = result.names[cls]
group = classify_group(name)
distance = distance_estimator.estimate(bbox, name, track_id)
roi = roi_associator.associate(
    bbox, roi_state.polygon, frame.shape, name, confidence, track_id,
    roi_state=roi_state.state, roi_confidence=roi_state.confidence,
    roi_uncertainty=roi_state.uncertainty,
)
in_path = roi["in_path"]                         # overlap + bottom-center + history
risk = risk_engine.evaluate(distance, ttc, closing_speed, in_path, name)
```

Aggregated across the frame: counts per group, class distribution,
`closest_distance` (min in-path), `min_ttc` (min wherever defined),
`overall_risk` (highest priority), `decision`.

The class-group mapping (`GROUP_DEFS` in `pipeline.py`) feeds the dashboard's
"Live Statistics" panel and works for both COCO names and custom model names
(car / truck / bus / van / …; person / pedestrian / …; motorcycle / bicycle /
scooter / …; traffic sign / light / stop sign / …) — anything unrecognized
lands in `other`.

### 3.5 Overlay Drawing (`_draw`)

Draws the current dynamic/fallback ROI polygon, sampled left/right boundaries,
and the confidence/uncertainty state when those layers are enabled. Every
YOLO11 detection retains a class/confidence/track/distance label plus a compact
ROI relevance tag (`IN`, `NEAR`, `OUT`, or `UNCERTAIN`); boxes are not removed
outside the corridor. Optional risk coloring and HUD remain controlled by the
existing overlay toggles.

### 3.6 Recent Detections & Thumbnails

- Top-5 detections by confidence per frame receive a **real crop** from the
  current frame: padded bbox → squared with the panel background → 96×96 →
  JPEG q70 → `data:image/jpeg;base64,…`.
- `_update_recent()` keeps a 12-entry deque, deduped by track ID, newest first;
  `/api/status` publishes the newest 8 (`stats["recent"]`).

### 3.7 FPS, Pacing & Histories

- Output FPS is measured as `1/dt` between completed annotated-frame outputs;
  EMA = `0.75·old + 0.25·new`. It is `null` until a real interval exists.
- A video is paced to valid capture/container FPS metadata when available.
  There is no assumed 30 FPS camera rate or invalid-metadata fallback; cameras
  run at their reported cadence and the measured output interval is shown.
- `perf.inference_ms` measures the YOLO11 call, `perf.processing_ms` measures
  capture-read-to-JPEG-encode latency excluding intentional pacing, and
  `perf.roi_ms` measures the independent OpenCV update.
- `fps_history` / `risk_history` (deque 180) receive measured samples and are
  published as the last 120 for the sparkline and risk timeline.

### 3.8 Device Selection

| `cfg["device"]` | Ultralytics `device=` | Notes |
|---|---|---|
| `"auto"` | `None` (library default) | GPU when a CUDA torch build is present |
| `"gpu"` | `"cuda:0"` if `torch.cuda.is_available()`, else `None` | Falls back silently when CUDA is absent |
| `"cpu"` | `"cpu"` | Forces CPU inference |

`_device_label()` reports the **actual** device from the loaded predictor
(`GPU (<name>)` or `CPU`) — it never echoes the requested setting.

### 3.9 Telemetry Snapshot (`_push_stats`)

Builds `engine.stats` (21 top-level keys) — full field list in §7. Two details
worth noting:

- `modules` describes Model / Dynamic ROI / Tracking / Video Source / Safety
  with `ok / warn / err / off` statuses for the right-hand System Status panel.
- `ego_speed` is hard-coded `None` with the source comment *"no speed sensor
  connected — UI shows 'not available'"*.

### 3.10 MJPEG Generator (`mjpeg_frames`)

Synchronous generator consumed by FastAPI's `StreamingResponse`:

```python
boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
while True:
    with self.frame_cond:
        if self.jpeg_seq == last:          # wait for the NEXT frame
            self.frame_cond.wait(timeout=1.5)
        data, last = self.latest_jpeg, self.jpeg_seq
    yield boundary + data + b"\r\n"
```

Each client therefore always transmits the newest frame and never builds a
backlog. `GET /api/video/frame` simply returns `latest_jpeg` as a single JPEG
(404 before the first frame).

# 4. FastAPI Server — `backend/server.py`

Thin transport layer over the engine. Constants: `MODELS_DIR`, `UPLOADS_DIR`,
`SAMPLES_DIR` under `backend/`; `FRONTEND_DIR` = `../frontend`. The engine is a
module-level singleton: `engine = PerceptionEngine(models_dir=MODELS_DIR)`.

## 4.1 Endpoint Reference

| Method | Path | Request body | Response |
|---|---|---|---|
| GET | `/healthz` | — | `{ok, state}` |
| GET | `/api/status` | — | Full telemetry snapshot |
| GET | `/api/report` | — | Session report (uptime, frames, detections, measured FPS EMA, last inference/processing/ROI latency, config, model, counts, tracks, safety) |
| GET | `/api/sources` | — | `{videos: [{name, path, kind, size_mb}]}` — `kind` = `sample` \| `upload` |
| POST | `/api/source/sample` | `{name}` | Loads from `samples/` (fallback `uploads/`); 404 if missing |
| POST | `/api/source/video` | multipart file | Streams upload to `uploads/` (1 MiB chunks), 400 MB cap → 413; starts playback |
| POST | `/api/source/camera` | `{index}` | Starts device capture (`cv2.VideoCapture(index)`) |
| POST | `/api/control/{action}` | — | `action` ∈ `pause · resume · start · stop · restart`; 404 on unknown |
| POST | `/api/config` | partial config dict | Deep-merged; returns the full updated config |
| GET | `/api/models` | — | `{models: [{name, size_mb, active}], active}` |
| POST | `/api/models/select` | `{name}` | Switches `model_name`; 404 if the file is absent |
| POST | `/api/models/upload` | multipart file | Accepts `.pt` only (400 otherwise); saved into `models/` |
| GET | `/api/video/stream` | — | `multipart/x-mixed-replace; boundary=frame` MJPEG stream |
| GET | `/api/video/frame` | — | Latest annotated JPEG; 404 before the first frame |
| WS | `/ws` | — | Immediate stats snapshot, then ≈5 Hz broadcasts; client pings keep it alive |
| GET | `/` (+ any static path) | — | The dashboard (StaticFiles mounted **after** all API routes) |

Notes:

- Uploads are sanitized with `os.path.basename` before writing.
- Model files must end in `.pt`; class names are read from the file itself by
  Ultralytics.
- CORS is not enabled — same-origin by design (the frontend is served by this app).

## 4.2 WebSocket Broadcasting

- `WSManager` keeps the set of connected clients; `disconnect()` prunes dead
  sockets when a send raises.
- `ws_endpoint` sends a snapshot immediately on connect (so the UI is never
  blank), then only receives keepalive text.
- A startup task loops every 200 ms: no clients → skip; otherwise broadcast the
  current stats (skipping repeats while the engine isn't running).

## 4.3 Boot Wiring

- Startup loads `samples/dashcam_city.mp4` automatically when present, so a
  fresh install demonstrates the full pipeline with zero configuration.
- The static mount is registered last, ensuring `/api/*`, `/ws` and `/healthz`
  take precedence over frontend files.

# 5. Measurement Client — `backend/measure_fps.py`

Standalone script (no imports from the app) that measures real pipeline
performance through the public API; it expects the server on
`http://127.0.0.1:8765`.

Sequence: boot wait (12 s) → read line → for `imgsz` in 320 / 416 / 640
(7 s settle each) → load `video.mp4` → read → 416 / 320 → set 320 →
`MEASURE DONE`.

Each line prints: FPS, inference ms, source name, resolution, `imgsz`. This is
the tool used to produce the tables in [RESULTS.md](RESULTS.md).

# 6. Frontend Reference

Static single-page app — no framework, no build step. Served by the backend.

## 6.1 `frontend/index.html` — Page Structure

Shell: top nav (9 pages), connection pill, clock, status pill, quick camera /
theme / settings buttons. Pages are `<section class="page" id="page-…">`
elements toggled by the router:

| Page | Highlights |
|---|---|
| `dashboard` | Video stage (img + transparent canvas + risk banner + paused veil), overlay menu, speed/heading widgets, Recent Detections (thumbnails), Class Distribution donut, Performance History graph |
| `video` | Current source (name, progress, time), transport controls, Sample Videos list, Upload dropzone, Live Camera (index, detect, start/stop) |
| `detection` | Confidence / IoU sliders, input-size segment, performance presets (Turbo / Normal), compute-device segment (auto / gpu / cpu), detections table |
| `tracking` | Tracking enable toggle, tracker segment (ByteTrack / BoT-SORT), IDs toggle, active-track table, track counters |
| `distance` | Focal length & smoothing sliders, ROI sliders (ego center, top/bottom width, height start), ROI polygon readout, schematic canvas, distance table |
| `safety` | Hero panel (risk / decision / closest / TTC), risk timeline canvas, threshold table, per-object TTC table |
| `viz` | Full-size annotated stream with the overlay-layer controls (lazy-loaded connection) |
| `model` | Active model card (device, input, inference), class list, available weights, custom `best.pt` upload |
| `settings` | Theme radios, stream transport segment, connection panel, pipeline diagram, reset button |
| (right panel, all pages) | Live Statistics (counts per group), System Status (module list), Performance (FPS / inference / input / extras), Safety summary, Quick Actions (distance test, risk test, tracks, report download) |

## 6.2 `frontend/css/style.css` — Design System

- CSS custom properties for both themes on `html[data-theme=…]`; dark is the
  default (deep navy engineering palette), light is a full alternative.
- Monospace (`ui-monospace`) for all numeric telemetry; accent `#38BDF8`; risk
  palette green / amber / orange / red shared with the backend overlay.
- Components: cards, stat grids, segment controls, sliders, tables, dropzones,
  toasts, status pills, risk banner, module list, progress bars.
- No external assets or fonts — the app renders fully offline.

## 6.3 `frontend/js/app.js` — Application Logic

| Module | Responsibility |
|---|---|
| Helpers | `$`, `$$`, `fmt`, `fmt0`, `clamp`, `ago`, `findSeq` (byte search for the MJPEG demuxer), `api()` fetch wrapper, debounced `postCfg()`, `toast()` |
| `Theme` | dark / light / system; persisted as `autosense.theme`; reacts to OS theme changes |
| Router | `PAGES` + hash routing (`#dashboard`, `#safety`, …); lazy work per page (`loadModels`, `loadSources`, viz source) |
| Clock | 1 s ticker for the header time/date |
| Telemetry | `startWS()` (WebSocket + ping + reconnect), `startPolling()` (350 ms), `markTransport()`, `onStats()` → `render(s)` |
| Renderers | `renderStatusPill`, `renderVideoHeader` (FPS + risk banner), `renderRightPanel`, `renderRecent`, `renderVideoPage`, `renderTables`, `renderSafety`, `renderModelCard`, `syncControls` |
| Canvases | `drawDonut` (class distribution), `drawSparkline` (FPS history), `drawRiskStrip` (risk timeline), `drawTrajectory` / `drawRoiSchematic` (top-down scene with detection points and ROI polygon); all re-drawn when the theme flips |
| `Video` | Transport manager: `fps()` (4 s rolling window), `paint()` (transparent canvas), `setMode(mjpeg\|polling)`, MJPEG demuxer (`pumpStream`), snapshot poller, stall watchdog, `startLegacyImg()` fallback (which goes straight to polling) |
| Controls | Sliders, segments, toggles, transport buttons, sample list, dropzones (`initUploads`), model select/upload, camera detection, settings reset |
| Turbo | `applyTurbo()` toggles `imgsz` between 320 / 640, persisted as `autosense.turbo`, restored on load |

Client-side persistence keys: `autosense.theme`, `autosense.stream`,
`autosense.turbo` (all cleared by the Settings reset button).

# 7. Telemetry Payload Reference (`GET /api/status`, `WS /ws`)

| Field | Content |
|---|---|
| `state` / `detail` | Engine state (`starting` / `idle` / `running` / `paused` / `error`) + human-readable detail |
| `mode` | UI-oriented mode (`live` / `paused` / `idle` / `error`) |
| `time` | Unix timestamp of the snapshot |
| `source` | Active source `{type, path \| index, name, width, height, fps, position_s, duration_s, total_frames}` |
| `model` | `{name, loaded, error, classes[], device}` — `classes` read from the model file |
| `config` | Full effective configuration (see §3.2) |
| `perf` | `{fps, inference_ms, processing_ms, roi_ms, input_w, input_h, frames_processed, uptime_s, session_detections}`; measured FPS is null until two completed frames |
| `counts` | `{vehicles, pedestrians, two_wheelers, traffic_signs, other, total}` for the current frame |
| `distribution` | `{class_name: count}` for the current frame |
| `detections` | Every current-frame object, including outside-ROI objects; `{track_id, class_name, group, confidence, bbox, distance, ttc, closing_speed, risk, in_path, roi_relevance, roi_label, roi_score, roi_raw_score, roi_overlap, bottom_center_in_roi, roi_signed_distance_px, roi_class_factor, roi_confidence, roi_state, roi_uncertainty, roi_track_history}`; optional `thumb` on the top five |
| `detection_total` | Total detections in the current frame |
| `tracks` | Active tracks (seen in the last 2 s): `{id, class_name, distance, risk, in_path, roi_relevance, roi_label, age_s, seen_frames}` |
| `safety` | `{overall_risk, decision, closest_distance, min_ttc, thresholds{distance{…}, ttc{…}}}` |
| `recent` | Newest 8 deduped detections with base64 thumbnails |
| `roi` | `{state, confidence, uncertainty, reason, source, candidate_count, frames_since_detection}` |
| `roi_polygon` | Polygon, `corners`, sampled left/right boundaries, search region, state/confidence/uncertainty and frame dimensions |
| `fps_history` / `risk_history` | Last 120 samples for the charts |
| `modules` | Model / Dynamic ROI / Tracking / Video Source / Safety status rows for the System Status panel |
| `ego_speed` | Always `null` — no speed sensor is connected |

# 8. Operational Notes & Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `device: "gpu"` still reports `CPU` | On Windows, `pip install torch` yields the CPU-only build; install the CUDA wheel (see Quick Start) and confirm `torch.cuda.is_available()` |
| `GET /api/video/frame` 404 | No frame processed yet (still booting / no source) — poll until `state=running` |
| Video not processing after start | Check `state` / `detail` in `/api/status`; verify the source file exists in `samples/` or `uploads/` |
| FPS lower than expected | Measured completed-frame throughput depends on actual inference, ROI/drawing time, video pacing and camera cadence; compare `inference_ms`, `processing_ms`, `roi_ms`, and source metadata |
| Sliders don't seem to apply | Edits are debounced 160 ms and apply on the next frame — check the value echoed in `config` |
| Model upload rejected | Only `.pt` files are accepted (400); video uploads are capped at 400 MB (413) |
| `ConnectionResetError` in the server console | Cosmetic Windows asyncio teardown noise when a tab closes; the client reconnects automatically |
| Port already in use | `uvicorn … --port 8000` fails fast — stop the previous server or pick another port (frontend is same-origin, no config change needed) |

# 9. Extension Recipes

**Add an overlay layer** — add a default key in `cfg["overlays"]`, honor it in
`PerceptionEngine._draw()`, and add a toggle wired through `postCfg` on the
Visualizations page.

**Add a tunable** — add the default to `cfg`, read it from the per-frame
snapshot, and add the control in `app.js` (`syncControls` keeps it in sync when
other clients change it).

**Add an API route** — append it to `backend/server.py` above the static mount;
use `engine.get_config()` / `update_config()` so it stays consistent with the UI.

**Add a metric** — compute it in the engine and include it in `_push_stats()`;
the frontend `render(s)` can consume it immediately, and `/api/report` picks it
up from the same stats structure where relevant.

**Swap the model** — drop any Ultralytics-compatible `.pt` into
`backend/models/`, then `POST /api/models/select` (or use Model → Available
Weights). Class names and grouping are derived from the file, so custom classes
work without code changes.