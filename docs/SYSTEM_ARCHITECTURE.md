# AutoSense — System Architecture

Document scope: how the pieces of AutoSense fit together at runtime — components,
threads, data flows, transports, state handling and failure behavior. For
file-by-file API details see [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md);
for measured behavior see [RESULTS.md](RESULTS.md).

## 1. Architecture at a Glance

```
 ┌─────────────────────────── browser (frontend/) ───────────────────────────┐
 │  index.html · css/style.css · js/app.js                                   │
 │                                                                           │
 │   MJPEG demuxer / snapshot poller       WebSocket client / HTTP poller    │
 │            │ (frames)                           │ (telemetry)             │
 └────────────┼────────────────────────────────────┼──────────────────────────┘
              │ GET /api/video/stream            │ WS /ws   or  GET /api/status
              │ GET /api/video/frame             │
 ┌────────────┴────────────────────────────────────┴──────────────────────────┐
 │                      FastAPI process (backend/server.py)                   │
 │  ── REST control ──  /api/config · /api/control/* · /api/source/* ·        │
 │                      /api/models* · /api/report · /api/sources · /healthz  │
 │  ── publishers ───   latest_jpeg (frame_cond)   ·   stats snapshot (~5 Hz) │
 │  ── static mount ──  "/" → ../frontend                                     │
 └──────────────────────────────────▲─────────────────────────────────────────┘
                                    │ in-process shared state
 ┌──────────────────────────────────┴─────────────────────────────────────────┐
 │               PerceptionEngine thread (backend/pipeline.py)                │
 │                                                                            │
 │  cv2.VideoCapture ─┬─► OpenCV dynamic ROI (Canny/Hough/robust curve fits) │
 │                    └─► YOLO11 track/predict → detection/track objects     │
 │  distance → ROI association → TTC → risk → decision → overlay / JPEG      │
 │  ─► measured frame metrics ─► latest_jpeg + stats                         │
 └────────────────────────────────────────────────────────────────────────────┘

Video source: bundled sample │ uploaded file │ live camera
```

Key architectural choice: **one process, two execution contexts.** The FastAPI
event loop never runs inference; the engine thread never serves HTTP. They meet
only at two synchronized hand-off points: the published JPEG (`latest_jpeg`,
guarded by `frame_cond`) and the immutable stats snapshot (`stats`, replaced
atomically after every processed frame).

## 2. Design Principles

1. **No fabricated data** — published values come from actual frame processing,
   measured timers, or explicit configuration; unavailable metrics are `null`,
   and uncertain ROI geometry is marked rather than presented as a fact.
2. **Single-file engine, single-file server** — `pipeline.py` owns the pipeline
   and metrics; `server.py` is a thin HTTP/WS shell.
3. **Config is data, not code** — one deep-merged `cfg` dict drives every tunable;
   the frontend edits it through `POST /api/config`.
4. **Degrade, never crash** — capture failures, model-load failures and inference
   exceptions become explicit states plus a placeholder frame.
5. **Client-side resilience** — the frontend can be refreshed or killed at any
   time; the backend keeps processing and republishes on reconnect.
6. **Transport choice per client** — MJPEG *or* polling; WebSocket *or* REST —
   both implemented, both able to fall back automatically.

## 3. Component Inventory

| Component | File | Responsibility |
|---|---|---|
| `PerceptionEngine` | `backend/pipeline.py` | Owns the capture thread, model lifecycle, per-frame pipeline, metrics, overlay drawing, JPEG publication |
| FastAPI app | `backend/server.py` | REST control API, WebSocket telemetry, MJPEG/JPEG endpoints, static frontend mount, demo auto-load |
| `ObjectDetector` | `backend/perception/detector.py` | Plain YOLO predict wrapper |
| `DistanceEstimator` | `backend/perception/distance.py` | Monocular distance (H·F/h) with per-track smoothing |
| `DrivingROI` | `backend/perception/roi.py` | Existing configurable trapezoid fallback and geometry controls |
| `DynamicROIEngine` | `backend/perception/dynamic_roi.py` | YOLO-independent grayscale/Canny/Hough robust boundary fits, smoothing, bounded reuse and fallback |
| `ROIObjectAssociator` | `backend/perception/roi_association.py` | Bbox overlap/bottom-center/class/confidence/track-history association; never filters detections |
| `ObjectTracker` | `backend/tracking/tracker.py` | YOLO + ByteTrack wrapper and box-extraction helper |
| `TTCEstimator` | `backend/safety/ttc.py` | Closing speed & time-to-collision per track |
| `CollisionRisk` | `backend/safety/collision.py` | Rule table → SAFE / WARNING / DANGER / CRITICAL |
| `DecisionEngine` | `backend/safety/decision.py` | Risk → action (CONTINUE / WARNING / BRAKE) |
| `SafetyOverlay` | `backend/visualization/overlay.py` | Reference drawing routines (object labels, ROI, dashboard HUD) |
| Dashboard UI | `frontend/index.html` + `css/style.css` + `js/app.js` | 9-page SPA: video transports, telemetry rendering, controls |
| Measurement client | `backend/measure_fps.py` | Sweeps sources / input sizes and prints engine metrics |

Dependency direction is strictly one-way: `server.py → pipeline.py → {perception,
tracking, safety}`. `visualization/overlay.py` is a style reference that
`pipeline._draw()` re-implements with toggleable layers. No module imports
`server.py`, and the perception modules never touch web concerns.

## 4. Deployment & Runtime Topology

- **One Python process** (`uvicorn server:app`) on port **8000** by default.
  It serves the REST/WS API *and* the static frontend (`frontend/` mounted at `/`).
- **One engine thread** (`PerceptionEngine`, daemon) created when `server.py` is
  imported; it survives across requests and owns the video capture.
- **One browser client** (or several) consuming `/ws` + `/api/video/stream`.
- **Storage:** `backend/models/` (weights), `backend/samples/` (bundled videos),
  `backend/uploads/` (user videos, 400 MB cap per file). `.gitignore` excludes
  `*.pt` / `*.mp4`, so media assets are provisioned per machine.
- **Boot sequence:** import `server.py` → engine thread starts (`state=starting`)
  → model loads lazily inside the thread → if
  `samples/dashcam_city.mp4` exists it is registered as the default source;
  otherwise the dashboard waits for an uploaded video or camera → first frame
  processed ⇒ `state=running`.

## 5. Concurrency Model

| Execution context | Created by | Touches | Synchronization |
|---|---|---|---|
| Engine thread (`_loop`) | `PerceptionEngine.__init__` | capture, model, pipeline modules, `stats`, `latest_jpeg`, `_last_detections` | `self.lock` for `cfg` / source switches; `self.frame_cond` for frame publication |
| FastAPI event loop | uvicorn | request handlers; `engine.get_stats()` / `get_config()` | reads via lock-protected getters |
| WS broadcaster task | `@app.on_event("startup")` | `WSManager.broadcast` every 0.2 s | async; skips when no clients or nothing changed while not running |
| MJPEG generator (per client) | `/api/video/stream` | waits on `frame_cond`, yields newest JPEG | condition variable with 1.5 s timeout |
| WS connection handlers | `/ws` per client | receive keepalives; snapshot on connect | `WSManager.clients` set |

Rules that keep this safe:

- **Config is copied, never shared mid-iteration.** The engine calls
  `self.get_config()` once per loop iteration (deepcopy under lock) and works on
  that snapshot.
- **Frames are immutable once published.** The engine stores encoded JPEG bytes;
  readers only swap references, so no tearing.
- **Histories are engine-only** deques (`fps_history`, `risk_history`, `recent`,
  `track_registry`); the stats dict embeds *copies* (`list(...)`) for the API.
- **Model reload is engine-triggered**, not request-triggered: a request only
  edits `cfg["model_name"]` / `cfg["device"]`; the thread notices and rebuilds
  `YOLO(...)` on its next iteration.

## 6. Frame Processing Pipeline

One iteration of `PerceptionEngine._loop()` — everything below happens on the
engine thread for a single source frame:

1. **Config snapshot** — `cfg = self.get_config()` (deepcopy under lock).
2. **Model lifecycle** — if ultralytics is missing → placeholder + `error`;
   if `model_name` / `device` changed → `YOLO(path)` reload, history flush,
   ByteTrack predictor reset.
3. **Source lifecycle** — no source / not playing → `idle` + placeholder;
   capture reopens when flagged (`_reset_capture`); video EOF loops
   (`loop=True`) or transitions to `idle` / `error` with a placeholder.
4. **Pause gate** — while `cfg["paused"]`, republish the last stats and sleep
   150 ms without reading frames.
5. **Read frame** — `cap.read()`; failures handled as in step 3.
6. **Dynamic ROI update** — grayscale/blur/Canny/perspective mask/Hough and
   robust fitting run independently of YOLO. Weak evidence invokes bounded
   reuse or the configured fallback and carries uncertainty metadata.
7. **Inference** — `model.track(frame, persist=True, tracker=cfg, conf, iou,
   imgsz, device)` when tracking is on, else `model.predict(...)`;
   `infer_ms` is measured around this call.
8. **Per-detection chain** (per box):
   `classify_group` → counters / distribution → `distance_estimator.estimate`
   → `ROIObjectAssociator.associate` (box overlap + bottom-center + class /
   confidence + optional track history) → TTC/risk → track registry → append
   detection dict. Every detection remains in the list outside the ROI too.
9. **Aggregates** — `overall_risk` = max risk by priority;
   `closest_distance` = min in-path distance; `min_ttc` = min across all
   detections; `decision = DecisionEngine.decide(overall_risk)`.
10. **Thumbnails** — top-5 detections by confidence get 96 px square JPEG crops
   (25 % padding) as data-URIs; `recent` deque (12) deduped per track.
11. **Overlay drawing** — `_draw()` applies the enabled layers (see §6.1).
12. **Publish & metrics** — JPEG (quality 80) into `latest_jpeg`; processing
   latency covers read-to-encode, FPS is measured between completed frames and
   includes intentional source pacing; `_push_stats()` rebuilds the snapshot.

### 6.1 Overlay layers (server-side rendering)

| Layer | Config key | Effect |
|---|---|---|
| Boxes | `overlays.boxes` | 2 px rectangle; risk-colored when `risk` is on, else accent (in-path) / green |
| Confidence | `overlays.confidence` | `0.35`-style value appended to the label |
| IDs | `overlays.ids` | `#<track_id>` appended to the class name |
| Distance | `overlays.distance` | `12.4m` appended to the label |
| TTC / risk semantics | `overlays.risk` | Adds TTC/risk information and risk colors; the ROI relevance tag remains visible for every object |
| ROI | `overlays.roi` | Current dynamic corridor only; the configured fallback is rendered separately |
| Lane | `overlays.lane` | Sampled left/right dynamic boundaries and detected/low-confidence/reused/fallback state label |
| Static ROI | `overlays.static_roi` (default on) | Shows/hides the editable configured blue `DrivingROI` trapezoid; independent of dynamic estimation |
| Search-mask debug | `overlays.search_region` (default off) | Optional outline of the dynamic estimator's internal perspective search mask |
| HUD | (part of `overlays.risk`) | Compact panel: `RISK`, `ACTION`, measured `FPS / OBJ / CLOSE` |

The blue trapezoid is the configured `DrivingROI.get_polygon()` geometry,
rendered separately from the dynamic estimate. Its height, widths, and center
follow the static ROI sliders even while dynamic estimation is active. The
estimator's `ROIFrameState.search_region` is a separate internal perspective
mask, optionally shown only by `overlays.search_region`; its center is controlled
by `roi.search_center_ratio`, not the static trapezoid center. The video canvas
paints the server JPEG. Schematic canvases receive separate `static_roi_polygon`
and `roi_polygon` telemetry and honor the independent layer toggles.

Risk colors (BGR): SAFE `(128,222,74)`, WARNING `(36,191,251)`,
DANGER `(60,147,251)`, CRITICAL `(113,113,248)`, outside corridor
`(150,150,150)`, accent `(248,189,56)`.

## 7. Video Transport Architecture

**Server side.** The engine encodes one JPEG per processed frame and stores it in
`latest_jpeg` with a monotonically increasing `jpeg_seq`. Two endpoints expose it:

- `GET /api/video/stream` — synchronous generator yielding
  `--frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg>\r\n` multipart chunks.
  It waits on `frame_cond` for a **new** sequence number (1.5 s timeout), so
  slow clients skip intermediate frames instead of queuing them.
- `GET /api/video/frame` — the latest JPEG as a single response (polling
  transport; 404 until the first frame exists).

**Client side** (`frontend/js/app.js` → `Video` module):

- **MJPEG mode (default).** `fetch("/api/video/stream")` + `AbortController`,
  streaming reader that scans for the `--frame` boundary, decodes with
  `createImageBitmap`, and paints onto a **transparent canvas** layered over the
  `img` element. Up to 4 reconnect attempts with backoff before falling back.
- **Polling mode.** `setInterval(tick, 140)` fetching `/api/video/frame?t=…`
  (`cache: no-store`), decoded and painted the same way.
- **Stall watchdog.** Every 1.5 s: if no frame was painted for ≥ 3 s while the
  engine reports `running`, the transport is silently restarted (mode preserved).
- **Why canvas always:** the legacy "point an `<img>` at the stream" path is
  known to white-stall in Chromium; the code therefore never uses it for the
  main view — every fallback repaints through the canvas.
- **Delivery FPS** in the header is measured client-side over a rolling 4 s
  window of painted frames, independent of the backend FPS metric.
- The Visualizations page has a **lazy second consumer** that only attaches while
  that page is open (MJPEG with cache-buster, or every 4th polling tick).

## 8. Telemetry Architecture

**Producer.** Every processed frame ends in `_push_stats(...)`, which rebuilds
`engine.stats` — a plain JSON-serializable dict. The engine keeps that dict as an
atomic snapshot; readers never see partial state. Top-level keys:

```
state · detail · mode · time · source · model · config · perf · counts ·
distribution · detections · detection_total · tracks · safety · recent ·
roi · roi_polygon · static_roi_polygon · fps_history · risk_history · modules · ego_speed
```

**Distributors.**

| Channel | Cadence | Notes |
|---|---|---|
| `WS /ws` | snapshot on connect, then a broadcast every 0.2 s (≈5 Hz) while clients are attached | Broadcaster skips identical signatures while not running; client pings every 15 s |
| `GET /api/status` | on demand | Initial render before WS attaches; polling fallback |
| `GET /api/report` | on demand | Session summary: uptime, frames, detections, measured FPS EMA, last inference/processing/ROI latency, configuration, model, counts, tracks, safety |

**Consumer (`frontend/js/app.js`).**

- `startWS()` connects `ws(s)://<host>/ws`; on message → `onStats()`; on close →
  `startPolling()` (350 ms interval) plus up to 4 quiet reconnect attempts.
- Every handler funnels into `render(s)`, which updates all pages and (on every
  second tick) redraws the canvases: class donut, FPS sparkline, risk timeline,
  trajectory / ROI schematic.
- The connection pill shows the active transport (`WebSocket` → "Live
  telemetry", `HTTP polling` → "Polling telemetry", or "Disconnected").

## 9. Configuration Flow

```
UI control ──debounced POST /api/config──► engine.update_config(patch)
                                            └─ _deep_merge(cfg, patch)  (under lock)
engine thread ── get_config() deepcopy per iteration ──► applied to:
    • tracker / imgsz / conf / iou / device → next model call
    • focal_length / smoothing              → DistanceEstimator histories rebuilt
    • roi.*                                 → fallback DrivingROI updated; dynamic config applied per frame
    • overlays.*                            → next _draw()
    • model_name / device                   → YOLO reload on next iteration
```

- The frontend batches slider edits with a 160 ms debounce (`postCfg`), so a drag
  produces a handful of patches, not hundreds.
- `Turbo FPS` (input size 320) is additionally persisted client-side in
  `localStorage` and re-applied on page load, because the backend holds config in
  memory only.
- Because the engine reads a copied snapshot, mid-frame config changes can never
  corrupt an in-flight frame; they take effect atomically on the next iteration.

## 10. Engine State Machine

```
        ┌──────────┐  model ok + source + first frame
        │ starting │──────────────────────────────────► running ⇄ paused
        └────┬─────┘                                      │  ▲
             │                                            │  │ resume
             │ no model / no source / open failure        ▼  │
             ▼                                        idle ◄─┘ (video finished, stop)
        ┌──────────┐  inference exception / camera failure
        │  error   │◄────────────────────────────────────────
        └──────────┘
```

| State | Meaning | UI signal |
|---|---|---|
| `starting` | Engine thread booting | status pill idle, placeholder frame |
| `idle` | No source loaded / playing, or video finished | "Idle", placeholder frame with reason |
| `running` | Frames processed successfully | "Live Mode", streaming video |
| `paused` | Source loaded, frame loop suspended | "Paused" veil over the last frame |
| `error` | Model load / inference / camera failure | "System Error", `detail` string in the Connection panel |

## 11. Failure Handling & Resilience

| Failure | Detection | Response |
|---|---|---|
| ultralytics missing | import guard `ULTRALYTICS_OK` | `error` + "MODEL RUNTIME UNAVAILABLE" placeholder, retries each second |
| Model file unreadable | `YOLO(path)` exception | `error` + "MODEL NOT LOADED" placeholder; config change can recover |
| Video missing / unopenable | `cap.isOpened()` false | `error` + "SOURCE NOT AVAILABLE" placeholder |
| Video EOF | `cap.read()` false | Loop if `loop=True`, else `idle` + "STREAM ENDED" |
| Camera frame grab failure | `cap.read()` false on camera source | `error` + "CAMERA ERROR" |
| Inference exception | try/except around track/predict | `error` state with message; loop continues |
| WebSocket drop | client `onclose` | Fallback to 350 ms polling; quiet reconnects |
| MJPEG stall | client watchdog (3 s without paint) | Transport restarted silently; legacy `<img>` path avoided entirely |
| Slow MJPEG client | `frame_cond` sequence check | Client skips frames instead of buffering |

## 12. Extension Points

| To add… | Touch |
|---|---|
| A new overlay layer | `cfg["overlays"]` + `PerceptionEngine._draw()` + the Visualizations page toggles |
| A new tunable | `cfg` default in `PerceptionEngine.__init__` + a control in `app.js` (`postCfg`) |
| A new API route | `backend/server.py` (keep it above the static mount) |
| A new metric in telemetry | Build it in `_push_stats()`; the frontend `render(s)` picks it up |
| A different detector/tracker | Point `model_name` / `tracker` config at new Ultralytics weights or tracker YAML |
| A new UI page | `PAGES` array + `<section class="page" id="page-…">` in `index.html` + nav item |