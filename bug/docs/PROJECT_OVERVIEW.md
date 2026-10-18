# AutoSense — Project Overview

> **SEE • TRACK • UNDERSTAND** — *AI for Safer Roads*

| | |
|---|---|
| **Project** | AutoSense — Autonomous Vehicle Perception Dashboard |
| **Kind** | Full-stack web application (engineering / research prototype) |
| **Repository** | `Sarthak156/AutoSense` — branch `main`, documented at commit `27a569c` |
| **Backend** | Python 3.11 · FastAPI · Ultralytics YOLO11n · ByteTrack · OpenCV |
| **Frontend** | Vanilla HTML/CSS/JavaScript (no build step) |
| **Status** | Working end-to-end — pipeline, API and UI verified against live footage on 27 Sep 2026 (see [RESULTS.md](RESULTS.md)) |

---

## 1. What AutoSense Is

AutoSense is a full-stack perception dashboard for **monocular autonomous-driving
research**. It accepts an ordinary video — a bundled dashcam sample, an uploaded
file, or a live camera feed — and runs the complete perception pipeline of the
[Vehicle_perception](https://github.com/Sarthak156/Vehicle_perception) project on
every frame:

```text
Video → YOLO11n → Detection → ByteTrack → Distance → TTC / Risk → Frontend
   └────→ OpenCV classical CV → Dynamic ROI / bounded fallback ────────┘
```

The backend preserves the existing detector, tracker, monocular distance
estimation, TTC and rule-based collision risk, wrapped in a threaded
`PerceptionEngine`. The forward corridor is estimated in a separate
training-free OpenCV branch (`dynamic_roi.py`) and falls back to the original
configurable trapezoid when evidence is weak. YOLO11 remains the only object
detection model; no learned lane model is added. The frontend streams annotated
video and live telemetry from the engine.

Nothing in the UI is invented. Output FPS is measured from completed frame
intervals, processing and inference latency use `perf_counter()`, object counts
come from current-frame YOLO detections, distances come from
`perception/distance.py`, WARNING/DANGER/CRITICAL levels come from
`safety/collision.py`, and actions come from `safety/decision.py`. Lane geometry
is explicitly marked uncertain when evidence is weak; ego speed and heading
remain unavailable because no sensors are connected.

## 2. Why It Exists

- **Run a real perception stack, not a mock-up.** Every metric on screen traces
  back to a genuine model inference on a genuine video frame.
- **Make the pipeline observable.** Detection thresholds, tracker choice, focal
  calibration, ROI geometry and risk thresholds are all adjustable at runtime,
  and their effect is visible immediately on the annotated stream.
- **Zero-install client.** The dashboard is static HTML/JS served by the same
  process as the API — open a browser and the demo video starts processing.
- **Honest engineering.** When a module is unavailable (no ultralytics, no video,
  no frame yet) the system says so through explicit states and placeholder
  frames instead of showing fake values.

## 3. Objectives

| Objective | How the project delivers it |
|---|---|
| **SEE** — detect road users | YOLO11n inference (`perception/detector.py`, `pipeline.py`) with live confidence / IoU / input-size control |
| **TRACK** — give objects identity | ByteTrack (selectable BoT-SORT) via `model.track()`, per-ID distance smoothing, active-track registry |
| **UNDERSTAND** — judge danger | Monocular distance → TTC → rule-based risk (SAFE/WARNING/DANGER/CRITICAL) → decision (CONTINUE/WARNING/BRAKE) |
| **SHOW** — expose everything | Annotated MJPEG stream, server-side overlay layers, WebSocket telemetry at ~5 Hz, session report |

## 4. Feature Overview

| Area | What it does |
|---|---|
| **Dashboard** | Live annotated video, overlay toggles, recent detections with real frame-crop thumbnails, class-distribution donut, FPS history graph, live statistics and safety panel |
| **Video Input** | Upload video, load bundled samples, live camera (device index), transport controls (play/pause/resume/restart/stop) |
| **Detection** | Confidence / IoU thresholds, YOLO input-size presets (320 / 416 / 640 …), live detection table with per-object distance / TTC / risk |
| **Tracking** | Enable/disable ByteTrack, switch ByteTrack ↔ BoT-SORT, active-track table (ID, class, age, seen frames) |
| **Distance & ROI** | Focal-length calibration, static fallback corridor, dynamic OpenCV search/edge/Hough controls, confidence/reuse status and live schematic |
| **Safety Analysis** | Overall risk, decision (CONTINUE / WARNING / BRAKE), per-object TTC & closing speed, threshold table, risk timeline |
| **Visualizations** | Server-side overlay layer control (boxes, confidence, IDs, distance, dynamic ROI, lane boundaries/uncertainty, risk HUD) |
| **Model** | Active weights, class list, runtime GPU/CPU selection, upload custom trained `best.pt` |
| **Settings** | Dark / Light / System themes, MJPEG ↔ snapshot-polling transport, connection panel, session report download |

Panel-by-panel details are in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) §7.

## 5. Who It Is For

- **Students & researchers** exploring ADAS / perception pipelines who want a
  real model (not a mock) behind an understandable UI.
- **Reviewers & supervisors** who need measured evidence — every panel traces
  back to a formula and a log line (see [RESULTS.md](RESULTS.md)).
- **Engineers** prototyping detection / tracking / safety parameters, e.g.
  sweeping confidence thresholds or comparing trackers on their own footage.

## 6. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Detection | Ultralytics YOLO11n (`yolo11n.pt`) | 80 COCO classes; any `.pt` can be uploaded and selected |
| Tracking | ByteTrack / BoT-SORT (Ultralytics built-ins) | `bytetrack.yaml` default; switchable at runtime |
| Vision I/O | OpenCV (`cv2`) | Capture, drawing, JPEG encode |
| Math | NumPy | ROI polygon tests, image operations |
| Backend | FastAPI + uvicorn | REST + WebSocket + static file serving in one process |
| Inference runtime | PyTorch (CPU or CUDA) | GPU mode used in the measured runs (RTX 2050) |
| Frontend | Vanilla HTML / CSS / JavaScript | No framework, no build step, no npm |
| Config | JSON over REST | Deep-merged live into the engine |

## 7. Repository Layout

```
autosense/
├── backend/
│   ├── perception/       ← existing modules plus the new classical-CV ROI
│   │   ├── detector.py   YOLO predict wrapper
│   │   ├── distance.py   monocular distance (H·F/h) + smoothing
│   │   ├── roi.py        configurable trapezoid fallback
│   │   ├── dynamic_roi.py OpenCV-only dynamic corridor geometry
│   │   └── roi_association.py box/track-to-ROI relevance (no inference)
│   ├── tracking/
│   │   └── tracker.py    ByteTrack wrapper + box extraction
│   ├── safety/
│   │   ├── ttc.py        closing speed + time-to-collision
│   │   ├── collision.py  rule-based risk levels
│   │   └── decision.py   CONTINUE / WARNING / BRAKE
│   ├── visualization/
│   │   └── overlay.py    reference drawing routines
│   ├── pipeline.py       PerceptionEngine — threaded wrapper, metrics, MJPEG
│   ├── server.py         FastAPI REST + WebSocket + static frontend
│   ├── measure_fps.py    standalone FPS measurement client
│   ├── models/           runtime weights; not included in the Git checkout
│   ├── samples/          optional local demo clips; not included in the Git checkout
│   └── uploads/          user-uploaded videos (created/used at runtime)
├── frontend/
│   ├── index.html        dashboard shell (9 pages)
│   ├── css/style.css     dark + light themes
│   └── js/app.js         router, telemetry, video transports, charts
└── requirements.txt      backend dependencies
```

## 8. Quick Start

**Prerequisites:** Python 3.10+ and, for GPU inference on Windows, the CUDA build
of PyTorch (the plain `pip install torch` wheel is CPU-only).

```bash
cd backend
pip install -r ../requirements.txt     # ultralytics, fastapi, uvicorn, opencv, torch…

# optional, NVIDIA GPUs — replace CPU torch/torchvision with the matching
# CUDA build selected at https://pytorch.org/get-started/locally/
# Do not reinstall requirements.txt after installing the CUDA build.

python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. No sample video is committed; use Video Input to
upload a clip or start a camera. A local `backend/samples/dashcam_city.mp4` is
loaded automatically if you add it. Verify from another terminal:

```bash
curl http://localhost:8000/healthz        # {"ok": true, "state": "running"}
curl http://localhost:8000/api/status     # full telemetry snapshot
```

Media assets (`yolo11n.pt`, sample videos) are not committed to git; the model
is fetched on first use by Ultralytics, and any `.mp4` can be dropped into
`backend/samples/` or uploaded through the UI. GPU selection lives in the UI
(Detection → Compute Device); measured numbers for this exact stack are in
[RESULTS.md](RESULTS.md).

## 9. The "No Fabricated Data" Principle

This project treats invented telemetry as a bug. Concretely:

| Value | Source of truth |
|---|---|
| FPS | Timer around processed frames, EMA-smoothed in `pipeline.py` |
| Inference time | `time.perf_counter()` around the YOLO call |
| Object counts | Detections of the *current* frame, grouped by class |
| Distance | `DistanceEstimator.estimate()` (H·F/h + track smoothing) |
| TTC / closing speed | `TTCEstimator.update()` from distance/time deltas |
| WARNING / DANGER / CRITICAL | `CollisionRisk.evaluate()` rule table |
| CONTINUE / WARNING / BRAKE | `DecisionEngine.decide()` mapping |
| Ego speed, heading | Published as `null` — no sensor is connected |
| Lane / ROI geometry | Training-free grayscale/blur/Canny/Hough fitting with confidence, bounded reuse and a configured fallback; weak evidence is explicitly marked uncertain |

If the model is missing, the video cannot open, or no frame has been processed,
the dashboard shows explicit idle/error states and a labelled placeholder frame
(`NO INPUT SOURCE`, `MODEL NOT LOADED`, `SOURCE NOT AVAILABLE`, `STREAM ENDED`)
instead of stale or imaginary values.

## 10. Non-Goals and Safety Notice

- **Not a driving system.** The decision output (CONTINUE / WARNING / BRAKE) is a
  UI/research signal only. Nothing in this repository actuates a vehicle, and
  `safety/collision.py` explicitly documents itself as a prototype.
- **Not a certified ADAS.** Distances are monocular estimates that depend on an
  assumed object height and a focal length you calibrate; they carry real error
  and must not be used for safety-critical decisions.
- **No semantic lane guarantee.** Classical edges/Hough cannot reliably
  distinguish lane paint from shadows, road seams, crosswalks or curbs, nor
  determine ego-lane topology on multi-lane roads. Reused/fallback geometry is
  explicitly uncertain; the ROI is for research visualization only.
- **Not a cloud service.** No authentication, no multi-tenant storage — deploy
  on localhost or trusted networks only.
- **No simulated sensors.** Where a sensor is absent (speed, IMU, GNSS), the UI
  shows "not available" instead of an estimate.

## 11. Documentation Map

| Document | Read it for |
|---|---|
| **PROJECT_OVERVIEW.md** (this file) | What the project is, features, stack, quick start |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, threading, data flows, transports, state machine |
| [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) | Module-by-module reference, config keys, telemetry schema, API, troubleshooting |
| [RESULTS.md](RESULTS.md) | Measured FPS / inference / API results on a real machine, with analysis |
| [FUTURE_SCOPE.md](FUTURE_SCOPE.md) | Roadmap: logging, ego-speed, edge deployment, research ideas |

## 12. Project Status

- **Validated commit:** `27a569c` on `main`, with functional verification and
  performance measurements recorded on 27 Sep 2026 ([RESULTS.md](RESULTS.md)).
- **Verified working:** end-to-end detection → tracking → distance → TTC → risk →
  overlay → MJPEG/telemetry on all three bundled samples and a 1080p upload;
  CPU and GPU device modes; WebSocket and polling transports.
- **Known limitations** are catalogued in [RESULTS.md](RESULTS.md) §6 and
  [FUTURE_SCOPE.md](FUTURE_SCOPE.md) §1 — most notably the absence of automated
  tests and the rule-based (non-learned) risk model.