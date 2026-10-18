# AutoSense — Autonomous Vehicle Perception Dashboard

**SEE • TRACK • UNDERSTAND** — *AI for Safer Roads*

AutoSense is the existing full-stack YOLO11 perception dashboard with an
additional **training-free dynamic ROI engine**. YOLO11 remains the only object
detection model. The ROI engine is classical OpenCV computer vision; it does
not use YOLOP, a lane-segmentation network, a newly trained model, or any other
learned lane detector.

```text
Video / camera
  ├─ YOLO11 → detections → optional ByteTrack / BoT-SORT → distance / TTC / risk
  └─ OpenCV → grayscale → Gaussian blur → Canny → perspective mask → Hough
             → robust left/right curve fits → smoothed dynamic ROI / fallback
```

Every YOLO detection is retained. Objects are associated with the corridor and
labelled **in**, **near**, **outside**, or **uncertain**; ROI relevance never
filters the detection list. If lane evidence is poor, the UI reports uncertainty,
uses bounded recent geometry, and then falls back to the configured trapezoid.
Classical CV is not a reliable lane-semantics system and AutoSense is not a
certified ADAS or vehicle-control system.

## Setup and run

Requirements: Python 3.10 or 3.11 is recommended. Install the dependencies from
the repository root:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch` in `requirements.txt` provides a CPU installation on most standard pip
setups. For NVIDIA acceleration, first install the requirements above, then
replace the CPU PyTorch packages with the build matching the machine's CUDA/driver
using the [official PyTorch install selector](https://pytorch.org/get-started/locally/).
Do not reinstall `requirements.txt` afterward, or pip may replace the CUDA build.
GPU availability is reported from the actual runtime; choosing GPU in the UI
does not fabricate GPU capability.

Start the backend from its directory:

```bash
cd backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. The first YOLO11n load may download Ultralytics'
official `yolo11n.pt` checkpoint when the machine has network access. That
default checkpoint is COCO-trained (80 classes), so the Model page correctly
shows 80 names. A dataset such as [BDD100K in YOLO format](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo)
contains images, labels, and a 10-class `data.yaml`; it is not itself a trained
`.pt` checkpoint. To use BDD100K's `traffic light` and `traffic sign` labels,
provide an existing YOLO11 checkpoint trained on that dataset. Uploading a
`.pt` file on the Model page now selects it automatically, and the displayed
class names come from that checkpoint. Offline, you can place the checkpoint in
`backend/models/` and select it there. Checkpoints are ignored by Git in this
repository; this feature does not train or replace model weights.

No sample videos or weights are committed. Use **Video Input → Upload Video**
or start a connected camera. If you have a local demo clip, put it in
`backend/samples/`; the app will list it. Runtime uploads are stored locally in
`backend/uploads/`.

### Automated tests

```bash
# From repository root
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The dynamic ROI and association tests use synthetic OpenCV frames and do not
load YOLO weights or start the web server.

## Dynamic ROI controls

Open **Distance & ROI** to configure the original fallback trapezoid and the
new classical-CV engine:

- enable/disable dynamic geometry;
- set the dynamic search center independently from the static trapezoid, plus the perspective horizon and top/bottom widths;
- tune Canny low/high and Hough vote thresholds;
- set the minimum geometry confidence, temporal smoothing, and maximum reuse
  frames.

The original **Static Fallback ROI** sliders edit an independent blue trapezoid;
it remains visible alongside the dynamic corridor while its layer is enabled.
The dynamic estimator's internal perspective search mask is separate and can be
adjusted with its own center/horizon/width controls. In **Layers**, the static
trapezoid and optional search-mask debug outline have independent visibility;
hiding the static trapezoid does not disable or change dynamic estimation.

The **ROI Corridor** overlay draws the current estimated corridor.
**Lane Boundaries & Confidence** draws the dynamic boundaries and marks
low-confidence, reused, fallback, and disabled states. In the dashboard's System
Status and video stage, the current ROI state and confidence are also visible.

The object association uses bottom-center inclusion, bounding-box/polygon overlap,
class context, YOLO confidence, and a short tracked-object score history when a
track ID is available. The original detection class list, tracking controls,
ROI calibration controls, risk analysis, themes, uploads, and dashboard pages
remain in place.

## Measurements

- `perf.fps` is a rolling EMA of completed annotated-frame intervals (including
  intentional source-video pacing). It is `null` until there are two completed
  frames; it is not the video's advertised FPS.
- `source.fps` is the capture/container's reported metadata when it is valid;
  otherwise it is `null` (including cameras that do not report a rate).
- `perf.inference_ms` is measured YOLO11 inference time.
- `perf.processing_ms` is measured capture-read-to-JPEG-encode frame latency,
  excluding intentional source pacing.
- `perf.roi_ms` measures the independent classical-CV ROI update.

No camera rate, processing delay, or FPS is filled in with a fixed or simulated
number.

## Project layout

```text
backend/
  perception/
    detector.py          YOLO detector wrapper
    dynamic_roi.py       OpenCV dynamic ROI engine (YOLO-independent)
    roi_association.py   Detection-to-ROI relevance, no inference
    roi.py               Existing configurable fallback trapezoid
    distance.py          Existing monocular distance estimator
  pipeline.py            Threaded perception, tracking, metrics and overlays
  server.py              FastAPI REST / WebSocket / static frontend
frontend/
  index.html              Existing nine-page dashboard
  css/style.css           Existing dark/light theme system
  js/app.js               Existing controls and live telemetry

docs/DYNAMIC_ROI.md       Algorithm, configuration, tests and limitations
```

See [docs/DYNAMIC_ROI.md](docs/DYNAMIC_ROI.md) for implementation details,
limitations, telemetry fields, and test coverage.
