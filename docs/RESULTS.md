# AutoSense — Results

**Historical baseline:** measured behavior captured on **27 September 2026**
against commit `27a569c` (branch `main`), before the dynamic-ROI integration
and the current measured-pacing implementation. Every number below was produced
by that running system on the machine described in §2; it is not a benchmark of
the current checkout. Current metric semantics are documented in
[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) and
[DYNAMIC_ROI.md](DYNAMIC_ROI.md). Raw historical evidence: live `/api/status`
samples, WebSocket snapshots and `backend/measure_fps.py`.

## 1. Methodology

Two complementary measurement passes were used:

1. **Sweep pass — `backend/measure_fps.py`.** The repository's own measurement
   client boots the server, waits until the engine reports `running`, then walks
   four demo sources and several input sizes, printing the engine's per-frame
   metrics at each step: `fps`, `inference_ms`, source resolution, `imgsz`,
   frames processed and the safety summary. This pass deliberately includes
   cold-start and source-switching behavior.
2. **Verification pass — direct API probes.** A fresh server instance was
   queried over HTTP/WebSocket: `/healthz`, `/api/models`, `/api/sources`,
   `/api/report`, `/api/video/frame`, `/ws`, plus a CPU-vs-GPU device
   comparison through `POST /api/config  {"device": ...}`.

Interpretation notes:

- For this historical baseline, `fps` was an exponential moving average
  (`0.75·old + 0.25·new` per processed frame) of processed frames per second.
- That commit rate-limited video and camera sources using assumed cadence
  (including a 30 fps camera budget). The current checkout instead uses valid
  video metadata for pacing, does not assume camera FPS, and exposes measured
  `processing_ms` and `roi_ms`; see the current technical documentation.
- Numbers are machine- and thermal-state dependent; §5 states the variance we
  observed so readers can weigh single readings correctly.

## 2. Test Environment

| Item | Value |
|---|---|
| Operating system | Windows 10.0.26200 (64-bit) |
| CPU | 13th Gen Intel Core i5-13420H |
| System RAM | 15.6 GB |
| GPU | NVIDIA GeForce RTX 2050 (Laptop) |
| Python | 3.11.9 (`.venv`) |
| PyTorch | 2.14.0+cu130 — `torch.cuda.is_available() == True` |
| Ultralytics | 8.4.157 |
| OpenCV | 5.0.0 |
| FastAPI / uvicorn | 0.141.1 |
| Model | `yolo11n.pt` (15.9 MB, 80 COCO classes) |

## 3. Functional Verification (HTTP / WebSocket surface)

All checks below were executed against the live server and returned the
indicated results.

| Check | Request | Observed result |
|---|---|---|
| Health | `GET /healthz` | `{"ok": true, "state": "running"}` |
| Model registry | `GET /api/models` | `[("yolo11n.pt", active=true)]` — weights reported with size and active flag |
| Source listing | `GET /api/sources` | 3 samples (`dashcam_city.mp4`, `pedestrians.mp4`, `tunnel_drive.mp4`) + 1 upload (`video.mp4`), each with kind and size |
| Telemetry | `GET /api/status` | Full snapshot, 20 top-level keys, updated continuously |
| Annotated frame | `GET /api/video/frame` | `HTTP 200 · image/jpeg · 54,804 bytes · JPEG magic ffd8` |
| Telemetry push | `WS /ws` | Snapshot delivered on connect: `state=running`, `fps=24.3`, 20 keys |
| Session report | `GET /api/report` | Keys: `report, generated_at, session, configuration, model, current_counts, active_tracks, safety` |
| Session counters | `/api/report` after ~9 s uptime | `frames_processed=22`, `session_detections=121`, `avg_fps=23.7`, `last_inference_ms=39.1` |
| Device switch (CPU) | `POST /api/config {"device":"cpu"}` | Engine reloaded on CPU: `fps=16.2`, `inference_ms=54.4`, `device=CPU` |
| Device switch (GPU) | `POST /api/config {"device":"auto"}` | Back on GPU: `fps=26.4`, `inference_ms=30.9`, `device=GPU (RTX 2050)` |
| Boot to first frame | server start → `state=running` | 15.7 s (model load + capture open + first processed frame, warm disk cache) |

Observed pipeline behavior on real footage:

- **dashcam_city.mp4** — 4 objects in frame at sample time (vehicles and riders);
  cumulative 121 detections over the first 22 frames; TTC computed from distance
  deltas (e.g. `min_ttc = 0.91 s` for one approaching object — outside the ego
  corridor, so overall risk correctly stayed `SAFE`, see §5).
- **tunnel_drive.mp4** — 5–6 objects in frame; at 640 input the engine escalated
  to `overall_risk = WARNING` with `decision = WARNING`,
  `closest_distance = 11.78 m`, `min_ttc = 4.63 s` — a real hazard trace through
  the rule chain (see §5).
- **pedestrians.mp4** — pedestrian detections at 768×432; `SAFE` / `CONTINUE`
  in the sampled frame.
- **video.mp4 (1080p upload)** — served from `backend/uploads/`, 1080p capture
  confirmed in telemetry, processing sustained at ~6–7 fps.

## 4. Performance Results

### 4.1 Sweep pass — `measure_fps.py` (cold run, multi-source)

Engine-reported FPS (EMA) and measured inference time at each sampling moment:

| Source | Resolution | imgsz | FPS | Inference (ms) | Objects in frame |
|---|---|---|---|---|---|
| dashcam_city.mp4 (boot sample) | 960×540 | 640 | 9.3 | 94.4 | 4 |
| dashcam_city.mp4 | 960×540 | 320 | 8.7 | 98.1 | 4 |
| dashcam_city.mp4 | 960×540 | 416 | 8.3 | 101.2 | 2 |
| dashcam_city.mp4 (back to 640) | 960×540 | 640 | 11.4 | 58.0 | 3 |
| video.mp4 (uploaded, 1080p) | 1920×1080 | 640 | 7.3 | 86.7 | 3 |
| video.mp4 (uploaded, 1080p) | 1920×1080 | 320 | 6.3 | 108.3 | 3 |
| tunnel_drive.mp4 | 960×540 | 320 | 10.0 | 87.7 | 6 |
| tunnel_drive.mp4 | 960×540 | 640 | 9.3 | 105.1 | 5 |
| pedestrians.mp4 | 768×432 | 640 | 11.5 | 52.9 | 1 |

This pass intentionally includes cold-start, source-switching and GPU
power-state transitions; §4.3 explains why these numbers sit below the warm
steady state of §4.2.

### 4.2 Verification pass — fresh server, steady state

| Condition | FPS | Inference (ms) | Device reported |
|---|---|---|---|
| GPU (`device: auto`), dashcam_city @640 | **26.4** | 30.9 | `GPU (RTX 2050)` |
| CPU (`device: cpu`), dashcam_city @640 | 16.2 | 54.4 | `CPU` |
| GPU, boot snapshot (WebSocket) | 24.3 | — | `GPU (RTX 2050)` |
| GPU, ~9 s session (report) | 23.7 (session avg) | 39.1 (last frame) | `GPU (RTX 2050)` |

Reference points for the same session: 22 frames / 121 cumulative detections in
the first 9 s; dashcam_city's source cadence is 30 fps, which the engine enforces
as its rate cap.

### 4.3 Analysis

- **Historical source-rate cap dominated when inference was fast.** In the
  measured baseline, dashcam_city.mp4 ran at 30 fps and the then-current engine
  slept out the remainder of each 33 ms frame budget; the 26.4 fps reading was
  about 88% of that historical cap. This is not a claim about the current
  pacing implementation.
- **GPU vs CPU on this machine:** ≈ 1.6× higher FPS and ≈ 1.8× lower inference
  time on the RTX 2050 than on the i5-13420H alone. Both remain real-time for
  540p-class footage; 1080p input raised inference cost visibly (86.7 ms at 640).
- **Input size trades accuracy for speed**, but is not the only factor: on this
  laptop the dominant costs are model dispatch and GPU power state — some
  320 px runs measured *slower* per frame than 640 px runs, because the dGPU
  down-clocks under light load. The Turbo toggle (320 px) matters most on
  CPU-bound machines.
- **Run-to-run variance is real and quantified.** Identical configurations
  measured 8.3–11.5 fps in the cold sweep and 24–26 fps in the warm pass. One
  deliberately contended run (several processes sharing the GPU) showed
  inference times of **24 s** — a > 100× degradation — which is why §6.2
  recommends isolating the machine when recording benchmark numbers.
- **Cold start:** server boot to first processed frame took 15.7 s (model load,
  capture open, first inference — warm disk cache).

## 5. Safety-Layer Validation (rule-chain traces)

Two real incidents from the sweep demonstrate the full rule chain working on
footage, matching the thresholds in `safety/collision.py` exactly:

**Trace A — tunnel_drive.mp4 @640 (hazard correctly flagged):**

```
closest in-path distance = 11.78 m   (distance_warning = 12.0 m → triggered)
min TTC                  = 4.63 s    (ttc_warning = 4.0 s → not triggered)
⇒ overall_risk = WARNING   ⇒ DecisionEngine ⇒ decision = WARNING
```

**Trace B — dashcam_city.mp4 @640 (low TTC safely ignored):**

```
min TTC (any object)     = 0.91 s    (below ttc_critical = 1.0 s)
nearest in-path distance = none within thresholds
⇒ overall_risk = SAFE      ⇒ decision = CONTINUE
```

Trace B is the interesting one: `min_ttc` is computed across **all** detections
for information, while risk escalation considers **in-path objects only** — the
fast-approaching object was outside the ego corridor, and the system correctly
refused to escalate. Together the traces show the pipeline is not echoing
numbers: detection → distance → TTC → in-path gating → risk → decision each
contributed to the outcome.

## 6. Known Behaviors & Limitations Observed

### 6.1 Expected-but-notable behaviors

- `GET /api/video/frame` returns **404** until the first frame is processed
  (seen in early logs and reproduced) — the frontend treats this as normal and
  re-polls.
- The engine emits a **placeholder frame** (`NO INPUT SOURCE`, etc.) rather than
  stale video when no frame is flowing; telemetry reports `idle` / `error`.
- On Windows, closing a tab mid-stream logs
  `ConnectionResetError [WinError 10054]` from the asyncio proactor layer — a
  cosmetic uvicorn/asyncio artifact on WebSocket teardown, not a functional fault.

### 6.2 Measurement caveats

- All figures are from **one machine** (§2); treat them as a reference point,
  not a guarantee, and re-run `measure_fps.py` on the target hardware.
- GPU results depend on the dGPU power state. Contention with other processes
  degraded inference from ~30 ms to **24 s** in a deliberately overloaded run;
  run benchmarks on an idle machine.
- The historical FPS field was EMA-smoothed and source-capped. Current `perf.fps`
  is measured completed-frame throughput (including valid video pacing), with
  no assumed camera rate; allow intervals to settle after a source/config change.

### 6.3 Functional limitations (current phase)

- Monocular distance assumes a fixed focal length and class heights — accuracy
  varies with calibration quality and object pose.
- Risk thresholds are global constants (no per-class policies yet).
- No persistence: server restart resets runtime config to defaults.
- No authentication — localhost / trusted-network use only.

## 7. Reproducing These Results

```bash
# 1) environment (§2 versions)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2) start the server (the sweep client expects port 8765)
cd backend
python -m uvicorn server:app --host 127.0.0.1 --port 8765

# 3) in a second terminal — run the project's own measurement client
python measure_fps.py

# 4) spot-check the API surface
curl http://127.0.0.1:8765/healthz
curl http://127.0.0.1:8765/api/status
curl -o frame.jpg http://127.0.0.1:8765/api/video/frame
```

`measure_fps.py` prints one metric line per step and `MEASURE DONE` at the end;
compare your lines against §4.1.

## 8. Conclusion

On the measured machine, AutoSense processes 540p-class footage in **real time on
GPU** (24–26 fps against a 30 fps source cap, ~31 ms inference) and remains
usable on CPU alone (16 fps, ~54 ms), while correctly driving its safety state
machine from real detections — including a true WARNING escalation at
11.78 m / 4.63 s and a correctly suppressed false alarm. The full surface (REST,
WebSocket, MJPEG, polling, device switching, report export) was verified
end-to-end. The significant caveats are benchmark variance from GPU power state
and the research-grade, rule-based nature of the risk model.