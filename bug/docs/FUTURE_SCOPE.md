# AutoSense — Future Scope

This document describes where the project can go next. It separates what exists
(and is therefore a foundation) from what remains proposed, and orders proposals
into phases by cost/benefit. The current training-free OpenCV ROI overlay is
implemented; future items in the phase tables remain proposals. `ego_speed` is
still published as `null` because no speed sensor is connected.

## 1. Current Boundaries (as shipped)

| Area | Today | Consequence |
|---|---|---|
| Input | One stream at a time (sample, upload, or camera index) | No multi-camera or multi-stream dashboards |
| Ego state | No speed / heading / yaw sensors — UI shows unavailable | TTC is derived purely from monocular distance deltas |
| Lane geometry | Dynamic ROI uses classical grayscale/blur/Canny/perspective-mask/Hough fitting, smoothing, bounded reuse and a configurable fallback; it does not infer lane semantics | In-path relevance is geometric and can be wrong; uncertainty is surfaced |
| Risk logic | Fixed rule table (distance & TTC thresholds in `safety/collision.py`) | Thresholds are constants, not per-class or scenario-learned |
| Persistence | Config and session state live in memory; media in local folders | Server restart resets `imgsz` etc.; no database |
| Security | No authentication — intended for localhost / dev networks | Must not be exposed to untrusted networks |
| Testing | Synthetic dynamic ROI and object-association suites plus manual API verification | No automated end-to-end browser/model test suite yet |
| Deployment | Single uvicorn process; CPU or CUDA PyTorch | No container, no edge-device packaging |

## 2. Guiding Principles for Future Work

1. **Keep the honesty invariant** — new features must publish real measurements
   or an explicit "unavailable", never simulated values.
2. **Keep the transport-agnostic backend** — new modalities (speed, GNSS, IMU,
   lane data) enter through the config/telemetry layer, not through UI hacks.
3. **Prefer runtime-selectable modules** — the tracker switch (ByteTrack ↔
   BoT-SORT) and the overlay-layer toggles are the pattern to copy.
4. **Measure everything** — every phase should end with numbers, like
   [RESULTS.md](RESULTS.md).

## 3. Phase 1 — Near-Term (harden what exists)

Foundations already present in code: the dynamic classical-CV ROI engine and
`overlays.lane`, the `ego_speed` field, the `recent` / `fps_history` /
`risk_history` buffers, the report endpoint and the calibration sliders.

| Item | Description | Grounding in current code |
|---|---|---|
| **Session logging & export** | Persist per-frame telemetry to CSV/JSONL (timestamp, fps, counts, risk, closest distance, per-track TTC) so sessions can be re-analysed offline; extend `GET /api/report` with download links | `_push_stats()` already builds the full payload every frame |
| **Incident snapshots** | Save an annotated JPEG + telemetry slice automatically when risk ≥ WARNING, and on demand from the UI | Risk escalation computed per frame; `_crop_thumb` / JPEG encode reusable |
| **Ego-speed input** | Accept a speed source (manual entry or OBD/GPS import) so TTC can account for ego motion; keep publishing `null` when absent | `stats["ego_speed"]` is already in the payload and rendered as "—" |
| **Calibration assist** | A wizard that estimates `focal_length` from a known object height/distance, plus a sanity overlay | `focal_length` is already a live-tunable config value |
| **Per-class risk policy** | Make the constants in `CollisionRisk` class-aware (pedestrians / cyclists escalate earlier) and editable from the Safety page | `CollisionRisk.evaluate()` already receives `class_name` but ignores it |
| **Broader automated tests** | Add suites for distance, TTC, risk and decision logic plus a FastAPI `TestClient` smoke test for every endpoint | `tests/test_dynamic_roi.py` and `tests/test_roi_association.py` cover synthetic lane geometry and association without weights |
| **Config persistence** | Persist runtime config (imgsz, thresholds, ROI) server-side so restarts don't reset it | The frontend already persists a subset in `localStorage` as a workaround |

## 4. Phase 2 — Deployment & Scale

Goal: make AutoSense usable beyond a developer laptop — on edge hardware, with
recorded evidence and cleaner models.

| Item | Description | Notes |
|---|---|---|
| **Model format acceleration** | ONNX / TensorRT export of YOLO11n (`model.export(format=...)`), INT8 option, benchmarked against the current PyTorch path | Ultralytics provides the export path; the pipeline only needs a "weights + format" registry |
| **Multi-stream support** | Several `PerceptionEngine` instances (or a worker pool) with per-stream telemetry namespaces; UI gains a stream switcher | Each engine is already self-contained (own thread, config, histories) |
| **Edge packaging** | Dockerfile + compose file, plus a Jetson (JetPack) variant; pinned torch/ultralytics versions; healthcheck on `/healthz` | Matches the single-process deployment model |
| **Recording & replay studio** | Record the annotated stream to file, store incident clips, replay any session with its telemetry timeline | Frames are already encoded centrally; add an ffmpeg / OpenCV writer toggle |
| **Dataset & retraining loop** | Store raw frames + labels for hard cases (missed detections, false positives); schedule retraining on uploaded footage | Pairs with the model-upload endpoint that already accepts `best.pt` |
| **Authentication & access control** | API keys or reverse-proxy auth, per-source permissions, upload rate limiting | Required before any non-localhost deployment |
| **Observability** | Structured JSON logs; `/metrics` (Prometheus) for FPS / inference / queue depth; alerts on inference degradation | Uses existing counters (`frames_processed`, `inference_ms`, `fps`) |
| **Cross-platform releases** | CI matrix (Windows / Linux) building wheels and a packaged release that bundles a default model | `.gitignore` excludes media, so releases must ship assets explicitly |

## 5. Phase 3 — Research Directions

Longer-horizon ideas, kept honest about uncertainty:

| Direction | Sketch |
|---|---|
| **Sensor fusion** | Fuse radar / LiDAR or a depth camera with the monocular branch; keep the UI able to show per-source provenance |
| **Ego-motion compensation** | Estimate camera motion (optical flow / visual odometry) so distances and TTC account for ego displacement |
| **Learned risk models** | Replace or augment the rule table with a calibrated probabilistic model (e.g. survival analysis on TTC), keeping the rule engine as a cross-check |
| **Trajectory forecasting** | Predict short-horizon object paths (constant-velocity → learned predictors) for earlier warnings |
| **Scene understanding** | Drivable-area / lane semantics (crossings, sidewalk vs roadway) to contextualize risk |
| **Driver monitoring** | Secondary camera branch for attention / distraction signals next to the road-facing branch |
| **Fleet / V2X dashboards** | Aggregate many AutoSense instances: alert maps, session comparison, fleet-wide model rollouts |

## 6. Engineering Quality Track (parallel, ongoing)

| Item | Why |
|---|---|
| Automated test suite | The safety math (distance, TTC, risk, decision) deserves regression tests before any threshold changes |
| Type hints + docstring pass | The engine is stable enough to type-check; helps refactors |
| Split `pipeline.py` internals | Extract drawing and stats-building into helper modules once tests exist (behavior-preserving) |
| Lint / format in CI | Ruff / Black or equivalent, plus a smoke test that boots the server and hits `/healthz` |
| Frontend module split | `app.js` is ~1,100 lines; ES modules would improve navigation without adding a build step |

## 7. Explicit Non-Goals

To keep the roadmap honest, these stay out of scope for this project:

- **Vehicle actuation.** AutoSense will not drive or brake anything; decisions
  are advisory UI output only.
- **Safety certification.** Nothing here is certified or certifiable ADAS; the
  collision model is explicitly a research prototype.
- **Faking unavailable signals.** No simulated speed / heading / lane data will
  ever be displayed as if measured.
- **Public-internet exposure without auth.** The auth phase comes first; until
  then the app is a localhost / lab tool.

## 8. Closing Note

The project's core promise — *every number on screen is produced by the real
pipeline* — is also its roadmap constraint: each phase above is only "done" when
its outputs are measurable and honest, in the same style as
[RESULTS.md](RESULTS.md). Contributions should follow the same pattern: build the
feature, measure it, document the measurement.