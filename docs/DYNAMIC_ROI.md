# Dynamic ROI — classical OpenCV engine

The dynamic corridor is implemented in `backend/perception/dynamic_roi.py` and
is deliberately independent of the detector. It accepts an image, fallback
polygon, and plain configuration mapping; its tests run without importing or
initializing YOLO. Object association is a second inference-free module in
`backend/perception/roi_association.py`.

## Processing stages

1. Convert the BGR frame to grayscale and apply a 5×5 Gaussian blur.
2. Run Canny edge detection with runtime low/high thresholds.
3. Create a configurable perspective trapezoid mask. Its center comes from the
   independent `search_center_ratio` control; the search horizon, top width,
   and bottom width can be adjusted in **Distance & ROI**. Static fallback
   geometry edits do not move or reshape this dynamic search mask.
4. Apply probabilistic Hough line detection (`HoughLinesP`). Mostly horizontal
   segments are rejected as lane candidates; long transverse segments are
   tracked as a warning signal for intersections/crosswalk-like scenes.
5. Cluster candidate segments by their extrapolated lower-frame positions.
   A deterministic RANSAC consensus fit rejects outliers, then an iteratively
   reweighted geometric fit selects a line or quadratic `x(y)` model. The
   quadratic is retained only if it materially improves the fit over a line and
   has enough vertical evidence.
6. Select a plausible left/right pair using support, lower-frame width bounds,
   and the independent dynamic search-center prior. A sampled corridor polygon
   is built from the two curves.
7. Smooth valid normalized geometry across frames. A failed estimate can reuse
   the last valid geometry only for `max_reuse_frames`, with decaying confidence;
   after that the existing `DrivingROI` trapezoid is returned.

A fit below `min_confidence` is rejected. Fits that pass that threshold but do
not reach `reliable_confidence` are still labelled `low_confidence` and
`uncertain`; the dashboard does not call them reliable. `reused`, `fallback`,
and `disabled` states are always uncertain.

## Runtime controls

The settings are nested under `config.roi` and can be sent to `POST /api/config`
without changing detector thresholds or weights:

```json
{
  "roi": {
    "dynamic_enabled": true,
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
    "center_tolerance_ratio": 0.30
  }
}
```

Ratios are relative to frame dimensions; the search width fields are full
widths. `smoothing` is the weight of a newly observed valid geometry (closer to
1.0 reacts faster). The original static controls (`height_start`,
`top_width_ratio`, `bottom_width_ratio`, `ego_center_ratio`, `bottom_margin`)
define the editable `DrivingROI` fallback trapezoid and its blue display guide.
Dynamic search uses `search_center_ratio` (default `0.62`) independently,
alongside its horizon and width controls; changing the static center does not
change the dynamic search prior. Values are range-checked by
`DynamicROIConfig.from_mapping` before use.

The `lane` overlay displays left/right boundary curves and the current certainty
state. The `roi` overlay displays only the active dynamic corridor. The static
blue trapezoid is a separate layer, visible by default and controlled by
`overlays.static_roi`; hiding it is display-only. The estimator's
`ROIFrameState.search_region` is an internal mask and has its own optional
`overlays.search_region` debug outline. `static_roi_polygon` and `roi_polygon`
telemetry remain separate for the frontend schematic and coordinate display.
The video stage, System Status module, ROI controls, and ROI schematic expose
the dynamic state/confidence. If the camera aspect ratio changes materially,
stale geometry is cleared before processing the new view.

## Object association contract

`ROIObjectAssociator.associate(...)` receives a plain bbox, polygon, frame shape,
class name, confidence, optional track ID, and ROI certainty metadata. It
combines:

- bottom-center point-in-polygon membership;
- rasterized bounding-box/polygon overlap;
- detector confidence and a light class-context prior;
- a short score history for tracked IDs (no history is invented without an ID).

It returns a score, overlap, relevance label, uncertainty flag and a legacy
`in_path` boolean for the existing safety calculations. The function never
filters an input detection. The pipeline sends all YOLO detections to the
frontend, including boxes outside the corridor; only risk relevance is
classified separately.

## Telemetry and metrics

`roi` reports `state`, `confidence`, `uncertainty`, `reason`, `source`,
`candidate_count`, and `frames_since_detection`. `roi_polygon` carries the
current `ROIFrameState` geometry (estimated, reused, or fallback), corners,
left/right sampled boundaries when available, internal search mask, frame
dimensions, and uncertainty metadata. `static_roi_polygon` separately carries
the configured editable fallback trapezoid.

`perf.fps` is measured between completed annotated-frame outputs and smoothed as
an EMA. It remains `null` before two output frames exist. For video sources,
normal source pacing is included in the throughput measurement; for cameras,
there is no assumed frame rate. `source.fps` contains reported source metadata
only when it is plausible, otherwise `null`.

`perf.inference_ms` is measured around the existing YOLO11 call.
`perf.processing_ms` is measured from frame acquisition start through JPEG
encoding (excluding intentional source pacing). `perf.roi_ms` is measured around
the independent OpenCV ROI update. None of these fields is populated by a fixed
or fabricated value.

## Tests

Run from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/test_dynamic_roi.py` uses generated frames for curved lanes, multiple
lane markings and ego-center selection, absent/one-sided markings, hard shadows,
intersection/crosswalk bars, several resolutions/aspect ratios, disabled mode,
and temporary evidence failures with bounded reuse and confidence decay.
`tests/test_roi_association.py` verifies inside/outside retention, overlap-based
association, class/confidence inputs, track-score history, and fallback
uncertainty. No model weights or internet access are needed for these tests.

## Limitations and safety note

This is a classical edge/line heuristic, not a learned semantic lane detector.
It cannot reliably determine which painted line belongs to the ego lane, infer
road topology, distinguish every lane marking from shadows/cracks/curbs, or
understand intersections, merges, construction markings, or unusual camera
mounting. Night, glare, rain, worn paint, occlusion, rolling shutter, camera
pitch changes, wide-angle distortion, and low-resolution footage can all break
its assumptions. Multiple lanes are disambiguated only by the configurable
center/width prior; that is not semantic lane identification.

The fallback trapezoid is operator-configured geometry, not measured lane data.
Reused geometry is stale and explicitly uncertain. Even a high-confidence
classical fit can be wrong. This feature is for a research/visualization
prototype; do not use it for braking, steering, or safety-critical decisions.
