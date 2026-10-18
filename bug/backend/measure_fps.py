"""Measures the real backend pipeline FPS per input size on both sources."""
import json, time, urllib.request

BASE = "http://127.0.0.1:8765"

def api(path, method="GET", body=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def read(label):
    s = api("/api/status")
    print(f"{label:26s} fps={s['perf']['fps']:5.1f}  infer={s['perf']['inference_ms']}ms  "
          f"src={s['source']['name']} {s['source']['width']}x{s['source']['height']}  imgsz={s['config']['imgsz']}",
          flush=True)

time.sleep(12)                       # boot warm-up (dashcam auto-loads)
read("dashcam @640 (default)")
for sz in (320, 416, 640):
    api("/api/config", "POST", {"imgsz": sz}); time.sleep(7)
    read(f"dashcam @{sz}")

api("/api/source/sample", "POST", {"name": "video.mp4"})
time.sleep(14)                       # new source + EMA settle
read("video.mp4 @640")
for sz in (416, 320):
    api("/api/config", "POST", {"imgsz": sz}); time.sleep(7)
    read(f"video.mp4 @{sz}")
api("/api/config", "POST", {"imgsz": 320})
print("MEASURE DONE")
