"""
AutoSense — FastAPI server
==========================

Serves the dashboard frontend (static), a REST control API, an MJPEG video
stream of the annotated perception output, and a WebSocket telemetry channel.

All telemetry originates from PerceptionEngine (backend/pipeline.py), which
wraps the original Vehicle_perception modules. No fabricated values.
"""

import asyncio
import json
import os
import shutil
import time

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, JSONResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import PerceptionEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

os.makedirs(UPLOADS_DIR, exist_ok=True)

engine = PerceptionEngine(models_dir=MODELS_DIR)

app = FastAPI(title="AutoSense", version="1.0.0")


# ---------------------------------------------------------------------------
# Models / sources
# ---------------------------------------------------------------------------

class SampleSource(BaseModel):
    name: str

class CameraSource(BaseModel):
    index: int = 0

class ModelSelect(BaseModel):
    name: str


def _video_listing():
    out = []
    for folder, kind in ((SAMPLES_DIR, "sample"), (UPLOADS_DIR, "upload")):
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                p = os.path.join(folder, f)
                out.append({"name": f, "path": p, "kind": kind,
                            "size_mb": round(os.path.getsize(p) / 1e6, 2)})
    return out


@app.get("/api/sources")
def list_sources():
    return {"videos": _video_listing()}


@app.post("/api/source/sample")
def load_sample(body: SampleSource):
    path = os.path.join(SAMPLES_DIR, body.name)
    if not os.path.isfile(path):
        # maybe it is an upload
        path = os.path.join(UPLOADS_DIR, body.name)
    if not os.path.isfile(path):
        return JSONResponse({"error": "video not found"}, status_code=404)
    engine.set_video_source(path, name=body.name)
    return {"ok": True, "source": engine.get_config()["source"]}


@app.post("/api/source/camera")
def start_camera(body: CameraSource):
    engine.set_camera_source(body.index)
    return {"ok": True, "source": engine.get_config()["source"]}


@app.post("/api/source/video")
async def upload_video(file: UploadFile = File(...)):
    safe = os.path.basename(file.filename or "upload.mp4")
    dest = os.path.join(UPLOADS_DIR, safe)
    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > 400 * 1024 * 1024:
                fh.close()
                os.remove(dest)
                return JSONResponse({"error": "file too large (max 400MB)"},
                                    status_code=413)
            fh.write(chunk)
    engine.set_video_source(dest, name=safe)
    return {"ok": True, "source": engine.get_config()["source"],
            "size_mb": round(size / 1e6, 2)}


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

@app.post("/api/control/{action}")
def control(action: str):
    actions = {
        "pause": engine.pause,
        "resume": engine.resume,
        "start": engine.start,
        "stop": engine.stop,
        "restart": engine.restart,
    }
    fn = actions.get(action)
    if fn is None:
        return JSONResponse({"error": "unknown action"}, status_code=404)
    fn()
    return {"ok": True, "state": engine.state}


@app.post("/api/config")
def update_config(patch: dict):
    cfg = engine.update_config(patch)
    return {"ok": True, "config": cfg}


@app.get("/api/models")
def models():
    return {"models": engine.available_models(),
            "active": engine.get_config()["model_name"]}


@app.post("/api/models/select")
def select_model(body: ModelSelect):
    if not os.path.isfile(os.path.join(MODELS_DIR, body.name)):
        return JSONResponse({"error": "model not found"}, status_code=404)
    engine.update_config({"model_name": body.name})
    return {"ok": True}


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    safe = os.path.basename(file.filename or "model.pt")
    if not safe.endswith(".pt"):
        return JSONResponse({"error": "only .pt files are supported"},
                            status_code=400)
    dest = os.path.join(MODELS_DIR, safe)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"ok": True, "name": safe,
            "size_mb": round(os.path.getsize(dest) / 1e6, 2)}


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@app.get("/api/status")
def status():
    return engine.get_stats()


@app.get("/api/report")
def report():
    return engine.get_report()


@app.get("/api/video/stream")
def video_stream():
    return StreamingResponse(
        engine.mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store",
                 "X-Accel-Buffering": "no"},
    )


@app.get("/api/video/frame")
def video_frame():
    """Latest annotated frame as a plain JPEG (polling transport)."""
    data = engine.latest_jpeg
    if data is None:
        return JSONResponse({"error": "no frame yet"}, status_code=404)
    from fastapi.responses import Response
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


# ---------------------------------------------------------------------------
# WebSocket broadcaster (stats pushed ~5 Hz; frontend polls as fallback)
# ---------------------------------------------------------------------------

class WSManager:
    def __init__(self):
        self.clients = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, message: str):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        # push an immediate snapshot on connect
        await ws.send_text(json.dumps(engine.get_stats()))
        while True:
            await ws.receive_text()  # keepalive/pings from client
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


@app.on_event("startup")
async def _broadcaster():
    async def _loop():
        last_sig = None
        while True:
            await asyncio.sleep(0.2)
            if not ws_manager.clients:
                continue
            s = engine.get_stats()
            sig = (s.get("perf", {}).get("frames_processed"),
                   s.get("state"))
            if sig == last_sig and s.get("state") not in ("running",):
                continue
            last_sig = sig
            await ws_manager.broadcast(json.dumps(s))
    asyncio.create_task(_loop())


# ---------------------------------------------------------------------------
# Frontend (static) — mounted last so /api routes win
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True, "state": engine.state}

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="app")


# ---------------------------------------------------------------------------
# Default demo wiring: load bundled sample automatically on boot
# ---------------------------------------------------------------------------

_default = os.path.join(SAMPLES_DIR, "dashcam_city.mp4")
if os.path.isfile(_default):
    engine.set_video_source(_default, name="dashcam_city.mp4")
