from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageOps
import io
import os
import time
import threading

import torch
import face_alignment

app = FastAPI()

# CORS: in production you should lock this down to your frontend domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Configuration (tweakable via env vars)
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Downscale big images to avoid huge RAM/CPU spikes
MAX_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1280"))  # 1024–1600 is a good range
# Safety: reject extremely large uploads (bytes)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))  # 8MB default
# How long we allow inference to "take" before we say it's slow (for logging)
SLOW_INFER_MS = int(os.getenv("SLOW_INFER_MS", "2500"))

# ----------------------------
# Global model + lock (prevents concurrent inference crashes/timeouts)
# ----------------------------
_fa = None
_model_lock = threading.Lock()
_model_loaded_at = None


def _load_model():
    """Load the FAN model once and cache globally."""
    global _fa, _model_loaded_at
    if _fa is not None:
        return _fa

    # Create model
    _fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=DEVICE,
        flip_input=False,
    )
    _model_loaded_at = time.time()
    return _fa


def _prepare_image_bytes(data: bytes) -> np.ndarray:
    """Decode image, fix EXIF orientation, convert to RGB, downscale, return numpy."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (>{MAX_UPLOAD_BYTES} bytes).")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Fix iPhone / camera rotations
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Downscale if huge
    w, h = img.size
    max_side = max(w, h)
    if max_side > MAX_SIDE:
        scale = MAX_SIDE / float(max_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

    return np.array(img)


def _infer_landmarks(img_np: np.ndarray):
    """Run inference with a global lock to protect memory/CPU."""
    fa = _load_model()

    start = time.time()
    with _model_lock:
        preds = fa.get_landmarks(img_np)
    ms = int((time.time() - start) * 1000)

    if ms > SLOW_INFER_MS:
        # lightweight logging (shows in Railway logs)
        print(f"[warn] slow inference: {ms}ms, img={img_np.shape}")

    if preds is None or len(preds) == 0:
        raise HTTPException(status_code=422, detail="No face detected.")
    return preds[0], ms


# ----------------------------
# Startup warmup (optional but helpful)
# ----------------------------
@app.on_event("startup")
def startup_event():
    """
    Try to load the model at startup.
    If memory is tight, you can comment this out and rely on /warmup or first request.
    """
    try:
        _load_model()
        print(f"[startup] FAN loaded on {DEVICE}")
    except Exception as e:
        # Don't crash the container just because warm startup failed
        print(f"[startup] model load failed (will lazy-load on first request): {e}")


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "model": "FAN face-alignment",
        "device": DEVICE,
        "modelLoaded": _fa is not None,
        "maxImageSide": MAX_SIDE,
    }


@app.get("/health")
def health():
    # Useful if you want Railway health checks
    return {
        "ok": True,
        "status": "healthy",
        "device": DEVICE,
        "modelLoaded": _fa is not None,
        "modelLoadedSecondsAgo": None if _model_loaded_at is None else int(time.time() - _model_loaded_at),
    }


@app.post("/warmup")
def warmup():
    """
    Hit this once after deploy (or periodically) to reduce cold-start pain.
    """
    try:
        _load_model()
        # tiny dummy image (no face) just to exercise the pipeline
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        try:
            with _model_lock:
                _fa.get_landmarks(dummy)
        except Exception:
            # it's okay if no face detected; we're just warming things up
            pass
        return {"ok": True, "warmed": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.post("/landmarks")
async def landmarks(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    data = await file.read()
    img_np = _prepare_image_bytes(data)

    pts, infer_ms = _infer_landmarks(img_np)

    return {
        "width": int(img_np.shape[1]),
        "height": int(img_np.shape[0]),
        "landmarks68": pts.tolist(),
        "inference_ms": infer_ms,
    }
