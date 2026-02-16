from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import face_alignment
import torch

app = FastAPI()

# ✅ In production, replace "*" with your Netlify domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=device,
    flip_input=False
)

@app.get("/")
def root():
    return {"ok": True, "model": "FAN face-alignment", "device": device}

@app.post("/landmarks")
async def landmarks(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(img)

    preds = fa.get_landmarks(img_np)
    if preds is None or len(preds) == 0:
        raise HTTPException(status_code=422, detail="No face detected.")

    pts = preds[0]  # (68,2)

    return {
        "width": img_np.shape[1],
        "height": img_np.shape[0],
        "landmarks68": pts.tolist()
    }
