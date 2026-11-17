"""FastAPI inference service.

    uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   liveness + whether a model is loaded
    POST /predict  multipart image -> score, verdict, latency
    GET  /stats    rolling drift signal
"""

from __future__ import annotations

import io
import os
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.data import build_transform
from src.model import CheckpointError, PatchCore

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/patchcore_carpet.pt")
# Optional overrides. Left unset, the input size and the threshold come from the checkpoint.
# INPUT_SIZE, if set, must equal the checkpoint's; a different value refuses to start.
INPUT_SIZE = os.getenv("INPUT_SIZE")
THRESHOLD = os.getenv("THRESHOLD")

model: PatchCore | None = None
transform = None
threshold: float | None = None
threshold_source: str | None = None

# Rolling window of recent scores. Cheap, memory-bounded drift signal: if the
# live mean drifts away from the value observed at validation, the input
# distribution has moved and the model needs re-checking.
recent_scores: deque[float] = deque(maxlen=500)


class Prediction(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    threshold: float
    latency_ms: float


def _load_model() -> None:
    """Load the checkpoint if there is one. An incompatible checkpoint raises CheckpointError,
    so the server refuses to start rather than serving scores that mean something else."""
    global model, transform, threshold, threshold_source
    model, transform, threshold, threshold_source = None, None, None, None
    recent_scores.clear()

    if not os.path.exists(MODEL_PATH):
        print(f"[warn] no model at {MODEL_PATH}; /predict will 503")
        return

    loaded = PatchCore.from_checkpoint(MODEL_PATH, input_size=int(INPUT_SIZE) if INPUT_SIZE else None)
    if THRESHOLD:
        value, source = float(THRESHOLD), "THRESHOLD environment variable"
    elif loaded.threshold:
        value, source = float(loaded.threshold["value"]), "checkpoint calibration"
    else:
        raise CheckpointError(
            f"{MODEL_PATH} has no calibrated threshold (it predates calibration or was trained with "
            "--calibration-fraction 0); set THRESHOLD explicitly"
        )
    model, threshold, threshold_source = loaded, value, source
    transform = build_transform(loaded.input_size)
    print(f"[ok] loaded {MODEL_PATH} (input {loaded.input_size}, threshold {value:.4f} from {source})")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Anomaly Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "input_size": model.input_size if model else None,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "seed": model.seed if model else None,
    }


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)) -> Prediction:
    if model is None:
        raise HTTPException(503, "Model not loaded")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Not a readable image")

    t0 = time.perf_counter()
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        scores, _ = model.predict(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000

    score = float(scores[0])
    recent_scores.append(score)

    return Prediction(
        anomaly_score=score,
        is_anomalous=score > threshold,
        threshold=threshold,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/stats")
def stats() -> dict:
    if not recent_scores:
        return {"n": 0, "message": "no predictions yet"}

    arr = np.array(recent_scores)
    return {
        "n": len(arr),
        "mean_score": round(float(arr.mean()), 4),
        "p95_score": round(float(np.percentile(arr, 95)), 4),
        "anomaly_rate": round(float((arr > threshold).mean()), 4) if threshold is not None else None,
    }