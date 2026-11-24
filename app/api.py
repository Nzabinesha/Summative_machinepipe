"""FastAPI service: predict, upload, train/retrain, evaluate, status."""
import io
import os
import socket
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from src import model as model_lib
from src import prediction
from src.preprocessing import (CLASS_NAMES, FOLDER_NAMES, IMAGE_EXTS, NEW_DIR,
                               TEST_DIR, TRAIN_DIR, count_images, load_folder,
                               save_upload)

STARTED_AT = time.time()
HOSTNAME = socket.gethostname()
REQUESTS = {"predict": 0, "errors": 0}
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() == "true"
THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", model_lib.RETRAIN_THRESHOLD))

_job_lock = threading.Lock()
_reload_lock = threading.Lock()
_loaded_mtime = None
_stats_cache = {"key": None, "value": None}


def _ensure_fresh_model():
    """Reload the model if the file changed (e.g. another container retrained it)."""
    global _loaded_mtime
    mtime = model_lib.MODEL_PATH.stat().st_mtime   # raises FileNotFoundError if missing
    if _loaded_mtime != mtime:
        with _reload_lock:
            if _loaded_mtime != mtime:
                prediction.reload_model()
                _loaded_mtime = mtime


@asynccontextmanager
async def lifespan(app):
    try:
        _ensure_fresh_model()
    except Exception as e:
        print(f"WARNING: could not load model: {e}")
    yield


app = FastAPI(title="Fashion-MNIST ML Pipeline API", lifespan=lifespan)


# ---------- helpers ----------
def _is_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def _class_counts(folder):
    counts = []
    for name in FOLDER_NAMES:
        d = Path(folder) / name
        counts.append(sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                      if d.is_dir() else 0)
    return counts


def _run_job(fn, kwargs):
    try:
        fn(**kwargs)
        _ensure_fresh_model()
    except Exception as e:
        model_lib._set_status(state="error", message=str(e))
    finally:
        _job_lock.release()


def _start_job(fn, **kwargs):
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "A training job is already running in this container")
    model_lib._set_status(state="queued", epoch=0, total_epochs=0, message="Starting")
    threading.Thread(target=_run_job, args=(fn, kwargs), daemon=True).start()


def _maybe_auto_retrain() -> bool:
    """Retrain trigger: enough new images waiting and AUTO_RETRAIN is on."""
    if AUTO_RETRAIN and model_lib.should_retrain(THRESHOLD):
        try:
            _start_job(model_lib.retrain)
            return True
        except HTTPException:
            return False
    return False


def _upload_response(saved, skipped):
    return {"saved": saved, "skipped": skipped,
            "pending_new_images": model_lib.pending_new_images(),
            "auto_retrain_started": _maybe_auto_retrain()}


# ---------- monitoring ----------
@app.get("/health")
def health():
    return {"status": "ok", "container": HOSTNAME}


@app.get("/status")
def status():
    metrics = {k: v for k, v in model_lib.read_metrics().items() if k != "confusion_matrix"}
    return {
        "uptime_seconds": round(time.time() - STARTED_AT),
        "started_at": datetime.fromtimestamp(STARTED_AT).isoformat(timespec="seconds"),
        "container": HOSTNAME,
        "model_loaded": model_lib.MODEL_PATH.exists(),
        "metrics": metrics,
        "train_images": count_images(TRAIN_DIR),
        "pending_new_images": model_lib.pending_new_images(),
        "retrain_threshold": THRESHOLD,
        "auto_retrain": AUTO_RETRAIN,
        "training": model_lib.get_training_status(),
        "requests": REQUESTS,
    }


@app.get("/metrics")
def metrics():
    return model_lib.read_metrics()


@app.get("/class-distribution")
def class_distribution():
    return {"classes": CLASS_NAMES, "folders": FOLDER_NAMES,
            "train": _class_counts(TRAIN_DIR), "new": _class_counts(NEW_DIR)}


@app.get("/dataset-stats")
def dataset_stats():
    """Per-class mean brightness and average image (for the visualizations page)."""
    key = count_images(TRAIN_DIR)
    if _stats_cache["key"] == key and _stats_cache["value"]:
        return _stats_cache["value"]
    X, y = load_folder(TRAIN_DIR, max_per_class=200)
    brightness, mean_images = [], []
    for i in range(10):
        Xi = X[y == i]
        if len(Xi):
            brightness.append(float(Xi.mean() * 255))
            mean_images.append(Xi.mean(axis=0)[:, :, 0].round(3).tolist())
        else:
            brightness.append(0.0)
            mean_images.append(np.zeros((28, 28)).tolist())
    value = {"classes": CLASS_NAMES, "mean_brightness": brightness, "mean_images": mean_images}
    _stats_cache.update(key=key, value=value)
    return value


# ---------- prediction ----------
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read()
    try:
        _ensure_fresh_model()
    except FileNotFoundError:
        raise HTTPException(503, "No trained model available. Call POST /train first.")
    except Exception as e:
        raise HTTPException(503, f"Model could not be loaded: {e}")
    try:
        result = prediction.predict_image(data)
    except Exception:
        REQUESTS["errors"] += 1
        raise HTTPException(400, "Could not read the image")
    REQUESTS["predict"] += 1
    result["container"] = HOSTNAME
    return result


# ---------- data upload ----------
@app.post("/upload")
def upload(class_name: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk upload of images that all belong to one class."""
    if class_name not in FOLDER_NAMES:
        raise HTTPException(400, f"class_name must be one of {FOLDER_NAMES}")
    saved = skipped = 0
    for f in files:
        data = f.file.read()
        if Path(f.filename).suffix.lower() in IMAGE_EXTS and _is_image(data):
            save_upload(data, f.filename, class_name)
            saved += 1
        else:
            skipped += 1
    return _upload_response(saved, skipped)


@app.post("/upload-zip")
def upload_zip(file: UploadFile = File(...)):
    """Zip with one folder per class, e.g. Sneaker/1.png, Bag/2.png."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(file.file.read()))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Not a valid zip file")
    saved = skipped = 0
    for info in archive.infolist():
        parts = Path(info.filename).parts
        if info.is_dir() or len(parts) < 2 or "__MACOSX" in parts or parts[-1].startswith("."):
            continue
        class_name, fname = parts[-2], parts[-1]
        data = archive.read(info)
        if (class_name in FOLDER_NAMES and Path(fname).suffix.lower() in IMAGE_EXTS
                and _is_image(data)):
            save_upload(data, fname, class_name)
            saved += 1
        else:
            skipped += 1
    return _upload_response(saved, skipped)


# ---------- training, retraining, evaluation ----------
@app.post("/retrain", status_code=202)
def retrain(epochs: int = 3):
    if model_lib.pending_new_images() == 0:
        raise HTTPException(400, "No new images uploaded yet")
    _start_job(model_lib.retrain, epochs=epochs)
    return {"message": "Retraining started"}


@app.post("/train", status_code=202)
def train(epochs: int = 5, subset: int = 10000):
    _start_job(model_lib.train, epochs=epochs, subset=subset)
    return {"message": "Training started"}


@app.post("/evaluate")
def evaluate_model():
    """Evaluate the model currently in production on data/test."""
    try:
        _ensure_fresh_model()
    except FileNotFoundError:
        raise HTTPException(503, "No trained model available")
    except Exception as e:
        raise HTTPException(503, f"Model could not be loaded: {e}")
    X, y = load_folder(TEST_DIR)
    if len(y) == 0:
        raise HTTPException(400, "No test images in data/test")
    result = model_lib.evaluate(prediction.get_model(), X, y)
    saved = model_lib.read_metrics()
    saved.update(result, evaluated_at=datetime.now().isoformat(timespec="seconds"))
    model_lib._write_metrics(saved)
    return {k: v for k, v in saved.items() if k != "confusion_matrix"}
