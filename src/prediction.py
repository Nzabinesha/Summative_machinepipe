"""Load the saved model and predict a single image."""
import threading
import time

import numpy as np
import tensorflow as tf

from src.model import MODEL_PATH
from src.preprocessing import CLASS_NAMES, prepare_image

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def reload_model():
    """Call this after retraining so the API serves the new model."""
    global _model
    with _lock:
        _model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(source, top_k=3):
    """source: file path, bytes, or file-like object."""
    x = prepare_image(source)
    start = time.perf_counter()
    probs = get_model()(x, training=False).numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000

    order = np.argsort(probs)[::-1]
    return {
        "label": CLASS_NAMES[int(order[0])],
        "confidence": float(probs[order[0]]),
        "top_predictions": [{"label": CLASS_NAMES[int(i)], "confidence": float(probs[i])}
                            for i in order[:top_k]],
        "latency_ms": round(latency_ms, 2),
    }


if __name__ == "__main__":
    import sys
    print(predict_image(sys.argv[1]))
    