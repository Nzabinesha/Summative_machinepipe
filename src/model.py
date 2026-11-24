"""Model creation, training, evaluation, retraining and the retrain trigger."""
import json
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers

from src.preprocessing import (CLASS_NAMES, FOLDER_NAMES, MODELS_DIR, NEW_DIR,
                               TEST_DIR, TRAIN_DIR, count_images,
                               load_fashion_mnist, load_folder)

MODEL_PATH = MODELS_DIR / "fashion_cnn.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"
STATUS_PATH = MODELS_DIR / "training_status.json"
ARCHIVE_DIR = MODELS_DIR / "archive"
RETRAIN_THRESHOLD = 50   # auto-retrain when this many new images are waiting

# Progress info that the API/UI can read
TRAINING_STATUS = {"state": "idle", "epoch": 0, "total_epochs": 0,
                   "message": "", "updated_at": None}


def _set_status(**kwargs):
    TRAINING_STATUS.update(kwargs, updated_at=datetime.now().isoformat(timespec="seconds"))
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(TRAINING_STATUS))
    except OSError:
        pass


def get_training_status():
    """Read progress from a shared file so every container sees the same status."""
    try:
        return json.loads(STATUS_PATH.read_text())
    except (OSError, ValueError):
        return dict(TRAINING_STATUS)


class _StatusCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        _set_status(epoch=epoch + 1,
                    message=f"val_accuracy={logs.get('val_accuracy', 0):.4f}")


# ---------- Model creation ----------
def build_cnn():
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        augment,
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------- Model testing ----------
def evaluate(model, X, y):
    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(y, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds, labels=range(10)).tolist(),
        "n_samples": int(len(y)),
    }


def read_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def _write_metrics(metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


# ---------- Training from scratch ----------
def train(epochs=10, batch_size=64, subset=None):
    """Train a fresh CNN on Fashion-MNIST. `subset` limits training size for quick runs."""
    _set_status(state="training", epoch=0, total_epochs=epochs, message="Loading data")
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    X_train = (X_train / 255.0).reshape(-1, 28, 28, 1).astype("float32")
    X_test = (X_test / 255.0).reshape(-1, 28, 28, 1).astype("float32")
    if subset:
        X_train, y_train = X_train[:subset], y_train[:subset]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    model = build_cnn()
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=batch_size, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
                         _StatusCallback()])

    metrics = evaluate(model, X_test, y_test)
    metrics.update(model_version=read_metrics().get("model_version", 0) + 1,
                   trained_at=datetime.now().isoformat(timespec="seconds"),
                   trained_on="fashion-mnist (train)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    _write_metrics(metrics)
    _set_status(state="idle", message=f"Trained v{metrics['model_version']}")
    return metrics


# ---------- Retraining ----------
def pending_new_images():
    return count_images(NEW_DIR)


def should_retrain(threshold=RETRAIN_THRESHOLD):
    """Retrain trigger: enough new labelled images have been uploaded."""
    return pending_new_images() >= threshold


def retrain(epochs=3, batch_size=32):
    """Fine-tune the saved model on new uploads + the existing training images."""
    n_new = pending_new_images()
    if n_new == 0:
        raise ValueError("No new images in data/new/. Upload some first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run train() first.")

    _set_status(state="retraining", epoch=0, total_epochs=epochs,
                message=f"Loading {n_new} new images")
    try:
        X_new, y_new = load_folder(NEW_DIR)
        X_old, y_old = load_folder(TRAIN_DIR)
        X_test, y_test = load_folder(TEST_DIR)

        X = np.concatenate([X_old, X_new]) if len(X_old) else X_new
        y = np.concatenate([y_old, y_new]) if len(y_old) else y_new
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        model = tf.keras.models.load_model(MODEL_PATH)
        before = evaluate(model, X_test, y_test)

        # lower learning rate so fine-tuning does not wipe out what was learned
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
                  batch_size=batch_size, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
                             _StatusCallback()])
        after = evaluate(model, X_test, y_test)

        # keep the old model so you can roll back
        old = read_metrics()
        version = old.get("model_version", 1)
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(MODEL_PATH, ARCHIVE_DIR / f"fashion_cnn_v{version}.keras")

        model.save(MODEL_PATH)
        after.update(model_version=version + 1,
                     trained_at=datetime.now().isoformat(timespec="seconds"),
                     trained_on=f"retrain: {n_new} new + {len(X_old)} old images",
                     previous_accuracy=before["accuracy"])
        _write_metrics(after)

        # new images become part of the training set
        for class_dir in NEW_DIR.iterdir():
            if class_dir.is_dir():
                target = TRAIN_DIR / class_dir.name
                target.mkdir(parents=True, exist_ok=True)
                for f in class_dir.iterdir():
                    shutil.move(str(f), str(target / f.name))

        _set_status(state="idle", message=f"Retrained: v{after['model_version']}")
        return after
    except Exception as e:
        _set_status(state="error", message=str(e))
        raise


if __name__ == "__main__":
    print(train(epochs=2, subset=5000))
