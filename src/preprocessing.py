"""Data acquisition and preprocessing."""
import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
NEW_DIR = DATA_DIR / "new"          # uploaded images waiting for retraining
MODELS_DIR = BASE_DIR / "models"

IMG_SIZE = 28
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Folder-safe names ("/" is not allowed in a folder name)
FOLDER_NAMES = [n.replace("/", "-") for n in CLASS_NAMES]
FOLDER_TO_INDEX = {name: i for i, name in enumerate(FOLDER_NAMES)}


# ---------- 1. Data acquisition ----------
def load_fashion_mnist():
    """Download Fashion-MNIST (used by train())."""
    import tensorflow as tf
    return tf.keras.datasets.fashion_mnist.load_data()


# ---------- 2. Processing ----------
def prepare_image(source):
    """Turn a path, bytes, or file-like object into a (1, 28, 28, 1) array in [0, 1].

    Fashion-MNIST items are light on a dark background, so photos with a
    light background are inverted. Non-square images are padded, not squashed.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source).convert("L")

    arr = np.asarray(img, dtype="float32")
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    img = ImageOps.pad(img, (IMG_SIZE, IMG_SIZE), color=int(np.median(border)))

    arr = np.asarray(img, dtype="float32") / 255.0
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 0.5:
        arr = 1.0 - arr
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def load_folder(folder, max_per_class=None, seed=42):
    """Load images from folder/<ClassFolder>/*.png into (X, y). Returns empty arrays if none."""
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for name, idx in FOLDER_TO_INDEX.items():
        class_dir = folder / name
        if not class_dir.is_dir():
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if max_per_class and len(files) > max_per_class:
            files = list(rng.choice(files, max_per_class, replace=False))
        for f in files:
            X.append(prepare_image(str(f))[0])
            y.append(idx)
    if not X:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 1), "float32"), np.empty((0,), "int64")
    return np.stack(X).astype("float32"), np.array(y, dtype="int64")


def count_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def save_upload(file_bytes, filename, class_name, dest=NEW_DIR):
    """Save one uploaded image under dest/<class_name>/ with a unique name."""
    if class_name not in FOLDER_TO_INDEX:
        raise ValueError(f"Unknown class '{class_name}'. Use one of {FOLDER_NAMES}")
    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")
    class_dir = Path(dest) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    out = class_dir / f"{uuid.uuid4().hex[:12]}{ext}"
    out.write_bytes(file_bytes)
    return out
