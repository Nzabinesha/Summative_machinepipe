"""Streamlit UI: uptime, visualizations, prediction, upload and retrain."""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Fashion ML Pipeline", layout="wide")


def api(method, path, timeout=60, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            return None, f"{r.status_code}: {detail}"
        return r.json(), None
    except requests.RequestException as e:
        return None, f"API unreachable: {e}"


def fmt_uptime(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


# ---------------- Dashboard ----------------
def page_dashboard():
    st.header("Model dashboard")
    s, err = api("GET", "/status")
    if err:
        st.error(err)
        return
    m = s["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Service uptime", fmt_uptime(s["uptime_seconds"]))
    c2.metric("Model version", m.get("model_version", "-"))
    c3.metric("Accuracy", f"{m['accuracy']:.4f}" if "accuracy" in m else "-")
    c4.metric("F1 (macro)", f"{m['f1']:.4f}" if "f1" in m else "-")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Precision (macro)", f"{m['precision']:.4f}" if "precision" in m else "-")
    c6.metric("Recall (macro)", f"{m['recall']:.4f}" if "recall" in m else "-")
    c7.metric("Training images", s["train_images"])
    c8.metric("New images waiting", f"{s['pending_new_images']} / {s['retrain_threshold']}")

    t = s["training"]
    st.write(f"**Training state:** {t['state']}  |  **Predictions served:** "
             f"{s['requests']['predict']}  |  **Served by container:** `{s['container']}`")
    if m.get("trained_at"):
        st.caption(f"Last trained: {m['trained_at']} ({m.get('trained_on', '')})")
    if m.get("previous_accuracy") is not None:
        st.caption(f"Accuracy before last retrain: {m['previous_accuracy']:.4f}")

    col_a, col_b = st.columns(2)
    if col_a.button("Refresh"):
        st.rerun()
    if col_b.button("Evaluate model on test set"):
        with st.spinner("Evaluating..."):
            res, err = api("POST", "/evaluate", timeout=300)
        if err:
            st.error(err)
        else:
            st.success(f"Accuracy {res['accuracy']:.4f} on {res['n_samples']} test images")


# ---------------- Visualizations ----------------
def page_visualizations():
    st.header("Data visualizations")
    dist, err = api("GET", "/class-distribution")
    if err:
        st.error(err)
        return
    stats, err = api("GET", "/dataset-stats", timeout=120)
    if err:
        st.error(err)
        return
    classes = dist["classes"]

    st.subheader("1. Class distribution")
    df = pd.DataFrame({"Training images": dist["train"],
                       "New uploads (waiting)": dist["new"]}, index=classes)
    st.bar_chart(df)
    lo, hi = min(dist["train"]), max(dist["train"])
    if hi == 0:
        st.warning("No training images found.")
    elif hi - lo <= 0.1 * hi:
        st.info("Interpretation: the classes are balanced, so no class weights or resampling "
                "are needed and accuracy is a fair metric.")
    else:
        st.warning(f"Interpretation: the data is imbalanced ({classes[int(np.argmax(dist['train']))]} "
                   f"has the most, {classes[int(np.argmin(dist['train']))]} the fewest). "
                   "Check per-class recall, not only accuracy.")

    st.subheader("2. Average brightness per class")
    b = stats["mean_brightness"]
    st.bar_chart(pd.Series(b, index=classes, name="Mean pixel value (0-255)"))
    dark, bright = classes[int(np.argmin(b))], classes[int(np.argmax(b))]
    st.info(f"Interpretation: {dark} is the darkest class ({min(b):.0f}) and {bright} the "
            f"brightest ({max(b):.0f}). Brightness reflects how much of the 28x28 frame the item "
            "fills, so it separates footwear from tops, but it cannot separate similar tops "
            "(T-shirt, Shirt, Pullover, Coat), which have close values.")

    st.subheader("3. Average image per class")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, name, img in zip(axes.ravel(), classes, stats["mean_images"]):
        ax.imshow(np.array(img), cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    st.pyplot(fig)
    st.info("Interpretation: Trousers, Bags and footwear have sharp, distinctive average shapes, "
            "so they are easy to classify. Tops and coats have blurry, overlapping averages, "
            "which explains why the model confuses them with each other.")

    st.subheader("4. Confusion matrix and per-class recall")
    m, err = api("GET", "/metrics")
    if err or "confusion_matrix" not in (m or {}):
        st.info("No confusion matrix yet. Press 'Evaluate model on test set' on the Dashboard.")
        return
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), classes, rotation=90)
    ax.set_yticks(range(10), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    st.pyplot(fig)
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    st.bar_chart(pd.Series(recall, index=classes, name="Recall"))
    st.info(f"Interpretation: the weakest class is {classes[int(np.argmin(recall))]} "
            f"(recall {recall.min():.2f}); most of its errors go to visually similar classes.")


# ---------------- Predict ----------------
def page_predict():
    st.header("Predict a clothing image")
    file = st.file_uploader("Upload an image (png, jpg)", type=["png", "jpg", "jpeg"])
    if file:
        left, right = st.columns(2)
        left.image(file, caption="Uploaded image", width=220)
        if right.button("Predict", type="primary"):
            res, err = api("POST", "/predict",
                           files={"file": (file.name, file.getvalue(), file.type)})
            if err:
                right.error(err)
            else:
                right.metric("Prediction", res["label"], f"{res['confidence']:.1%} confidence")
                right.caption(f"Latency {res['latency_ms']} ms, container {res['container']}")
                right.bar_chart(pd.DataFrame(res["top_predictions"]).set_index("label"))


# ---------------- Upload and retrain ----------------
def watch_training():
    bar = st.progress(0.0, text="Starting...")
    while True:
        s, err = api("GET", "/status")
        if err:
            st.error(err)
            return
        t = s["training"]
        frac = t["epoch"] / max(t["total_epochs"], 1)
        bar.progress(min(frac, 1.0),
                     text=f"{t['state']}: epoch {t['epoch']}/{t['total_epochs']}  {t['message']}")
        if t["state"] == "error":
            st.error(f"Training failed: {t['message']}")
            return
        if t["state"] == "idle":
            m = s["metrics"]
            text = f"{t['message'] or 'Done'} | model v{m.get('model_version')}, accuracy {m.get('accuracy', 0):.4f}"
            if m.get("previous_accuracy") is not None:
                text += f" (before: {m['previous_accuracy']:.4f})"
            st.success(text)
            return
        time.sleep(3)


def page_retrain():
    st.header("Upload new data and retrain")
    dist, err = api("GET", "/class-distribution")
    if err:
        st.error(err)
        return

    st.subheader("1. Upload images of one class")
    label = st.selectbox("Class of these images", dist["classes"])
    folder = dist["folders"][dist["classes"].index(label)]
    files = st.file_uploader("Choose multiple images", type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True, key="multi")
    if files and st.button("Upload images"):
        res, err = api("POST", "/upload", data={"class_name": folder},
                       files=[("files", (f.name, f.getvalue(), f.type)) for f in files])
        if err:
            st.error(err)
        else:
            st.success(f"Saved {res['saved']} images, skipped {res['skipped']}. "
                       f"{res['pending_new_images']} images are now waiting for retraining.")

    st.subheader("2. Or upload a zip (one folder per class)")
    z = st.file_uploader("Zip file, e.g. Sneaker/1.png, Bag/2.png", type=["zip"], key="zip")
    if z and st.button("Upload zip"):
        res, err = api("POST", "/upload-zip", timeout=300,
                       files={"file": (z.name, z.getvalue(), "application/zip")})
        if err:
            st.error(err)
        else:
            st.success(f"Saved {res['saved']} images, skipped {res['skipped']}. "
                       f"{res['pending_new_images']} images are now waiting for retraining.")

    st.subheader("3. Retrain")
    s, _ = api("GET", "/status")
    if s:
        st.write(f"New images waiting: **{s['pending_new_images']}** "
                 f"(automatic trigger at {s['retrain_threshold']}, "
                 f"auto-retrain {'ON' if s['auto_retrain'] else 'OFF'})")
    epochs = st.slider("Fine-tuning epochs", 1, 10, 3)
    if st.button("Retrain model", type="primary"):
        res, err = api("POST", "/retrain", params={"epochs": epochs})
        if err:
            st.error(err)
        else:
            watch_training()

    with st.expander("Train from scratch (first-time setup)"):
        ep = st.slider("Epochs", 1, 20, 5, key="ep")
        sub = st.number_input("Training images", 1000, 60000, 10000, step=1000)
        if st.button("Train new model"):
            res, err = api("POST", "/train", params={"epochs": ep, "subset": int(sub)})
            if err:
                st.error(err)
            else:
                watch_training()


# ---------------- Navigation ----------------
PAGES = {"Dashboard": page_dashboard, "Visualizations": page_visualizations,
         "Predict": page_predict, "Upload & Retrain": page_retrain}
page = st.sidebar.radio("Page", list(PAGES))
h, _ = api("GET", "/health", timeout=5)
st.sidebar.write("API: online" if h else "API: offline")
PAGES[page]()
