# Fashion-MNIST ML Pipeline: train, deploy, monitor and retrain
 
An end-to-end machine learning pipeline for **image classification** (Fashion-MNIST, 10 clothing classes). It covers data acquisition, preprocessing, model creation, model testing, retraining with a trigger, a Python API, a monitoring/control UI, Docker deployment, and a Locust flood test.
 
## Links
 
| What | Link |
|---|---|
| GitHub repository | https://github.com/Nzabinesha/Summative_machinepipe |
| Demo video (YouTube) | **https://youtu.be/CVYHEsZqn2A**|
| Live UI (Render) | https://summative-machinepipe-2.onrender.com |
| Live API (Render) | https://summative-machinepipe.onrender.com |
| Interactive API docs | https://summative-machinepipe.onrender.com/docs |
 
> > **This is anote for some one assessing this project using render URL**: 
The live services run on Render's free tier. They go to sleep after 15 minutes without traffic, and the first request after that can take about a minute while they wake up. Please open the UI and wait for it to load.
 
## Project description
 
A user can classify a clothing image, look at visualizations of the dataset, upload new labelled images in bulk, and trigger retraining of the model from the UI. The API serves the model, and the UI shows uptime, metrics and training status.
 
**Required functionalities**
 
1. **Model prediction:** upload one image and get the predicted class, confidence and top 3 predictions (Predict page).
2. **Visualizations:** class distribution, average brightness per class, average image per class, and the confusion matrix with per-class recall, each with an interpretation (Visualizations page).
3. **Upload data:** bulk upload of images for one class, or a zip file with one folder per class (Upload & Retrain page).
4. **Trigger retraining:** a "Retrain model" button fine-tunes the model on the new data. An automatic trigger also exists (see below).
## Dataset and model
 
- **Data:** Fashion-MNIST, 70,000 grayscale 28x28 images in 10 balanced classes (60,000 train, 10,000 test). A copy of part of the data is stored as PNG files in `data/train` (500 per class) and `data/test` (100 per class), organized in one folder per class. This is the format used for uploads and retraining.
- **Preprocessing:** scale pixels to 0-1, add a channel dimension, stratified train/validation split, and augmentation (horizontal flip, small rotation, zoom). Uploaded photos are converted to grayscale, padded to a square, resized to 28x28 and inverted when the background is light.
- **Baseline model:** a dense network (Flatten, Dense 128, Dense 10).
- **Final model:** a CNN with two convolution blocks (Conv, BatchNorm, MaxPool), Dense 128 with L2 regularization, Dropout 0.4, and a softmax output. It was trained with early stopping and learning-rate reduction.
- **Results (notebook, full 10,000-image test set):**
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| Baseline dense | 0.8734 | 0.8766 | 0.8734 | 0.8731 | 0.9897 |
| Optimized CNN | 0.9169 | 0.9170 | 0.9169 | 0.9169 | 0.9951 |
 
 
The weakest class is **Shirt** (F1 about 0.75), which is mostly confused with T-shirt/top, Coat and Pullover. Footwear, trousers and bags are classified almost perfectly. The full analysis, with the confusion matrix and training curves, is in the notebook.
 
- **Model files:** `models/fashion_cnn.keras` (used by the API) and `models/fashion_cnn.h5` (same model, H5 format).
## Repository structure
 
```
Summative_machinepipe/
├── README.md
├── notebook/
│   └── Summative_machinepipe.ipynb   # preprocessing, training, evaluation, prediction tests
├── src/
│   ├── preprocessing.py              # data acquisition, image preparation, upload saving
│   ├── model.py                      # model creation, training, evaluation, retraining, retrain trigger
│   └── prediction.py                 # load model and predict one image
├── app/
│   ├── api.py                        # FastAPI service
│   └── ui.py                         # Streamlit UI
├── data/
│   ├── train/<Class>/*.png
│   └── test/<Class>/*.png
├── models/
│   ├── fashion_cnn.keras
│   ├── fashion_cnn.h5
│   └── metrics.json
├── results/screenshots/              # Locust and deployment screenshots
├── locustfile.py                     # flood test
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml                # api (scalable) + nginx load balancer + ui
├── nginx.conf
├── requirements-api.txt
└── requirements-ui.txt
```
 
## How the pipeline works
 
1. **Data acquisition:** `load_fashion_mnist()` in `src/preprocessing.py` downloads the dataset. Images saved as PNG files are used for uploads and retraining.
2. **Data processing:** `prepare_image()` and `load_folder()` turn images into model-ready arrays.
3. **Model creation:** `build_cnn()` and `train()` in `src/model.py`.
4. **Model testing:** `evaluate()` computes accuracy, precision, recall, F1 and the confusion matrix. The API exposes it as `POST /evaluate`, which tests the model that is currently deployed on the test images.
5. **Retraining:** uploaded images wait in `data/new/<Class>/`. `retrain()` loads the saved model, fine-tunes it with a low learning rate on the new images plus the existing training images, compares accuracy before and after, archives the previous model in `models/archive/`, saves the new model and moves the new images into `data/train/`. The API reloads the new model automatically.
6. **API:** FastAPI (`app/api.py`).
### Retraining trigger
 
- **Manual:** the "Retrain model" button in the UI (`POST /retrain`).
- **Automatic:** when the environment variable `AUTO_RETRAIN=true` is set, retraining starts as soon as at least `RETRAIN_THRESHOLD` new images (default 50) are waiting after an upload. Both variables are in `docker-compose.yml`.
## API endpoints
 
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/status` | Uptime, model version, metrics, training status, pending images |
| GET | `/metrics` | Latest metrics including the confusion matrix |
| GET | `/class-distribution` | Image counts per class (train and waiting uploads) |
| GET | `/dataset-stats` | Average brightness and average image per class |
| POST | `/predict` | Predict one image (multipart `file`) |
| POST | `/upload` | Bulk upload images for one class (`class_name`, `files`) |
| POST | `/upload-zip` | Upload a zip with one folder per class |
| POST | `/retrain` | Fine-tune on the uploaded images |
| POST | `/train` | Train a new model from scratch |
| POST | `/evaluate` | Evaluate the deployed model on `data/test` |
 
## Setup: run locally with Docker (recommended)
 
Requirements: Docker with Docker Compose.
 
```bash
git clone https://github.com/Nzabinesha/Summative_machinepipe.git
cd Summative_machinepipe
docker compose up -d --build
```
 
The first build takes several minutes because it installs TensorFlow. Then open:
 
- UI: http://localhost:8501
- API (through the Nginx load balancer): http://localhost:8080 (docs at http://localhost:8080/docs)
To run several API containers behind the load balancer:
 
```bash
docker compose up -d --scale api=3
docker compose restart nginx
```
 
To stop everything: `docker compose down`.
 
## Setup: run without Docker
 
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-api.txt -r requirements-ui.txt
uvicorn app.api:app --port 8000
# in a second terminal:
API_URL=http://localhost:8000 streamlit run app/ui.py
```
 
## Deployment (cloud)
 
The API and the UI are deployed as two separate Docker web services on **Render** (free tier), built from this repository (`Dockerfile.api` and `Dockerfile.ui`). The UI reads the API address from the `API_URL` environment variable. The "Evaluate model on test set" button on the Dashboard runs the evaluation of the model in production.
 
Limitations of the free tier: services sleep after 15 minutes of inactivity, the CPU and memory are small, only one instance can run, and files written at runtime (uploads and retrained models) are lost when the service restarts. For this reason the container-scaling test and the retraining demo were run on the Docker stack (see the demo video).
 
## Flood request simulation (Locust)
 
**Method.** `locustfile.py` simulates users who send images from the test set to `POST /predict` (plus a few `GET /health` calls). Each simulated user waits 0.1 to 0.5 seconds between requests. The test ran in a GitHub Codespace with Docker Compose: 1, 2 and 3 API containers behind an Nginx load balancer, 100 simulated users, ramp-up of 10 users per second, and the `/predict` results below. All runs had **0 failed requests**.
 
```bash
pip install locust
docker compose up -d --scale api=N        # N = 1, 2, 3
docker compose restart nginx
locust -f locustfile.py --host http://localhost:8080 -u 100 -r 10
```
 
**Results (`/predict`)**
 
| API containers | Users | Median (ms) | 95th percentile (ms) | Average (ms) | Requests/s | Failures |
|---|---|---|---|---|---|---|
| 1 (no-load baseline) | 1 | 13 | 20 | 14 | 3.2 | 0 |
| 1 | 100 | 1800 | 2100 | 1779 | 46.5 | 0 |
| 2 | 100 | 560 | 1900 | 859 | 75.6 | 0 |
| 3 | 100 | 550 | 2300 | 864 | 81.2 | 0 |
 
*(The 2- and 3-container runs include part of the ramp-up period, so their numbers are slightly rougher than the 1-container run. TODO: adjust this sentence if you rerun them.)*
 
**Interpretation.** With a single user the model answers in about 13 ms. With 100 users and one container, requests queue up: the median rose to 1.8 s and throughput reached about 47 requests/s. A second container cut the median to about 0.56 s (roughly 3x faster) and raised throughput to about 76 requests/s. A third container gave no further gain (median 0.55 s, about 81 requests/s), most likely because the Codespace's CPU cores were already fully used by two containers. Scaling out therefore helps until the host's CPU is saturated, and further gains would need more CPU cores or separate machines.
 
Screenshots of the Locust statistics and charts for each run are in `results/screenshots/` and you can access it in this project github repo.
