import random
from pathlib import Path

from locust import HttpUser, between, task

# 25 sample test images, loaded once
SAMPLES = [(p.name, p.read_bytes())
           for p in sorted((Path(__file__).parent / "data" / "test").rglob("*.png"))[::40]]


class ModelUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        name, data = random.choice(SAMPLES)
        self.client.post("/predict", files={"file": (name, data, "image/png")})

    @task(1)
    def health(self):
        self.client.get("/health")
        