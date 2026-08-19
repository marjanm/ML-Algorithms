"""
Model Deployment / Serving — FastAPI Demo
===========================================
Bridges the gap from training to production.

Demonstrates:
  1. Train a model and save with joblib
  2. Build a FastAPI app that loads the model and serves predictions
  3. Start the server, send test requests, verify responses
  4. Shut down — all self-contained

Run:
    python serving_demo.py
"""

import os, sys, time, json, signal
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
APP_FILE = os.path.join(OUTPUT_DIR, "app.py")


def train_and_save_model():
    """Train a simple model and persist it."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=1000, n_features=4, n_informative=3,
                                n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_PATH)
    return acc, X_test[:3].tolist()


def create_app_file():
    """Write the FastAPI app as a standalone file (loaded by uvicorn)."""
    app_code = '''"""FastAPI model serving endpoint."""
import os
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="ML Model API", version="1.0")


class PredictionRequest(BaseModel):
    features: List[List[float]]  # batch of feature vectors


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array(request.features)
    preds = model.predict(X).tolist()
    probs = model.predict_proba(X).tolist()
    return PredictionResponse(predictions=preds, probabilities=probs)


@app.get("/model-info")
def model_info():
    return {
        "type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "n_features": model.n_features_in_,
        "classes": model.classes_.tolist(),
    }
'''
    with open(APP_FILE, "w") as f:
        f.write(app_code)


def run_serving_demo():
    lines = [
        "=" * 65,
        "  MODEL DEPLOYMENT / SERVING  —  FastAPI Demo",
        "=" * 65, "",
    ]

    # Step 1: Train & save
    lines.append("  ── Step 1: Train and Save Model ──")
    acc, sample_inputs = train_and_save_model()
    model_size = os.path.getsize(MODEL_PATH) / 1024
    lines += [
        f"    Model: RandomForestClassifier (50 trees)",
        f"    Test accuracy: {acc:.4f}",
        f"    Saved to: model.joblib ({model_size:.1f} KB)",
    ]

    # Step 2: Create FastAPI app
    lines += ["", "  ── Step 2: Create FastAPI App ──"]
    create_app_file()
    lines += [
        f"    Created: app.py",
        "    Endpoints:",
        "      GET  /health      — liveness check",
        "      POST /predict     — send features, get predictions + probabilities",
        "      GET  /model-info  — model metadata",
    ]

    # Step 3: Start server & test
    lines += ["", "  ── Step 3: Start Server & Send Requests ──"]

    import subprocess, urllib.request

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8321",
         "--log-level", "warning"],
        cwd=OUTPUT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # wait for server to start
    server_ready = False
    for _ in range(30):
        time.sleep(0.5)
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:8321/health", timeout=2)
            if resp.status == 200:
                server_ready = True
                break
        except Exception:
            continue

    if not server_ready:
        lines.append("    ⚠ Server failed to start. Make sure fastapi & uvicorn are installed.")
        lines.append("      pip3 install fastapi uvicorn")
        proc.terminate()
    else:
        lines.append(f"    Server running at http://127.0.0.1:8321")

        # Health check
        resp = urllib.request.urlopen("http://127.0.0.1:8321/health")
        health = json.loads(resp.read())
        lines.append(f"    GET /health → {health}")

        # Model info
        resp = urllib.request.urlopen("http://127.0.0.1:8321/model-info")
        info = json.loads(resp.read())
        lines.append(f"    GET /model-info → {info}")

        # Prediction
        payload = json.dumps({"features": sample_inputs}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8321/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        lines.append(f"    POST /predict (3 samples):")
        for i, (pred, prob) in enumerate(zip(result["predictions"], result["probabilities"])):
            lines.append(f"      Sample {i}: class={pred}, P(class=1)={prob[1]:.4f}")

        # Latency test
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            req = urllib.request.Request(
                "http://127.0.0.1:8321/predict",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req)
            latencies.append((time.perf_counter() - start) * 1000)
        lines += [
            f"",
            f"    Latency test (50 requests, batch of 3):",
            f"      Mean:   {np.mean(latencies):.1f} ms",
            f"      Median: {np.median(latencies):.1f} ms",
            f"      P95:    {np.percentile(latencies, 95):.1f} ms",
            f"      P99:    {np.percentile(latencies, 99):.1f} ms",
        ]

        proc.terminate()
        proc.wait(timeout=5)
        lines.append(f"\n    Server shut down.")

    lines += [
        "", "  ── Step 4: What a Production Setup Adds ──",
        "    • Input validation & error handling (Pydantic does basic validation)",
        "    • Authentication (API keys, OAuth)",
        "    • Rate limiting & request queuing",
        "    • Model versioning (serve v1 and v2 side by side)",
        "    • Monitoring (latency, error rate, prediction distribution drift)",
        "    • Containerisation (Docker) for reproducible deployment",
        "    • Load balancing (multiple workers, horizontal scaling)",
        "    • Logging & alerting",
        "", "  ── API Example (curl) ──",
        '    curl -X POST http://localhost:8321/predict \\',
        '      -H "Content-Type: application/json" \\',
        '      -d \'{"features": [[0.5, -1.2, 0.3, 0.8]]}\'',
        "", "=" * 65,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_serving_demo()
