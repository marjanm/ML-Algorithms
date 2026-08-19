"""FastAPI model serving endpoint."""
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
