import logging
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")

MODEL_PATH = Path(__file__).resolve(
).parents[1] / "models" / "final_model.joblib"
REQUEST_COUNT = Counter(
    "heart_api_requests_total", "Total API requests", [
        "endpoint", "method", "status"])
PREDICTION_COUNT = Counter(
    "heart_api_predictions_total",
    "Total predictions",
    ["prediction"])
LATENCY = Histogram(
    "heart_api_request_latency_seconds",
    "Request latency",
    ["endpoint"])


class PatientRecord(BaseModel):
    age: float = Field(..., example=63)
    sex: float = Field(..., example=1)
    cp: float = Field(..., example=3)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: float = Field(..., example=1)
    restecg: float = Field(..., example=0)
    thalach: float = Field(..., example=150)
    exang: float = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=0)
    ca: float = Field(..., example=0)
    thal: float = Field(..., example=1)


app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    else:
        logger.warning("Model file not found. Run python src/train.py first.")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    with LATENCY.labels(endpoint=request.url.path).time():
        response = await call_next(request)
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code).inc()
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(record: PatientRecord) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.")
    df = pd.DataFrame([record.model_dump()])
    pred = int(model.predict(df)[0])
    confidence = float(model.predict_proba(df)[0][pred])
    PREDICTION_COUNT.labels(prediction=str(pred)).inc()
    logger.info("prediction=%s confidence=%.4f input=%s",
                pred, confidence, record.model_dump())
    return {"prediction": pred, "risk_label": "heart_disease" if pred ==
            1 else "no_heart_disease", "confidence": confidence}
