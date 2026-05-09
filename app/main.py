"""FastAPI service for the fraud detection model.

Run from project root:
    .venv/Scripts/python.exe -m uvicorn app.main:app --reload

Then visit:
    http://127.0.0.1:8000/docs        Swagger UI (interactive)
    http://127.0.0.1:8000/redoc       ReDoc (alternative)
    http://127.0.0.1:8000/health      Liveness check
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    Transaction,
)
from src.model import predict_proba

MODEL_PATH = Path("models/fraud_model.joblib")

# Module-level dict populated at startup so request handlers can read it.
# Using a dict (rather than a global) keeps mutation explicit.
_state: dict = {"bundle": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model once.
    print(f"[lifespan] loading model bundle from {MODEL_PATH}...")
    _state["bundle"] = joblib.load(MODEL_PATH)
    print(f"[lifespan] model loaded; threshold = {_state['bundle']['threshold']:.4f}")
    yield
    # Shutdown: nothing to clean up, but the hook is here if we ever need it.
    _state["bundle"] = None


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Credit card fraud detection — XGBoost classifier with cost-tuned threshold. "
        "See /docs for the interactive Swagger UI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Phase 4 Next.js frontend (different origin) to call us.
# Lock down `allow_origins` to your real frontend domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _bundle_or_503():
    """Return the loaded bundle, or raise a 503 if it isn't ready yet."""
    bundle = _state["bundle"]
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return bundle


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_state["bundle"] is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    bundle = _bundle_or_503()
    md = bundle["metadata"]
    return ModelInfoResponse(
        model_name=md["model_name"],
        model_type=md["model_type"],
        trained_at_utc=md["trained_at_utc"],
        threshold=bundle["threshold"],
        feature_columns=bundle["feature_columns"],
        training_info=md["training"],
        test_metrics=md["test_metrics_at_chosen_threshold"],
    )


def _score_dataframe(bundle, X: pd.DataFrame) -> list[float]:
    # Column ORDER must match training; reorder defensively.
    X = X[bundle["feature_columns"]]
    return [float(s) for s in predict_proba(bundle, X)]


@app.post("/predict", response_model=PredictionResponse)
def predict_one(transaction: Transaction):
    bundle = _bundle_or_503()
    X = pd.DataFrame([transaction.model_dump()])
    score = _score_dataframe(bundle, X)[0]
    threshold = bundle["threshold"]
    return PredictionResponse(
        fraud_probability=score,
        is_fraud=score >= threshold,
        threshold=threshold,
        model_name=bundle["metadata"]["model_name"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_many(request: BatchTransactionRequest):
    bundle = _bundle_or_503()
    if not request.transactions:
        return BatchPredictionResponse(predictions=[])
    X = pd.DataFrame([t.model_dump() for t in request.transactions])
    scores = _score_dataframe(bundle, X)
    threshold = bundle["threshold"]
    model_name = bundle["metadata"]["model_name"]
    return BatchPredictionResponse(predictions=[
        PredictionResponse(
            fraud_probability=s,
            is_fraud=s >= threshold,
            threshold=threshold,
            model_name=model_name,
        )
        for s in scores
    ])
