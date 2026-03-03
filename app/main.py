"""
Fraud Detection API — FastAPI application.

Endpoints:
  GET  /health          — health check + model/kafka status
  POST /score           — score a single transaction
  POST /score/batch     — score a batch of transactions
  POST /score/explain   — score + LLM explanation
  GET  /evaluate        — run evaluation on held-out test set
  GET  /drift           — drift monitoring status
  GET  /metrics         — Prometheus metrics
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import settings
from app.models.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    DriftStatusResponse,
    EvaluationRequest,
    EvaluationResponse,
    FraudScoreResponse,
    HealthResponse,
    TransactionRequest,
)
from app.streaming.drift_monitor import DriftMonitor

# ─── Logging ─────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# ─── Global state ────────────────────────────────────────────────────────────
_model = None
_drift_monitor = DriftMonitor()
_model_loaded = False


def _load_model():
    """Load ensemble model (lazy, only once)."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    try:
        from app.core.ensemble import FraudEnsemble
        _model = FraudEnsemble.load(settings.model_path)
        _model_loaded = True
        logger.info("model_loaded", path=settings.model_path)
    except Exception as e:
        logger.warning("model_not_loaded", error=str(e))
        _model_loaded = False
    return _model


def _get_model():
    model = _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first: python scripts/train.py",
        )
    return model


# ─── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", env=settings.environment)
    _load_model()
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection using stacked AutoEncoder + XGBoost ensemble. "
        "IEEE-CIS Fraud Detection dataset. SHAP explainability. Kafka + Spark streaming."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request logging middleware ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency, 2),
    )
    return response


# ─── Internal scoring helper ──────────────────────────────────────────────────

def _score_transaction(tx_dict: dict, include_explanation: bool = False) -> FraudScoreResponse:
    """Core scoring logic shared by single and batch endpoints."""
    from app.utils.inference_utils import build_inference_features
    from app.agent.explainer import get_explainer

    model = _get_model()
    t0 = time.perf_counter()

    # Build features — zero-pads missing columns to match training dimension
    from app.utils.inference_utils import build_inference_features
    X, feat_cols = build_inference_features(tx_dict, settings.model_path)

    if X.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No valid feature columns found in transaction.")

    # Score
    fraud_score = float(model.predict_proba(X)[0])
    anomaly_score = float(model.anomaly_scores(X)[0])
    ae_threshold = model.ae_trainer.threshold or 0.05
    is_fraud = fraud_score >= settings.threshold_ensemble
    anomaly_flag = anomaly_score > ae_threshold

    # SHAP
    shap_result = model.explain_single(X[0])
    top_risk = shap_result.get("top_features", [])[:5]

    # LLM explanation (optional)
    explanation = None
    if include_explanation:
        try:
            explainer = get_explainer(use_llm=True)
            explanation = explainer.explain(
                transaction=tx_dict,
                shap_explanation=shap_result,
                anomaly_score=anomaly_score,
                ae_threshold=ae_threshold,
                fraud_score=fraud_score,
            )
        except Exception as e:
            logger.warning("explanation_failed", error=str(e))
            explanation = f"Explanation unavailable: {e}"

    latency_ms = (time.perf_counter() - t0) * 1000

    # Update drift monitor
    _drift_monitor.record(fraud_score)

    return FraudScoreResponse(
        transaction_id=tx_dict.get("TransactionID"),
        fraud_score=round(fraud_score, 6),
        is_fraud=is_fraud,
        anomaly_score=round(anomaly_score, 6),
        anomaly_flag=anomaly_flag,
        top_risk_factors=top_risk,
        explanation=explanation,
        latency_ms=round(latency_ms, 2),
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — returns model and Kafka connectivity status."""
    kafka_ok = False
    try:
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
        admin.close()
        kafka_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        kafka_connected=kafka_ok,
    )


@app.post("/score", response_model=FraudScoreResponse, tags=["Scoring"])
async def score_transaction(tx: TransactionRequest):
    """
    Score a single transaction for fraud.

    Returns fraud probability, anomaly score, and top SHAP risk factors.
    Sub-100ms latency in production.
    """
    return _score_transaction(tx.model_dump())


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
async def score_batch(request: BatchScoreRequest):
    """Score a batch of transactions. Returns all results with summary stats."""
    results = []
    t0 = time.perf_counter()

    for tx in request.transactions:
        result = _score_transaction(
            tx.model_dump(),
            include_explanation=request.include_explanations,
        )
        results.append(result)

    total_latency = (time.perf_counter() - t0) * 1000
    flagged = sum(1 for r in results if r.is_fraud)

    return BatchScoreResponse(
        results=results,
        total=len(results),
        flagged=flagged,
        avg_latency_ms=round(total_latency / max(len(results), 1), 2),
    )


@app.post("/score/explain", response_model=FraudScoreResponse, tags=["Scoring"])
async def score_with_explanation(tx: TransactionRequest):
    """
    Score a transaction and generate LLM natural-language fraud justification.
    Uses LangChain + GPT-4o-mini. Falls back to rule-based if no API key.
    """
    return _score_transaction(tx.model_dump(), include_explanation=True)


@app.get("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate(n_samples: int = 1000, threshold: float = 0.5):
    """
    Evaluate model on a random sample of the held-out test set.
    Returns AUC-PR, AUC-ROC, precision, recall, F1, confusion matrix.
    """
    model = _get_model()

    try:
        test_df = pd.read_parquet("data/test_features.parquet")
        if len(test_df) > n_samples:
            test_df = test_df.sample(n=n_samples, random_state=42)

        from app.core.features import get_feature_columns
        feat_cols = get_feature_columns(test_df)
        X = test_df[feat_cols].fillna(0).values
        y = test_df["isFraud"].values

        metrics = model.evaluate(X, y)
        report = metrics.get("classification_report", {})
        fraud_report = report.get("1", {})

        return EvaluationResponse(
            auc_pr=metrics["auc_pr"],
            auc_roc=metrics["auc_roc"],
            precision=fraud_report.get("precision", 0.0),
            recall=fraud_report.get("recall", 0.0),
            f1=fraud_report.get("f1-score", 0.0),
            confusion_matrix=metrics["confusion_matrix"],
            n_samples=len(test_df),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Test data not found. Run: python scripts/prepare_data.py",
        )


@app.get("/drift", response_model=DriftStatusResponse, tags=["Monitoring"])
async def drift_status():
    """Return current drift monitoring status and any active alerts."""
    summary = _drift_monitor.summary()
    alerts = _drift_monitor.check()
    return DriftStatusResponse(
        **summary,
        alerts=[
            {
                "metric": a.metric,
                "severity": a.severity,
                "message": a.message,
                "current": a.current_value,
                "baseline": a.baseline_value,
            }
            for a in alerts
        ],
    )


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
