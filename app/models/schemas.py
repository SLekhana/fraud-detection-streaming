"""Pydantic v2 request/response models."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class TransactionRequest(BaseModel):
    TransactionID: Optional[int] = None
    TransactionDT: int
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: Optional[str] = None
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class FraudScoreResponse(BaseModel):
    transaction_id: Optional[int] = None
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    anomaly_score: float
    anomaly_flag: bool
    top_risk_factors: list[dict[str, Any]] = []
    explanation: Optional[str] = None
    latency_ms: float
    model_version: str = "1.0.0"


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionRequest]
    include_explanations: bool = False


class BatchScoreResponse(BaseModel):
    results: list[FraudScoreResponse]
    total: int
    flagged: int
    avg_latency_ms: float


class EvaluationRequest(BaseModel):
    n_samples: int = Field(default=1000, ge=100, le=50000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class EvaluationResponse(BaseModel):
    auc_pr: float
    auc_roc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    n_samples: int


class DriftStatusResponse(BaseModel):
    window_size: int
    avg_fraud_score: float
    fraud_rate: float
    baseline_set: bool
    baseline_precision: Optional[float]
    baseline_recall: Optional[float]
    alerts: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    kafka_connected: bool
    version: str = "1.0.0"
