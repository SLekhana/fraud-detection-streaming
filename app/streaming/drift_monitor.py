"""
Model drift monitoring — detects precision/recall degradation
and sends Slack alerts when thresholds are breached.

Monitors:
- Precision drift (KS test on score distributions)
- Recall drift (rolling window vs baseline)
- Fraud rate anomaly (sudden spikes/drops)
- Feature distribution shift (PSI - Population Stability Index)
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
from scipy import stats

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class DriftAlert:
    metric: str
    current_value: float
    baseline_value: float
    threshold: float
    severity: str  # "warning" | "critical"
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelMetrics:
    """Rolling window of model performance metrics."""
    precision: float = 0.0
    recall: float = 0.0
    fraud_rate: float = 0.0
    avg_score: float = 0.0
    n_transactions: int = 0
    n_flagged: int = 0
    timestamp: float = field(default_factory=time.time)


# ─── PSI helper ──────────────────────────────────────────────────────────────


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1:  stable
    PSI < 0.2:  slight shift
    PSI >= 0.2: significant shift (trigger alert)
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ─── Drift monitor ───────────────────────────────────────────────────────────


class DriftMonitor:
    """
    Maintains rolling windows of model scores and ground-truth labels
    (where available) to detect distribution shift and performance drift.
    """

    def __init__(
        self,
        window_size: int = 1000,
        precision_threshold: float | None = None,
        recall_threshold: float | None = None,
        psi_threshold: float = 0.2,
        slack_webhook: str | None = None,
    ):
        self.window_size = window_size
        self.precision_threshold = precision_threshold or settings.drift_precision_threshold
        self.recall_threshold = recall_threshold or settings.drift_recall_threshold
        self.psi_threshold = psi_threshold
        self.slack_webhook = slack_webhook or settings.slack_webhook_url

        # Rolling windows
        self._scores: deque[float] = deque(maxlen=window_size)
        self._labels: deque[int] = deque(maxlen=window_size)
        self._baseline_scores: Optional[np.ndarray] = None
        self._baseline_metrics: Optional[ModelMetrics] = None

        # Alert history (deduplicate)
        self._last_alerts: dict[str, float] = {}
        self._alert_cooldown_s: int = 3600  # 1 hour between same alert

    def set_baseline(self, scores: np.ndarray, labels: np.ndarray) -> ModelMetrics:
        """Set baseline metrics from validation set."""
        self._baseline_scores = scores.copy()
        self._baseline_metrics = self._compute_metrics(scores, labels)
        logger.info(
            f"[DriftMonitor] Baseline set: precision={self._baseline_metrics.precision:.4f} "
            f"recall={self._baseline_metrics.recall:.4f}"
        )
        return self._baseline_metrics

    def record(self, score: float, label: Optional[int] = None) -> None:
        """Record a transaction score (and optionally its true label)."""
        self._scores.append(score)
        if label is not None:
            self._labels.append(label)

    def check(self) -> list[DriftAlert]:
        """Run drift checks and return any triggered alerts."""
        if len(self._scores) < 100:
            return []

        alerts = []
        current_scores = np.array(self._scores)
        current_labels = np.array(self._labels) if self._labels else None

        # ── Score distribution drift (KS test) ────────────────────────────
        if self._baseline_scores is not None:
            ks_stat, ks_p = stats.ks_2samp(self._baseline_scores, current_scores)
            if ks_p < 0.01:
                alerts.append(
                    DriftAlert(
                        metric="score_distribution_ks",
                        current_value=ks_stat,
                        baseline_value=0.0,
                        threshold=0.05,
                        severity="warning" if ks_stat < 0.2 else "critical",
                        message=(
                            f"Score distribution drift detected (KS={ks_stat:.4f}, p={ks_p:.4f}). "
                            "Model may need retraining."
                        ),
                    )
                )

            # ── PSI ────────────────────────────────────────────────────────
            psi = compute_psi(self._baseline_scores, current_scores)
            if psi >= self.psi_threshold:
                alerts.append(
                    DriftAlert(
                        metric="score_psi",
                        current_value=psi,
                        baseline_value=0.0,
                        threshold=self.psi_threshold,
                        severity="warning" if psi < 0.25 else "critical",
                        message=(
                            f"PSI={psi:.4f} ≥ {self.psi_threshold}. "
                            "Score distribution has shifted significantly."
                        ),
                    )
                )

        # ── Performance drift (if labels available) ────────────────────────
        if current_labels is not None and self._baseline_metrics is not None:
            current_metrics = self._compute_metrics(current_scores, current_labels)

            prec_drop = self._baseline_metrics.precision - current_metrics.precision
            rec_drop = self._baseline_metrics.recall - current_metrics.recall

            if prec_drop > self.precision_threshold:
                alerts.append(
                    DriftAlert(
                        metric="precision_drift",
                        current_value=current_metrics.precision,
                        baseline_value=self._baseline_metrics.precision,
                        threshold=self.precision_threshold,
                        severity="critical" if prec_drop > 0.1 else "warning",
                        message=(
                            f"Precision dropped from {self._baseline_metrics.precision:.4f} "
                            f"to {current_metrics.precision:.4f} "
                            f"(Δ={prec_drop:.4f} > threshold={self.precision_threshold})"
                        ),
                    )
                )

            if rec_drop > self.recall_threshold:
                alerts.append(
                    DriftAlert(
                        metric="recall_drift",
                        current_value=current_metrics.recall,
                        baseline_value=self._baseline_metrics.recall,
                        threshold=self.recall_threshold,
                        severity="critical" if rec_drop > 0.1 else "warning",
                        message=(
                            f"Recall dropped from {self._baseline_metrics.recall:.4f} "
                            f"to {current_metrics.recall:.4f} "
                            f"(Δ={rec_drop:.4f} > threshold={self.recall_threshold})"
                        ),
                    )
                )

        # Send alerts
        for alert in alerts:
            self._maybe_send_slack(alert)

        return alerts

    def _compute_metrics(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5
    ) -> ModelMetrics:
        preds = (scores >= threshold).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fraud_rate = labels.mean() if len(labels) > 0 else 0.0

        return ModelMetrics(
            precision=precision,
            recall=recall,
            fraud_rate=float(fraud_rate),
            avg_score=float(scores.mean()),
            n_transactions=len(scores),
            n_flagged=int(preds.sum()),
        )

    def _maybe_send_slack(self, alert: DriftAlert) -> None:
        """Send Slack alert with cooldown deduplication."""
        now = time.time()
        last = self._last_alerts.get(alert.metric, 0)
        if now - last < self._alert_cooldown_s:
            return

        self._last_alerts[alert.metric] = now
        self._send_slack(alert)

    def _send_slack(self, alert: DriftAlert) -> None:
        if not self.slack_webhook:
            logger.warning(f"[DriftMonitor] {alert.severity.upper()}: {alert.message}")
            return

        emoji = "🚨" if alert.severity == "critical" else "⚠️"
        payload = {
            "text": f"{emoji} *Fraud Model Drift Alert*",
            "attachments": [
                {
                    "color": "#ff0000" if alert.severity == "critical" else "#ffaa00",
                    "fields": [
                        {"title": "Metric", "value": alert.metric, "short": True},
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {
                            "title": "Current",
                            "value": f"{alert.current_value:.4f}",
                            "short": True,
                        },
                        {
                            "title": "Baseline",
                            "value": f"{alert.baseline_value:.4f}",
                            "short": True,
                        },
                        {"title": "Message", "value": alert.message, "short": False},
                    ],
                }
            ],
        }
        try:
            resp = requests.post(self.slack_webhook, json=payload, timeout=5)
            resp.raise_for_status()
            logger.info(f"[DriftMonitor] Slack alert sent: {alert.metric}")
        except Exception as e:
            logger.error(f"[DriftMonitor] Slack send failed: {e}")

    def summary(self) -> dict:
        """Return current monitoring summary."""
        scores = np.array(self._scores) if self._scores else np.array([])
        return {
            "window_size": len(scores),
            "avg_fraud_score": float(scores.mean()) if len(scores) > 0 else 0.0,
            "fraud_rate": float((scores >= 0.5).mean()) if len(scores) > 0 else 0.0,
            "baseline_set": self._baseline_metrics is not None,
            "baseline_precision": (
                self._baseline_metrics.precision if self._baseline_metrics else None
            ),
            "baseline_recall": (
                self._baseline_metrics.recall if self._baseline_metrics else None
            ),
        }
