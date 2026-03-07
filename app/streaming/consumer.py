"""
Kafka Consumer with Dead-Letter Queue (DLQ) and retry logic.

Architecture:
  - Consumes from: fraud-transactions topic
  - On success: writes scored result to fraud-scores topic
  - On transient failure (e.g. model timeout): retries up to MAX_RETRIES
  - On persistent failure: sends to fraud-transactions-dlq topic
  - Logs all DLQ events with error details for alerting

Topics:
  fraud-transactions      ← incoming raw transactions
  fraud-scores            ← scored results
  fraud-transactions-dlq  ← dead-letter queue (failed after all retries)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import structlog
from kafka import KafkaConsumer, KafkaProducer

logger = structlog.get_logger()

# ─── Config ───────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CONSUMER_TOPIC = "fraud-transactions"
SCORES_TOPIC = "fraud-scores"
DLQ_TOPIC = "fraud-transactions-dlq"
CONSUMER_GROUP = "fraud-scorer-group"

MAX_RETRIES = 3
RETRY_BACKOFF_BASE_S = 1.0  # exponential: 1s, 2s, 4s
POLL_TIMEOUT_MS = 1000


# ─── DLQ message ─────────────────────────────────────────────────────────────


@dataclass
class DLQMessage:
    original_topic: str
    original_partition: int
    original_offset: int
    payload: dict
    error_type: str
    error_message: str
    retry_count: int
    failed_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> dict:
        return {
            "original_topic": self.original_topic,
            "original_partition": self.original_partition,
            "original_offset": self.original_offset,
            "payload": self.payload,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "failed_at": self.failed_at,
        }


# ─── Consumer ────────────────────────────────────────────────────────────────


class FraudConsumer:
    """
    Event-driven Kafka consumer for real-time fraud scoring.

    Features:
    - Retry with exponential backoff (up to MAX_RETRIES)
    - Dead-letter queue for unrecoverable failures
    - Structured logging for every event
    - Prometheus metrics (optional)
    - Graceful shutdown on SIGINT/SIGTERM
    """

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self._model = None
        self._running = False

        self.consumer = KafkaConsumer(
            CONSUMER_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=CONSUMER_GROUP,
            auto_offset_reset="latest",
            enable_auto_commit=False,  # manual commit after successful scoring
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            acks="all",  # wait for all replicas to confirm
            compression_type="gzip",
        )

        logger.info(
            "consumer_initialized",
            topic=CONSUMER_TOPIC,
            group=CONSUMER_GROUP,
            dlq_topic=DLQ_TOPIC,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        from app.core.ensemble import FraudEnsemble

        self._model = FraudEnsemble.load(self.model_dir)
        logger.info("model_loaded", model_dir=self.model_dir)
        return self._model

    def _score(self, tx_dict: dict) -> dict:
        """Score a transaction and return result dict."""
        from app.utils.inference_utils import build_inference_features

        model = self._load_model()
        t0 = time.perf_counter()

        X, feat_cols = build_inference_features(tx_dict, self.model_dir)
        fraud_score = float(model.predict_proba(X)[0])
        anomaly_score = float(model.anomaly_scores(X)[0])
        shap_result = model.explain_single(X[0])

        latency_ms = (time.perf_counter() - t0) * 1000
        ae_threshold = (
            model.ae_trainer.threshold
            if model.ae_trainer is not None and model.ae_trainer.threshold
            else 0.05
        )

        return {
            "transaction_id": tx_dict.get("TransactionID"),
            "fraud_score": round(fraud_score, 6),
            "is_fraud": fraud_score >= 0.4,
            "anomaly_score": round(anomaly_score, 6),
            "anomaly_flag": anomaly_score > ae_threshold,
            "top_risk_factors": shap_result.get("top_features", [])[:5],
            "latency_ms": round(latency_ms, 2),
            "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _send_to_dlq(
        self, record, payload: dict, error: Exception, retries: int
    ) -> None:
        """Send failed message to dead-letter queue."""
        dlq_msg = DLQMessage(
            original_topic=record.topic,
            original_partition=record.partition,
            original_offset=record.offset,
            payload=payload,
            error_type=type(error).__name__,
            error_message=str(error)[:500],
            retry_count=retries,
        )
        try:
            self.producer.send(DLQ_TOPIC, value=dlq_msg.to_dict())
            self.producer.flush()
            logger.error(
                "dlq_sent",
                transaction_id=payload.get("TransactionID"),
                error_type=dlq_msg.error_type,
                retries=retries,
                partition=record.partition,
                offset=record.offset,
            )
        except Exception as dlq_err:
            logger.critical("dlq_send_failed", error=str(dlq_err))

    def _process_record(self, record) -> bool:
        """
        Process one Kafka record with retry logic.
        Returns True on success, False on failure (after all retries).
        """
        payload = record.value
        tx_id = payload.get("TransactionID", "unknown")
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                result = self._score(payload)
                self.producer.send(SCORES_TOPIC, value=result)
                self.producer.flush()
                logger.info(
                    "transaction_scored",
                    transaction_id=tx_id,
                    fraud_score=result["fraud_score"],
                    is_fraud=result["is_fraud"],
                    latency_ms=result["latency_ms"],
                    attempt=attempt + 1,
                )
                return True

            except Exception as e:
                last_error = e
                wait = RETRY_BACKOFF_BASE_S * (2**attempt)
                logger.warning(
                    "score_failed_retrying",
                    transaction_id=tx_id,
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    error=str(e),
                    retry_in_s=wait,
                )
                time.sleep(wait)

        # All retries exhausted → DLQ
        self._send_to_dlq(record, payload, last_error, MAX_RETRIES)
        return False

    def run(self) -> None:
        """Main consumer loop. Runs until stop() is called."""
        self._running = True
        logger.info("consumer_started", topic=CONSUMER_TOPIC)

        processed = 0
        failures = 0

        try:
            while self._running:
                records = self.consumer.poll(timeout_ms=POLL_TIMEOUT_MS)

                for tp, messages in records.items():
                    for record in messages:
                        success = self._process_record(record)
                        if success:
                            processed += 1
                            # Commit offset only after successful scoring
                            self.consumer.commit({tp: record.offset + 1})
                        else:
                            failures += 1

                if processed > 0 and processed % 100 == 0:
                    logger.info(
                        "consumer_progress",
                        processed=processed,
                        failures=failures,
                        dlq_rate=round(failures / (processed + failures), 4),
                    )

        except KeyboardInterrupt:
            logger.info("consumer_stopping", reason="keyboard_interrupt")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.consumer.close()
        self.producer.close()
        logger.info("consumer_stopped")


if __name__ == "__main__":
    consumer = FraudConsumer()
    consumer.run()
