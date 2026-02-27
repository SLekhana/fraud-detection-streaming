"""
Kafka Consumer — reads from fraud.transactions.raw, scores each transaction,
publishes results to fraud.scores topic.

Features:
- Configurable retry with exponential backoff
- Dead-letter queue for unprocessable messages
- Prometheus metrics (latency, throughput, fraud rate)
- Graceful shutdown on SIGTERM
"""
from __future__ import annotations

import json
import logging
import signal
import time
from typing import Callable, Optional

import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from prometheus_client import Counter, Histogram, Gauge

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Prometheus metrics ──────────────────────────────────────────────────────
TRANSACTIONS_PROCESSED = Counter(
    "fraud_transactions_processed_total", "Total transactions processed"
)
TRANSACTIONS_FLAGGED = Counter(
    "fraud_transactions_flagged_total", "Total transactions flagged as fraud"
)
PROCESSING_LATENCY = Histogram(
    "fraud_scoring_latency_seconds",
    "Transaction scoring latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)
FRAUD_RATE_GAUGE = Gauge("fraud_rate_current", "Current fraud rate (rolling)")
DLQ_MESSAGES = Counter("fraud_dlq_messages_total", "Messages sent to DLQ")


class FraudScoringConsumer:
    """
    Consumes raw transactions from Kafka, scores them using the ensemble
    model, and publishes scored results to fraud.scores topic.
    """

    def __init__(
        self,
        score_fn: Callable[[dict], dict],
        bootstrap_servers: Optional[str] = None,
        consumer_group: Optional[str] = None,
        input_topic: Optional[str] = None,
        output_topic: Optional[str] = None,
        dlq_topic: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_ms: int = 500,
    ):
        """
        Args:
            score_fn: Callable that takes a transaction dict and returns
                      a scored dict with {fraud_score, is_fraud, anomaly_score, ...}.
        """
        self.score_fn = score_fn
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.consumer_group = consumer_group or settings.kafka_consumer_group
        self.input_topic = input_topic or settings.kafka_topic_transactions
        self.output_topic = output_topic or settings.kafka_topic_scores
        self.dlq_topic = dlq_topic or settings.kafka_topic_dlq
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        self._running = False

        # Rolling fraud rate tracker
        self._recent_scores: list[float] = []
        self._window_size = 1000

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _build_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            auto_offset_reset="earliest",
            enable_auto_commit=False,  # manual commit for at-least-once semantics
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_records=100,
        )

    def _build_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )

    def run(self) -> None:
        """Main consumer loop. Blocks until shutdown signal received."""
        self._running = True
        consumer = self._build_consumer()
        producer = self._build_producer()
        logger.info(
            f"[Consumer] Listening on {self.input_topic} "
            f"(group={self.consumer_group})"
        )

        try:
            while self._running:
                records = consumer.poll(timeout_ms=1000)
                for tp, messages in records.items():
                    for message in messages:
                        self._process_message(message, producer)
                    consumer.commit()
        except Exception as e:
            logger.error(f"[Consumer] Fatal error: {e}")
            raise
        finally:
            consumer.close()
            producer.close()
            logger.info("[Consumer] Shutdown complete")

    def _process_message(self, message, producer: KafkaProducer) -> None:
        """Score a single transaction message with retry logic."""
        tx = message.value
        tx_id = tx.get("TransactionID", "unknown")

        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                scored = self.score_fn(tx)
                latency = time.perf_counter() - start

                # Publish to output topic
                producer.send(self.output_topic, value=scored)

                # Update metrics
                TRANSACTIONS_PROCESSED.inc()
                PROCESSING_LATENCY.observe(latency)

                if scored.get("is_fraud", False):
                    TRANSACTIONS_FLAGGED.inc()

                self._update_fraud_rate(scored.get("fraud_score", 0.0))

                logger.debug(
                    f"[Consumer] tx={tx_id} score={scored.get('fraud_score', 0):.4f} "
                    f"latency={latency * 1000:.1f}ms"
                )
                return

            except Exception as e:
                logger.warning(
                    f"[Consumer] Attempt {attempt + 1}/{self.max_retries} "
                    f"failed for tx={tx_id}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep((self.retry_backoff_ms * (2 ** attempt)) / 1000.0)

        # All retries exhausted → DLQ
        logger.error(f"[Consumer] Sending tx={tx_id} to DLQ after {self.max_retries} retries")
        self._send_to_dlq(producer, tx, error="Max retries exceeded")
        DLQ_MESSAGES.inc()

    def _send_to_dlq(self, producer: KafkaProducer, tx: dict, error: str) -> None:
        payload = {
            "original_message": tx,
            "error": error,
            "timestamp": time.time(),
        }
        try:
            producer.send(self.dlq_topic, value=payload)
        except KafkaError as e:
            logger.critical(f"[Consumer] DLQ send failed: {e}")

    def _update_fraud_rate(self, score: float) -> None:
        self._recent_scores.append(score)
        if len(self._recent_scores) > self._window_size:
            self._recent_scores.pop(0)
        if self._recent_scores:
            FRAUD_RATE_GAUGE.set(
                sum(1 for s in self._recent_scores if s >= 0.5) / len(self._recent_scores)
            )

    def _handle_sigterm(self, signum, frame) -> None:
        logger.info("[Consumer] Shutdown signal received")
        self._running = False
