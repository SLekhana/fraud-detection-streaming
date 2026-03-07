"""
Kafka Producer — streams transaction events to fraud.transactions.raw topic.

In production this would be triggered by payment gateway webhooks.
In development/testing, use the simulate() method to replay IEEE-CIS data.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Generator

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

from app.core.config import settings

logger = logging.getLogger(__name__)


class TransactionProducer:
    """
    Kafka producer that serialises transaction dicts to JSON
    and publishes to the transactions topic.

    Dead-letter queue: failed messages are sent to fraud.dlq.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str | None = None,
        dlq_topic: str | None = None,
        retries: int = 3,
    ):
        self.topic = topic or settings.kafka_topic_transactions
        self.dlq_topic = dlq_topic or settings.kafka_topic_dlq
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.retries = retries
        self._producer: KafkaProducer | None = self._get_producer()

    def _get_producer(self) -> KafkaProducer:
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k else None,
                retries=self.retries,
                acks="all",  # wait for all replicas
                compression_type="gzip",
                batch_size=16384,
                linger_ms=5,
            )
        return self._producer

    def send(self, transaction: dict, key: str | None = None) -> bool:
        """
        Send a single transaction to Kafka.
        Returns True on success, False on failure (sent to DLQ).
        """
        try:
            producer = self._get_producer()
            future = producer.send(
                self.topic,
                value=transaction,
                key=key or str(transaction.get("TransactionID", "")),
            )
            future.get(timeout=10)
            return True
        except (KafkaError, Exception) as e:
            logger.error(
                f"Failed to send transaction {transaction.get('TransactionID')}: {e}"
            )
            self._send_to_dlq(transaction, error=str(e))
            return False

    def _send_to_dlq(self, transaction: dict, error: str) -> None:
        """Send failed message to dead-letter queue."""
        try:
            producer = self._get_producer()
            dlq_payload = {
                "original_message": transaction,
                "error": error,
                "timestamp": time.time(),
            }
            producer.send(self.dlq_topic, value=dlq_payload)
        except Exception as e:
            logger.critical(f"DLQ send failed: {e}")

    def simulate(
        self,
        df: pd.DataFrame,
        delay_ms: float = 10.0,
        max_records: int | None = None,
    ) -> Generator[dict, None, None]:
        """
        Replay IEEE-CIS transactions at configurable rate.
        Yields sent transaction dicts.

        Args:
            df: IEEE-CIS transaction DataFrame.
            delay_ms: Delay between messages (simulates real-time rate).
            max_records: Stop after this many records (None = all).
        """
        count = 0
        for _, row in df.iterrows():
            if max_records and count >= max_records:
                break

            tx = row.to_dict()
            # Normalise NaN → None for JSON serialisation
            tx = {k: (None if pd.isna(v) else v) for k, v in tx.items()}

            success = self.send(tx)
            if success:
                count += 1
                logger.debug(f"Sent transaction {tx.get('TransactionID')} ({count})")

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            yield tx

        logger.info(f"Simulation complete: {count} transactions sent")

    def flush(self) -> None:
        if self._producer:
            self._producer.flush()

    def close(self) -> None:
        if self._producer:
            self._producer.close()
            self._producer = None

    def __enter__(self) -> "TransactionProducer":
        return self

    def __exit__(self, *args) -> None:
        self.flush()
        self.close()
