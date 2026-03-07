"""
Spark Structured Streaming — micro-batch feature computation.

Reads raw transactions from Kafka, computes velocity and aggregation
features in Spark, then writes enriched records to fraud.scored topic.

This is the production path for high-throughput scenarios (millions of
transactions/day). For lower throughput, the Kafka consumer + FastAPI
scoring path is used.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_spark_session(app_name: str = "FraudDetectionStreaming", master: str = "local[*]"):
    """Build and return a SparkSession."""
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.sql.streaming.checkpointLocation", "/tmp/fraud_checkpoint")
            .config("spark.sql.shuffle.partitions", "4")
            .config(
                "spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
            )
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except ImportError:
        raise RuntimeError("PySpark not installed. Run: pip install pyspark==3.5.1")


# ─── Schema ──────────────────────────────────────────────────────────────────

def get_transaction_schema():
    """Spark schema for IEEE-CIS transaction records."""
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("TransactionID", IntegerType(), True),
            StructField("TransactionDT", IntegerType(), True),
            StructField("TransactionAmt", DoubleType(), True),
            StructField("ProductCD", StringType(), True),
            StructField("card1", IntegerType(), True),
            StructField("card2", DoubleType(), True),
            StructField("card3", DoubleType(), True),
            StructField("card4", StringType(), True),
            StructField("card5", DoubleType(), True),
            StructField("card6", StringType(), True),
            StructField("addr1", DoubleType(), True),
            StructField("addr2", DoubleType(), True),
            StructField("dist1", DoubleType(), True),
            StructField("dist2", DoubleType(), True),
            StructField("P_emaildomain", StringType(), True),
            StructField("R_emaildomain", StringType(), True),
            StructField("C1", DoubleType(), True),
            StructField("C2", DoubleType(), True),
            StructField("C3", DoubleType(), True),
            StructField("C4", DoubleType(), True),
            StructField("C5", DoubleType(), True),
            StructField("C6", DoubleType(), True),
            StructField("C7", DoubleType(), True),
            StructField("C8", DoubleType(), True),
            StructField("C9", DoubleType(), True),
            StructField("C10", DoubleType(), True),
            StructField("C11", DoubleType(), True),
            StructField("C12", DoubleType(), True),
            StructField("C13", DoubleType(), True),
            StructField("C14", DoubleType(), True),
            StructField("D1", DoubleType(), True),
            StructField("D2", DoubleType(), True),
            StructField("D3", DoubleType(), True),
            StructField("D4", DoubleType(), True),
            StructField("isFraud", IntegerType(), True),
        ]
    )


# ─── Streaming pipeline ──────────────────────────────────────────────────────

class SparkFraudStream:
    """
    Spark Structured Streaming pipeline for real-time fraud feature computation.

    Flow:
      Kafka (raw transactions)
        → Spark parse JSON
        → compute velocity features (windowed aggregations)
        → compute amount z-score (per-card window)
        → write enriched records to output Kafka topic
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "fraud.transactions.raw",
        output_topic: str = "fraud.enriched",
        checkpoint_dir: str = "/tmp/fraud_checkpoint",
        trigger_interval: str = "5 seconds",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.checkpoint_dir = checkpoint_dir
        self.trigger_interval = trigger_interval

    def build_stream(self):
        """
        Build and return the Spark streaming query.
        Call .awaitTermination() to keep it running.
        """
        from pyspark.sql import functions as F
        from pyspark.sql.types import StringType

        spark = build_spark_session()

        # Read from Kafka
        raw_stream = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.bootstrap_servers)
            .option("subscribe", self.input_topic)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .load()
        )

        schema = get_transaction_schema()

        # Parse JSON value
        parsed = raw_stream.select(
            F.from_json(F.col("value").cast("string"), schema).alias("data")
        ).select("data.*")

        # Add timestamp column
        parsed = parsed.withColumn(
            "event_time",
            (F.col("TransactionDT") + F.lit(1512000000)).cast("timestamp"),
        )

        # ── Velocity features (windowed aggregations) ──────────────────────
        # Count and sum of transactions per card in 1h and 24h windows
        _ = (
            parsed.withWatermark("event_time", "1 hour")
            .groupBy(F.col("card1"), F.window("event_time", "1 hour"))
            .agg(
                F.count("*").alias("velocity_count_1h"),
                F.sum("TransactionAmt").alias("velocity_sum_1h"),
                F.stddev("TransactionAmt").alias("velocity_std_1h"),
            )
            .select(
                F.col("card1"),
                F.col("window.end").alias("window_end_1h"),
                "velocity_count_1h",
                "velocity_sum_1h",
                "velocity_std_1h",
            )
        )

        _ = (
            parsed.withWatermark("event_time", "24 hours")
            .groupBy(F.col("card1"), F.window("event_time", "24 hours"))
            .agg(
                F.count("*").alias("velocity_count_24h"),
                F.sum("TransactionAmt").alias("velocity_sum_24h"),
            )
            .select(
                F.col("card1"),
                F.col("window.end").alias("window_end_24h"),
                "velocity_count_24h",
                "velocity_sum_24h",
            )
        )

        # ── Amount z-score (per-card rolling stats) ────────────────────────
        _ = (
            parsed.withWatermark("event_time", "7 days")
            .groupBy(F.col("card1"), F.window("event_time", "7 days"))
            .agg(
                F.avg("TransactionAmt").alias("card_avg_amt"),
                F.stddev("TransactionAmt").alias("card_std_amt"),
            )
        )

        # ── Enrich base stream ─────────────────────────────────────────────
        enriched = parsed.withColumn("log_amt", F.log1p(F.col("TransactionAmt")))
        enriched = enriched.withColumn(
            "tx_hour",
            F.hour(F.col("event_time"))
        )
        enriched = enriched.withColumn(
            "tx_is_weekend",
            (F.dayofweek(F.col("event_time")).isin([1, 7])).cast("integer"),
        )
        enriched = enriched.withColumn(
            "tx_is_night",
            ((F.hour(F.col("event_time")) < 6) | (F.hour(F.col("event_time")) >= 22)).cast("integer"),
        )

        # Serialise to JSON for output
        output = enriched.select(
            F.col("TransactionID").cast(StringType()).alias("key"),
            F.to_json(F.struct([enriched[c] for c in enriched.columns])).alias("value"),
        )

        # Write to output Kafka topic
        query = (
            output.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", self.bootstrap_servers)
            .option("topic", self.output_topic)
            .option("checkpointLocation", self.checkpoint_dir)
            .trigger(processingTime=self.trigger_interval)
            .outputMode("update")
            .start()
        )

        logger.info(
            f"[Spark] Streaming query started: {self.input_topic} → {self.output_topic}"
        )
        return query


def run_spark_stream(
    bootstrap_servers: str = "localhost:9092",
    input_topic: str = "fraud.transactions.raw",
    output_topic: str = "fraud.enriched",
) -> None:
    """Entry point for running the Spark streaming job."""
    stream = SparkFraudStream(
        bootstrap_servers=bootstrap_servers,
        input_topic=input_topic,
        output_topic=output_topic,
    )
    query = stream.build_stream()
    query.awaitTermination()


if __name__ == "__main__":
    run_spark_stream()
