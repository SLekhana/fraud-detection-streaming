"""
Application configuration — loaded from environment / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # OpenAI
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o-mini"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_transactions: str = "fraud.transactions.raw"
    kafka_topic_scores: str = "fraud.scores"
    kafka_topic_dlq: str = "fraud.dlq"
    kafka_consumer_group: str = "fraud-scorer"

    # Spark
    spark_master: str = "local[*]"
    spark_app_name: str = "FraudDetectionStreaming"

    # Database
    database_url: str = "postgresql://fraud:fraud@localhost:5432/frauddb"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Model paths
    model_path: str = "models/"
    autoencoder_path: str = "models/autoencoder.pt"
    xgboost_path: str = "models/xgboost.json"
    threshold_autoencoder: float = 0.05
    threshold_ensemble: float = 0.5

    # Drift alerting
    slack_webhook_url: str = ""
    drift_precision_threshold: float = 0.05
    drift_recall_threshold: float = 0.05
    drift_check_interval_seconds: int = 300

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
