from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    model_path: str = "models/v3"
    environment: str = "development"
    kafka_bootstrap_servers: str = "localhost:9092"
    threshold_ensemble: float = 0.5
    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "info"
    redis_host: str = "localhost"
    redis_port: int = 6379
    kafka_topic: str = "fraud.transactions.raw"
    model_dir: str = "models"
    data_dir: str = "data"
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o-mini"
    drift_precision_threshold: float = 0.05
    drift_recall_threshold: float = 0.05
    slack_webhook_url: str = ""


settings = Settings()
