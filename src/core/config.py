"""
Configuration de l'application
"""

import os
import secrets

from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

_INSECURE_DEFAULTS = {"change-this-secret-key", "minioadmin", "minioadmin"}


def _require_env(name: str, default: str, insecure_values: set[str] | None = None) -> str:
    """Retourne la valeur de la variable d'env ; lève une erreur si non définie ou non sécurisée en production."""
    value = os.getenv(name, default)
    if insecure_values and value in insecure_values and not os.getenv("DEBUG", "false").lower() == "true":
        import warnings
        warnings.warn(
            f"[SECURITY] {name} utilise une valeur par défaut non sécurisée. "
            f"Définissez {name} via variable d'environnement avant le déploiement en production.",
            stacklevel=2,
        )
    return value


class Settings:
    """Configuration de l'application"""

    # API
    API_TITLE: str = "PredictML API - Multi Models"
    API_VERSION: str = "2.0.0"
    SECRET_KEY: str = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("API_PORT", "8000"))

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/sklearn_api"
    )

    # MinIO Object Storage
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = _require_env("MINIO_ACCESS_KEY", "minioadmin", {"minioadmin"})
    MINIO_SECRET_KEY: str = _require_env("MINIO_SECRET_KEY", "minioadmin", {"minioadmin"})
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "models")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # Upload
    MAX_MODEL_SIZE_MB: int = int(os.getenv("MAX_MODEL_SIZE_MB", "500"))

    # Redis Cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))

    # MLflow Experiment Tracking
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MLFLOW_EXPERIMENT_PREFIX: str = os.getenv("MLFLOW_EXPERIMENT_PREFIX", "predictml")
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")
    MLFLOW_REGISTER_MODELS: bool = os.getenv("MLFLOW_REGISTER_MODELS", "true").lower() == "true"
    MLFLOW_ENABLE: bool = os.getenv("MLFLOW_ENABLE", "true").lower() == "true"

    # Prometheus metrics — token requis pour protéger /metrics (vide = accès public)
    METRICS_TOKEN: str = os.getenv("METRICS_TOKEN", "")

    # OpenTelemetry
    ENABLE_OTEL: bool = os.getenv("ENABLE_OTEL", "false").lower() == "true"
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "predictml-api")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )

    # SMTP / Alertes e-mail (désactivées par défaut — configurer via variables d'env)
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM: str = os.getenv("SMTP_FROM", "")
    SMTP_STARTTLS: bool = os.getenv("SMTP_STARTTLS", "true").lower() == "true"
    ALERT_EMAIL_TO: list = [
        e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(",") if e.strip()
    ]
    STREAMLIT_URL: str = os.getenv("STREAMLIT_URL", "http://localhost:8501")
    ENABLE_EMAIL_ALERTS: bool = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
    WEEKLY_REPORT_ENABLED: bool = os.getenv("WEEKLY_REPORT_ENABLED", "false").lower() == "true"
    WEEKLY_REPORT_DAY: str = os.getenv("WEEKLY_REPORT_DAY", "monday")
    WEEKLY_REPORT_HOUR: int = int(os.getenv("WEEKLY_REPORT_HOUR", "8"))
    # Seuil de baisse d'accuracy (ex: 0.10 = chute de 10 pts → alerte)
    PERFORMANCE_DRIFT_ALERT_THRESHOLD: float = float(
        os.getenv("PERFORMANCE_DRIFT_ALERT_THRESHOLD", "0.10")
    )
    # Taux d'erreur déclenchant une alerte (ex: 0.10 = 10 %)
    ERROR_RATE_ALERT_THRESHOLD: float = float(os.getenv("ERROR_RATE_ALERT_THRESHOLD", "0.10"))


settings = Settings()
