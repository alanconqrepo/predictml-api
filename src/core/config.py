"""
Configuration de l'application
"""

import os

from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

_INSECURE_DEFAULTS = {"change-this-secret-key", "minioadmin", "minioadmin"}

_MISSING = object()


def _require_env(
    name: str, default: object = _MISSING, insecure_values: set[str] | None = None
) -> str:
    """Retourne la valeur de la variable d'env.

    Si la variable n'est pas définie et qu'aucun default n'est fourni, lève
    EnvironmentError — utilisé pour les variables obligatoires comme SECRET_KEY.
    """
    raw = os.getenv(name)
    if raw is None:
        if default is _MISSING:
            raise EnvironmentError(
                f"[CONFIG] La variable d'environnement '{name}' est obligatoire mais non définie. "
                f'Générez une valeur avec : python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        value = str(default)
    else:
        value = raw
    if insecure_values and value in insecure_values:
        if os.getenv("DEBUG", "false").lower() == "true":
            import warnings

            warnings.warn(
                f"[SECURITY] {name} utilise une valeur par défaut non sécurisée. "
                f"Définissez {name} via variable d'environnement avant le déploiement en production.",
                stacklevel=2,
            )
        else:
            raise EnvironmentError(
                f"[SECURITY] {name} utilise une valeur non sécurisée interdite en production. "
                f"Définissez {name} avec une valeur forte avant le déploiement. "
                f'Générez-en une avec : python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
    return value


class Settings:
    """Configuration de l'application"""

    # API
    API_TITLE: str = "PredictML API - Multi Models"
    API_VERSION: str = "2.0.0"
    SECRET_KEY: str = _require_env("SECRET_KEY", insecure_values=set())
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "")
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@predictml.local")

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("API_PORT", "8000"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL") or (
        "postgresql+asyncpg://postgres:postgres@localhost:{port}/{db}".format(
            port=os.getenv("POSTGRES_INTERNAL_PORT", "5432"),
            db=os.getenv("POSTGRES_DB", "sklearn_api"),
        )
    )
    # Empty string = no replica configured; falls back to DATABASE_URL
    DATABASE_READ_REPLICA_URL: str = os.getenv("DATABASE_READ_REPLICA_URL", "")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "40"))

    # MinIO Object Storage
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = _require_env("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = _require_env("MINIO_SECRET_KEY", "minioadmin", {"minioadmin"})
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "models")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # Upload
    MAX_MODEL_SIZE_MB: int = int(os.getenv("MAX_MODEL_SIZE_MB", "500"))

    # Redis Cache
    REDIS_URL: str = os.getenv("REDIS_URL") or (
        "redis://:{pw}@localhost:{port}/0".format(
            pw=os.getenv("REDIS_PASSWORD", ""),
            port=os.getenv("REDIS_PORT", "6379"),
        )
        if os.getenv("REDIS_PASSWORD")
        else "redis://localhost:{}/0".format(os.getenv("REDIS_PORT", "6379"))
    )
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    # Sentinel hosts — "sentinel1:26379,sentinel2:26379,sentinel3:26379"
    # When set, the API connects via Sentinel instead of directly to REDIS_URL.
    REDIS_SENTINEL_HOSTS: str = os.getenv("REDIS_SENTINEL_HOSTS", "")

    # MLflow Experiment Tracking
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:{}".format(
        os.getenv("MLFLOW_PORT", "5000")
    )
    MLFLOW_EXPERIMENT_PREFIX: str = os.getenv("MLFLOW_EXPERIMENT_PREFIX", "predictml")
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")
    MLFLOW_REGISTER_MODELS: bool = os.getenv("MLFLOW_REGISTER_MODELS", "true").lower() == "true"
    MLFLOW_ENABLE: bool = os.getenv("MLFLOW_ENABLE", "true").lower() == "true"

    # Prometheus metrics — obligatoire en production (DEBUG=False) ; vide interdit hors dev
    METRICS_TOKEN: str = os.getenv("METRICS_TOKEN", "")

    # OpenTelemetry
    ENABLE_OTEL: bool = os.getenv("ENABLE_OTEL", "false").lower() == "true"
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "predictml-api")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or (
        "http://localhost:{}".format(os.getenv("GRAFANA_GRPC_PORT", "4317"))
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
    STREAMLIT_URL: str = os.getenv("STREAMLIT_URL") or "http://localhost:{}".format(
        os.getenv("STREAMLIT_PORT", "8501")
    )
    ENABLE_EMAIL_ALERTS: bool = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
    WEEKLY_REPORT_ENABLED: bool = os.getenv("WEEKLY_REPORT_ENABLED", "false").lower() == "true"
    WEEKLY_REPORT_DAY: str = os.getenv("WEEKLY_REPORT_DAY", "monday")
    WEEKLY_REPORT_HOUR: int = int(os.getenv("WEEKLY_REPORT_HOUR", "8"))

    # Token expiration (days) — 0 = pas d'expiration
    TOKEN_LIFETIME_DAYS: int = int(os.getenv("TOKEN_LIFETIME_DAYS", "90"))
    # Seuil de baisse d'accuracy (ex: 0.10 = chute de 10 pts → alerte)
    PERFORMANCE_DRIFT_ALERT_THRESHOLD: float = float(
        os.getenv("PERFORMANCE_DRIFT_ALERT_THRESHOLD", "0.10")
    )
    # Taux d'erreur déclenchant une alerte (ex: 0.10 = 10 %)
    ERROR_RATE_ALERT_THRESHOLD: float = float(os.getenv("ERROR_RATE_ALERT_THRESHOLD", "0.10"))

    # Analytics safety caps — protect aggregation queries from full-table scans
    MAX_ROWS_ANALYTICS: int = int(os.getenv("MAX_ROWS_ANALYTICS", "50000"))
    ANALYTICS_MAX_DAYS: int = int(os.getenv("ANALYTICS_MAX_DAYS", "90"))

    # Redis Streams — queue asynchrone des writes de prédictions
    PREDICTION_STREAM_ENABLED: bool = (
        os.getenv("PREDICTION_STREAM_ENABLED", "false").lower() == "true"
    )
    PREDICTION_STREAM_NAME: str = os.getenv("PREDICTION_STREAM_NAME", "predictions:new")
    PREDICTION_STREAM_DLQ: str = os.getenv("PREDICTION_STREAM_DLQ", "predictions:dlq")
    PREDICTION_STREAM_BATCH_SIZE: int = int(os.getenv("PREDICTION_STREAM_BATCH_SIZE", "100"))
    PREDICTION_STREAM_FLUSH_MS: int = int(os.getenv("PREDICTION_STREAM_FLUSH_MS", "500"))
    PREDICTION_STREAM_MAX_RETRIES: int = int(os.getenv("PREDICTION_STREAM_MAX_RETRIES", "3"))
    PREDICTION_STREAM_MAXLEN: int = int(os.getenv("PREDICTION_STREAM_MAXLEN", "100000"))


settings = Settings()
