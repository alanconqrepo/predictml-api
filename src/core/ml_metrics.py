"""
Business Prometheus metrics for predictml-api.

These instruments complement the generic HTTP metrics from
prometheus-fastapi-instrumentator by adding per-model, per-version, and
per-deployment-mode granularity.

Compatible with Prometheus multiprocess mode (PROMETHEUS_MULTIPROC_DIR
configured in docker-compose.yml) — counters are defined at module level.
"""

from prometheus_client import Counter, Histogram

predictions_total = Counter(
    "predictml_predictions_total",
    "Total number of predictions processed",
    ["model_name", "version", "mode", "status"],
    # mode   : "production" | "ab" | "shadow"
    # status : "success" | "error"
)

inference_duration_seconds = Histogram(
    "predictml_inference_duration_seconds",
    "End-to-end ML inference duration (model loading + prediction)",
    ["model_name", "version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

retrain_total = Counter(
    "predictml_retrain_total",
    "Number of retraining runs triggered (manual or scheduled)",
    ["model_name", "status"],
    # status : "success" | "failure"
)

drift_detected_total = Counter(
    "predictml_drift_detected_total",
    "Drift events detected by the supervision scheduler",
    ["model_name", "drift_type", "severity"],
    # drift_type : "feature" | "performance" | "output" | "error_rate"
    # severity   : "warning" | "critical"
)
