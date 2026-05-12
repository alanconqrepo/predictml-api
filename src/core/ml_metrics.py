"""
Métriques Prometheus métier pour predictml-api.

Ces instruments complètent les métriques HTTP génériques de prometheus-fastapi-instrumentator
en ajoutant une granularité par modèle, version et mode de déploiement.

Compatible avec le mode multiprocess Prometheus (PROMETHEUS_MULTIPROC_DIR configuré
dans docker-compose.yml) — les compteurs sont définis au niveau module.
"""

from prometheus_client import Counter, Histogram

predictions_total = Counter(
    "predictml_predictions_total",
    "Nombre total de prédictions traitées",
    ["model_name", "version", "mode", "status"],
    # mode   : "production" | "ab" | "shadow"
    # status : "success" | "error"
)

inference_duration_seconds = Histogram(
    "predictml_inference_duration_seconds",
    "Durée d'inférence ML de bout en bout (chargement modèle + prédiction)",
    ["model_name", "version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

retrain_total = Counter(
    "predictml_retrain_total",
    "Nombre de ré-entraînements déclenchés (manuels ou planifiés)",
    ["model_name", "status"],
    # status : "success" | "failure"
)

drift_detected_total = Counter(
    "predictml_drift_detected_total",
    "Événements de drift détectés par le scheduler de supervision",
    ["model_name", "drift_type", "severity"],
    # drift_type : "feature" | "performance" | "output" | "error_rate"
    # severity   : "warning" | "critical"
)
