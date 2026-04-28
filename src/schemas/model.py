"""
Schémas Pydantic pour la création et la réponse de modèles
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelDeleteResponse(BaseModel):
    """Résumé de la suppression de toutes les versions d'un modèle"""

    name: str
    deleted_versions: List[str]
    mlflow_runs_deleted: List[str]
    minio_objects_deleted: List[str]


class FeatureStats(BaseModel):
    """Statistiques d'une feature issues des données d'entraînement"""

    mean: float
    std: float
    min: float
    max: float
    null_rate: Optional[float] = None


class AlertThresholds(BaseModel):
    """Seuils d'alerte configurables par modèle (surcharge les variables d'env globales)"""

    accuracy_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_rate_max: Optional[float] = Field(None, ge=0.0, le=1.0)
    drift_auto_alert: Optional[bool] = None


class ModelUpdateInput(BaseModel):
    """Champs modifiables d'un modèle (tous optionnels)"""

    description: Optional[str] = None
    is_production: Optional[bool] = None
    accuracy: Optional[float] = None
    features_count: Optional[int] = None
    classes: Optional[List[Any]] = None
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    feature_baseline: Optional[Dict[str, FeatureStats]] = None

    tags: Optional[List[str]] = None
    webhook_url: Optional[str] = None

    # A/B Testing & Shadow Deployment
    traffic_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    deployment_mode: Optional[str] = None

    alert_thresholds: Optional[AlertThresholds] = None

    model_config = {"from_attributes": True}


class ModelCreateResponse(BaseModel):
    """Réponse après création ou mise à jour d'un modèle"""

    id: int
    name: str
    version: str
    description: Optional[str]
    algorithm: Optional[str]
    mlflow_run_id: Optional[str]
    minio_bucket: Optional[str]
    minio_object_key: Optional[str]
    file_size_bytes: Optional[int]
    accuracy: Optional[float]
    f1_score: Optional[float]
    features_count: Optional[int]
    classes: Optional[List[Any]]
    training_params: Optional[Dict[str, Any]]
    confidence_threshold: Optional[float] = None
    feature_baseline: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    is_active: bool
    is_production: bool
    status: str = "active"
    traffic_weight: Optional[float] = None
    deployment_mode: Optional[str] = None
    train_script_object_key: Optional[str] = None
    promotion_policy: Optional[Dict[str, Any]] = None
    retrain_schedule: Optional[Dict[str, Any]] = None
    parent_version: Optional[str] = None
    alert_thresholds: Optional[Dict[str, Any]] = None
    created_at: datetime
    user_id_creator: Optional[int]
    creator_username: Optional[str] = None

    model_config = {"from_attributes": True}


class ModelGetResponse(BaseModel):
    """Réponse détaillée d'un modèle — métadonnées complètes + infos de chargement"""

    # Identification
    id: int
    name: str
    version: str

    # Métadonnées
    description: Optional[str]
    algorithm: Optional[str]
    features_count: Optional[int]
    classes: Optional[List[Any]]
    training_params: Optional[Dict[str, Any]]
    training_metrics: Optional[Dict[str, Any]]
    training_dataset: Optional[str]
    trained_by: Optional[str]
    training_date: Optional[datetime]

    # Performance
    accuracy: Optional[float]
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    confidence_threshold: Optional[float] = None

    # Baseline features
    feature_baseline: Optional[Dict[str, Any]] = None

    # Tags et webhook
    tags: Optional[List[str]] = None
    webhook_url: Optional[str] = None

    # Stockage
    mlflow_run_id: Optional[str]
    minio_bucket: Optional[str]
    minio_object_key: Optional[str]
    file_size_bytes: Optional[int]
    file_hash: Optional[str]

    # Créateur
    user_id_creator: Optional[int]
    creator_username: Optional[str]

    # Status
    is_active: bool
    is_production: bool
    status: str = "active"
    traffic_weight: Optional[float] = None
    deployment_mode: Optional[str] = None
    promotion_policy: Optional[Dict[str, Any]] = None
    parent_version: Optional[str] = None
    alert_thresholds: Optional[Dict[str, Any]] = None
    training_stats: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime]
    deprecated_at: Optional[datetime]

    # Infos de chargement
    model_loaded: bool
    model_type: Optional[str]  # ex: "RandomForestClassifier"
    feature_names: Optional[List[str]]  # model.feature_names_in_ si disponible
    load_instructions: Optional[Dict[str, Any]]

    model_config = {"from_attributes": True}


class PerClassMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int


class PeriodPerformance(BaseModel):
    period: str
    matched_count: int
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None


class VersionTimelineEntry(BaseModel):
    version: str
    deployed_at: datetime
    accuracy: Optional[float] = None
    mae: Optional[float] = None
    f1_score: Optional[float] = None
    sample_count: int
    trained_at: Optional[datetime] = None
    n_rows_trained: Optional[int] = None


class PerformanceTimelineResponse(BaseModel):
    model_name: str
    timeline: List[VersionTimelineEntry]


class ModelPerformanceResponse(BaseModel):
    model_name: str
    model_version: Optional[str]
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    total_predictions: int
    matched_predictions: int
    model_type: str  # "classification" ou "regression"

    # Classification
    accuracy: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_weighted: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classes: Optional[List[Any]] = None
    per_class_metrics: Optional[Dict[str, PerClassMetrics]] = None

    # Régression
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None

    # Agrégation temporelle
    by_period: Optional[List[PeriodPerformance]] = None


class FeatureDriftResult(BaseModel):
    """Résultat du drift pour une feature individuelle"""

    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    baseline_min: Optional[float] = None
    baseline_max: Optional[float] = None
    production_mean: Optional[float] = None
    production_std: Optional[float] = None
    production_count: int = 0
    z_score: Optional[float] = None
    psi: Optional[float] = None
    null_rate_production: Optional[float] = None
    null_rate_baseline: Optional[float] = None
    null_rate_status: Optional[str] = None  # "ok" | "warning" | "critical"
    drift_status: str  # "ok" | "warning" | "critical" | "insufficient_data" | "no_baseline"


class DriftReportResponse(BaseModel):
    """Rapport de drift complet pour un modèle"""

    model_name: str
    model_version: Optional[str]
    period_days: int
    predictions_analyzed: int
    baseline_available: bool
    drift_summary: str  # "ok" | "warning" | "critical" | "no_baseline" | "insufficient_data"
    features: Dict[str, FeatureDriftResult]


# ---------------------------------------------------------------------------
# Historique et rollback
# ---------------------------------------------------------------------------


class ModelHistoryEntry(BaseModel):
    """Une entrée d'historique pour un changement d'état de modèle"""

    id: int
    model_name: str
    model_version: str
    changed_by_user_id: Optional[int] = None
    changed_by_username: Optional[str] = None
    action: str
    snapshot: Dict[str, Any]
    changed_fields: Optional[List[str]] = None
    timestamp: datetime

    model_config = {"from_attributes": True}


class ModelHistoryResponse(BaseModel):
    """Réponse pour GET /models/{name}/history et GET /models/{name}/{version}/history"""

    model_name: str
    version: Optional[str] = None  # None = toutes les versions
    entries: List[ModelHistoryEntry]
    total: int


class RollbackResponse(BaseModel):
    """Réponse pour POST /models/{name}/{version}/rollback/{history_id}"""

    model_name: str
    version: str
    rolled_back_to_history_id: int
    new_history_id: int
    restored_fields: List[str]
    snapshot: Dict[str, Any]


class DeprecateModelResponse(BaseModel):
    """Réponse pour PATCH /models/{name}/{version}/deprecate"""

    name: str
    version: str
    status: str
    is_production: bool
    deprecated_at: Optional[datetime]
    deprecated_by: str


# ---------------------------------------------------------------------------
# A/B Testing & Shadow Deployment
# ---------------------------------------------------------------------------


class ABVersionStats(BaseModel):
    """Statistiques par version pour un rapport de comparaison A/B"""

    version: str
    deployment_mode: Optional[str] = None
    traffic_weight: Optional[float] = None
    total_predictions: int
    shadow_predictions: int
    error_rate: float
    avg_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None
    prediction_distribution: Dict[str, int]
    agreement_rate: Optional[float] = None  # taux de concordance shadow vs prod (via id_obs)


class ABSignificance(BaseModel):
    """Résultat du test de significativité statistique entre les deux versions A/B principales"""

    metric: str
    test: str
    p_value: float
    significant: bool
    confidence_level: float
    winner: Optional[str] = None
    min_samples_needed: int
    current_samples: Dict[str, int]


class ABCompareResponse(BaseModel):
    """Réponse de GET /models/{name}/ab-compare"""

    model_name: str
    period_days: int
    versions: List[ABVersionStats]
    ab_significance: Optional[ABSignificance] = None


# ---------------------------------------------------------------------------
# Politique d'auto-promotion post-retrain
# ---------------------------------------------------------------------------


class PromotionPolicy(BaseModel):
    """Politique d'auto-promotion appliquée après un ré-entraînement"""

    min_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_mae: Optional[float] = Field(None, gt=0.0)
    max_latency_p95_ms: Optional[float] = Field(None, gt=0.0)
    min_sample_validation: int = Field(10, ge=1)
    auto_promote: bool = False


class PolicyUpdateResponse(BaseModel):
    """Réponse de PATCH /models/{name}/policy"""

    model_name: str
    promotion_policy: Optional[PromotionPolicy]
    updated_versions: int


# ---------------------------------------------------------------------------
# Ré-entraînement
# ---------------------------------------------------------------------------


class RetrainHistoryEntry(BaseModel):
    """Une entrée d'historique de ré-entraînement pour un modèle"""

    timestamp: datetime
    source_version: Optional[str]
    new_version: str
    trained_by: Optional[str]
    accuracy: Optional[float]
    f1_score: Optional[float]
    auto_promoted: Optional[bool]
    auto_promote_reason: Optional[str]
    n_rows: Optional[int]
    train_start_date: Optional[str]
    train_end_date: Optional[str]

    model_config = {"from_attributes": True}


class RetrainHistoryResponse(BaseModel):
    """Réponse de GET /models/{name}/retrain-history"""

    model_name: str
    history: List[RetrainHistoryEntry]
    total: int


# ---------------------------------------------------------------------------
# Feature importance globale (SHAP agrégé)
# ---------------------------------------------------------------------------


class FeatureImportanceItem(BaseModel):
    mean_abs_shap: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    model_name: str
    version: str
    sample_size: int
    feature_importance: Dict[str, FeatureImportanceItem]


class RetrainRequest(BaseModel):
    """Requête pour POST /models/{name}/{version}/retrain"""

    start_date: str  # YYYY-MM-DD — date de début des données d'entraînement
    end_date: str  # YYYY-MM-DD — date de fin des données d'entraînement
    new_version: Optional[str] = None  # auto-généré si absent
    set_production: bool = False  # passer la nouvelle version en production


class RetrainResponse(BaseModel):
    """Réponse de POST /models/{name}/{version}/retrain"""

    model_name: str
    source_version: str
    new_version: str
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    new_model_metadata: Optional[ModelCreateResponse] = None
    auto_promoted: Optional[bool] = None
    auto_promote_reason: Optional[str] = None
    training_stats: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Planification du ré-entraînement automatique
# ---------------------------------------------------------------------------


class RetrainScheduleInput(BaseModel):
    """Payload pour PATCH /models/{name}/{version}/schedule"""

    cron: Optional[str] = None
    lookback_days: int = Field(30, ge=1)
    auto_promote: bool = False
    enabled: bool = True


class ScheduleUpdateResponse(BaseModel):
    """Réponse de PATCH /models/{name}/{version}/schedule"""

    model_name: str
    version: str
    retrain_schedule: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Validation du schéma d'entrée
# ---------------------------------------------------------------------------


class InputValidationError(BaseModel):
    """Erreur de validation d'une feature d'entrée"""

    type: str  # "missing_feature" | "unexpected_feature"
    feature: str


class InputValidationWarning(BaseModel):
    """Avertissement lors de la validation d'une feature d'entrée"""

    type: str  # "type_coercion"
    feature: str
    from_type: str
    to_type: str


class ValidateInputResponse(BaseModel):
    """Réponse de POST /models/{name}/{version}/validate-input"""

    valid: bool
    errors: List[InputValidationError]
    warnings: List[InputValidationWarning]
    expected_features: Optional[List[str]]


# ---------------------------------------------------------------------------
# Tendance de confiance
# ---------------------------------------------------------------------------


class ConfidenceTrendPoint(BaseModel):
    date: str
    mean_confidence: float
    p25: float
    p75: float
    predictions: int
    low_confidence_count: int


class ConfidenceTrendOverall(BaseModel):
    mean_confidence: float
    p25_confidence: float
    p75_confidence: float
    low_confidence_rate: float


class ConfidenceTrendResponse(BaseModel):
    model_name: str
    version: Optional[str]
    period_days: int
    overall: ConfidenceTrendOverall
    trend: List[ConfidenceTrendPoint]


# ---------------------------------------------------------------------------
# Distribution de confiance
# ---------------------------------------------------------------------------


class ConfidenceBin(BaseModel):
    bin_min: float
    bin_max: float
    count: int
    pct: float


class ConfidenceDistributionResponse(BaseModel):
    model_name: str
    version: Optional[str]
    period_days: int
    sample_count: int
    mean_confidence: float
    pct_high_confidence: float
    pct_uncertain: float
    histogram: List[ConfidenceBin]


# ---------------------------------------------------------------------------
# Calibration des probabilités
# ---------------------------------------------------------------------------


class ReliabilityBin(BaseModel):
    """Un bucket de la courbe de calibration (reliability diagram)"""

    confidence_bin: str
    predicted_rate: float
    observed_rate: float
    count: int


class CalibrationResponse(BaseModel):
    """Réponse de GET /models/{name}/calibration"""

    model_name: str
    version: Optional[str]
    sample_size: int
    brier_score: Optional[float]
    calibration_status: str  # "ok" | "overconfident" | "underconfident" | "insufficient_data"
    mean_confidence: Optional[float]
    mean_accuracy: Optional[float]
    overconfidence_gap: Optional[float]
    reliability: List[ReliabilityBin]


# ---------------------------------------------------------------------------
# Rapport de performance consolidé
# ---------------------------------------------------------------------------


class PerformanceReportResponse(BaseModel):
    """Réponse de GET /models/{name}/performance-report"""

    model_name: str
    generated_at: datetime
    period_days: int
    performance: Optional[ModelPerformanceResponse] = None
    drift: Optional[DriftReportResponse] = None
    feature_importance: Optional[FeatureImportanceResponse] = None
    calibration: Optional[CalibrationResponse] = None
    ab_comparison: Optional[ABCompareResponse] = None


# ---------------------------------------------------------------------------
# Calcul du baseline depuis la production
# ---------------------------------------------------------------------------


class ComputeBaselineResponse(BaseModel):
    """Réponse de POST /models/{name}/{version}/compute-baseline"""

    model_name: str
    version: str
    predictions_used: int
    dry_run: bool
    baseline: Dict[str, FeatureStats]


# ---------------------------------------------------------------------------
# Préchauffage du cache (warmup)
# ---------------------------------------------------------------------------


class WarmupResponse(BaseModel):
    """Réponse de POST /models/{name}/{version}/warmup"""

    model_name: str
    version: str
    already_cached: bool
    load_time_ms: float
    cache_key: str


# ---------------------------------------------------------------------------
# Comparaison multi-versions
# ---------------------------------------------------------------------------


class ModelVersionSummary(BaseModel):
    """Résumé d'une version pour GET /models/{name}/compare"""

    version: str
    is_production: bool
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    drift_status: Optional[str] = None
    brier_score: Optional[float] = None
    trained_at: Optional[datetime] = None
    n_rows_trained: Optional[int] = None


class ModelCompareResponse(BaseModel):
    """Réponse de GET /models/{name}/compare"""

    model_name: str
    compared_at: datetime
    versions: List[ModelVersionSummary]


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


class ReadinessCheck(BaseModel):
    """Résultat d'un check individuel de readiness"""

    model_config = {"populate_by_name": True}

    pass_: bool = Field(..., alias="pass")
    detail: Optional[str] = None


class ReadinessChecks(BaseModel):
    """Ensemble des 4 checks de readiness"""

    is_production: ReadinessCheck
    file_accessible: ReadinessCheck
    baseline_computed: ReadinessCheck
    no_critical_drift: ReadinessCheck


class ReadinessResponse(BaseModel):
    """Réponse de GET /models/{name}/readiness"""

    model_name: str
    version: str
    ready: bool
    checked_at: datetime
    checks: ReadinessChecks


class LeaderboardEntry(BaseModel):
    """Entrée du classement global de performance — GET /models/leaderboard"""

    rank: int
    name: str
    version: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    drift_status: str = "unknown"
    predictions_count: int = 0
