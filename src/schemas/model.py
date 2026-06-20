"""
Pydantic schemas for model creation and responses
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ModelDeleteResponse(BaseModel):
    """Summary of the deletion of all versions of a model"""

    name: str
    deleted_versions: List[str]
    mlflow_runs_deleted: List[str]
    minio_objects_deleted: List[str]


class FeatureStats(BaseModel):
    """Statistics for a feature derived from training data"""

    mean: float
    std: float
    min: float
    max: float
    null_rate: Optional[float] = None


class AlertThresholds(BaseModel):
    """Per-model configurable alert thresholds (overrides global environment variables)"""

    accuracy_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    auc_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_rate_max: Optional[float] = Field(None, ge=0.0, le=1.0)
    drift_auto_alert: Optional[bool] = None


class ModelUpdateInput(BaseModel):
    """Editable fields of a model (all optional)"""

    description: Optional[str] = None
    is_production: Optional[bool] = None
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    features_count: Optional[int] = None
    classes: Optional[List[Any]] = None
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    feature_baseline: Optional[Dict[str, FeatureStats]] = None
    categorical_baseline: Optional[Dict[str, Dict[str, float]]] = None
    feature_importances: Optional[Dict[str, float]] = None
    training_dataset: Optional[str] = None

    hyperparameters: Optional[Dict[str, Any]] = None

    tags: Optional[List[str]] = None
    webhook_url: Optional[str] = None

    # A/B Testing & Shadow Deployment
    traffic_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    deployment_mode: Optional[str] = None

    alert_thresholds: Optional[AlertThresholds] = None
    training_stats: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}


class ModelCreateResponse(BaseModel):
    """Response after model creation or update"""

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
    auc: Optional[float] = None
    f1_score: Optional[float]
    features_count: Optional[int]
    classes: Optional[List[Any]]
    training_params: Optional[Dict[str, Any]]
    hyperparameters: Optional[Dict[str, Any]] = None
    confidence_threshold: Optional[float] = None
    feature_baseline: Optional[Dict[str, Any]] = None
    categorical_baseline: Optional[Dict[str, Dict[str, float]]] = None
    feature_importances: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    is_active: bool
    is_production: bool
    status: str = "active"
    traffic_weight: Optional[float] = None
    deployment_mode: Optional[str] = None
    train_script_object_key: Optional[str] = None
    requirements_object_key: Optional[str] = None
    promotion_policy: Optional[Dict[str, Any]] = None
    retrain_schedule: Optional[Dict[str, Any]] = None
    parent_version: Optional[str] = None
    alert_thresholds: Optional[Dict[str, Any]] = None
    model_task: Optional[str] = None
    created_at: datetime
    user_id_creator: Optional[int]
    creator_username: Optional[str] = None

    model_config = {"from_attributes": True}


class ModelGetResponse(BaseModel):
    """Detailed model response — full metadata + loading info"""

    # Identification
    id: int
    name: str
    version: str

    # Metadata
    description: Optional[str]
    algorithm: Optional[str]
    features_count: Optional[int]
    classes: Optional[List[Any]]
    training_params: Optional[Dict[str, Any]]
    training_metrics: Optional[Dict[str, Any]]
    hyperparameters: Optional[Dict[str, Any]] = None
    training_dataset: Optional[str]
    trained_by: Optional[str]
    training_date: Optional[datetime]

    # Performance
    accuracy: Optional[float]
    auc: Optional[float] = None
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    confidence_threshold: Optional[float] = None

    # Feature baseline
    feature_baseline: Optional[Dict[str, Any]] = None
    categorical_baseline: Optional[Dict[str, Dict[str, float]]] = None
    feature_importances: Optional[Dict[str, float]] = None

    # Tags and webhook
    tags: Optional[List[str]] = None
    webhook_url: Optional[str] = None

    # Storage
    mlflow_run_id: Optional[str]
    minio_bucket: Optional[str]
    minio_object_key: Optional[str]
    file_size_bytes: Optional[int]
    file_hash: Optional[str]
    train_script_object_key: Optional[str] = None
    requirements_object_key: Optional[str] = None

    # Creator
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

    model_task: Optional[str] = (
        None  # "regression" | "classification_binary" | "classification_multiclass"
    )

    # Loading info
    model_loaded: bool
    model_type: Optional[str]  # e.g. "RandomForestClassifier"
    feature_names: Optional[List[str]]  # model.feature_names_in_ if available
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
    auc: Optional[float] = None
    f1_weighted: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None


class VersionTimelineEntry(BaseModel):
    version: str
    deployed_at: datetime
    accuracy: Optional[float] = None
    auc: Optional[float] = None
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
    model_type: str  # "classification" or "regression"

    # Classification
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_weighted: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classes: Optional[List[Any]] = None
    per_class_metrics: Optional[Dict[str, PerClassMetrics]] = None

    # ROC curve (binary classification with probabilities available)
    roc_curve_fpr: Optional[List[float]] = None
    roc_curve_tpr: Optional[List[float]] = None

    # Regression
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None

    # Temporal aggregation
    by_period: Optional[List[PeriodPerformance]] = None


class FeatureDriftResult(BaseModel):
    """Drift result for an individual numeric feature"""

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


class CategoricalDriftResult(BaseModel):
    """Drift result for an individual categorical feature (PSI on frequencies)"""

    baseline_distribution: Dict[str, float]  # {category: frequency at training}
    production_distribution: Dict[str, float]  # {category: frequency in production}
    production_count: int = 0
    psi: Optional[float] = None
    drift_status: str  # "ok" | "warning" | "critical" | "insufficient_data" | "no_baseline"


class DriftReportResponse(BaseModel):
    """Complete drift report for a model"""

    model_name: str
    model_version: Optional[str]
    period_days: int
    predictions_analyzed: int
    baseline_available: bool
    drift_summary: str  # "ok" | "warning" | "critical" | "no_baseline" | "insufficient_data"
    features: Dict[str, FeatureDriftResult]
    categorical_features: Dict[str, CategoricalDriftResult] = {}


# ---------------------------------------------------------------------------
# Output drift (label shift monitoring)
# ---------------------------------------------------------------------------


class OutputDriftClassResult(BaseModel):
    """Per-class distribution for output drift"""

    label: str
    baseline_ratio: float
    current_ratio: float
    delta: float


class OutputDriftResponse(BaseModel):
    """Output distribution drift report (label shift)"""

    model_name: str
    model_version: Optional[str]
    period_days: int
    predictions_analyzed: int
    status: str  # "ok" | "warning" | "critical" | "no_baseline" | "insufficient_data"
    psi: Optional[float] = None
    baseline_distribution: Optional[Dict[str, float]] = None
    current_distribution: Optional[Dict[str, float]] = None
    by_class: Optional[List[OutputDriftClassResult]] = None


# ---------------------------------------------------------------------------
# History and rollback
# ---------------------------------------------------------------------------


class ModelHistoryEntry(BaseModel):
    """A history entry for a model state change"""

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
    """Response for GET /models/{name}/history and GET /models/{name}/{version}/history"""

    model_name: str
    version: Optional[str] = None  # None = all versions
    entries: List[ModelHistoryEntry]
    total: int


class RollbackResponse(BaseModel):
    """Response for POST /models/{name}/{version}/rollback/{history_id}"""

    model_name: str
    version: str
    rolled_back_to_history_id: int
    new_history_id: int
    restored_fields: List[str]
    snapshot: Dict[str, Any]


class DeprecateModelResponse(BaseModel):
    """Response for PATCH /models/{name}/{version}/deprecate"""

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
    """Per-version statistics for an A/B comparison report"""

    version: str
    deployment_mode: Optional[str] = None
    traffic_weight: Optional[float] = None
    total_predictions: int
    shadow_predictions: int
    error_rate: float
    avg_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None
    prediction_distribution: Dict[str, int]
    agreement_rate: Optional[float] = None  # shadow vs prod agreement rate (via id_obs)


class ABSignificance(BaseModel):
    """Result of the statistical significance test between the two main A/B versions"""

    metric: str
    test: str
    p_value: float
    significant: bool
    confidence_level: float
    winner: Optional[str] = None
    min_samples_needed: int
    current_samples: Dict[str, int]


class ABCompareResponse(BaseModel):
    """Response for GET /models/{name}/ab-compare"""

    model_name: str
    period_days: int
    versions: List[ABVersionStats]
    ab_significance: Optional[ABSignificance] = None


class ShadowCompareResponse(BaseModel):
    """Response for GET /models/{name}/shadow-compare"""

    model_name: str
    shadow_version: Optional[str] = None
    production_version: Optional[str] = None
    period_days: int
    n_comparable: int
    agreement_rate: Optional[float] = None
    shadow_confidence_delta: Optional[float] = None
    shadow_latency_delta_ms: Optional[float] = None
    shadow_accuracy: Optional[float] = None
    production_accuracy: Optional[float] = None
    accuracy_available: bool
    recommendation: str


# ---------------------------------------------------------------------------
# Auto-promotion policy post-retrain
# ---------------------------------------------------------------------------


class PromotionPolicy(BaseModel):
    """Auto-promotion policy applied after retraining"""

    min_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_mae: Optional[float] = Field(None, gt=0.0)
    max_latency_p95_ms: Optional[float] = Field(None, gt=0.0)
    min_sample_validation: int = Field(10, ge=1)
    auto_promote: bool = False
    min_golden_test_pass_rate: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Circuit breaker — auto-demotion of the production model
    auto_demote: bool = False
    demote_on_drift: Literal["warning", "critical"] = "critical"
    demote_on_accuracy_below: Optional[float] = Field(None, ge=0.0, le=1.0)
    demote_cooldown_hours: int = Field(24, ge=0)


class PolicyUpdateResponse(BaseModel):
    """Response for PATCH /models/{name}/policy"""

    model_name: str
    promotion_policy: Optional[PromotionPolicy]
    updated_versions: int


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------


class RetrainHistoryEntry(BaseModel):
    """A retraining history entry for a model"""

    timestamp: datetime
    source_version: Optional[str]
    new_version: str
    trained_by: Optional[str]
    accuracy: Optional[float]
    auc: Optional[float] = None
    f1_score: Optional[float]
    auto_promoted: Optional[bool]
    auto_promote_reason: Optional[str]
    n_rows: Optional[int]
    train_start_date: Optional[str]
    train_end_date: Optional[str]

    model_config = {"from_attributes": True}


class RetrainHistoryResponse(BaseModel):
    """Response for GET /models/{name}/retrain-history"""

    model_name: str
    history: List[RetrainHistoryEntry]
    total: int


# ---------------------------------------------------------------------------
# Global feature importance (aggregated SHAP)
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
    """Request body for POST /models/{name}/{version}/retrain"""

    start_date: str  # YYYY-MM-DD — start date of training data
    end_date: str  # YYYY-MM-DD — end date of training data
    new_version: Optional[str] = None  # auto-generated if absent
    set_production: bool = False  # promote the new version to production


class RetrainResponse(BaseModel):
    """Response for POST /models/{name}/{version}/retrain"""

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
# Automatic retraining scheduling
# ---------------------------------------------------------------------------


class RetrainScheduleInput(BaseModel):
    """Payload for PATCH /models/{name}/{version}/schedule"""

    cron: Optional[str] = None
    lookback_days: int = Field(30, ge=1)
    auto_promote: bool = False
    enabled: bool = True
    trigger_on_drift: Optional[Literal["warning", "critical"]] = None
    drift_retrain_cooldown_hours: int = Field(24, ge=1)


class ScheduleUpdateResponse(BaseModel):
    """Response for PATCH /models/{name}/{version}/schedule"""

    model_name: str
    version: str
    retrain_schedule: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Input schema validation
# ---------------------------------------------------------------------------


class InputValidationError(BaseModel):
    """Validation error for an input feature"""

    type: str  # "missing_feature" | "unexpected_feature"
    feature: str


class InputValidationWarning(BaseModel):
    """Warning during input feature validation"""

    type: str  # "type_coercion"
    feature: str
    from_type: str
    to_type: str


class ValidateInputResponse(BaseModel):
    """Response for POST /models/{name}/{version}/validate-input"""

    valid: bool
    errors: List[InputValidationError]
    warnings: List[InputValidationWarning]
    expected_features: Optional[List[str]]


# ---------------------------------------------------------------------------
# Confidence trend
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
# Confidence distribution
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
# Probability calibration
# ---------------------------------------------------------------------------


class ReliabilityBin(BaseModel):
    """A bucket of the calibration curve (reliability diagram)"""

    confidence_bin: str
    predicted_rate: float
    observed_rate: float
    count: int


class CalibrationResponse(BaseModel):
    """Response for GET /models/{name}/calibration"""

    model_name: str
    version: Optional[str]
    sample_size: int
    # Common field — "classification" | "regression"
    model_type: str = "classification"
    # ── Classification ──
    brier_score: Optional[float] = None
    calibration_status: str = "insufficient_data"
    # "ok" | "overconfident" | "underconfident" | "insufficient_data"
    # "biased_high" | "biased_low"  (regression)
    mean_confidence: Optional[float] = None
    mean_accuracy: Optional[float] = None
    overconfidence_gap: Optional[float] = None
    reliability: List[ReliabilityBin] = []
    # ── Regression ──
    mae: Optional[float] = None  # Mean Absolute Error
    rmse: Optional[float] = None  # Root Mean Square Error
    r2: Optional[float] = None  # Coefficient of determination
    bias: Optional[float] = None  # mean(ŷ − y): positive = overestimation
    scatter_data: Optional[List[dict]] = None  # [{pred, obs}, …] sample ≤ 300


# ---------------------------------------------------------------------------
# Consolidated performance report
# ---------------------------------------------------------------------------


class PerformanceReportResponse(BaseModel):
    """Response for GET /models/{name}/performance-report"""

    model_name: str
    generated_at: datetime
    period_days: int
    performance: Optional[ModelPerformanceResponse] = None
    drift: Optional[DriftReportResponse] = None
    feature_importance: Optional[FeatureImportanceResponse] = None
    calibration: Optional[CalibrationResponse] = None
    ab_comparison: Optional[ABCompareResponse] = None


# ---------------------------------------------------------------------------
# Model Card Export
# ---------------------------------------------------------------------------


class ModelCardPerformanceSummary(BaseModel):
    model_type: str
    matched_predictions: int
    total_predictions: int
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None


class ModelCardDriftSummary(BaseModel):
    drift_summary: str
    baseline_available: bool
    predictions_analyzed: int
    top_drifting_features: Optional[List[str]] = None
    last_check_at: Optional[datetime] = None


class ModelCardCalibrationSummary(BaseModel):
    calibration_status: str
    brier_score: Optional[float] = None
    sample_size: int


class ModelCardTopFeature(BaseModel):
    feature: str
    mean_abs_shap: float


class ModelCardFeatureImportanceSummary(BaseModel):
    top_features: List[ModelCardTopFeature]
    sample_size: int


class ModelCardRetrainInfo(BaseModel):
    last_retrain_date: Optional[datetime] = None
    trained_by: Optional[str] = None
    n_rows_trained: Optional[int] = None
    next_run_at: Optional[datetime] = None


class ModelCardCoverage(BaseModel):
    coverage_rate: float
    labeled_count: int
    total_predictions: int


class ModelCardResponse(BaseModel):
    """Response for GET /models/{name}/{version}/card"""

    model_name: str
    version: str
    generated_at: datetime
    algorithm: Optional[str] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    tags: Optional[List[Any]] = None
    classes: Optional[List[Any]] = None
    features_count: Optional[int] = None
    trained_by: Optional[str] = None
    training_dataset: Optional[str] = None
    created_at: Optional[datetime] = None
    is_production: bool
    performance: Optional[ModelCardPerformanceSummary] = None
    drift: Optional[ModelCardDriftSummary] = None
    calibration: Optional[ModelCardCalibrationSummary] = None
    feature_importance: Optional[ModelCardFeatureImportanceSummary] = None
    retrain: Optional[ModelCardRetrainInfo] = None
    coverage: Optional[ModelCardCoverage] = None


# ---------------------------------------------------------------------------
# Baseline computation from production data
# ---------------------------------------------------------------------------


class ComputeBaselineResponse(BaseModel):
    """Response for POST /models/{name}/{version}/compute-baseline"""

    model_name: str
    version: str
    predictions_used: int
    dry_run: bool
    baseline: Dict[str, FeatureStats]


# ---------------------------------------------------------------------------
# Cache warmup
# ---------------------------------------------------------------------------


class WarmupResponse(BaseModel):
    """Response for POST /models/{name}/{version}/warmup"""

    model_name: str
    version: str
    already_cached: bool
    load_time_ms: float
    cache_key: str


# ---------------------------------------------------------------------------
# Multi-version comparison
# ---------------------------------------------------------------------------


class ModelVersionSummary(BaseModel):
    """Version summary for GET /models/{name}/compare"""

    version: str
    is_production: bool
    model_task: Optional[str] = (
        None  # "regression" | "classification_binary" | "classification_multiclass"
    )
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    f1_score: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    drift_status: Optional[str] = None
    brier_score: Optional[float] = None
    trained_at: Optional[datetime] = None
    n_rows_trained: Optional[int] = None
    prediction_count: Optional[int] = None
    shadow_prediction_count: Optional[int] = None
    # Regression eval metrics (from training_metrics JSON)
    mae_eval: Optional[float] = None
    rmse_eval: Optional[float] = None
    r2_eval: Optional[float] = None
    # Live metrics
    live_accuracy: Optional[float] = None
    live_auc: Optional[float] = None
    live_f1: Optional[float] = None
    live_mae: Optional[float] = None
    live_rmse: Optional[float] = None
    live_r2: Optional[float] = None


class ModelCompareResponse(BaseModel):
    """Response for GET /models/{name}/compare"""

    model_name: str
    compared_at: datetime
    versions: List[ModelVersionSummary]


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


class ReadinessCheck(BaseModel):
    """Result of an individual readiness check"""

    model_config = {"populate_by_name": True}

    pass_: bool = Field(..., alias="pass")
    detail: Optional[str] = None


class ReadinessChecks(BaseModel):
    """Set of 4 readiness checks"""

    is_production: ReadinessCheck
    file_accessible: ReadinessCheck
    baseline_computed: ReadinessCheck
    no_critical_drift: ReadinessCheck


class ReadinessResponse(BaseModel):
    """Response for GET /models/{name}/readiness"""

    model_name: str
    version: str
    ready: bool
    checked_at: datetime
    checks: ReadinessChecks


class LeaderboardEntry(BaseModel):
    """Entry in the global performance leaderboard — GET /models/leaderboard"""

    rank: int
    name: str
    version: str
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    f1_score: Optional[float] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    drift_status: str = "unknown"
    predictions_count: int = 0
    deployment_mode: Optional[str] = None
    is_production: bool = False
