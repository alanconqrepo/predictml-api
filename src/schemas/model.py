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
