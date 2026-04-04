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


class ModelUpdateInput(BaseModel):
    """Champs modifiables d'un modèle (tous optionnels)"""
    description: Optional[str] = None
    is_production: Optional[bool] = None
    accuracy: Optional[float] = None
    features_count: Optional[int] = None
    classes: Optional[List[Any]] = None

    model_config = {"from_attributes": True}


class ModelCreateResponse(BaseModel):
    """Réponse après création d'un modèle"""
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
    is_active: bool
    is_production: bool
    created_at: datetime

    model_config = {"from_attributes": True}
