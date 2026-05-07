"""
Modèle ModelMetadata pour gérer les métadonnées et le versioning des modèles
"""

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.core.utils import _utcnow
from src.db.database import Base


class DeploymentMode:
    """Constantes pour le mode de déploiement d'une version de modèle."""

    PRODUCTION = "production"
    AB_TEST = "ab_test"
    SHADOW = "shadow"


class ModelStatus:
    """Constantes pour le cycle de vie d'une version de modèle."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelMetadata(Base):
    """Métadonnées et versioning des modèles ML"""

    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)

    # Identification
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False, index=True)

    # Stockage MinIO
    minio_bucket = Column(String(100), nullable=True)
    minio_object_key = Column(
        String(255), nullable=True
    )  # Chemin dans MinIO (None si mlflow_run_id)
    file_size_bytes = Column(Integer, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA256
    pkl_hmac_signature = Column(String(64), nullable=True)  # HMAC-SHA256 of raw pkl bytes

    # Métadonnées du modèle
    description = Column(Text, nullable=True)
    algorithm = Column(String(100), nullable=True)  # RandomForest, LogisticRegression, etc.
    features_count = Column(Integer, nullable=True)
    classes = Column(JSON, nullable=True)  # Liste des classes

    # Performance
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    training_metrics = Column(JSON, nullable=True)  # Métriques supplémentaires

    # Seuil de confiance
    confidence_threshold = Column(
        Float, nullable=True
    )  # Si proba max < seuil → low_confidence=True

    # MLflow
    mlflow_run_id = Column(String(255), nullable=True, index=True)

    # Script de ré-entraînement
    train_script_object_key = Column(String(255), nullable=True)  # Chemin MinIO vers train.py

    # Politique d'auto-promotion post-retrain
    promotion_policy = Column(JSON, nullable=True)

    # Planification du ré-entraînement automatique
    retrain_schedule = Column(JSON, nullable=True)

    # Traçabilité de la lignée des modèles
    parent_version = Column(
        String(50), nullable=True
    )  # Version source lors d'un retrain ou upload dérivé

    # Seuils d'alerte par modèle (surcharge les variables d'env globales)
    alert_thresholds = Column(JSON, nullable=True)

    # Snapshot des données d'entraînement (dates, n_rows, feature_stats, label_distribution)
    training_stats = Column(JSON, nullable=True)

    # Créateur
    user_id_creator = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    creator = relationship("User", foreign_keys=[user_id_creator], back_populates="created_models")

    # Training info
    trained_by = Column(String(100), nullable=True)
    training_date = Column(DateTime, nullable=True)
    training_dataset = Column(String(255), nullable=True)
    training_params = Column(JSON, nullable=True)
    feature_baseline = Column(JSON, nullable=True)  # {feature: {mean, std, min, max}}

    # Tags et webhook
    tags = Column(JSON, nullable=True)  # Liste de tags ex: ["production", "finance"]
    webhook_url = Column(String(500), nullable=True)  # URL callback post-prédiction

    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_production = Column(Boolean, default=False, nullable=False)
    status = Column(String(20), nullable=False, default=ModelStatus.ACTIVE)

    # A/B Testing & Shadow Deployment
    # traffic_weight : fraction de trafic (0.0–1.0) pour le routage A/B ; None = non géré
    traffic_weight = Column(Float, nullable=True)
    # deployment_mode : "production" | "ab_test" | "shadow" | None (comportement legacy)
    deployment_mode = Column(String(20), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    deprecated_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        """Représentation lisible des métadonnées du modèle."""
        return f"<ModelMetadata(name='{self.name}', version='{self.version}', active={self.is_active})>"
