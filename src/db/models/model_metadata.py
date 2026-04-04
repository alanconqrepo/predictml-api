"""
Modèle ModelMetadata pour gérer les métadonnées et le versioning des modèles
"""
from datetime import datetime, timezone

def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Text

from src.db.database import Base


class ModelMetadata(Base):
    """Métadonnées et versioning des modèles ML"""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)

    # Identification
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False, index=True)

    # Stockage MinIO
    minio_bucket = Column(String(100), nullable=True)
    minio_object_key = Column(String(255), nullable=True)  # Chemin dans MinIO (None si mlflow_run_id)
    file_size_bytes = Column(Integer, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA256

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

    # MLflow
    mlflow_run_id = Column(String(255), nullable=True, index=True)

    # Training info
    trained_by = Column(String(100), nullable=True)
    training_date = Column(DateTime, nullable=True)
    training_dataset = Column(String(255), nullable=True)
    training_params = Column(JSON, nullable=True)

    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_production = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    deprecated_at = Column(DateTime, nullable=True)

    # Contrainte unique: un seul nom+version
    __table_args__ = (
        {'sqlite_autoincrement': True}
    )

    def __repr__(self):
        return f"<ModelMetadata(name='{self.name}', version='{self.version}', active={self.is_active})>"
