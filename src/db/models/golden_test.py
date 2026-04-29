"""Modèle GoldenTest — cas de test de régression pré-déploiement"""

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String

from src.core.utils import _utcnow
from src.db.database import Base


class GoldenTest(Base):
    """Cas de test Golden Set pour la validation d'un modèle avant promotion."""

    __tablename__ = "golden_tests"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    input_features = Column(JSON, nullable=False)
    expected_output = Column(String(500), nullable=False)
    description = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    created_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
