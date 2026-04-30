"""
ModelHistory — audit trail pour chaque changement d'état d'un ModelMetadata
"""

import enum

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import relationship

from src.core.utils import _utcnow
from src.db.database import Base


class HistoryActionType(str, enum.Enum):
    CREATED = "created"
    UPDATED = "updated"
    SET_PRODUCTION = "set_production"
    DEPRECATED = "deprecated"
    DELETED = "deleted"
    ROLLBACK = "rollback"
    AUTO_DEMOTE = "auto_demote"


class ModelHistory(Base):
    """Entrée d'historique pour un changement d'état d'un modèle ML."""

    __tablename__ = "model_history"

    id = Column(Integer, primary_key=True, index=True)

    # Quel modèle (stocké en string — survit à la suppression du modèle)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False, index=True)

    # Qui a fait le changement
    changed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    changed_by_username = Column(String(50), nullable=True)  # dénormalisé pour affichage

    # Type d'action
    # native_enum=False → stockage String, compatible SQLite (tests) et PostgreSQL
    action = Column(
        SQLEnum(HistoryActionType, native_enum=False, name="historyactiontype"),
        nullable=False,
        index=True,
    )

    # Snapshot complet des champs mutables au moment de l'action
    snapshot = Column(JSON, nullable=False)

    # Liste des noms de champs modifiés (None pour created/deleted/rollback)
    changed_fields = Column(JSON, nullable=True)

    # Horodatage
    timestamp = Column(DateTime, default=_utcnow, nullable=False, index=True)

    # Relation optionnelle vers l'utilisateur
    changed_by = relationship("User", foreign_keys=[changed_by_user_id])

    def __repr__(self) -> str:
        return (
            f"<ModelHistory(id={self.id}, model={self.model_name}:{self.model_version}, "
            f"action={self.action}, by={self.changed_by_username})>"
        )
