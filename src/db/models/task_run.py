"""
Modèle TaskRun — suivi persistant des jobs de tâches longues (retrain, etc.).

Chaque enregistrement correspond à une exécution planifiée ou manuelle.
Le statut passe de queued → running → success | failed | cancelled.
Les logs complets sont archivés dans la colonne ``logs`` après complétion
(source primaire : liste Redis ``retrain_logs:{id}`` pendant l'exécution).
"""

import uuid

from sqlalchemy import JSON, Column, DateTime, String, Text
from sqlalchemy.types import CHAR, TypeDecorator

from src.core.utils import _utcnow
from src.db.database import Base


class GUID(TypeDecorator):
    """Champ UUID cross-database : UUID natif sur PostgreSQL, CHAR(36) sur SQLite."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import UUID as PG_UUID

            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value) if not isinstance(value, uuid.UUID) else value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(value)
        return value


class TaskRun(Base):
    """Enregistrement d'une exécution de tâche longue."""

    __tablename__ = "task_runs"

    # Identifiant UUID stable (utilisé comme clé Redis pour les logs)
    id = Column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # Type de tâche : 'retrain' (manuel) | 'scheduled_retrain'
    task_type = Column(String(50), nullable=False, index=True)

    # Contexte métier
    model_name = Column(String(100), nullable=True, index=True)
    model_version = Column(String(50), nullable=True)  # version source du retrain
    new_version = Column(String(50), nullable=True)  # version produite (rempli après succès)

    # Qui a déclenché : username ou "scheduler"
    triggered_by = Column(String(100), nullable=True)

    # Cycle de vie : queued → running → success | failed | cancelled
    status = Column(String(20), nullable=False, default="queued", index=True)

    enqueued_at = Column(DateTime, nullable=False, default=_utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Résultat structuré (métriques, auto_promoted, etc.) — rempli après complétion
    result = Column(JSON, nullable=True)

    # Message d'erreur si status == 'failed'
    error = Column(Text, nullable=True)

    # Logs complets archivés après complétion (copie de la liste Redis)
    logs = Column(Text, nullable=True)
