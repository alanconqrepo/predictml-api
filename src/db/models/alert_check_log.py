"""
Modèle AlertCheckLog — historique des checks d'alerting supervisés.

Chaque enregistrement correspond à un check d'une métrique précise pour un modèle donné.
Créé par supervision_reporter.run_alert_check() toutes les 6 h.
"""

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String

from src.core.utils import _utcnow
from src.db.database import Base


class AlertCheckLog(Base):
    """Enregistrement d'un check d'alerte de supervision."""

    __tablename__ = "alert_check_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Quand le check a été effectué
    checked_at = Column(DateTime, nullable=False, default=_utcnow, index=True)

    # Type de check : "error_spike" | "auc" | "performance_drift" | "feature_drift" | "output_drift"
    check_type = Column(String(50), nullable=False, index=True)

    # Modèle concerné
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)

    # Résultat : "ok" | "alert_triggered" | "skipped_no_predictions" | "error"
    result = Column(String(30), nullable=False)

    # Envois effectués
    alert_sent = Column(Boolean, nullable=False, default=False)
    webhook_sent = Column(Boolean, nullable=False, default=False)

    # Nombre de prédictions production (non-shadow) depuis le dernier check
    new_predictions_count = Column(Integer, nullable=True)

    # Détails métriques : {"error_rate": 0.08, "threshold": 0.05} etc.
    details = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AlertCheckLog(id={self.id}, model='{self.model_name}', "
            f"type='{self.check_type}', result='{self.result}')>"
        )
