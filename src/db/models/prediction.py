"""
Modèle Prediction pour logger toutes les prédictions
"""

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from src.core.utils import _utcnow
from src.db.database import Base


class Prediction(Base):
    """Modèle pour stocker l'historique des prédictions"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    # Utilisateur
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Modèle utilisé
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)

    # Observation
    id_obs = Column(String(255), nullable=True, index=True)

    # Input/Output
    input_features = Column(JSON, nullable=False)  # Liste ou dict des features
    prediction_result = Column(JSON, nullable=False)  # Résultat de la prédiction
    probabilities = Column(JSON, nullable=True)  # Probabilités par classe (si disponible)

    # Confidence (max des probabilités pour les classifieurs, None pour la régression)
    max_confidence = Column(Float, nullable=True, index=True)

    # Performance
    response_time_ms = Column(Float, nullable=False)  # Temps de réponse en ms

    # Métadonnées
    timestamp = Column(DateTime, default=_utcnow, nullable=False, index=True)
    client_ip = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(Text, nullable=True)

    # Status
    status = Column(String(20), default="success", nullable=False)  # success, error
    error_message = Column(Text, nullable=True)

    # Shadow deployment : True si prédiction effectuée par un modèle shadow (non retournée au client)
    is_shadow = Column(Boolean, default=False, nullable=False)

    # Relations
    user = relationship("User", back_populates="predictions")

    def __repr__(self) -> str:
        """Représentation lisible de la prédiction."""
        return f"<Prediction(id={self.id}, model='{self.model_name}', user_id={self.user_id})>"
