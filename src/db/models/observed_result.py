"""
Modèle ObservedResult — données réelles observées pour comparer aux prédictions
"""

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from src.core.utils import _utcnow
from src.db.database import Base


class ObservedResult(Base):
    """Résultats réellement observés, liés à une observation et un modèle"""

    __tablename__ = "observed_results"

    id = Column(Integer, primary_key=True, index=True)

    # Clé métier — unicité sur (id_obs, model_name)
    id_obs = Column(String(255), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)

    # Résultat observé (même type JSON que prediction_result)
    observed_result = Column(JSON, nullable=False)

    # Horodatage de l'observation
    date_time = Column(DateTime, nullable=False, default=_utcnow, index=True)

    # Utilisateur qui a soumis la donnée
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Contrainte d'unicité — upsert sur ce couple
    __table_args__ = (
        UniqueConstraint("id_obs", "model_name", name="uq_observed_result_obs_model"),
    )

    # Relation
    user = relationship("User", back_populates="observed_results")

    def __repr__(self) -> str:
        """Représentation lisible du résultat observé."""
        return f"<ObservedResult(id_obs='{self.id_obs}', model='{self.model_name}')>"
