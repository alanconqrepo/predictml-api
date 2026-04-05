"""
Modèle User pour la gestion multi-utilisateurs
"""
from datetime import datetime, timezone

def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from src.db.database import Base


class UserRole(str, enum.Enum):
    """Rôles utilisateur"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class User(Base):
    """Modèle utilisateur"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    api_token = Column(String(255), unique=True, index=True, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Quotas
    rate_limit_per_day = Column(Integer, default=1000, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relations
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    created_models = relationship("ModelMetadata", back_populates="creator", foreign_keys="[ModelMetadata.user_id_creator]")
    observed_results = relationship("ObservedResult", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
