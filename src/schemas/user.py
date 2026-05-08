"""
Schémas Pydantic pour la gestion des utilisateurs
"""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserUpdateInput(BaseModel):
    """Données pour mettre à jour un utilisateur existant"""

    is_active: Optional[bool] = None
    role: Optional[str] = Field(None, pattern="^(admin|user|readonly)$")
    rate_limit: Optional[int] = Field(None, ge=1, le=100000)
    regenerate_token: Optional[bool] = False


class UserCreateInput(BaseModel):
    """Données pour créer un nouvel utilisateur"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    role: str = Field("user", pattern="^(admin|user|readonly)$")
    rate_limit: int = Field(1000, ge=1, le=100000)


class UserResponse(BaseModel):
    """Réponse utilisateur (inclut le token — à transmettre de façon sécurisée)"""

    id: int
    username: str
    email: str
    role: str
    is_active: bool
    rate_limit_per_day: int
    api_token: str
    created_at: datetime
    last_login: Optional[datetime]
    token_expires_at: Optional[datetime]

    model_config = {"from_attributes": True}


class QuotaResponse(BaseModel):
    """Quota journalier de l'utilisateur courant"""

    rate_limit_per_day: int
    used_today: int
    remaining_today: int
    reset_at: datetime


class UserUsageByModel(BaseModel):
    """Statistiques d'usage pour un modèle donné"""

    model_name: str
    calls: int
    errors: int
    avg_latency_ms: Optional[float]


class UserUsageByDay(BaseModel):
    """Statistiques d'usage pour un jour donné"""

    date: date
    calls: int


class UserUsageResponse(BaseModel):
    """Statistiques d'usage d'un utilisateur sur une période"""

    user_id: int
    username: str
    period_days: int
    total_calls: int
    by_model: List[UserUsageByModel]
    by_day: List[UserUsageByDay]
