"""
Schémas Pydantic pour la gestion des utilisateurs
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


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

    model_config = {"from_attributes": True}
