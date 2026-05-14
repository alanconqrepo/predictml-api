"""
Schémas Pydantic pour les demandes de création de compte
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class AccountRequestCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    message: Optional[str] = Field(None, max_length=500)
    role_requested: str = Field("user", pattern="^(user|readonly)$")


class AccountRequestResponse(BaseModel):
    id: int
    username: str
    email: str
    message: Optional[str]
    role_requested: str
    status: str
    rejection_reason: Optional[str]
    requested_at: datetime
    reviewed_at: Optional[datetime]
    reviewer_id: Optional[int]

    model_config = {"from_attributes": True}


class AccountRequestRejectInput(BaseModel):
    reason: Optional[str] = Field(None, max_length=300)
