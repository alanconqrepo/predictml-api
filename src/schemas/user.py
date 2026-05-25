"""
Pydantic schemas for user management
"""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserUpdateInput(BaseModel):
    """Data to update an existing user"""

    is_active: Optional[bool] = None
    role: Optional[str] = Field(None, pattern="^(admin|user|readonly)$")
    rate_limit: Optional[int] = Field(None, ge=1, le=100000)
    regenerate_token: Optional[bool] = False


class UserCreateInput(BaseModel):
    """Data to create a new user"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    role: str = Field("user", pattern="^(admin|user|readonly)$")
    rate_limit: int = Field(1000, ge=1, le=100000)


class UserResponse(BaseModel):
    """User response (includes the token — transmit securely)"""

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
    """Daily quota for the current user"""

    rate_limit_per_day: int
    used_today: int
    remaining_today: int
    reset_at: datetime


class UserUsageByModel(BaseModel):
    """Usage statistics for a given model"""

    model_name: str
    calls: int
    errors: int
    avg_latency_ms: Optional[float]


class UserUsageByDay(BaseModel):
    """Usage statistics for a given day"""

    date: date
    calls: int


class UserUsageByModelDay(BaseModel):
    """Usage statistics by model and by day"""

    model_name: str
    date: date
    calls: int


class UserUsageResponse(BaseModel):
    """Usage statistics for a user over a period"""

    user_id: int
    username: str
    period_days: int
    total_calls: int
    by_model: List[UserUsageByModel]
    by_day: List[UserUsageByDay]
    by_model_day: List[UserUsageByModelDay]
