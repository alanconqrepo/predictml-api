"""
User management endpoints
"""

import secrets
from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.audit import audit_log
from src.core.rate_limit import limiter
from src.core.security import require_admin, verify_token
from src.db.database import get_db
from src.db.models import User, UserRole
from src.schemas.user import (
    QuotaResponse,
    UserCreateInput,
    UserResponse,
    UserUpdateInput,
    UserUsageResponse,
)
from src.services.db_service import DBService

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/me/regenerate-token", response_model=UserResponse)
@limiter.limit("3/minute")
async def regenerate_my_token(
    request: Request,
    current_user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Regenerates the Bearer token of the authenticated user (self-service).

    The old token is immediately invalidated. The new token is returned only once.

    Accessible by all roles.
    """
    user = await DBService.update_user(db, current_user.id, regenerate_token=True)
    audit_log("user.token_self_regen", actor_id=current_user.id, resource=f"user:{current_user.id}")
    return user


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(verify_token)):
    """
    Returns the profile of the authenticated user.

    Accessible by all roles.
    """
    return current_user


@router.get("/me/quota", response_model=QuotaResponse)
async def get_my_quota(
    current_user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the daily quota of the authenticated user.

    Accessible by all roles.
    """
    used = await DBService.get_user_prediction_count_today(db, current_user.id)
    remaining = max(0, current_user.rate_limit_per_day - used)
    reset_at = datetime.combine(date.today() + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return QuotaResponse(
        rate_limit_per_day=current_user.rate_limit_per_day,
        used_today=used,
        remaining_today=remaining,
        reset_at=reset_at,
    )


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_user(
    request: Request,
    payload: UserCreateInput,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Creates a new user and generates their Bearer token.

    - **username** and **email** must be unique.
    - **role**: `user` (default), `admin`, or `readonly`.
    - **rate_limit**: max predictions per day (default: 1000).
    - The generated **api_token** is returned only once — save it.

    Reserved for administrators.
    """
    # Check username and email uniqueness
    existing = await db.execute(
        select(User).where((User.username == payload.username) | (User.email == payload.email))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this name or email already exists.",
        )

    api_token = secrets.token_urlsafe(32)
    user = await DBService.create_user(
        db,
        username=payload.username,
        email=payload.email,
        api_token=api_token,
        role=payload.role,
        rate_limit=payload.rate_limit,
    )
    audit_log(
        "user.create",
        actor_id=_.id,
        resource=f"user:{user.id}",
        details={"username": user.username},
    )
    return user


@router.get("", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Lists users with pagination.

    - **skip**: number of users to skip (default: 0)
    - **limit**: maximum number of users returned (default: 100, max: 500)

    Reserved for administrators.
    """
    return await DBService.get_all_users(db, skip=skip, limit=limit)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieves a user by their ID.

    Accessible by the administrator or by the user themselves.
    """
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you can only view your own profile.",
        )

    user = await DBService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )
    return user


@router.get("/{user_id}/usage", response_model=UserUsageResponse)
async def get_user_usage(
    user_id: int,
    days: Optional[int] = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns usage statistics for a user over a period.

    - **by_model**: calls, errors and average latency per model.
    - **by_day**: number of calls per day over the period.
    - **by_model_day**: calls per model per day.

    Pass `start_date` and `end_date` (YYYY-MM-DD) for a custom range,
    or `days` for the last N days (default: 30).

    Accessible by the administrator or by the user themselves.
    """
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you can only view your own statistics.",
        )

    user = await DBService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    if days is None or days < 1:
        days = 30

    usage = await DBService.get_user_usage(
        db, user_id, days, start_date=start_date, end_date=end_date
    )
    return UserUsageResponse(
        user_id=user.id,
        username=user.username,
        period_days=usage.get("actual_days", days),
        total_calls=usage["total_calls"],
        by_model=usage["by_model"],
        by_day=usage["by_day"],
        by_model_day=usage["by_model_day"],
    )


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    payload: UserUpdateInput,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Updates a user: active status, role, rate limit, or token renewal.

    - **is_active**: enable or disable the account.
    - **role**: change the role (`admin`, `user`, `readonly`).
    - **rate_limit**: modify the daily quota.
    - **regenerate_token**: if `true`, generates a new Bearer token.

    Reserved for administrators.
    """
    if payload.is_active is False and current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own account.",
        )
    user = await DBService.update_user(
        db,
        user_id,
        is_active=payload.is_active,
        role=payload.role,
        rate_limit_per_day=payload.rate_limit,
        regenerate_token=payload.regenerate_token,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )
    if payload.regenerate_token:
        audit_log("user.token_regen", actor_id=current_user.id, resource=f"user:{user_id}")
    else:
        audit_log("user.update", actor_id=current_user.id, resource=f"user:{user_id}")
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Deletes a user and all their predictions (cascade).

    An administrator cannot delete themselves.

    Reserved for administrators.
    """
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account.",
        )

    deleted = await DBService.delete_user(db, user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )
    audit_log("user.delete", actor_id=current_user.id, resource=f"user:{user_id}")
