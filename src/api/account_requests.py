"""
Endpoints for account creation requests (admin approval workflow)
"""

import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.audit import audit_log
from src.core.rate_limit import limiter
from src.core.security import require_admin
from src.db.database import get_db
from src.db.models import AccountRequestStatus, User
from src.schemas.account_request import (
    AccountRequestCreate,
    AccountRequestRejectInput,
    AccountRequestResponse,
)
from src.schemas.user import UserResponse
from src.services.db_service import DBService

router = APIRouter(prefix="/account-requests", tags=["account-requests"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=AccountRequestResponse)
@limiter.limit("5/hour")
async def submit_account_request(
    request: Request,
    payload: AccountRequestCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Submits an account creation request (public endpoint, no authentication required).

    An administrator must approve the request. The token will be transmitted manually.

    - **username**: 3 to 50 characters.
    - **email**: valid email address.
    - **message**: optional message to the admin (max 500 characters).
    - **role_requested**: `user` (default) or `readonly`.
    """
    # Check that no pending request exists for this email
    existing_pending = await DBService.get_pending_request_by_email(db, payload.email)
    if existing_pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A request is already pending for this email.",
        )

    # Check that no account already exists with this email or username
    result = await db.execute(
        select(User).where((User.email == payload.email) | (User.username == payload.username))
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account already exists with this email or username.",
        )

    req = await DBService.create_account_request(
        db,
        username=payload.username,
        email=payload.email,
        message=payload.message,
        role_requested=payload.role_requested,
    )
    return req


@router.get("/pending-count")
async def get_pending_count(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Returns the number of account requests pending approval.

    Reserved for administrators.
    """
    count = await DBService.count_pending_account_requests(db)
    return {"pending_count": count}


@router.get("", response_model=list[AccountRequestResponse])
async def list_account_requests(
    status_filter: str = Query(None, alias="status"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Lists account creation requests with optional filtering by status.

    - **status**: `pending`, `approved`, `rejected` (optional)

    Reserved for administrators.
    """
    return await DBService.get_account_requests(db, status=status_filter, skip=skip, limit=limit)


@router.patch("/{request_id}/approve")
async def approve_account_request(
    request_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Approves an account request: creates the user and returns their Bearer token.

    The token is returned only once — the admin must transmit it manually.

    Reserved for administrators.
    """
    req = await DBService.get_account_request_by_id(db, request_id)
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found.",
        )
    if req.status != AccountRequestStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The request is already in status '{req.status.value}'.",
        )

    # Check that username/email are not already taken (race condition)
    result = await db.execute(
        select(User).where((User.email == req.email) | (User.username == req.username))
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this username or email already exists.",
        )

    api_token = secrets.token_urlsafe(32)
    user = await DBService.create_user(
        db,
        username=req.username,
        email=req.email,
        api_token=api_token,
        role=req.role_requested,
        rate_limit=1000,
    )

    await DBService.approve_account_request(db, request_id, reviewer_id=admin.id)

    audit_log(
        "account_request.approve",
        actor_id=admin.id,
        resource=f"account_request:{request_id}",
        details={"username": user.username, "created_user_id": user.id},
    )

    return {
        "request_id": request_id,
        "created_user": UserResponse.model_validate(user),
    }


@router.patch("/{request_id}/reject", response_model=AccountRequestResponse)
async def reject_account_request(
    request_id: int,
    payload: AccountRequestRejectInput,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Rejects an account creation request with an optional reason.

    Reserved for administrators.
    """
    req = await DBService.get_account_request_by_id(db, request_id)
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found.",
        )
    if req.status != AccountRequestStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The request is already in status '{req.status.value}'.",
        )

    updated = await DBService.reject_account_request(
        db, request_id, reviewer_id=admin.id, reason=payload.reason
    )

    audit_log(
        "account_request.reject",
        actor_id=admin.id,
        resource=f"account_request:{request_id}",
        details={"reason": payload.reason},
    )

    return updated
