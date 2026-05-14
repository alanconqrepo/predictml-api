"""
Endpoints pour les demandes de création de compte (workflow d'approbation admin)
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
    Soumet une demande de création de compte (endpoint public, sans authentification).

    Un administrateur devra approuver la demande. Le token sera transmis manuellement.

    - **username** : 3 à 50 caractères.
    - **email** : adresse email valide.
    - **message** : message optionnel à l'intention de l'admin (max 500 caractères).
    - **role_requested** : `user` (défaut) ou `readonly`.
    """
    # Vérifier absence de demande pending pour cet email
    existing_pending = await DBService.get_pending_request_by_email(db, payload.email)
    if existing_pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Une demande est déjà en attente pour cet email.",
        )

    # Vérifier qu'aucun compte n'existe déjà avec cet email ou ce username
    result = await db.execute(
        select(User).where((User.email == payload.email) | (User.username == payload.username))
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un compte existe déjà avec cet email ou ce nom d'utilisateur.",
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
    Retourne le nombre de demandes de compte en attente d'approbation.

    Réservé aux administrateurs.
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
    Liste les demandes de création de compte avec filtrage optionnel par statut.

    - **status** : `pending`, `approved`, `rejected` (optionnel)

    Réservé aux administrateurs.
    """
    return await DBService.get_account_requests(db, status=status_filter, skip=skip, limit=limit)


@router.patch("/{request_id}/approve")
async def approve_account_request(
    request_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Approuve une demande de compte : crée l'utilisateur et retourne son token Bearer.

    Le token n'est retourné qu'une seule fois — l'admin doit le transmettre manuellement.

    Réservé aux administrateurs.
    """
    req = await DBService.get_account_request_by_id(db, request_id)
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demande {request_id} introuvable.",
        )
    if req.status != AccountRequestStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"La demande est déjà en statut « {req.status.value} ».",
        )

    # Vérifier que le username/email ne sont pas déjà pris (course condition)
    result = await db.execute(
        select(User).where((User.email == req.email) | (User.username == req.username))
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un compte avec ce nom d'utilisateur ou cet email existe déjà.",
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
    Rejette une demande de création de compte avec une raison optionnelle.

    Réservé aux administrateurs.
    """
    req = await DBService.get_account_request_by_id(db, request_id)
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demande {request_id} introuvable.",
        )
    if req.status != AccountRequestStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"La demande est déjà en statut « {req.status.value} ».",
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
