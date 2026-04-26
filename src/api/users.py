"""
Endpoints pour la gestion des utilisateurs
"""

import secrets
from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import require_admin, verify_token
from src.db.database import get_db
from src.db.models import User, UserRole
from src.schemas.user import QuotaResponse, UserCreateInput, UserResponse, UserUpdateInput, UserUsageResponse
from src.services.db_service import DBService

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(verify_token)):
    """
    Retourne le profil de l'utilisateur authentifié.

    Accessible par tous les rôles.
    """
    return current_user


@router.get("/me/quota", response_model=QuotaResponse)
async def get_my_quota(
    current_user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne le quota journalier de l'utilisateur authentifié.

    Accessible par tous les rôles.
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
async def create_user(
    payload: UserCreateInput,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Crée un nouvel utilisateur et génère son token Bearer.

    - **username** et **email** doivent être uniques.
    - **role** : `user` (défaut), `admin`, ou `readonly`.
    - **rate_limit** : nombre max de prédictions par jour (défaut: 1000).
    - Le **api_token** généré est retourné une seule fois — conservez-le.

    Réservé aux administrateurs.
    """
    # Vérifier l'unicité username et email
    existing = await db.execute(
        select(User).where((User.username == payload.username) | (User.email == payload.email))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un utilisateur avec ce nom ou cet email existe déjà.",
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
    return user


@router.get("", response_model=List[UserResponse])
async def list_users(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Liste tous les utilisateurs.

    Réservé aux administrateurs.
    """
    return await DBService.get_all_users(db)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère un utilisateur par son ID.

    Accessible par l'administrateur ou par l'utilisateur lui-même.
    """
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé : vous ne pouvez consulter que votre propre profil.",
        )

    user = await DBService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Utilisateur {user_id} introuvable.",
        )
    return user


@router.get("/{user_id}/usage", response_model=UserUsageResponse)
async def get_user_usage(
    user_id: int,
    days: Optional[int] = 30,
    current_user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne les statistiques d'usage d'un utilisateur sur les N derniers jours.

    - **by_model** : appels, erreurs et latence moyenne par modèle.
    - **by_day** : nombre d'appels par jour sur la période.

    Accessible par l'administrateur ou par l'utilisateur lui-même.
    """
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé : vous ne pouvez consulter que vos propres statistiques.",
        )

    user = await DBService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Utilisateur {user_id} introuvable.",
        )

    if days is None or days < 1:
        days = 30

    usage = await DBService.get_user_usage(db, user_id, days)
    return UserUsageResponse(
        user_id=user.id,
        username=user.username,
        period_days=days,
        total_calls=usage["total_calls"],
        by_model=usage["by_model"],
        by_day=usage["by_day"],
    )


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    payload: UserUpdateInput,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Met à jour un utilisateur : statut actif, rôle, rate limit, ou renouvellement du token.

    - **is_active** : activer ou désactiver le compte.
    - **role** : changer le rôle (`admin`, `user`, `readonly`).
    - **rate_limit** : modifier le quota journalier.
    - **regenerate_token** : si `true`, génère un nouveau token Bearer.

    Réservé aux administrateurs.
    """
    if payload.is_active is False and current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez pas désactiver votre propre compte.",
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
            detail=f"Utilisateur {user_id} introuvable.",
        )
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Supprime un utilisateur et toutes ses prédictions (cascade).

    Un administrateur ne peut pas se supprimer lui-même.

    Réservé aux administrateurs.
    """
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vous ne pouvez pas supprimer votre propre compte.",
        )

    deleted = await DBService.delete_user(db, user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Utilisateur {user_id} introuvable.",
        )
