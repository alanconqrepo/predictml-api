"""
Gestion de la sécurité et de l'authentification
"""
from datetime import datetime
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_db
from src.db.models import User, UserRole
from src.services.db_service import DBService

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Vérifie le token d'authentification Bearer dans la base de données

    Args:
        credentials: Credentials HTTP Bearer
        db: Session de base de données

    Returns:
        L'utilisateur authentifié

    Raises:
        HTTPException: Si le token est invalide ou l'utilisateur inactif
    """
    # Récupérer l'utilisateur par token
    user = await DBService.get_user_by_token(db, credentials.credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou utilisateur introuvable",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte utilisateur désactivé",
        )

    # Vérifier le rate limit
    today_count = await DBService.get_user_prediction_count_today(db, user.id)
    if today_count >= user.rate_limit_per_day:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit dépassé ({user.rate_limit_per_day} requêtes/jour). "
                   f"Vous avez effectué {today_count} prédictions aujourd'hui.",
        )

    # Mettre à jour la dernière connexion (async, non bloquant)
    await DBService.update_user_last_login(db, user.id)

    return user


async def require_admin(user: User = Depends(verify_token)) -> User:
    """Vérifie que l'utilisateur authentifié est admin."""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé aux administrateurs.",
        )
    return user
