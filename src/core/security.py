"""
Security and authentication helpers
"""

from datetime import datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_db
from src.db.models import User, UserRole
from src.services.db_service import DBService

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Verify the Bearer authentication token against the database.

    Args:
        credentials: HTTP Bearer credentials
        db: Database session

    Returns:
        The authenticated user

    Raises:
        HTTPException: If the token is invalid, expired, or the user is inactive
    """
    # Retrieve the user by token
    user = await DBService.get_user_by_token(db, credentials.credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or user not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    if user.token_expires_at and user.token_expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login timestamp (async, non-blocking)
    await DBService.update_user_last_login(db, user.id)

    return user


async def check_prediction_rate_limit(
    user: User = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Verify that the user has not exceeded their daily prediction quota.

    Should only be used on POST /predict — other endpoints remain accessible
    even when the quota is reached.

    Raises:
        HTTPException 429: If today's prediction count >= rate_limit_per_day
    """
    today_count = await DBService.get_user_prediction_count_today(db, user.id)
    if today_count >= user.rate_limit_per_day:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({user.rate_limit_per_day} requests/day). "
            f"You have made {today_count} predictions today.",
        )
    return user


async def require_admin(user: User = Depends(verify_token)) -> User:
    """Verify that the authenticated user is an admin."""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access restricted to administrators.",
        )
    return user
