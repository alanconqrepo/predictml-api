"""
Configuration de la base de données
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src.core.config import settings

# Créer le moteur de base de données
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True if settings.DEBUG else False,
    future=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=300,
    pool_pre_ping=True,
    pool_timeout=30,
)

# Session maker
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base pour les modèles
Base = declarative_base()


async def get_db() -> AsyncSession:
    """
    Dependency pour obtenir une session de base de données

    Yields:
        AsyncSession: Session de base de données
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Vérifie la connexion à la base de données.

    La création et l'évolution du schéma sont gérées par Alembic (alembic upgrade head).
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def close_db():
    """Ferme les connexions à la base de données"""
    await engine.dispose()
