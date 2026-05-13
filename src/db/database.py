"""
Configuration de la base de données
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src.core.config import settings

_ENGINE_KWARGS = dict(
    echo=True if settings.DEBUG else False,
    future=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=300,
    pool_pre_ping=True,
    pool_timeout=30,
    # PgBouncer transaction mode does not support server-side prepared statements
    connect_args={"prepared_statement_cache_size": 0},
)

# Write engine — primary (all mutations)
engine = create_async_engine(settings.DATABASE_URL, **_ENGINE_KWARGS)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Read engine — replica when DATABASE_READ_REPLICA_URL is set, otherwise reuses write engine
if settings.DATABASE_READ_REPLICA_URL:
    read_engine = create_async_engine(settings.DATABASE_READ_REPLICA_URL, **_ENGINE_KWARGS)
    ReadAsyncSessionLocal = async_sessionmaker(
        read_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    _separate_read_engine = True
else:
    read_engine = engine
    ReadAsyncSessionLocal = AsyncSessionLocal
    _separate_read_engine = False

# Base pour les modèles
Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency for write/transactional endpoints (primary)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_read_db() -> AsyncSession:
    """Dependency for analytics/read-only endpoints. Routes to read replica when configured."""
    async with ReadAsyncSessionLocal() as session:
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
    if _separate_read_engine:
        async with read_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))


async def close_db():
    """Ferme les connexions à la base de données"""
    await engine.dispose()
    if _separate_read_engine:
        await read_engine.dispose()
