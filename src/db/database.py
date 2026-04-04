"""
Configuration de la base de données
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from src.core.config import settings

# Créer le moteur de base de données
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True if settings.DEBUG else False,
    future=True,
    pool_pre_ping=True,
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
    """Initialise la base de données (crée les tables)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Ferme les connexions à la base de données"""
    await engine.dispose()
