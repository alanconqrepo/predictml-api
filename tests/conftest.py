"""
Configuration pytest - Fixes pour asyncpg + TestClient sur Windows
"""
import asyncio
import sys
import os

# Fix event loop sur Windows (requis par asyncpg)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Forcer les endpoints locaux pour les tests
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9002")

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool

from src.core.config import settings
from src.db.database import get_db, Base
from src.db.models import User, Prediction, ModelMetadata  # noqa: F401 — enregistre les modèles dans Base
from src.main import app


# Moteur sans pool de connexions (NullPool) pour éviter les conflits d'event loop
# entre les requêtes successives du TestClient
_test_engine = create_async_engine(
    settings.DATABASE_URL,
    poolclass=NullPool,
    echo=False,
)

_TestSessionLocal = async_sessionmaker(
    _test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def _setup():
    """Crée les tables et initialise la DB de test"""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


asyncio.run(_setup())


async def _override_get_db():
    async with _TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Remplacer la dépendance DB par la version NullPool
app.dependency_overrides[get_db] = _override_get_db
