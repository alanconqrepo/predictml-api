"""
Configuration pytest - Fixes pour asyncpg + TestClient sur Windows
"""
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Fix event loop sur Windows (requis par asyncpg)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Forcer les endpoints locaux pour les tests
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9002")
os.environ.setdefault("REDIS_URL", "redis://localhost:6399/0")  # port fantaisiste — jamais contacté

# Mock MinIO globalement — les tests ne nécessitent pas de vrai serveur MinIO
_minio_mock = MagicMock()
_minio_mock.upload_model_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_model/v1.0.0.pkl",
    "size": 512,
    "etag": "mock-etag-abc123",
}
_minio_mock.delete_model.return_value = True
_minio_mock.download_model.side_effect = Exception("MinIO non disponible en tests")

import src.api.models  # noqa: E402 — doit être importé avant le patch
patch("src.api.models.minio_service", _minio_mock).start()

# Remplacer le client Redis du singleton par un FakeRedis en mémoire
# (aucun serveur Redis requis pour les tests)
import fakeredis.aioredis  # noqa: E402
from src.services.model_service import model_service  # noqa: E402

model_service._redis = fakeredis.aioredis.FakeRedis()

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from src.db.database import get_db, Base
from src.db.models import GoldenTest, User, Prediction, ModelMetadata, ObservedResult  # noqa: F401 — enregistre les modèles dans Base
from src.main import app


# Base de données SQLite en mémoire — aucun PostgreSQL requis pour les tests
_test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
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
    """Recrée les tables depuis zéro pour garantir un schéma à jour"""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
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
