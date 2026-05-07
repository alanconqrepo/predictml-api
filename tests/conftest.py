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
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-do-not-use-in-production")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9002")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-minio-access-key")
os.environ.setdefault("MINIO_SECRET_KEY", "test-minio-secret-key-safe-value")
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
import src.tasks.retrain_scheduler  # noqa: E402 — doit être importé avant le patch
patch("src.api.models.minio_service", _minio_mock).start()

# Mock MLflow service globalement — les tests ne nécessitent pas de serveur MLflow
_mlflow_mock = MagicMock()
_mlflow_mock.log_retrain_run.return_value = "mock-mlflow-run-id-abc123"
_mlflow_mock.update_run_tags.return_value = True
_mlflow_mock.delete_run.return_value = True
_mlflow_mock.log_production_snapshot.return_value = "mock-monitoring-run-id"

# Patch dans le namespace de api.models (import module-level déjà lié)
patch("src.api.models.mlflow_service", _mlflow_mock).start()
# Patch dans le module source : couvre les imports lazy du scheduler (from src.services.mlflow_service import mlflow_service)
patch("src.services.mlflow_service.mlflow_service", _mlflow_mock).start()

# Remplacer le client Redis du singleton par un FakeRedis en mémoire
# (aucun serveur Redis requis pour les tests)
import fakeredis.aioredis  # noqa: E402
from src.services.model_service import model_service, _sign_for_cache  # noqa: E402


class _SigningFakeRedis:
    """
    Wrapper autour de FakeRedis qui signe automatiquement les entrées du cache modèle.

    Toute écriture sur une clé ``model:*`` (via set ou setex) est transparentement
    enveloppée avec un HMAC-SHA256 avant le stockage, de la même façon que le fait
    model_service.load_model() en production. Cela permet aux fonctions _inject_cache()
    des tests de continuer à écrire des données non-signées (elles sont signées ici).
    """

    def __init__(self, redis):
        self._r = redis

    def _maybe_sign(self, key: object, value: bytes) -> bytes:
        if isinstance(key, (str, bytes)):
            k = key.decode() if isinstance(key, bytes) else key
            if k.startswith("model:") and isinstance(value, bytes):
                return _sign_for_cache(value)
        return value

    async def set(self, key, value, *args, **kwargs):
        value = self._maybe_sign(key, value)
        return await self._r.set(key, value, *args, **kwargs)

    async def setex(self, key, ttl, value):
        value = self._maybe_sign(key, value)
        return await self._r.setex(key, ttl, value)

    async def get(self, key):
        return await self._r.get(key)

    async def keys(self, pattern):
        return await self._r.keys(pattern)

    async def delete(self, *keys):
        return await self._r.delete(*keys)

    def __getattr__(self, name):
        return getattr(self._r, name)


model_service._redis = _SigningFakeRedis(fakeredis.aioredis.FakeRedis())

import tempfile

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool

from src.db.database import get_db, Base
from src.db.models import GoldenTest, User, Prediction, ModelMetadata, ObservedResult  # noqa: F401 — enregistre les modèles dans Base
from src.main import app


# SQLite fichier temporaire — évite l'invalidation de connexion aiosqlite/StaticPool
# en Python 3.13 où asyncio.run() ferme l'executor et tue le thread aiosqlite.
# Avec NullPool + fichier, chaque session ouvre une connexion fraîche au même fichier.
_test_db_file = os.path.join(tempfile.gettempdir(), f"predictml_test_{os.getpid()}.db")

_test_engine = create_async_engine(
    f"sqlite+aiosqlite:///{_test_db_file}",
    connect_args={"check_same_thread": False},
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
