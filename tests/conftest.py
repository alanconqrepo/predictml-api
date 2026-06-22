"""
pytest configuration - Fixes for asyncpg + TestClient on Windows
"""
import asyncio
import io
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import joblib

# prometheus_fastapi_instrumentator ≥ 8.0 iterates over app.routes and accesses
# route.path unconditionally. Starlette 1.3+ adds _IncludedRouter objects (from
# include_router()) which do NOT have a path attribute, causing AttributeError on
# every test request. Patch _get_route_name before importing the app so the
# middleware never raises.
from prometheus_fastapi_instrumentator import routing as _pfi_routing
from starlette.routing import Match, Mount as _Mount


def _patched_get_route_name(scope, routes, route_name=None):
    for route in routes:
        if not hasattr(route, "path"):
            continue
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            route_name = route.path
            child_scope = {**scope, **child_scope}
            if isinstance(route, _Mount) and route.routes:
                child_route_name = _patched_get_route_name(
                    child_scope, route.routes, route_name
                )
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        elif match == Match.PARTIAL and route_name is None:
            route_name = route.path
    return None


_pfi_routing._get_route_name = _patched_get_route_name


def make_model_bytes(model) -> bytes:
    """Serialize a sklearn model to joblib bytes."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()

# Fix event loop on Windows (required by asyncpg)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Force local endpoints for tests
os.environ.setdefault("RATELIMIT_ENABLED", "0")  # disable IP-based rate limit in tests
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-do-not-use-in-production")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9002")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-minio-access-key")
os.environ.setdefault("MINIO_SECRET_KEY", "test-minio-secret-key-safe-value")
os.environ.setdefault("REDIS_URL", "redis://localhost:6399/0")  # bogus port — never contacted
os.environ.setdefault("METRICS_TOKEN", "test-metrics-token-for-pytest")

# Patch minio.Minio class BEFORE any src imports so MinIOService.__init__ never
# creates a real HTTP client, regardless of import order. This prevents connection
# attempts to localhost:9002 on Windows (where urllib3 probes eagerly).
_mock_minio_client = MagicMock()
_mock_minio_client.bucket_exists.return_value = True
_mock_minio_client.make_bucket.return_value = None
patch("minio.Minio", return_value=_mock_minio_client).start()

# Mock MinIO globally — tests do not require a real MinIO server
_minio_mock = MagicMock()
_upload_return = {
    "bucket": "models",
    "object_name": "mock_model/v1.0.0.joblib",
    "size": 512,
    "etag": "mock-etag-abc123",
}
_minio_mock.upload_model_bytes.return_value = _upload_return
_minio_mock.upload_file_bytes.return_value = _upload_return
_minio_mock.delete_model.return_value = True
_minio_mock.download_model.side_effect = Exception("MinIO non disponible en tests")
# Async versions of MinIO methods (used from async contexts)
_minio_mock.async_upload_model_bytes = AsyncMock(return_value=_upload_return)
_minio_mock.async_upload_file_bytes = AsyncMock(return_value=_upload_return)
_minio_mock.async_download_file_bytes = AsyncMock(return_value=b"fake-model-bytes")

import src.api.models  # noqa: E402 — must be imported before patching
patch("src.api.models.minio_service", _minio_mock).start()
patch("src.services.minio_service.minio_service", _minio_mock).start()
# model_service is imported transitively by src.api.models before patching,
# so its local binding to minio_service points to the original client.
# Patch it explicitly to prevent real MinIO connections in tests.
patch("src.services.model_service.minio_service", _minio_mock).start()

# Mock MLflow service globally — tests do not require an MLflow server
_mlflow_mock = MagicMock()
_mlflow_mock.log_retrain_run.return_value = "mock-mlflow-run-id-abc123"
_mlflow_mock.update_run_tags.return_value = True
_mlflow_mock.delete_run.return_value = True
_mlflow_mock.log_production_snapshot.return_value = "mock-monitoring-run-id"

# Patch in the api.models namespace (module-level import already bound)
patch("src.api.models.mlflow_service", _mlflow_mock).start()
# Patch in the source module: covers lazy imports from the scheduler and retrain_service
patch("src.services.mlflow_service.mlflow_service", _mlflow_mock).start()

# Mock ARQ pool — tests do not require an ARQ/Redis server for jobs
_arq_mock_pool = MagicMock()
_arq_mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="test-arq-job-id-abc123"))

async def _fake_get_arq_pool():
    return _arq_mock_pool

import src.core.arq_pool as _arq_pool_module  # noqa: E402
patch.object(_arq_pool_module, "get_arq_pool", side_effect=_fake_get_arq_pool).start()
# Patch in the api.models namespace where get_arq_pool is used via module
patch("src.api.models.arq_pool_module", _arq_pool_module).start()

# Replace the singleton Redis client with an in-memory FakeRedis
# (no Redis server required for tests)
import fakeredis  # noqa: E402
import fakeredis.aioredis  # noqa: E402
from src.services.model_service import model_service, _sign_for_cache  # noqa: E402


class _SigningFakeRedis:
    """
    FakeRedis wrapper that automatically signs model cache entries.

    Any write to a ``model:*`` key (via set or setex) is transparently
    wrapped with an HMAC-SHA256 before storage, exactly as model_service.load_model()
    does in production. This allows test _inject_cache() functions to keep writing
    unsigned data (they are signed here).

    Uses a shared FakeServer across all FakeRedis instances, enabling a separate
    client per event loop (avoids the "bound to a different event loop" error when
    multiple asyncio.run() calls are made from different test modules).
    """

    def __init__(self, server: "fakeredis.FakeServer"):
        self._server = server
        self._clients: dict = {}  # id(event_loop) -> FakeRedis instance

    def _get_client(self) -> "fakeredis.aioredis.FakeRedis":
        """Return (or create) a FakeRedis client bound to the current event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            loop_id = -1
        if loop_id not in self._clients:
            self._clients[loop_id] = fakeredis.aioredis.FakeRedis(server=self._server)
        return self._clients[loop_id]

    def _maybe_sign(self, key: object, value: bytes) -> bytes:
        if isinstance(key, (str, bytes)):
            k = key.decode() if isinstance(key, bytes) else key
            if k.startswith("model:") and isinstance(value, bytes):
                return _sign_for_cache(value)
        return value

    async def set(self, key, value, *args, **kwargs):
        value = self._maybe_sign(key, value)
        return await self._get_client().set(key, value, *args, **kwargs)

    async def setex(self, key, ttl, value):
        value = self._maybe_sign(key, value)
        return await self._get_client().setex(key, ttl, value)

    async def get(self, key):
        return await self._get_client().get(key)

    async def keys(self, pattern):
        return await self._get_client().keys(pattern)

    async def delete(self, *keys):
        return await self._get_client().delete(*keys)

    def __getattr__(self, name):
        # Delegate to the current-loop client via an async wrapper.
        # Do NOT use a client from another loop (avoids "bound to a different event loop").
        async def method(*args, **kwargs):
            return await getattr(self._get_client(), name)(*args, **kwargs)

        return method


_fake_server = fakeredis.FakeServer()
model_service._redis = _SigningFakeRedis(_fake_server)

import tempfile

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool

from src.db.database import get_db, get_read_db, Base
from src.db.models import AccountRequest, AlertCheckLog, GoldenTest, User, Prediction, ModelMetadata, ObservedResult, TaskRun  # noqa: F401 — registers models in Base
from src.main import app


# Temporary SQLite file — avoids connection invalidation with aiosqlite/StaticPool
# in Python 3.13 where asyncio.run() closes the executor and kills the aiosqlite thread.
# With NullPool + file, each session opens a fresh connection to the same file.
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
    """Recreate all tables from scratch to ensure an up-to-date schema."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


asyncio.run(_setup())


def pytest_sessionfinish(session, exitstatus):
    """Dispose the SQLAlchemy engine after the test session.

    aiosqlite runs each connection in a background thread. With NullPool, those
    threads stay alive until the engine is explicitly disposed. On Windows,
    Python 3.13 waits for all threads to finish before exiting the process, so
    pytest hangs indefinitely unless we dispose the engine here.
    """
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_test_engine.dispose())
        loop.close()
    except Exception:
        pass
    finally:
        try:
            import os
            if os.path.exists(_test_db_file):
                os.remove(_test_db_file)
        except Exception:
            pass


async def _override_get_db():
    async with _TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Override DB dependencies with the NullPool version (read + write → same SQLite)
app.dependency_overrides[get_db] = _override_get_db
app.dependency_overrides[get_read_db] = _override_get_db

# Redirect AsyncSessionLocal to the test DB (SQLite).
# Needed for direct sessions (not injected via Depends) in retrain code.
patch("src.db.database.AsyncSessionLocal", new=_TestSessionLocal).start()
patch("src.api.models.AsyncSessionLocal", new=_TestSessionLocal).start()
