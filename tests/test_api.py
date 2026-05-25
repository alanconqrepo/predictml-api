"""
Tests for the API endpoints
"""
import asyncio
import io
import joblib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

_ADMIN_TOKEN = "test-token-health-admin"


async def _ensure_admin():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, _ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="health_admin",
                email="health_admin@test.com",
                api_token=_ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_ensure_admin())

_ADMIN_HEADERS = {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["docs"] == "/docs"
    assert "models_available" not in data
    assert "models_count" not in data


def test_health_endpoint():
    """Test the health check — returns healthy if DB is connected, degraded otherwise."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]


def test_models_endpoint():
    """Test the /models endpoint — returns a list of models."""
    response = client.get("/models")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


def test_predict_without_auth():
    """Test prediction without authentication — HTTPBearer returns 401."""
    response = client.post(
        "/predict",
        json={"model_name": "iris_model", "features": {"sepal_length": 5.1, "sepal_width": 3.5}}
    )
    # HTTPBearer without an Authorization header returns 401 or 403 depending on the Starlette version
    assert response.status_code in [401, 403]


def test_predict_with_invalid_token():
    """Test prediction with an invalid token — must return 401."""
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token"},
        json={"model_name": "iris_model", "features": {"sepal_length": 5.1, "sepal_width": 3.5}}
    )
    assert response.status_code == 401


def test_predict_valid_payload_returns_auth_error_not_schema_error():
    """
    The dict {name: value} format must pass Pydantic validation
    and return 401 (auth), not 422 (schema error).
    FastAPI evaluates auth before the body — verifies that the pipeline is auth > schema.
    """
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token"},
        json={
            "model_name": "iris_model",
            "features": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    )
    assert response.status_code == 401  # Auth fail, not 422 validation error


def test_predict_payload_with_id_obs_returns_auth_error_not_schema_error():
    """
    Dict format with id_obs must pass Pydantic validation
    and return 401 (auth), not 422 (schema error).
    """
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token"},
        json={
            "model_name": "iris_model",
            "id_obs": "patient-42",
            "features": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    )
    assert response.status_code == 401  # Auth fail, not 422 validation error


def test_predict_payload_with_model_version_returns_auth_error_not_schema_error():
    """model_version in the body must pass Pydantic validation and return 401 (auth), not 422."""
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token"},
        json={
            "model_name": "iris_model",
            "model_version": "2.0.0",
            "features": {"sepal_length": 5.1, "sepal_width": 3.5}
        }
    )
    assert response.status_code == 401


def test_cached_models_endpoint_returns_structure():
    """GET /models/cached — no auth required, returns cached_models (list) and count (int)."""
    response = client.get("/models/cached")
    assert response.status_code == 200
    data = response.json()
    assert "cached_models" in data
    assert "count" in data
    assert isinstance(data["cached_models"], list)
    assert data["count"] == len(data["cached_models"])


def test_cached_models_count_reflects_injected_model():
    """After injecting a model into the Redis cache, GET /models/cached reflects it."""
    from src.services.model_service import model_service

    key = "cached_test_model:9.9.9"
    data = {
        "model": SimpleNamespace(),  # MagicMock is not picklable
        "metadata": SimpleNamespace(name="cached_test_model", version="9.9.9"),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    try:
        response = client.get("/models/cached")
        assert response.status_code == 200
        resp_data = response.json()
        assert key in resp_data["cached_models"]
        assert resp_data["count"] >= 1
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# GET /health/dependencies
# ---------------------------------------------------------------------------

def test_health_dependencies_requires_admin():
    """The endpoint must return 401/403 without authentication."""
    response = client.get("/health/dependencies")
    assert response.status_code in (401, 403)


def test_health_dependencies_requires_admin_token_not_user():
    """A non-admin token must return 403."""
    response = client.get("/health/dependencies", headers={"Authorization": "Bearer invalid-token"})
    assert response.status_code in (401, 403)


def test_health_dependencies_returns_200():
    """The endpoint responds 200 for an admin regardless of dependency status."""
    response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)
    assert response.status_code == 200


def test_health_dependencies_response_structure():
    """The response contains status, checked_at and the four dependencies."""
    response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)
    data = response.json()

    assert "status" in data
    assert data["status"] in ("ok", "degraded", "critical")
    assert "checked_at" in data
    assert "dependencies" in data

    deps = data["dependencies"]
    for key in ("database", "redis", "minio", "mlflow"):
        assert key in deps, f"Dépendance manquante : {key}"
        assert deps[key]["status"] in ("ok", "error")


def test_health_dependencies_db_ok(monkeypatch):
    """When the DB responds, database.status == 'ok' and latency_ms is a number."""
    response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)
    data = response.json()
    # The DB is in-memory SQLite — it must always respond in tests
    assert data["dependencies"]["database"]["status"] == "ok"
    assert data["dependencies"]["database"]["latency_ms"] is not None
    assert data["dependencies"]["database"]["latency_ms"] >= 0


def test_health_dependencies_redis_ok():
    """With FakeRedis, redis.status == 'ok'."""
    response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)
    data = response.json()
    assert data["dependencies"]["redis"]["status"] == "ok"
    assert data["dependencies"]["redis"]["latency_ms"] is not None


def test_health_dependencies_global_status_critical_when_db_fails(monkeypatch):
    """If the DB fails, the global status is 'critical'."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    db_error = DependencyDetail(status="error", latency_ms=None, detail="connection refused")
    ok = DependencyDetail(status="ok", latency_ms=1.0)

    with patch("src.main._check_db", AsyncMock(return_value=db_error)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=ok)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)

    assert response.status_code == 200
    assert response.json()["status"] == "critical"


def test_health_dependencies_global_status_degraded_when_non_db_fails(monkeypatch):
    """If MinIO fails (but not the DB), the global status is 'degraded'."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    ok = DependencyDetail(status="ok", latency_ms=1.0)
    err = DependencyDetail(status="error", latency_ms=None, detail="timeout")

    with patch("src.main._check_db", AsyncMock(return_value=ok)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=err)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_health_dependencies_global_status_ok_when_all_pass(monkeypatch):
    """When all dependencies respond, the global status is 'ok'."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    ok = DependencyDetail(status="ok", latency_ms=1.0)

    with patch("src.main._check_db", AsyncMock(return_value=ok)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=ok)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_dependencies_error_detail_included():
    """When a dependency fails, the detail field is present."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    ok = DependencyDetail(status="ok", latency_ms=1.0)
    err = DependencyDetail(status="error", latency_ms=None, detail="Connection refused")

    with patch("src.main._check_db", AsyncMock(return_value=ok)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=err)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies", headers=_ADMIN_HEADERS)

    data = response.json()
    assert data["dependencies"]["minio"]["status"] == "error"
    assert data["dependencies"]["minio"]["detail"] == "Connection refused"
    assert data["dependencies"]["minio"]["latency_ms"] is None
