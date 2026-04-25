"""
Tests pour les endpoints de l'API
"""
import asyncio
import pickle
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test de l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "active"
    assert "models_available" in data
    assert "models_count" in data


def test_health_endpoint():
    """Test du health check - retourne healthy si DB connectée, degraded sinon"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]


def test_models_endpoint():
    """Test de l'endpoint /models - retourne une liste de modèles"""
    response = client.get("/models")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


def test_predict_without_auth():
    """Test de prédiction sans authentification - HTTPBearer retourne 401"""
    response = client.post(
        "/predict",
        json={"model_name": "iris_model", "features": {"sepal_length": 5.1, "sepal_width": 3.5}}
    )
    # HTTPBearer sans header Authorization retourne 401 ou 403 selon la version de Starlette
    assert response.status_code in [401, 403]


def test_predict_with_invalid_token():
    """Test de prédiction avec un token invalide - doit retourner 401"""
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token"},
        json={"model_name": "iris_model", "features": {"sepal_length": 5.1, "sepal_width": 3.5}}
    )
    assert response.status_code == 401


def test_predict_valid_payload_returns_auth_error_not_schema_error():
    """
    Le format dict {nom: valeur} doit passer la validation Pydantic
    et retourner 401 (auth), pas 422 (schema error).
    FastAPI évalue l'auth avant le body — vérifie que le pipeline est bien auth > schema.
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
    assert response.status_code == 401  # Auth fail, pas 422 validation error


def test_predict_payload_with_id_obs_returns_auth_error_not_schema_error():
    """
    Format dict avec id_obs doit passer la validation Pydantic
    et retourner 401 (auth), pas 422 (schema error).
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
    assert response.status_code == 401  # Auth fail, pas 422 validation error


def test_predict_payload_with_model_version_returns_auth_error_not_schema_error():
    """model_version dans le body doit passer la validation Pydantic et retourner 401 (auth), pas 422."""
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
    """GET /models/cached — pas d'auth requise, retourne cached_models (liste) et count (int)."""
    response = client.get("/models/cached")
    assert response.status_code == 200
    data = response.json()
    assert "cached_models" in data
    assert "count" in data
    assert isinstance(data["cached_models"], list)
    assert data["count"] == len(data["cached_models"])


def test_cached_models_count_reflects_injected_model():
    """Après injection d'un modèle dans le cache Redis, GET /models/cached le reflète."""
    from src.services.model_service import model_service

    key = "cached_test_model:9.9.9"
    data = {
        "model": SimpleNamespace(),  # MagicMock n'est pas picklable
        "metadata": SimpleNamespace(name="cached_test_model", version="9.9.9"),
    }
    asyncio.run(model_service._redis.set(f"model:{key}", pickle.dumps(data)))
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

def test_health_dependencies_returns_200():
    """L'endpoint répond 200 quelle que soit l'état des dépendances."""
    response = client.get("/health/dependencies")
    assert response.status_code == 200


def test_health_dependencies_response_structure():
    """La réponse contient status, checked_at et les quatre dépendances."""
    response = client.get("/health/dependencies")
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
    """Quand la DB répond, database.status == 'ok' et latency_ms est un nombre."""
    response = client.get("/health/dependencies")
    data = response.json()
    # La DB est SQLite en mémoire — elle doit toujours répondre en test
    assert data["dependencies"]["database"]["status"] == "ok"
    assert data["dependencies"]["database"]["latency_ms"] is not None
    assert data["dependencies"]["database"]["latency_ms"] >= 0


def test_health_dependencies_redis_ok():
    """Avec FakeRedis, redis.status == 'ok'."""
    response = client.get("/health/dependencies")
    data = response.json()
    assert data["dependencies"]["redis"]["status"] == "ok"
    assert data["dependencies"]["redis"]["latency_ms"] is not None


def test_health_dependencies_global_status_critical_when_db_fails(monkeypatch):
    """Si la DB échoue, le statut global est 'critical'."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    db_error = DependencyDetail(status="error", latency_ms=None, detail="connection refused")
    ok = DependencyDetail(status="ok", latency_ms=1.0)

    with patch("src.main._check_db", AsyncMock(return_value=db_error)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=ok)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies")

    assert response.status_code == 200
    assert response.json()["status"] == "critical"


def test_health_dependencies_global_status_degraded_when_non_db_fails(monkeypatch):
    """Si MinIO échoue (mais pas la DB), le statut global est 'degraded'."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    ok = DependencyDetail(status="ok", latency_ms=1.0)
    err = DependencyDetail(status="error", latency_ms=None, detail="timeout")

    with patch("src.main._check_db", AsyncMock(return_value=ok)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=err)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_health_dependencies_global_status_ok_when_all_pass(monkeypatch):
    """Quand toutes les dépendances répondent, le statut global est 'ok'."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    ok = DependencyDetail(status="ok", latency_ms=1.0)

    with patch("src.main._check_db", AsyncMock(return_value=ok)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=ok)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_dependencies_error_detail_included():
    """Quand une dépendance échoue, le champ detail est présent."""
    from unittest.mock import AsyncMock, patch

    from src.schemas.health import DependencyDetail

    ok = DependencyDetail(status="ok", latency_ms=1.0)
    err = DependencyDetail(status="error", latency_ms=None, detail="Connection refused")

    with patch("src.main._check_db", AsyncMock(return_value=ok)), \
         patch("src.main._check_redis", AsyncMock(return_value=ok)), \
         patch("src.main._check_minio", AsyncMock(return_value=err)), \
         patch("src.main._check_mlflow", AsyncMock(return_value=ok)):
        response = client.get("/health/dependencies")

    data = response.json()
    assert data["dependencies"]["minio"]["status"] == "error"
    assert data["dependencies"]["minio"]["detail"] == "Connection refused"
    assert data["dependencies"]["minio"]["latency_ms"] is None
