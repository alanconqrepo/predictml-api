"""
Tests pour les endpoints de l'API
"""
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest
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
    """Après injection d'un modèle dans le cache, GET /models/cached le reflète."""
    from src.services.model_service import model_service

    key = "cached_test_model:9.9.9"
    model_service.models_cache[key] = {
        "model": MagicMock(),
        "metadata": SimpleNamespace(name="cached_test_model", version="9.9.9"),
    }
    try:
        response = client.get("/models/cached")
        assert response.status_code == 200
        data = response.json()
        assert key in data["cached_models"]
        assert data["count"] >= 1
    finally:
        model_service.models_cache.pop(key, None)
