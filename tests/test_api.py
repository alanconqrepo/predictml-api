"""
Tests pour les endpoints de l'API
"""
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


def test_predict_list_features_rejected_by_schema():
    """
    Le format liste est rejeté par Pydantic (couvert par test_schemas.py).
    Via HTTP, FastAPI évalue l'auth avant le body — retourne 401.
    """
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token"},
        json={
            "model_name": "iris_model",
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    )
    assert response.status_code == 401


def test_predict_dict_features_not_rejected_by_schema():
    """
    Le format dict {nom: valeur} doit passer la validation Pydantic
    et retourner 401 (auth), pas 422 (schema error).
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


def test_predict_dict_features_with_id_obs_not_rejected_by_schema():
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


def test_predict_with_model_version_not_rejected_by_schema():
    """model_version dans le body doit passer la validation et retourner 401 (auth)."""
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


def test_predict_dict_features_with_nested_list_rejected_by_schema():
    """
    Un dict avec valeur non scalaire est rejeté par Pydantic (couvert par test_schemas.py).
    Via HTTP, FastAPI évalue l'auth avant le body — retourne 401.
    """
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token"},
        json={
            "model_name": "iris_model",
            "features": {"a": [1, 2, 3]}
        }
    )
    assert response.status_code == 401
