"""
Tests pour l'endpoint POST /models
"""
import asyncio
import io
import pickle

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-post-models"
TEST_MODEL_NAME = "test_model_post_models"


def make_pkl_bytes() -> bytes:
    """Crée un modèle sklearn minimal sérialisé."""
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    return pickle.dumps(model)


async def _setup_user():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_post_models",
                email="test_post_models@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup_user())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_create_model_without_auth():
    """POST /models sans header Authorization → 401/403"""
    response = client.post(
        "/models",
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": "no_auth_model", "version": "1.0.0"},
    )
    assert response.status_code in [401, 403]


def test_create_model_with_invalid_token():
    """POST /models avec token invalide → 401"""
    response = client.post(
        "/models",
        headers={"Authorization": "Bearer invalid-token"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": "invalid_token_model", "version": "1.0.0"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Cas nominaux
# ---------------------------------------------------------------------------

def test_create_model_success():
    """POST /models avec token valide + fichier pkl → 201 + champs attendus"""
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={
            "name": TEST_MODEL_NAME,
            "version": "1.0.0",
            "description": "Modèle de test",
            "algorithm": "LogisticRegression",
            "accuracy": "0.97",
            "f1_score": "0.97",
            "features_count": "4",
        },
    )
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == TEST_MODEL_NAME
    assert data["version"] == "1.0.0"
    assert data["algorithm"] == "LogisticRegression"
    assert data["accuracy"] == pytest.approx(0.97)
    assert data["is_active"] is True
    assert data["is_production"] is False
    assert data["minio_object_key"] == f"{TEST_MODEL_NAME}/v1.0.0.pkl"
    assert "id" in data
    assert "created_at" in data


def test_create_model_with_mlflow_run_id():
    """POST /models avec mlflow_run_id → bien enregistré dans la réponse"""
    run_id = "abc123def456abc123def456abc123de"
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={
            "name": f"{TEST_MODEL_NAME}_mlflow",
            "version": "1.0.0",
            "mlflow_run_id": run_id,
        },
    )
    assert response.status_code == 201
    assert response.json()["mlflow_run_id"] == run_id


def test_create_model_with_classes_and_training_params():
    """POST /models avec classes JSON et training_params JSON → bien désérialisés"""
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={
            "name": f"{TEST_MODEL_NAME}_params",
            "version": "1.0.0",
            "classes": "[0, 1, 2]",
            "training_params": '{"n_estimators": 100, "max_depth": 3}',
        },
    )
    assert response.status_code == 201

    data = response.json()
    assert data["classes"] == [0, 1, 2]
    assert data["training_params"] == {"n_estimators": 100, "max_depth": 3}


# ---------------------------------------------------------------------------
# Cas d'erreur
# ---------------------------------------------------------------------------

def test_create_model_duplicate_name():
    """POST /models avec un nom déjà existant → 409"""
    # Premier enregistrement
    client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": f"{TEST_MODEL_NAME}_dup", "version": "1.0.0"},
    )

    # Deuxième enregistrement avec le même nom → 409
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": f"{TEST_MODEL_NAME}_dup", "version": "2.0.0"},
    )
    assert response.status_code == 409
    assert "existe déjà" in response.json()["detail"]


def test_create_model_empty_file():
    """POST /models avec fichier vide → 400"""
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(b""), "application/octet-stream")},
        data={"name": f"{TEST_MODEL_NAME}_empty", "version": "1.0.0"},
    )
    assert response.status_code == 400


def test_create_model_missing_name():
    """POST /models sans name → 422"""
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"version": "1.0.0"},
    )
    assert response.status_code == 422


def test_create_model_missing_version():
    """POST /models sans version → 422"""
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": f"{TEST_MODEL_NAME}_noversion"},
    )
    assert response.status_code == 422


def test_create_model_missing_file():
    """POST /models sans fichier → 422"""
    response = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        data={"name": f"{TEST_MODEL_NAME}_nofile", "version": "1.0.0"},
    )
    assert response.status_code == 422
