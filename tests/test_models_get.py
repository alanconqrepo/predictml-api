"""
Tests pour l'endpoint GET /models/{name}/{version}
"""
import asyncio
import io
import joblib

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-get-models"
TEST_MODEL_NAME = "test_model_get"


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    _jbuf = io.BytesIO()
    joblib.dump(model, _jbuf)
    return _jbuf.getvalue()


async def _setup_user():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_get_models",
                email="test_get_models@test.com",
                api_token=TEST_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup_user())


def _create_model(name: str, version: str = "1.0.0") -> dict:
    response = client.post(
        "/models",
        data={"name": name, "version": version},
        files={"file": (f"{name}.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    return response.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_model_not_found():
    response = client.get("/models/inexistant/9.9.9")
    assert response.status_code == 404


def test_get_model_returns_metadata():
    """La route retourne les métadonnées correctes."""
    _create_model(TEST_MODEL_NAME, "1.0.0")
    response = client.get(f"/models/{TEST_MODEL_NAME}/1.0.0")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == TEST_MODEL_NAME
    assert data["version"] == "1.0.0"
    assert "model_loaded" in data
    assert "model_type" in data
    assert "feature_names" in data
    assert "load_instructions" in data


def test_get_model_returns_creator_fields():
    """La route retourne user_id_creator et creator_username."""
    _create_model(TEST_MODEL_NAME, "2.0.0")
    response = client.get(f"/models/{TEST_MODEL_NAME}/2.0.0")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id_creator"] is not None
    assert data["creator_username"] == "test_get_models"


def test_get_model_not_loaded_returns_load_instructions():
    """MinIO mocké → model_loaded=False + load_instructions MinIO présent."""
    _create_model(TEST_MODEL_NAME, "3.0.0")
    response = client.get(f"/models/{TEST_MODEL_NAME}/3.0.0")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False
    assert data["load_instructions"] is not None
    assert data["load_instructions"]["source"] == "minio"
    assert "python_code" in data["load_instructions"]


def test_get_model_mlflow_only_returns_load_instructions():
    """Modèle MLflow uniquement → instructions MLflow."""
    client.post(
        "/models",
        data={"name": TEST_MODEL_NAME, "version": "4.0.0", "mlflow_run_id": "abc123run"},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    response = client.get(f"/models/{TEST_MODEL_NAME}/4.0.0")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False
    assert data["load_instructions"]["source"] == "mlflow"
    assert "abc123run" in data["load_instructions"]["python_code"]
