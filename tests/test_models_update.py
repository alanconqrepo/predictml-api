"""
Tests for the PATCH /models/{name}/{version} endpoint
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

TEST_TOKEN = "test-token-patch-models"
MODEL_A = "patch_model_a"
MODEL_B = "patch_model_b"


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_patch_models",
                email="test_patch@test.com",
                api_token=TEST_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())


def _create_model(name: str, version: str = "1.0.0", **extra_data) -> dict:
    """Create a model via POST /models and return the JSON response."""
    data = {"name": name, "version": version, **extra_data}
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data=data,
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_update_model_without_auth():
    """PATCH /models without auth → 401/403."""
    _create_model(f"{MODEL_A}_noauth")
    response = client.patch(
        f"/models/{MODEL_A}_noauth/1.0.0",
        json={"description": "test"},
    )
    assert response.status_code in [401, 403]


def test_update_model_with_invalid_token():
    """PATCH /models with invalid token → 401."""
    _create_model(f"{MODEL_A}_badtoken")
    response = client.patch(
        f"/models/{MODEL_A}_badtoken/1.0.0",
        headers={"Authorization": "Bearer invalid"},
        json={"description": "test"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_update_description():
    """PATCH → description updated."""
    _create_model(f"{MODEL_A}_desc")
    response = client.patch(
        f"/models/{MODEL_A}_desc/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"description": "nouvelle description"},
    )
    assert response.status_code == 200
    assert response.json()["description"] == "nouvelle description"


def test_update_accuracy_and_features_count():
    """PATCH → accuracy and features_count updated."""
    _create_model(f"{MODEL_A}_metrics")
    response = client.patch(
        f"/models/{MODEL_A}_metrics/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"accuracy": 0.95, "features_count": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accuracy"] == pytest.approx(0.95)
    assert data["features_count"] == 4


def test_update_classes():
    """PATCH → classes updated."""
    _create_model(f"{MODEL_A}_classes")
    response = client.patch(
        f"/models/{MODEL_A}_classes/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"classes": [0, 1, 2]},
    )
    assert response.status_code == 200
    assert response.json()["classes"] == [0, 1, 2]


def test_update_partial_fields_only():
    """PATCH → only the provided fields are modified, others remain unchanged."""
    _create_model(f"{MODEL_A}_partial", description="originale", accuracy="0.80")
    response = client.patch(
        f"/models/{MODEL_A}_partial/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"accuracy": 0.99},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accuracy"] == pytest.approx(0.99)
    assert data["description"] == "originale"  # unchanged


# ---------------------------------------------------------------------------
# is_production — exclusivity per model
# ---------------------------------------------------------------------------

def test_set_is_production_true():
    """PATCH is_production=true → model marked as production."""
    _create_model(f"{MODEL_B}_prod_single")
    response = client.patch(
        f"/models/{MODEL_B}_prod_single/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )
    assert response.status_code == 200
    assert response.json()["is_production"] is True


def test_is_production_exclusive_across_versions():
    """
    When v2.0.0 is set to is_production=True,
    v1.0.0 (which was production) must automatically switch to False.
    """
    model_name = f"{MODEL_B}_exclusive"

    # Create v1 and set it to production
    _create_model(model_name, version="1.0.0")
    client.patch(
        f"/models/{model_name}/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )

    # Create v2 and set it to production
    _create_model(model_name, version="2.0.0")
    r2 = client.patch(
        f"/models/{model_name}/2.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )
    assert r2.status_code == 200
    assert r2.json()["is_production"] is True

    # Verify that v1 is no longer in production
    r1 = client.get("/models")
    models = {m["name"] + "_" + m["version"]: m for m in r1.json()}
    assert models[f"{model_name}_1.0.0"]["is_production"] is False
    assert models[f"{model_name}_2.0.0"]["is_production"] is True


def test_set_is_production_false():
    """PATCH is_production=false → model removed from production."""
    _create_model(f"{MODEL_B}_unprod")
    client.patch(
        f"/models/{MODEL_B}_unprod/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )
    response = client.patch(
        f"/models/{MODEL_B}_unprod/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": False},
    )
    assert response.status_code == 200
    assert response.json()["is_production"] is False


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_update_model_not_found():
    """PATCH on a non-existent model → 404."""
    response = client.patch(
        "/models/inexistant_model/9.9.9",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"description": "test"},
    )
    assert response.status_code == 404


def test_update_model_empty_body():
    """PATCH with empty body → 200, no fields modified."""
    _create_model(f"{MODEL_A}_emptybody", description="stable")
    response = client.patch(
        f"/models/{MODEL_A}_emptybody/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={},
    )
    assert response.status_code == 200
    assert response.json()["description"] == "stable"


def test_update_returns_creator_fields():
    """PATCH returns user_id_creator and creator_username in the response."""
    _create_model(f"{MODEL_A}_creator")
    response = client.patch(
        f"/models/{MODEL_A}_creator/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"description": "updated"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id_creator"] is not None
    assert data["creator_username"] == "test_patch_models"
