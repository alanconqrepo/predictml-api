"""
Tests pour l'endpoint GET /predictions
"""
import asyncio
import io
import pickle
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-get-predictions"
TEST_MODEL_NAME = "test_model_predictions"
NOW = datetime.now(timezone.utc).replace(tzinfo=None)
START = (NOW - timedelta(hours=1)).isoformat()
END = (NOW + timedelta(hours=1)).isoformat()


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    return pickle.dumps(model)


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_get_predictions",
                email="test_get_predictions@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_predictions_without_auth():
    response = client.get(f"/predictions?name={TEST_MODEL_NAME}&start={START}&end={END}")
    assert response.status_code == 401


def test_get_predictions_start_after_end():
    response = client.get(
        f"/predictions?name={TEST_MODEL_NAME}&start={END}&end={START}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 422


def test_get_predictions_empty_result():
    """Aucune prédiction en base pour ce modèle → total=0, liste vide."""
    response = client.get(
        f"/predictions?name=modele_inexistant&start={START}&end={END}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["predictions"] == []


def test_get_predictions_pagination_fields():
    """La réponse contient bien total, limit, offset."""
    response = client.get(
        f"/predictions?name={TEST_MODEL_NAME}&start={START}&end={END}&limit=10&offset=0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["limit"] == 10
    assert data["offset"] == 0
    assert "predictions" in data


def test_get_predictions_missing_name():
    response = client.get(
        f"/predictions?start={START}&end={END}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 422


def test_get_predictions_missing_start():
    response = client.get(
        f"/predictions?name={TEST_MODEL_NAME}&end={END}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 422


def test_get_predictions_with_version_filter():
    """Filtre version — ne plante pas même si aucun résultat."""
    response = client.get(
        f"/predictions?name={TEST_MODEL_NAME}&version=9.9.9&start={START}&end={END}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_get_predictions_with_user_filter():
    """Filtre user — ne plante pas même si aucun résultat."""
    response = client.get(
        f"/predictions?name={TEST_MODEL_NAME}&user=nobody&start={START}&end={END}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0
