"""
Tests pour l'endpoint GET /predictions
"""
import asyncio
import pickle
from datetime import datetime, timedelta, timezone

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
    """Aucune prédiction en base pour ce modèle → total=0, liste vide, next_cursor=None."""
    response = client.get(
        f"/predictions?name=modele_inexistant&start={START}&end={END}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["predictions"] == []
    assert data["next_cursor"] is None


def test_get_predictions_pagination_fields():
    """La réponse contient bien total, limit, next_cursor et predictions."""
    response = client.get(
        f"/predictions?name={TEST_MODEL_NAME}&start={START}&end={END}&limit=10",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert data["limit"] == 10
    assert "next_cursor" in data
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


def test_get_predictions_cursor_navigation():
    """
    Insère 3 prédictions, pagine avec limit=2, vérifie que next_cursor est défini
    sur la première page et qu'on obtient le reste sans doublons sur la seconde.
    """
    cursor_model = "test_cursor_model"

    async def _insert_predictions():
        async with _TestSessionLocal() as db:
            user = await DBService.get_user_by_token(db, TEST_TOKEN)
            ids = []
            for i in range(3):
                p = await DBService.create_prediction(
                    db=db,
                    user_id=user.id,
                    model_name=cursor_model,
                    model_version="1.0.0",
                    input_features={"x": i},
                    prediction_result=i,
                    probabilities=None,
                    response_time_ms=1.0,
                    status="success",
                )
                ids.append(p.id)
            return ids

    inserted_ids = asyncio.run(_insert_predictions())

    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

    # Première page : limit=2, pas de curseur
    r1 = client.get(
        f"/predictions?name={cursor_model}&start={START}&end={END}&limit=2",
        headers=headers,
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert len(d1["predictions"]) == 2
    assert d1["next_cursor"] is not None

    ids_page1 = {p["id"] for p in d1["predictions"]}

    # Deuxième page : on passe next_cursor
    r2 = client.get(
        f"/predictions?name={cursor_model}&start={START}&end={END}&limit=2&cursor={d1['next_cursor']}",
        headers=headers,
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert len(d2["predictions"]) >= 1

    ids_page2 = {p["id"] for p in d2["predictions"]}

    # Aucun doublon entre les deux pages
    assert ids_page1.isdisjoint(ids_page2)

    # Les ids retournés sont bien parmi ceux insérés
    assert ids_page1 | ids_page2 <= set(inserted_ids)
