"""
Tests pour l'endpoint GET /predictions/{id}
"""
import asyncio

from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TOKEN_OWNER = "test-token-lookup-owner"
TOKEN_OTHER = "test-token-lookup-other"
TOKEN_ADMIN = "test-token-lookup-admin"


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TOKEN_OWNER):
            await DBService.create_user(
                db,
                username="lookup_owner",
                email="lookup_owner@test.com",
                api_token=TOKEN_OWNER,
                role="user",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, TOKEN_OTHER):
            await DBService.create_user(
                db,
                username="lookup_other",
                email="lookup_other@test.com",
                api_token=TOKEN_OTHER,
                role="user",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, TOKEN_ADMIN):
            await DBService.create_user(
                db,
                username="lookup_admin",
                email="lookup_admin@test.com",
                api_token=TOKEN_ADMIN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())


async def _create_prediction(token: str) -> int:
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, token)
        p = await DBService.create_prediction(
            db=db,
            user_id=user.id,
            model_name="iris",
            model_version="1.0.0",
            input_features={"petal_length": 5.1, "petal_width": 1.8},
            prediction_result="virginica",
            probabilities=[0.02, 0.11, 0.87],
            response_time_ms=23.0,
            status="success",
            id_obs="client_78",
        )
        return p.id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lookup_without_auth():
    response = client.get("/predictions/1")
    assert response.status_code == 401


def test_lookup_not_found():
    response = client.get(
        "/predictions/999999999",
        headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
    )
    assert response.status_code == 404
    assert "999999999" in response.json()["detail"]


def test_lookup_own_prediction_returns_200():
    pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
    response = client.get(
        f"/predictions/{pred_id}",
        headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == pred_id
    assert data["model_name"] == "iris"
    assert data["model_version"] == "1.0.0"
    assert data["id_obs"] == "client_78"
    assert data["prediction_result"] == "virginica"
    assert data["probabilities"] == [0.02, 0.11, 0.87]
    assert data["status"] == "success"
    assert data["username"] == "lookup_owner"


def test_lookup_other_user_prediction_returns_403():
    pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
    response = client.get(
        f"/predictions/{pred_id}",
        headers={"Authorization": f"Bearer {TOKEN_OTHER}"},
    )
    assert response.status_code == 403


def test_lookup_admin_can_see_any_prediction():
    pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
    response = client.get(
        f"/predictions/{pred_id}",
        headers={"Authorization": f"Bearer {TOKEN_ADMIN}"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == pred_id


def test_lookup_response_schema_fields():
    pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
    response = client.get(
        f"/predictions/{pred_id}",
        headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
    )
    assert response.status_code == 200
    data = response.json()
    for field in ("id", "model_name", "model_version", "id_obs", "input_features",
                  "prediction_result", "probabilities", "response_time_ms",
                  "timestamp", "status", "error_message", "username", "is_shadow"):
        assert field in data, f"Champ manquant : {field}"


def test_lookup_invalid_id_type():
    response = client.get(
        "/predictions/not-an-int",
        headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
    )
    assert response.status_code == 422
