"""
Tests pour GET /users/{user_id}/usage
"""
import asyncio
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src.db.models import Prediction
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-admin-usage"
USER_TOKEN = "test-token-regular-usage"
OTHER_TOKEN = "test-token-other-usage"
USERNAME_PREFIX = "test_usage_route"

_admin_id: int = 0
_user_id: int = 0


async def _setup():
    global _admin_id, _user_id
    async with _TestSessionLocal() as db:
        admin = await DBService.get_user_by_token(db, ADMIN_TOKEN)
        if not admin:
            admin = await DBService.create_user(
                db,
                username=f"{USERNAME_PREFIX}_admin",
                email=f"{USERNAME_PREFIX}_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        _admin_id = admin.id

        user = await DBService.get_user_by_token(db, USER_TOKEN)
        if not user:
            user = await DBService.create_user(
                db,
                username=f"{USERNAME_PREFIX}_regular",
                email=f"{USERNAME_PREFIX}_regular@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )
        _user_id = user.id

        if not await DBService.get_user_by_token(db, OTHER_TOKEN):
            await DBService.create_user(
                db,
                username=f"{USERNAME_PREFIX}_other",
                email=f"{USERNAME_PREFIX}_other@test.com",
                api_token=OTHER_TOKEN,
                role="user",
                rate_limit=10000,
            )

        # Seed predictions for the regular user
        existing = await db.execute(
            __import__("sqlalchemy", fromlist=["select"]).select(Prediction).where(
                Prediction.user_id == _user_id
            )
        )
        if not existing.scalars().first():
            now = datetime.utcnow()
            predictions = [
                # iris model — 3 success, 1 error — two different days
                # Use 23h instead of 24h to avoid cutoff boundary issues
                Prediction(
                    user_id=_user_id,
                    model_name="iris",
                    model_version="1.0.0",
                    input_features={"sepal_length": 5.1},
                    prediction_result={"class": "setosa"},
                    probabilities=None,
                    response_time_ms=20.0,
                    status="success",
                    timestamp=now - timedelta(hours=23),
                ),
                Prediction(
                    user_id=_user_id,
                    model_name="iris",
                    model_version="1.0.0",
                    input_features={"sepal_length": 5.2},
                    prediction_result={"class": "setosa"},
                    probabilities=None,
                    response_time_ms=25.0,
                    status="success",
                    timestamp=now - timedelta(hours=23),
                ),
                Prediction(
                    user_id=_user_id,
                    model_name="iris",
                    model_version="1.0.0",
                    input_features={"sepal_length": 5.3},
                    prediction_result={"class": "setosa"},
                    probabilities=None,
                    response_time_ms=30.0,
                    status="success",
                    timestamp=now - timedelta(days=2),
                ),
                Prediction(
                    user_id=_user_id,
                    model_name="iris",
                    model_version="1.0.0",
                    input_features={"sepal_length": 0.0},
                    prediction_result={},
                    probabilities=None,
                    response_time_ms=5.0,
                    status="error",
                    timestamp=now - timedelta(days=2),
                ),
                # churn model — 2 success
                Prediction(
                    user_id=_user_id,
                    model_name="churn",
                    model_version="2.0.0",
                    input_features={"age": 30},
                    prediction_result={"churn": False},
                    probabilities=None,
                    response_time_ms=40.0,
                    status="success",
                    timestamp=now - timedelta(days=3),
                ),
                Prediction(
                    user_id=_user_id,
                    model_name="churn",
                    model_version="2.0.0",
                    input_features={"age": 40},
                    prediction_result={"churn": True},
                    probabilities=None,
                    response_time_ms=50.0,
                    status="success",
                    timestamp=now - timedelta(days=3),
                ),
                # old prediction outside the 30-day window
                Prediction(
                    user_id=_user_id,
                    model_name="iris",
                    model_version="1.0.0",
                    input_features={"sepal_length": 4.0},
                    prediction_result={"class": "virginica"},
                    probabilities=None,
                    response_time_ms=10.0,
                    status="success",
                    timestamp=now - timedelta(days=60),
                ),
            ]
            for p in predictions:
                db.add(p)
            await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth & access control
# ---------------------------------------------------------------------------


def test_usage_without_auth():
    """GET /users/{id}/usage sans auth → 401/403"""
    r = client.get(f"/users/{_user_id}/usage")
    assert r.status_code in [401, 403]


def test_usage_other_user_as_non_admin():
    """Un utilisateur ne peut pas voir les stats d'un autre → 403"""
    r = client.get(
        f"/users/{_admin_id}/usage",
        headers={"Authorization": f"Bearer {OTHER_TOKEN}"},
    )
    assert r.status_code == 403


def test_usage_self_as_user():
    """Un utilisateur peut voir ses propres stats → 200"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 200


def test_usage_any_user_as_admin():
    """Un admin peut voir les stats de n'importe quel utilisateur → 200"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200


def test_usage_not_found():
    """GET /users/{id}/usage avec id inexistant → 404"""
    r = client.get(
        "/users/999999/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


def test_usage_response_fields():
    """La réponse contient tous les champs attendus"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "user_id" in data
    assert "username" in data
    assert "period_days" in data
    assert "total_calls" in data
    assert "by_model" in data
    assert "by_day" in data


def test_usage_response_user_id_and_username():
    """user_id et username correspondent à l'utilisateur demandé"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = r.json()
    assert data["user_id"] == _user_id
    assert data["username"] == f"{USERNAME_PREFIX}_regular"


def test_usage_period_days_default():
    """period_days vaut 30 par défaut"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.json()["period_days"] == 30


def test_usage_period_days_custom():
    """period_days reflète le paramètre ?days=7"""
    r = client.get(
        f"/users/{_user_id}/usage?days=7",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.json()["period_days"] == 7


# ---------------------------------------------------------------------------
# Aggregation correctness
# ---------------------------------------------------------------------------


def test_usage_total_calls_excludes_old():
    """total_calls exclut les prédictions hors fenêtre (> 30 jours)"""
    r = client.get(
        f"/users/{_user_id}/usage?days=30",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    # 4 iris + 2 churn = 6; the old one (60 days ago) is excluded
    assert r.json()["total_calls"] == 6


def test_usage_by_model_contains_both_models():
    """by_model liste les deux modèles utilisés"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    names = [m["model_name"] for m in r.json()["by_model"]]
    assert "iris" in names
    assert "churn" in names


def test_usage_by_model_iris_calls():
    """by_model iris → 4 appels"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    iris = next(m for m in r.json()["by_model"] if m["model_name"] == "iris")
    assert iris["calls"] == 4


def test_usage_by_model_iris_errors():
    """by_model iris → 1 erreur"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    iris = next(m for m in r.json()["by_model"] if m["model_name"] == "iris")
    assert iris["errors"] == 1


def test_usage_by_model_iris_avg_latency():
    """by_model iris → avg_latency_ms calculé sur les succès uniquement"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    iris = next(m for m in r.json()["by_model"] if m["model_name"] == "iris")
    # 3 success: 20.0, 25.0, 30.0 → avg = 25.0
    assert iris["avg_latency_ms"] == pytest.approx(25.0, abs=0.1)


def test_usage_by_model_churn_no_errors():
    """by_model churn → 0 erreurs"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    churn = next(m for m in r.json()["by_model"] if m["model_name"] == "churn")
    assert churn["errors"] == 0
    assert churn["calls"] == 2


def test_usage_by_day_is_sorted():
    """by_day est trié par date croissante"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    dates = [entry["date"] for entry in r.json()["by_day"]]
    assert dates == sorted(dates)


def test_usage_by_day_calls_sum_equals_total():
    """La somme des calls dans by_day == total_calls"""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = r.json()
    assert sum(d["calls"] for d in data["by_day"]) == data["total_calls"]


def test_usage_narrow_window_excludes_old():
    """days=1 → seules les prédictions du dernier jour sont incluses"""
    r = client.get(
        f"/users/{_user_id}/usage?days=1",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    # Only 2 iris predictions from yesterday are within the last 1 day
    assert r.json()["total_calls"] == 2


def test_usage_zero_calls_when_no_data():
    """Un utilisateur sans prédictions retourne total_calls=0 et listes vides"""
    # Get the "other" user who has no predictions
    users_r = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    other = next(u for u in users_r.json() if u["username"] == f"{USERNAME_PREFIX}_other")

    r = client.get(
        f"/users/{other['id']}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_calls"] == 0
    assert data["by_model"] == []
    assert data["by_day"] == []


def test_usage_by_model_avg_latency_none_when_all_errors():
    """avg_latency_ms est null si toutes les prédictions du modèle sont des erreurs"""
    # Inject a model with only errors for the regular user
    async def _add_error_only():
        async with _TestSessionLocal() as db:
            db.add(
                Prediction(
                    user_id=_user_id,
                    model_name="error_only_model",
                    model_version="1.0.0",
                    input_features={},
                    prediction_result={},
                    probabilities=None,
                    response_time_ms=1.0,
                    status="error",
                    timestamp=datetime.utcnow(),
                )
            )
            await db.commit()

    asyncio.run(_add_error_only())

    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    models = {m["model_name"]: m for m in r.json()["by_model"]}
    assert "error_only_model" in models
    assert models["error_only_model"]["avg_latency_ms"] is None
