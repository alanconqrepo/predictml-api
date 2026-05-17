"""
Tests pour GET /models/leaderboard
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

import src.api.models as models_module
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

BOARD_TOKEN = "test-token-leaderboard-x7q2"
MODEL_HIGH = "lb_high_acc"   # is_production=True, accuracy=0.95, f1=0.94, 10 preds @ 50ms
MODEL_LOW = "lb_low_acc"     # is_production=True, accuracy=0.80, f1=0.79, 5 preds @ 100ms
MODEL_NONPROD = "lb_nonprod"  # is_production=False, accuracy=0.99 — must NOT appear
VERSION = "1.0.0"


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, BOARD_TOKEN):
            user = await DBService.create_user(
                db,
                username="lb_test_user",
                email="lb_test@test.com",
                api_token=BOARD_TOKEN,
                role="user",
                rate_limit=10000,
            )
        else:
            from sqlalchemy import select
            from src.db.models import User
            result = await db.execute(select(User).where(User.api_token == BOARD_TOKEN))
            user = result.scalar_one()

        for name, is_prod, acc, f1 in [
            (MODEL_HIGH, True, 0.95, 0.94),
            (MODEL_LOW, True, 0.80, 0.79),
            (MODEL_NONPROD, False, 0.99, 0.98),
        ]:
            if not await DBService.get_model_metadata(db, name, VERSION):
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=VERSION,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{VERSION}.joblib",
                    is_active=True,
                    is_production=is_prod,
                    accuracy=acc,
                    f1_score=f1,
                )

        # 10 predictions for MODEL_HIGH at 50ms, 5 for MODEL_LOW at 100ms
        existing_high = await DBService.get_prediction_stats(db, days=1, model_name=MODEL_HIGH)
        if not existing_high:
            for i in range(10):
                await DBService.create_prediction(
                    db,
                    user_id=user.id,
                    model_name=MODEL_HIGH,
                    model_version=VERSION,
                    input_features={"f": float(i)},
                    prediction_result=i % 2,
                    probabilities=None,
                    response_time_ms=50.0,
                    client_ip="127.0.0.1",
                    user_agent="test",
                    status="success",
                    id_obs=f"lb_high_{i}",
                )
            for i in range(5):
                await DBService.create_prediction(
                    db,
                    user_id=user.id,
                    model_name=MODEL_LOW,
                    model_version=VERSION,
                    input_features={"f": float(i)},
                    prediction_result=i % 2,
                    probabilities=None,
                    response_time_ms=100.0,
                    client_ip="127.0.0.1",
                    user_agent="test",
                    status="success",
                    id_obs=f"lb_low_{i}",
                )


asyncio.run(_setup())


def setup_function():
    """Clear the leaderboard cache before each test for isolation."""
    models_module._leaderboard_cache.clear()


# ---------------------------------------------------------------------------
# Auth & validation
# ---------------------------------------------------------------------------

def test_leaderboard_requires_auth():
    r = client.get("/models/leaderboard")
    assert r.status_code in [401, 403]


def test_leaderboard_invalid_metric():
    r = client.get(
        "/models/leaderboard?metric=invalid_metric",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 422


def test_leaderboard_invalid_days_too_low():
    r = client.get(
        "/models/leaderboard?days=0",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 422


def test_leaderboard_invalid_days_too_high():
    r = client.get(
        "/models/leaderboard?days=366",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Schema & basic response
# ---------------------------------------------------------------------------

def test_leaderboard_schema_fields():
    r = client.get(
        "/models/leaderboard",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    entry = next(e for e in data if e["name"] == MODEL_HIGH)
    for field in ("rank", "name", "version", "accuracy", "f1_score", "latency_p95_ms", "drift_status", "predictions_count"):
        assert field in entry, f"Missing field: {field}"


def test_leaderboard_returns_only_production():
    r = client.get(
        "/models/leaderboard",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 200
    names = [e["name"] for e in r.json()]
    assert MODEL_HIGH in names
    assert MODEL_LOW in names
    assert MODEL_NONPROD not in names


def test_leaderboard_rank_ordering():
    r = client.get(
        "/models/leaderboard",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    ranks = [e["rank"] for e in data]
    assert ranks == list(range(1, len(data) + 1))


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_leaderboard_sort_by_accuracy():
    r = client.get(
        "/models/leaderboard?metric=accuracy",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    high = next(e for e in data if e["name"] == MODEL_HIGH)
    low = next(e for e in data if e["name"] == MODEL_LOW)
    assert high["rank"] < low["rank"]
    assert high["accuracy"] == pytest.approx(0.95)


def test_leaderboard_sort_by_f1_score():
    r = client.get(
        "/models/leaderboard?metric=f1_score",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    high = next(e for e in data if e["name"] == MODEL_HIGH)
    low = next(e for e in data if e["name"] == MODEL_LOW)
    assert high["rank"] < low["rank"]


def test_leaderboard_sort_by_predictions_count():
    r = client.get(
        "/models/leaderboard?metric=predictions_count",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    high = next(e for e in data if e["name"] == MODEL_HIGH)
    low = next(e for e in data if e["name"] == MODEL_LOW)
    # MODEL_HIGH has 10 predictions, MODEL_LOW has 5
    assert high["predictions_count"] == 10
    assert low["predictions_count"] == 5
    assert high["rank"] < low["rank"]


def test_leaderboard_sort_by_latency():
    r = client.get(
        "/models/leaderboard?metric=latency_p95_ms",
        headers={"Authorization": f"Bearer {BOARD_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    high = next(e for e in data if e["name"] == MODEL_HIGH)
    low = next(e for e in data if e["name"] == MODEL_LOW)
    # MODEL_HIGH latency=50ms < MODEL_LOW latency=100ms → MODEL_HIGH ranked first (ascending)
    assert high["rank"] < low["rank"]
    assert high["latency_p95_ms"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_leaderboard_cache_hit():
    """Second identical request returns cached data (same rank ordering)."""
    headers = {"Authorization": f"Bearer {BOARD_TOKEN}"}
    r1 = client.get("/models/leaderboard?metric=accuracy", headers=headers)
    r2 = client.get("/models/leaderboard?metric=accuracy", headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
    assert "accuracy:30" in models_module._leaderboard_cache
