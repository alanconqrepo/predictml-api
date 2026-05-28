"""
Tests for GET /predictions/unlabeled
"""

import asyncio
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.db.models import ObservedResult, Prediction
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TOKEN = "test-token-unlabeled-user"
MODEL = "unlabeled_model"
NOW = datetime.now(timezone.utc).replace(tzinfo=None)


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TOKEN):
            user = await DBService.create_user(
                db,
                username="unlabeled_user",
                email="unlabeled_user@test.com",
                api_token=TOKEN,
                role="user",
                rate_limit=10000,
            )
        else:
            from sqlalchemy import select

            from src.db.models import User

            result = await db.execute(select(User).where(User.api_token == TOKEN))
            user = result.scalar_one()

        # 3 predictions without observed result
        for i in range(3):
            db.add(
                Prediction(
                    user_id=user.id,
                    model_name=MODEL,
                    model_version="1.0.0",
                    id_obs=f"unlabeled-obs-{i}",
                    input_features={"f1": float(i)},
                    prediction_result="classA",
                    probabilities=[0.3 + i * 0.1, 0.7 - i * 0.1],
                    max_confidence=0.7 - i * 0.1,
                    response_time_ms=10.0,
                    status="success",
                    is_shadow=False,
                    timestamp=NOW,
                )
            )

        # 1 prediction WITH observed result
        db.add(
            Prediction(
                user_id=user.id,
                model_name=MODEL,
                model_version="1.0.0",
                id_obs="labeled-obs-0",
                input_features={"f1": 9.0},
                prediction_result="classA",
                probabilities=[0.9, 0.1],
                max_confidence=0.9,
                response_time_ms=10.0,
                status="success",
                is_shadow=False,
                timestamp=NOW,
            )
        )
        await db.commit()

        # Fetch user ID for observed result
        from sqlalchemy import select

        from src.db.models import User

        result = await db.execute(select(User).where(User.api_token == TOKEN))
        user = result.scalar_one()

        db.add(
            ObservedResult(
                id_obs="labeled-obs-0",
                model_name=MODEL,
                observed_result="classA",
                date_time=NOW,
                user_id=user.id,
            )
        )
        await db.commit()

        # 1 shadow prediction (should be excluded)
        db.add(
            Prediction(
                user_id=user.id,
                model_name=MODEL,
                model_version="1.0.0",
                id_obs="shadow-obs-0",
                input_features={"f1": 5.0},
                prediction_result="classA",
                probabilities=[0.6, 0.4],
                max_confidence=0.6,
                response_time_ms=10.0,
                status="success",
                is_shadow=True,
                timestamp=NOW,
            )
        )

        # 1 error prediction (should be excluded)
        db.add(
            Prediction(
                user_id=user.id,
                model_name=MODEL,
                model_version="1.0.0",
                id_obs="error-obs-0",
                input_features={"f1": 5.0},
                prediction_result=None,
                response_time_ms=5.0,
                status="error",
                is_shadow=False,
                timestamp=NOW,
            )
        )

        # 1 prediction without id_obs (should be excluded — not annotatable)
        db.add(
            Prediction(
                user_id=user.id,
                model_name=MODEL,
                model_version="1.0.0",
                id_obs=None,
                input_features={"f1": 5.0},
                prediction_result="classB",
                response_time_ms=10.0,
                status="success",
                is_shadow=False,
                timestamp=NOW,
            )
        )
        await db.commit()


asyncio.run(_setup())


def test_unlabeled_nominal():
    """Returns only non-labeled, non-shadow, successful predictions with id_obs."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_unlabeled"] == 3
    assert data["returned"] == 3
    assert data["strategy"] == "uncertainty"
    assert all(p["id_obs"] is not None for p in data["predictions"])


def test_unlabeled_strategy_uncertainty_ordering():
    """uncertainty strategy: lowest max_confidence first."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL, "strategy": "uncertainty"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    preds = r.json()["predictions"]
    confidences = [p["max_confidence"] for p in preds if p["max_confidence"] is not None]
    assert confidences == sorted(confidences), "uncertainty strategy must sort ASC by confidence"


def test_unlabeled_strategy_recent():
    """recent strategy returns 200 without error."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL, "strategy": "recent"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["strategy"] == "recent"


def test_unlabeled_strategy_random():
    """random strategy returns 200 without error."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL, "strategy": "random"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["strategy"] == "random"


def test_unlabeled_invalid_strategy():
    """Invalid strategy returns 422."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL, "strategy": "best_first"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 422


def test_unlabeled_limit():
    """limit parameter is respected."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL, "limit": 2},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["returned"] == 2
    assert data["total_unlabeled"] == 3  # total still reflects actual count


def test_unlabeled_excludes_labeled():
    """labeled-obs-0 must not appear in unlabeled results."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    id_obs_list = [p["id_obs"] for p in r.json()["predictions"]]
    assert "labeled-obs-0" not in id_obs_list


def test_unlabeled_excludes_shadow_and_errors():
    """shadow and error predictions are never returned."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    id_obs_list = [p["id_obs"] for p in r.json()["predictions"]]
    assert "shadow-obs-0" not in id_obs_list
    assert "error-obs-0" not in id_obs_list


def test_unlabeled_no_model_filter():
    """Without model_name filter, returns predictions from all models."""
    r = client.get(
        "/predictions/unlabeled",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["model_name"] is None


def test_unlabeled_empty_when_all_labeled():
    """Returns empty list when all predictions are labeled."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": "nonexistent_model_xyz"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_unlabeled"] == 0
    assert data["predictions"] == []


def test_unlabeled_requires_auth():
    """Returns 401 without a Bearer token."""
    r = client.get("/predictions/unlabeled", params={"model_name": MODEL})
    assert r.status_code == 401


def test_unlabeled_response_schema():
    """Response includes all expected fields."""
    r = client.get(
        "/predictions/unlabeled",
        params={"model_name": MODEL, "limit": 1},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert "id" in pred
    assert "id_obs" in pred
    assert "model_name" in pred
    assert "model_version" in pred
    assert "prediction_result" in pred
    assert "max_confidence" in pred
    assert "timestamp" in pred
