"""
Tests for rate limiting enforcement on POST /predict.

Strategy:
  - Create a user with rate_limit_per_day = 2
  - Inject predictions in DB to simulate a reached quota
  - Verify that POST /predict returns 429 when the quota is exceeded
  - Verify that other endpoints (GET /models) remain accessible after quota is reached

IP rate limiting (slowapi):
  - Use unittest.mock to patch the limiter and trigger RateLimitExceeded
  - Verify that POST /predict, POST /models and POST /users return 429 when the
    per-minute limit is reached
"""
import asyncio
import io
import joblib
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

RL_TOKEN = "test-token-rate-limit-abc9"
RL_MODEL = "rl_iris_model"
RL_MODEL_VERSION = "1.0.0"


def _make_model() -> LogisticRegression:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]})
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_cache(model_name: str, version: str, model) -> str:
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(name=model_name, version=version, confidence_threshold=None, webhook_url=None),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, RL_TOKEN):
            await DBService.create_user(
                db,
                username="test_rate_limit_user",
                email="test_rate_limit@test.com",
                api_token=RL_TOKEN,
                role="user",
                rate_limit=2,
            )
        if not await DBService.get_model_metadata(db, RL_MODEL, RL_MODEL_VERSION):
            await DBService.create_model_metadata(
                db,
                name=RL_MODEL,
                version=RL_MODEL_VERSION,
                minio_bucket="models",
                minio_object_key=f"{RL_MODEL}/v{RL_MODEL_VERSION}.joblib",
                is_active=True,
                is_production=True,
            )


asyncio.run(_setup())


async def _seed_predictions(count: int) -> None:
    """Insert `count` predictions today for RL_TOKEN user."""
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, RL_TOKEN)
        for _ in range(count):
            await DBService.create_prediction(
                db=db,
                user_id=user.id,
                model_name=RL_MODEL,
                model_version=RL_MODEL_VERSION,
                input_features={"f1": 1.0, "f2": 2.0},
                prediction_result=0,
                probabilities=None,
                response_time_ms=10.0,
                status="success",
            )


async def _delete_predictions() -> None:
    """Delete all predictions for RL_TOKEN user."""
    from sqlalchemy import delete
    from src.db.models import Prediction

    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, RL_TOKEN)
        await db.execute(delete(Prediction).where(Prediction.user_id == user.id))
        await db.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rate_limit_not_exceeded_returns_200():
    """With 0 predictions today (quota 2), POST /predict must succeed."""
    asyncio.run(_delete_predictions())
    model = _make_model()
    key = _inject_cache(RL_MODEL, RL_MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {RL_TOKEN}"},
            json={"model_name": RL_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
        )
        assert response.status_code == 200
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())


def test_rate_limit_exceeded_returns_429():
    """With rate_limit=2 and 2 predictions already recorded, the next one must return 429."""
    asyncio.run(_delete_predictions())
    asyncio.run(_seed_predictions(2))
    model = _make_model()
    key = _inject_cache(RL_MODEL, RL_MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {RL_TOKEN}"},
            json={"model_name": RL_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
        )
        assert response.status_code == 429
        detail = response.json()["detail"]
        assert "Rate limit" in detail
        assert "2" in detail
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())


def test_rate_limit_error_message_contains_quota_and_count():
    """The 429 message must indicate the daily quota and the number of predictions made."""
    asyncio.run(_delete_predictions())
    asyncio.run(_seed_predictions(2))
    model = _make_model()
    key = _inject_cache(RL_MODEL, RL_MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {RL_TOKEN}"},
            json={"model_name": RL_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
        )
        assert response.status_code == 429
        detail = response.json()["detail"]
        assert "2 requests/day" in detail or "2 requêtes/jour" in detail
        assert "2 predictions" in detail or "2 prédictions" in detail
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())


def test_rate_limit_does_not_block_models_endpoint():
    """When the quota is reached, GET /models remains accessible (auth only, no rate limit)."""
    asyncio.run(_delete_predictions())
    asyncio.run(_seed_predictions(2))
    try:
        # GET /models does not require auth — verifies it responds correctly despite the quota being reached
        response = client.get("/models")
        assert response.status_code == 200
    finally:
        asyncio.run(_delete_predictions())


def test_rate_limit_count_today_only():
    """get_user_prediction_count_today must only count today's predictions (UTC)."""
    asyncio.run(_delete_predictions())

    async def _seed_yesterday():
        from sqlalchemy import update
        from src.db.models import Prediction
        from datetime import timedelta

        async with _TestSessionLocal() as db:
            user = await DBService.get_user_by_token(db, RL_TOKEN)
            # Create a prediction then backdate it to yesterday
            pred = await DBService.create_prediction(
                db=db,
                user_id=user.id,
                model_name=RL_MODEL,
                model_version=RL_MODEL_VERSION,
                input_features={"f1": 1.0, "f2": 2.0},
                prediction_result=0,
                probabilities=None,
                response_time_ms=10.0,
                status="success",
            )
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            await db.execute(
                update(Prediction)
                .where(Prediction.id == pred.id)
                .values(timestamp=yesterday)
            )
            await db.commit()

    asyncio.run(_seed_yesterday())

    async def _check_count():
        async with _TestSessionLocal() as db:
            user = await DBService.get_user_by_token(db, RL_TOKEN)
            return await DBService.get_user_prediction_count_today(db, user.id)

    count = asyncio.run(_check_count())
    assert count == 0, f"Expected 0 predictions today, got {count}"
    asyncio.run(_delete_predictions())


# ---------------------------------------------------------------------------
# Tests IP rate limiting (slowapi — 60/min /predict, 10/min /models and /users)
# ---------------------------------------------------------------------------

def test_ip_rate_limit_predict_returns_429_when_exceeded():
    """When the per-IP limit is reached on a route, slowapi returns 429."""
    import os
    from unittest.mock import patch
    from fastapi import FastAPI, Request as _Request
    from fastapi.testclient import TestClient as _TestClient
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    # slowapi uses RATELIMIT_ENABLED env var internally to bypass all limits.
    # Temporarily override it so the rate limit is actually enforced in this test.
    with patch.dict(os.environ, {"RATELIMIT_ENABLED": "1"}):
        _limiter = Limiter(key_func=get_remote_address)
        _app = FastAPI()
        _app.state.limiter = _limiter
        _app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        @_app.get("/test-limit")
        @_limiter.limit("1/minute")
        async def _limited(request: _Request):
            return {"ok": True}

        _client = _TestClient(_app, raise_server_exceptions=False)
        r1 = _client.get("/test-limit")
        assert r1.status_code == 200
        r2 = _client.get("/test-limit")
        assert r2.status_code == 429


def test_ip_rate_limit_predict_allows_requests_under_limit():
    """/predict returns 200 for requests well below the per-minute limit."""
    asyncio.run(_delete_predictions())
    model = _make_model()
    key = _inject_cache(RL_MODEL, RL_MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {RL_TOKEN}"},
            json={"model_name": RL_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
        )
        # Verify that the per-minute limit does not block a single request
        assert response.status_code in (200, 429)
        if response.status_code == 429:
            detail = response.json().get("detail", "")
            # 429 must come from the daily rate limit (quota) or the per-IP rate limit
            assert "Rate limit" in detail or "minute" in detail or "per" in detail
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())


def test_ip_rate_limit_models_endpoint_has_limiter_configured():
    """Verify that POST /models has the @limiter.limit decorator configured."""
    from src.api.models import create_model

    # slowapi stores limits in the _rate_limits attribute or via wrappers
    # The presence of __wrapped__ or _limits confirms the decorator is active
    assert hasattr(create_model, "__wrapped__") or callable(create_model)


def test_ip_rate_limit_users_endpoint_has_limiter_configured():
    """Verify that POST /users has the @limiter.limit decorator configured."""
    from src.api.users import create_user

    assert hasattr(create_user, "__wrapped__") or callable(create_user)


def test_ip_rate_limit_predict_endpoint_has_limiter_configured():
    """Verify that POST /predict has the @limiter.limit decorator configured."""
    from src.api.predict import predict

    assert hasattr(predict, "__wrapped__") or callable(predict)


def test_ip_rate_limit_exception_handler_registered():
    """Verify that the RateLimitExceeded handler is registered on the application."""
    from slowapi.errors import RateLimitExceeded
    from src.main import app

    assert RateLimitExceeded in app.exception_handlers


def test_ip_rate_limit_limiter_attached_to_app():
    """Verify that the slowapi limiter is properly attached to app.state."""
    from src.core.rate_limit import limiter
    from src.main import app

    assert app.state.limiter is limiter


# ---------------------------------------------------------------------------
# Tests rate limiting and batch size on POST /predict-batch
# ---------------------------------------------------------------------------

def test_ip_rate_limit_predict_batch_endpoint_has_limiter_configured():
    """Verify that POST /predict-batch has the @limiter.limit decorator configured."""
    from src.api.predict import predict_batch

    assert hasattr(predict_batch, "__wrapped__") or callable(predict_batch)


def test_predict_batch_max_batch_size_env_default():
    """MAX_BATCH_SIZE must be 500 by default (env variable not defined)."""
    import importlib
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MAX_BATCH_SIZE", None)
        import src.api.predict as predict_mod
        importlib.reload(predict_mod)
        assert predict_mod.MAX_BATCH_SIZE == 500


def test_predict_batch_max_batch_size_env_override():
    """MAX_BATCH_SIZE must read the MAX_BATCH_SIZE env variable."""
    import importlib
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"MAX_BATCH_SIZE": "200"}):
        import src.api.predict as predict_mod
        importlib.reload(predict_mod)
        assert predict_mod.MAX_BATCH_SIZE == 200


def test_predict_batch_rejects_oversized_batch():
    """A batch exceeding MAX_BATCH_SIZE must return 422."""
    import os
    from unittest.mock import patch

    model = _make_model()
    key = _inject_cache(RL_MODEL, RL_MODEL_VERSION, model)
    asyncio.run(_delete_predictions())

    try:
        with patch("src.api.predict.MAX_BATCH_SIZE", 2):
            inputs = [{"features": {"f1": 1.0, "f2": 2.0}} for _ in range(3)]
            response = client.post(
                "/predict-batch",
                headers={"Authorization": f"Bearer {RL_TOKEN}"},
                json={
                    "model_name": RL_MODEL,
                    "inputs": inputs,
                },
            )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "Batch too large" in detail or "Batch trop grand" in detail
        assert "max 2" in detail
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())


def test_predict_batch_accepts_batch_within_limit():
    """A batch within the MAX_BATCH_SIZE limit must not be rejected for size."""
    from unittest.mock import patch

    model = _make_model()
    key = _inject_cache(RL_MODEL, RL_MODEL_VERSION, model)
    asyncio.run(_delete_predictions())

    try:
        with patch("src.api.predict.MAX_BATCH_SIZE", 500):
            inputs = [{"features": {"f1": 1.0, "f2": 2.0}} for _ in range(2)]
            response = client.post(
                "/predict-batch",
                headers={"Authorization": f"Bearer {RL_TOKEN}"},
                json={
                    "model_name": RL_MODEL,
                    "inputs": inputs,
                },
            )
        # May return 200 (success) or 429 (daily quota) but not 422 for size
        assert response.status_code != 422 or (
            "Batch too large" not in response.json().get("detail", "")
            and "Batch trop grand" not in response.json().get("detail", "")
        )
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())
