"""
Tests pour l'enforcement du rate limiting sur POST /predict.

Stratégie :
  - Créer un utilisateur avec rate_limit_per_day = 2
  - Injecter des prédictions en DB pour simuler le quota atteint
  - Vérifier que POST /predict retourne 429 quand le quota est dépassé
  - Vérifier que les autres endpoints (GET /models) restent accessibles après quota atteint
"""
import asyncio
import pickle
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
    asyncio.run(model_service._redis.set(f"model:{key}", pickle.dumps(data)))
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
                minio_object_key=f"{RL_MODEL}/v{RL_MODEL_VERSION}.pkl",
                is_active=True,
                is_production=True,
            )


asyncio.run(_setup())


async def _seed_predictions(count: int) -> None:
    """Insère `count` prédictions today pour l'utilisateur RL_TOKEN."""
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
    """Supprime toutes les prédictions de l'utilisateur RL_TOKEN."""
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
    """Avec 0 prédictions aujourd'hui (quota 2), POST /predict doit réussir."""
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
    """Avec rate_limit=2 et 2 prédictions déjà enregistrées, la suivante doit retourner 429."""
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
        assert "Rate limit dépassé" in detail
        assert "2" in detail
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())


def test_rate_limit_error_message_contains_quota_and_count():
    """Le message 429 doit indiquer le quota journalier et le nombre de prédictions effectuées."""
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
        assert "2 requêtes/jour" in detail
        assert "2 prédictions" in detail
    finally:
        asyncio.run(model_service.clear_cache(key))
        asyncio.run(_delete_predictions())


def test_rate_limit_does_not_block_models_endpoint():
    """Quand le quota est atteint, GET /models reste accessible (auth uniquement, pas de rate limit)."""
    asyncio.run(_delete_predictions())
    asyncio.run(_seed_predictions(2))
    try:
        # GET /models ne nécessite pas d'auth — vérifie qu'il répond bien malgré le quota atteint
        response = client.get("/models")
        assert response.status_code == 200
    finally:
        asyncio.run(_delete_predictions())


def test_rate_limit_count_today_only():
    """get_user_prediction_count_today ne doit compter que les prédictions du jour (UTC)."""
    asyncio.run(_delete_predictions())

    async def _seed_yesterday():
        from sqlalchemy import update
        from src.db.models import Prediction
        from datetime import timedelta

        async with _TestSessionLocal() as db:
            user = await DBService.get_user_by_token(db, RL_TOKEN)
            # Créer une prédiction puis la rétrodater à hier
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
