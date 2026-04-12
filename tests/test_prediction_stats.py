"""
Tests pour GET /predictions/stats — #18

Vérifie : auth, stats vides, counts, error_rate, filtre model_name, filtre days.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

STATS_TOKEN = "test-token-stats-zy9w"
STATS_MODEL_A = "stats_model_alpha"
STATS_MODEL_B = "stats_model_beta"
MODEL_VERSION = "1.0.0"


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, STATS_TOKEN):
            user = await DBService.create_user(
                db,
                username="stats_test_user",
                email="stats_test@test.com",
                api_token=STATS_TOKEN,
                role="user",
                rate_limit=10000,
            )
        else:
            from sqlalchemy import select
            from src.db.models import User
            result = await db.execute(
                select(User).where(User.api_token == STATS_TOKEN)
            )
            user = result.scalar_one()

        # Créer les entrées ModelMetadata
        for name in [STATS_MODEL_A, STATS_MODEL_B]:
            if not await DBService.get_model_metadata(db, name, MODEL_VERSION):
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=MODEL_VERSION,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{MODEL_VERSION}.pkl",
                    is_active=True,
                    is_production=True,
                )

        # Insérer 3 prédictions succès + 1 erreur pour STATS_MODEL_A
        for i in range(3):
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=STATS_MODEL_A,
                model_version=MODEL_VERSION,
                input_features={"f1": float(i)},
                prediction_result=i % 2,
                probabilities=None,
                response_time_ms=10.0 + i * 5,
                client_ip="127.0.0.1",
                user_agent="test",
                status="success",
                id_obs=f"obs_stats_{i}",
            )
        await DBService.create_prediction(
            db,
            user_id=user.id,
            model_name=STATS_MODEL_A,
            model_version=MODEL_VERSION,
            input_features={},
            prediction_result=None,
            probabilities=None,
            response_time_ms=5.0,
            client_ip="127.0.0.1",
            user_agent="test",
            status="error",
            id_obs=None,
        )

        # Insérer 2 prédictions succès pour STATS_MODEL_B
        for i in range(2):
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=STATS_MODEL_B,
                model_version=MODEL_VERSION,
                input_features={"f1": float(i)},
                prediction_result=i,
                probabilities=None,
                response_time_ms=20.0 + i,
                client_ip="127.0.0.1",
                user_agent="test",
                status="success",
                id_obs=f"obs_stats_b_{i}",
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_prediction_stats_requires_auth():
    """GET /predictions/stats sans token → 401/403."""
    r = client.get("/predictions/stats")
    assert r.status_code in [401, 403]


# ---------------------------------------------------------------------------
# Stats de base
# ---------------------------------------------------------------------------

def test_prediction_stats_returns_stats():
    """GET /predictions/stats retourne les deux modèles avec les bons counts."""
    r = client.get(
        "/predictions/stats",
        headers={"Authorization": f"Bearer {STATS_TOKEN}"},
        params={"days": 365},
    )
    assert r.status_code == 200
    body = r.json()
    assert "stats" in body
    assert body["days"] == 365

    by_name = {s["model_name"]: s for s in body["stats"]}
    assert STATS_MODEL_A in by_name
    assert STATS_MODEL_B in by_name

    a = by_name[STATS_MODEL_A]
    assert a["total_predictions"] == 4
    assert a["error_count"] == 1
    assert round(a["error_rate"], 4) == 0.25

    b = by_name[STATS_MODEL_B]
    assert b["total_predictions"] == 2
    assert b["error_count"] == 0
    assert b["error_rate"] == 0.0


def test_prediction_stats_response_time_percentiles():
    """p50 et p95 sont calculés uniquement sur les succès de STATS_MODEL_A."""
    r = client.get(
        "/predictions/stats",
        headers={"Authorization": f"Bearer {STATS_TOKEN}"},
        params={"days": 365, "model_name": STATS_MODEL_A},
    )
    assert r.status_code == 200
    body = r.json()
    stats = body["stats"]
    assert len(stats) == 1
    s = stats[0]
    # 3 succès avec response_time_ms = 10, 15, 20
    assert s["p50_response_time_ms"] is not None
    assert s["p95_response_time_ms"] is not None
    assert s["avg_response_time_ms"] is not None


# ---------------------------------------------------------------------------
# Filtre model_name
# ---------------------------------------------------------------------------

def test_prediction_stats_model_filter():
    """?model_name= limite les résultats à un seul modèle."""
    r = client.get(
        "/predictions/stats",
        headers={"Authorization": f"Bearer {STATS_TOKEN}"},
        params={"days": 365, "model_name": STATS_MODEL_B},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == STATS_MODEL_B
    assert len(body["stats"]) == 1
    assert body["stats"][0]["model_name"] == STATS_MODEL_B


def test_prediction_stats_unknown_model_returns_empty():
    """Modèle inexistant → stats liste vide, pas d'erreur."""
    r = client.get(
        "/predictions/stats",
        headers={"Authorization": f"Bearer {STATS_TOKEN}"},
        params={"days": 365, "model_name": "model_qui_nexiste_pas"},
    )
    assert r.status_code == 200
    assert r.json()["stats"] == []


# ---------------------------------------------------------------------------
# Filtre days
# ---------------------------------------------------------------------------

def test_prediction_stats_days_validation():
    """days=0 → 422 (ge=1). days=366 → 422 (le=365)."""
    for bad_days in [0, 366]:
        r = client.get(
            "/predictions/stats",
            headers={"Authorization": f"Bearer {STATS_TOKEN}"},
            params={"days": bad_days},
        )
        assert r.status_code == 422, f"Expected 422 for days={bad_days}"
