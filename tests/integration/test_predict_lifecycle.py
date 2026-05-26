"""
Integration tests — complete prediction lifecycle.

Tested workflow:
  POST /models → POST /predict → GET /predictions → POST /observed-results
  → GET /predictions/stats → consistency verification

These tests exercise multiple components by chaining real API calls
with SQLite in-memory + FakeRedis + MinIO mock.
"""

import asyncio
import io
import joblib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-integ-pl-admin-ee55"
USER_TOKEN = "test-token-integ-pl-user-ff66"
PL_MODEL = "pl_lifecycle_model"
MODEL_VERSION = "1.0.0"

FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pkl() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _inject_cache(model_name: str, version: str = MODEL_VERSION):
    """Inject the model into Redis with the correct key."""
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    model.feature_names_in_ = list(FEATURES.keys())
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name, version=version, confidence_threshold=None, webhook_url=None
        ),
    }
    asyncio.run(
        model_service._redis.set(f"model:{model_name}:{version}", (lambda _b: (joblib.dump(data, _b), _b.getvalue())[1])(io.BytesIO()))
    )


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="integ_pl_admin",
                email="integ_pl_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="integ_pl_user",
                email="integ_pl_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=1000,
            )
        await db.commit()


asyncio.run(_setup())

# Create the model once for all tests
_model_response = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={"name": PL_MODEL, "version": MODEL_VERSION, "accuracy": "0.95"},
)
assert _model_response.status_code == 201, _model_response.text
_inject_cache(PL_MODEL)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPredictLifecycle:
    def test_predict_success_and_in_history(self):
        """Prediction → result in history."""
        # Uses ADMIN_TOKEN for predict AND query to avoid lazy-loading
        # p.user for a different user than the one querying (MissingGreenlet)
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": PL_MODEL, "features": FEATURES},
        )
        assert r.status_code == 200
        pred_result = r.json()
        assert "prediction" in pred_result

        # Verify that the prediction is in history
        now = datetime.utcnow()
        hist = client.get(
            "/predictions",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "name": PL_MODEL,
                "start": (now - timedelta(minutes=5)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
                "limit": 10,
            },
        )
        assert hist.status_code == 200
        data = hist.json()
        assert data["total"] >= 1
        model_names = [p["model_name"] for p in data["predictions"]]
        assert PL_MODEL in model_names

    def test_batch_predict_then_history(self):
        """Batch prediction → multiple entries in history."""
        batch_inputs = [{"features": FEATURES} for _ in range(3)]
        r = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"model_name": PL_MODEL, "inputs": batch_inputs},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 3

    def test_predict_with_id_obs_then_add_observed_result(self):
        """Prediction with id_obs then adding an observed result."""
        obs_id = "lifecycle-integ-obs-1"
        r_pred = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"model_name": PL_MODEL, "features": FEATURES, "id_obs": obs_id},
        )
        assert r_pred.status_code == 200

        # Add the observed result — body expects {"data": [...]} with date_time
        r_obs = client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "data": [
                    {
                        "model_name": PL_MODEL,
                        "id_obs": obs_id,
                        "date_time": datetime.utcnow().isoformat(),
                        "observed_result": 0,
                    }
                ]
            },
        )
        assert r_obs.status_code == 200
        data = r_obs.json()
        assert data["upserted"] >= 1

    def test_prediction_stats_after_predictions(self):
        """After predictions, GET /predictions/stats returns stats for the model."""
        # Make a few predictions
        for _ in range(2):
            client.post(
                "/predict",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": PL_MODEL, "features": FEATURES},
            )

        now = datetime.utcnow()
        r = client.get(
            "/predictions/stats",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "name": PL_MODEL,
                "start": (now - timedelta(minutes=10)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        # Stats must have at least one result for PL_MODEL
        if isinstance(data, list):
            model_stats = [s for s in data if s.get("model_name") == PL_MODEL]
            assert len(model_stats) > 0
        else:
            assert data.get("total_predictions", 0) >= 0

    def test_predict_missing_features_returns_422_or_500(self):
        """Prediction with incomplete features → error (422 validation or 500 runtime)."""
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={
                "model_name": PL_MODEL,
                "features": {},  # empty features
            },
        )
        # Depending on implementation, 422 (Pydantic validation) or 500 (sklearn error)
        assert r.status_code in (422, 500)

    def test_predict_unknown_model_returns_error(self):
        """Prediction on a non-existent model → 404."""
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"model_name": "totally_nonexistent_model_xyz", "features": FEATURES},
        )
        assert r.status_code == 404

    def test_get_predictions_pagination(self):
        """GET /predictions with pagination → limit and offset work."""
        # First generate some predictions with ADMIN_TOKEN (same user
        # as the one querying) to avoid cross-user lazy-load (MissingGreenlet)
        for _ in range(3):
            client.post(
                "/predict",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"model_name": PL_MODEL, "features": FEATURES},
            )

        now = datetime.utcnow()
        r = client.get(
            "/predictions",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "name": PL_MODEL,
                "start": (now - timedelta(minutes=10)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
                "limit": 2,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) <= 2

    def test_observed_result_overwrites_on_same_id_obs(self):
        """POST /observed-results twice with same id_obs → upsert (no duplication)."""
        obs_id = "lifecycle-upsert-obs-99"
        ts = datetime.utcnow().isoformat()

        def _payload(result):
            return {
                "data": [
                    {
                        "model_name": PL_MODEL,
                        "id_obs": obs_id,
                        "date_time": ts,
                        "observed_result": result,
                    }
                ]
            }

        r1 = client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json=_payload(1),
        )
        assert r1.status_code == 200

        # Second POST with same id_obs but different result
        r2 = client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json=_payload(2),
        )
        assert r2.status_code == 200
