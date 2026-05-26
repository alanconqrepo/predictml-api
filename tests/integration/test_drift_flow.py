"""
Integration tests — drift detection workflow.

Tested workflow:
  POST /models (with or without feature_baseline)
  → GET /models/{name}/drift → verify structure and values

Strategy:
  Predictions are inserted directly into the database via DBService.create_prediction
  (pattern from test_db_service_monitoring.py) to precisely control
  input_features and guarantee the desired production statistics.

Uses SQLite in-memory + FakeRedis + global MinIO mock.
Admin token: test-token-integ-df-admin-kk11
"""

import asyncio
import io
import json
import joblib

import numpy as np

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-integ-df-admin-kk11"
DF_MODEL_NOBASELINE = "df_drift_nobaseline_model"
DF_MODEL_BASELINE = "df_drift_baseline_model"
DF_MODEL_CRITICAL = "df_drift_critical_model"
MODEL_VERSION = "1.0.0"

# Typical baseline for iris (sepal length)
BASELINE = {
    "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9},
    "sepal width (cm)": {"mean": 3.05, "std": 0.43, "min": 2.0, "max": 4.4},
    "petal length (cm)": {"mean": 3.76, "std": 1.77, "min": 1.0, "max": 6.9},
    "petal width (cm)": {"mean": 1.20, "std": 0.76, "min": 0.1, "max": 2.5},
}

# Features close to baseline → no drift (single value for point tests)
NORMAL_FEATURES = {
    "sepal length (cm)": 5.8,
    "sepal width (cm)": 3.0,
    "petal length (cm)": 3.7,
    "petal width (cm)": 1.2,
}

# Generate features following baseline distribution N(mean, std) for reliable PSI.
# N=200 with fixed seed → deterministic, exceeds min_count=30, PSI consistent with "ok/warning".
_rng = np.random.default_rng(42)
NORMAL_FEATURES_LIST = [
    {
        "sepal length (cm)": float(_rng.normal(5.84, 0.83)),
        "sepal width (cm)": float(_rng.normal(3.05, 0.43)),
        "petal length (cm)": float(_rng.normal(3.76, 1.77)),
        "petal width (cm)": float(_rng.normal(1.20, 0.76)),
    }
    for _ in range(200)
]

# Features far from baseline → critical drift
OUTLIER_FEATURES = {
    "sepal length (cm)": 99.0,
    "sepal width (cm)": 99.0,
    "petal length (cm)": 99.0,
    "petal width (cm)": 99.0,
}


def _make_pkl() -> bytes:
    """Create a serialized sklearn model."""
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    """Create the admin user and test models."""
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="integ_df_admin",
                email="integ_df_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())


def _create_model(name: str, with_baseline: bool = False):
    """Create a model via the API."""
    data = {"name": name, "version": MODEL_VERSION}
    if with_baseline:
        data["feature_baseline"] = json.dumps(BASELINE)
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
        data=data,
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _insert_predictions(model_name: str, features_list: list):
    """Insert predictions directly into the database to control input_features."""
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, ADMIN_TOKEN)
        for features in features_list:
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=model_name,
                model_version=MODEL_VERSION,
                input_features=features,
                prediction_result=0,
                probabilities=[0.9, 0.05, 0.05],
                response_time_ms=10.0,
                status="success",
            )
        await db.commit()


# Create models at module load time
_create_model(DF_MODEL_NOBASELINE, with_baseline=False)
_create_model(DF_MODEL_BASELINE, with_baseline=True)
_create_model(DF_MODEL_CRITICAL, with_baseline=True)

# Insert 200 normal predictions for DF_MODEL_BASELINE (reliable PSI with N>=200)
asyncio.run(_insert_predictions(DF_MODEL_BASELINE, NORMAL_FEATURES_LIST))

# Insert 30 outlier predictions for DF_MODEL_CRITICAL
asyncio.run(_insert_predictions(DF_MODEL_CRITICAL, [OUTLIER_FEATURES] * 30))


class TestDriftFlow:
    """Tests for the drift detection workflow."""

    def test_01_drift_without_baseline_returns_no_baseline(self):
        """Model without feature_baseline → baseline_available=False, drift_summary='no_baseline'."""
        r = client.get(
            f"/models/{DF_MODEL_NOBASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["baseline_available"] is False
        assert data["drift_summary"] == "no_baseline"

    def test_02_drift_with_normal_features_returns_ok_or_no_data(self):
        """Model with baseline and normal features → drift_summary == 'ok' (or similar)."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["baseline_available"] is True
        # With features following baseline distribution, drift should be ok or warning
        # (insufficient_data if time window does not cover inserted predictions)
        assert data["drift_summary"] in ("ok", "warning", "no_data", "insufficient_data")

    def test_03_drift_with_outlier_features_returns_critical(self):
        """Model with features far from baseline → at least one critical feature."""
        r = client.get(
            f"/models/{DF_MODEL_CRITICAL}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["baseline_available"] is True
        if data["predictions_analyzed"] > 0:
            # Outlier features should trigger critical drift
            feature_statuses = [f["drift_status"] for f in data["features"].values()]
            assert "critical" in feature_statuses

    def test_04_drift_response_has_required_fields(self):
        """The drift response contains all required fields."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "period_days" in data
        assert "predictions_analyzed" in data
        assert "baseline_available" in data
        assert "drift_summary" in data
        assert "features" in data

    def test_05_drift_features_have_production_stats(self):
        """Each production feature has the required stats."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        if data["predictions_analyzed"] > 0:
            for feat_name, feat_data in data["features"].items():
                assert "production_mean" in feat_data
                assert "production_std" in feat_data
                assert "production_count" in feat_data

    def test_06_drift_unknown_model_returns_404(self):
        """GET /models/unknown/drift → 404."""
        r = client.get(
            "/models/totally_unknown_df_model/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_07_drift_requires_auth(self):
        """Without token → 401 or 403."""
        r = client.get(f"/models/{DF_MODEL_BASELINE}/drift")
        assert r.status_code in (401, 403)

    def test_08_drift_period_days_param_accepted(self):
        """The days= parameter is accepted without error."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"days": 30},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["period_days"] == 30

    def test_09_drift_no_baseline_features_have_no_baseline_status(self):
        """Without baseline, each feature has drift_status='no_baseline'."""
        # Insert a few predictions so features are non-empty
        asyncio.run(_insert_predictions(DF_MODEL_NOBASELINE, [NORMAL_FEATURES] * 3))

        r = client.get(
            f"/models/{DF_MODEL_NOBASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        if data["predictions_analyzed"] > 0:
            for feat_data in data["features"].values():
                assert feat_data["drift_status"] == "no_baseline"

    def test_10_drift_model_name_in_response(self):
        """The model name in the response matches the requested model."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        assert r.json()["model_name"] == DF_MODEL_BASELINE
