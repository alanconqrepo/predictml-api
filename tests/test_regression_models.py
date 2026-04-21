"""
Tests end-to-end pour les modèles de régression.

Vérifie que l'API gère correctement les modèles de régression dans tous les
scénarios : prédiction, politique de promotion, A/B compare, drift monitoring.
"""
import asyncio
import io
import pickle
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, Ridge

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _minio_mock, _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-regression-admin-r9z"
USER_TOKEN = "test-token-regression-user-r9z"

REG_MODEL = "reg_linear_model"
REG_RIDGE_MODEL = "reg_ridge_model"
REG_AB_MODEL = "reg_ab_model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_regression() -> LinearRegression:
    x = pd.DataFrame({"area": [50.0, 80.0, 120.0, 60.0, 90.0], "rooms": [2.0, 3.0, 4.0, 2.0, 3.0]})
    y = [150000.0, 240000.0, 360000.0, 180000.0, 270000.0]
    return LinearRegression().fit(x, y)


def _make_ridge_regression() -> Ridge:
    x = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0], "x2": [2.0, 4.0, 6.0, 8.0, 10.0]})
    y = [1.5, 3.2, 4.8, 6.1, 7.9]
    return Ridge(alpha=1.0).fit(x, y)


def _pkl(model) -> bytes:
    return pickle.dumps(model)


def _inject_cache(model_name: str, version: str, model) -> str:
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
        ),
    }
    asyncio.run(model_service._redis.set(f"model:{key}", pickle.dumps(data)))
    return key


VALID_TRAIN_SCRIPT = """\
import os
import pickle
import json
from sklearn.linear_model import LinearRegression
import pandas as pd

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X = pd.DataFrame({"area": [50.0, 80.0, 120.0], "rooms": [2.0, 3.0, 4.0]})
y = [150000.0, 240000.0, 360000.0]
model = LinearRegression().fit(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(json.dumps({"mae": 1234.5, "rmse": 2000.0}))
"""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="reg_admin",
                email="reg_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=99999,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="reg_user",
                email="reg_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=99999,
            )
        for name, ver in [
            (REG_MODEL, "1.0.0"),
            (REG_RIDGE_MODEL, "1.0.0"),
            (REG_AB_MODEL, "1.0.0"),
            (REG_AB_MODEL, "2.0.0"),
        ]:
            if not await DBService.get_model_metadata(db, name, ver):
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=ver,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{ver}.pkl",
                    is_active=True,
                    is_production=(ver == "1.0.0"),
                    deployment_mode="ab_test" if name == REG_AB_MODEL else "production",
                    traffic_weight=0.5 if name == REG_AB_MODEL else None,
                )


asyncio.run(_setup())

_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


# ---------------------------------------------------------------------------
# POST /predict — régression
# ---------------------------------------------------------------------------


class TestRegressionPredict:
    def test_regression_predict_no_probability(self):
        """Un modèle de régression retourne une prédiction sans champ 'probability'."""
        model = _make_linear_regression()
        key = _inject_cache(REG_MODEL, "1.0.0", model)
        try:
            r = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": REG_MODEL, "features": {"area": 75.0, "rooms": 3.0}},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["model_name"] == REG_MODEL
            assert data["prediction"] is not None
            # Un régresseur ne fournit pas de probabilités
            assert data.get("probability") is None
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_regression_predict_returns_float(self):
        """La prédiction d'un régresseur est une valeur numérique (float)."""
        model = _make_linear_regression()
        key = _inject_cache(REG_MODEL, "1.0.0", model)
        try:
            r = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": REG_MODEL, "features": {"area": 100.0, "rooms": 3.0}},
            )
            assert r.status_code == 200
            pred = r.json()["prediction"]
            assert isinstance(pred, (int, float))
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_ridge_regression_predict(self):
        """Ridge Regression : prédiction sans probabilités."""
        model = _make_ridge_regression()
        key = _inject_cache(REG_RIDGE_MODEL, "1.0.0", model)
        try:
            r = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": REG_RIDGE_MODEL, "features": {"x1": 3.0, "x2": 6.0}},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["prediction"] is not None
            assert data.get("probability") is None
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_regression_predict_batch(self):
        """Batch de prédictions via /predict-batch sans probabilités."""
        model = _make_linear_regression()
        key = _inject_cache(REG_MODEL, "1.0.0", model)
        try:
            r = client.post(
                "/predict-batch",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={
                    "model_name": REG_MODEL,
                    "inputs": [
                        {"features": {"area": 50.0, "rooms": 2.0}},
                        {"features": {"area": 100.0, "rooms": 3.0}},
                    ],
                },
            )
            assert r.status_code == 200
            data = r.json()
            results = data["predictions"]
            assert len(results) == 2
            for item in results:
                assert item["prediction"] is not None
                assert item.get("probability") is None
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# POST /models — upload régression via multipart
# ---------------------------------------------------------------------------


class TestRegressionModelUpload:
    def test_upload_linear_regression(self):
        """Uploader un LinearRegression via POST /models → 201."""
        model = _make_linear_regression()
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={"file": ("model.pkl", io.BytesIO(_pkl(model)), "application/octet-stream")},
            data={"name": "reg_upload_test", "version": "1.0.0"},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "reg_upload_test"
        assert data["version"] == "1.0.0"

    def test_upload_regression_with_train_script(self):
        """Uploader un régresseur avec un script d'entraînement valide → 201."""
        model = _make_linear_regression()
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={
                "file": ("model.pkl", io.BytesIO(_pkl(model)), "application/octet-stream"),
                "train_file": ("train.py", io.BytesIO(VALID_TRAIN_SCRIPT.encode()), "text/x-python"),
            },
            data={"name": "reg_upload_train", "version": "1.0.0"},
        )
        assert r.status_code == 201


# ---------------------------------------------------------------------------
# PATCH /models/{name}/policy — max_mae pour régression
# ---------------------------------------------------------------------------


class TestRegressionPromotionPolicy:
    def test_set_max_mae_policy(self):
        """PATCH /models/{name}/policy avec max_mae → stocké sans erreur."""
        r = client.patch(
            f"/models/{REG_MODEL}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "max_mae": 5000.0,
                "min_sample_validation": 5,
                "auto_promote": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["promotion_policy"]["max_mae"] == 5000.0
        assert data["promotion_policy"]["auto_promote"] is True

    def test_set_max_mae_policy_negative_rejected(self):
        """max_mae ≤ 0 → 422 (validation Pydantic)."""
        r = client.patch(
            f"/models/{REG_MODEL}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"max_mae": -1.0, "auto_promote": True},
        )
        assert r.status_code == 422

    def test_policy_accepts_both_min_accuracy_and_max_mae(self):
        """Une policy peut contenir min_accuracy ET max_mae (utilisés selon le type détecté)."""
        r = client.patch(
            f"/models/{REG_MODEL}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "min_accuracy": 0.85,
                "max_mae": 1000.0,
                "max_latency_p95_ms": 300.0,
                "min_sample_validation": 10,
                "auto_promote": False,
            },
        )
        assert r.status_code == 200
        policy = r.json()["promotion_policy"]
        assert policy["min_accuracy"] == 0.85
        assert policy["max_mae"] == 1000.0
        assert policy["max_latency_p95_ms"] == 300.0


# ---------------------------------------------------------------------------
# GET /models/{name}/ab-compare — régression
# ---------------------------------------------------------------------------


class TestRegressionABCompare:
    def test_ab_compare_regression_returns_200(self):
        """GET /ab-compare sur un modèle de régression → 200."""
        r = client.get(
            f"/models/{REG_AB_MODEL}/ab-compare",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            params={"days": 30},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == REG_AB_MODEL
        assert "versions" in data
        assert "ab_significance" in data

    def test_ab_compare_regression_no_significance_without_data(self):
        """Sans prédictions, ab_significance est None."""
        r = client.get(
            f"/models/{REG_AB_MODEL}/ab-compare",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            params={"days": 30},
        )
        assert r.status_code == 200
        data = r.json()
        # Sans prédictions, pas de significance possible
        assert data["ab_significance"] is None


# ---------------------------------------------------------------------------
# GET /models/{name}/calibration — doit retourner 422 pour la régression
# ---------------------------------------------------------------------------


class TestRegressionCalibration:
    def test_calibration_not_available_for_regression(self):
        """GET /models/{name}/calibration → insufficient_data pour régression sans prédictions.

        Sans données (pas de prédictions avec observed_results), la réponse est 200
        avec calibration_status='insufficient_data'. La 422 n'est levée que s'il y a
        des paires mais aucune avec predict_proba.
        """
        r = client.get(
            f"/models/{REG_MODEL}/calibration",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["calibration_status"] == "insufficient_data"
        assert data["brier_score"] is None


# ---------------------------------------------------------------------------
# GET /models/{name}/confidence-trend — comportement régression
# ---------------------------------------------------------------------------


class TestRegressionConfidenceTrend:
    def test_confidence_trend_returns_empty_trend_for_regression(self):
        """GET /models/{name}/confidence-trend → trend vide pour la régression (pas de probabilités)."""
        r = client.get(
            f"/models/{REG_MODEL}/confidence-trend",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            params={"days": 30},
        )
        assert r.status_code == 200
        data = r.json()
        # Pour un régresseur, pas de probabilités → trend vide
        assert data["trend"] == []
        assert data["model_name"] == REG_MODEL
