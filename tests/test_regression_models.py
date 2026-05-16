"""
Tests end-to-end pour les modèles de régression.

Vérifie que l'API gère correctement les modèles de régression dans tous les
scénarios : prédiction, politique de promotion, A/B compare, drift monitoring.
"""
import asyncio
import io
import joblib
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
    _jbuf = io.BytesIO()
    joblib.dump(model, _jbuf)
    return _jbuf.getvalue()


def _inject_cache(model_name: str, version: str, model, feature_baseline=None) -> str:
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
            feature_baseline=feature_baseline,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


VALID_TRAIN_SCRIPT = """\
import os
import joblib
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
    joblib.dump(model, f)

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
                    minio_object_key=f"{name}/v{ver}.joblib",
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

REG_SHAP_LR_MODEL = "reg_shap_lr_model"
REG_SHAP_RIDGE_MODEL = "reg_shap_ridge_model"
REG_FI_MODEL = "reg_fi_model"
REG_IV_MODEL = "reg_iv_model"


async def _setup_extra():
    async with _TestSessionLocal() as db:
        for name, ver in [
            (REG_SHAP_LR_MODEL, "1.0.0"),
            (REG_SHAP_RIDGE_MODEL, "1.0.0"),
            (REG_FI_MODEL, "1.0.0"),
            (REG_IV_MODEL, "1.0.0"),
        ]:
            if not await DBService.get_model_metadata(db, name, ver):
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=ver,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{ver}.joblib",
                    is_active=True,
                    is_production=True,
                )

        # Seed predictions for feature-importance tests
        user = await DBService.get_user_by_token(db, USER_TOKEN)
        for _ in range(6):
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=REG_FI_MODEL,
                model_version="1.0.0",
                input_features={"area": 75.0, "rooms": 3.0},
                prediction_result=225000.0,
                probabilities=None,
                response_time_ms=5.0,
            )


asyncio.run(_setup_extra())


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
            files={"file": ("model.joblib", io.BytesIO(_pkl(model)), "application/octet-stream")},
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
                "file": ("model.joblib", io.BytesIO(_pkl(model)), "application/octet-stream"),
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


# ---------------------------------------------------------------------------
# POST /explain — SHAP pour régresseurs linéaires (LinearExplainer)
# ---------------------------------------------------------------------------


class TestRegressionSHAP:
    def test_explain_linear_regression_returns_200(self):
        """POST /explain sur LinearRegression → 200, model_type=linear."""
        model = _make_linear_regression()
        baseline = {"area": {"mean": 80.0, "std": 25.0, "min": 30.0, "max": 150.0},
                    "rooms": {"mean": 2.8, "std": 0.8, "min": 1.0, "max": 5.0}}
        key = _inject_cache(REG_SHAP_LR_MODEL, "1.0.0", model, feature_baseline=baseline)
        try:
            r = client.post(
                "/explain",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": REG_SHAP_LR_MODEL, "features": {"area": 75.0, "rooms": 3.0}},
            )
            assert r.status_code == 200
            result = r.json()
            assert result["model_type"] == "linear"
            assert isinstance(result["shap_values"], dict)
            assert set(result["shap_values"].keys()) == {"area", "rooms"}
            for v in result["shap_values"].values():
                assert isinstance(v, float)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_explain_ridge_regression_returns_200(self):
        """POST /explain sur Ridge → 200, model_type=linear, shap_values complet."""
        model = _make_ridge_regression()
        key = _inject_cache(REG_SHAP_RIDGE_MODEL, "1.0.0", model)
        try:
            r = client.post(
                "/explain",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": REG_SHAP_RIDGE_MODEL, "features": {"x1": 3.0, "x2": 6.0}},
            )
            assert r.status_code == 200
            result = r.json()
            assert result["model_type"] == "linear"
            assert set(result["shap_values"].keys()) == {"x1", "x2"}
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_explain_linear_regression_base_value_is_float(self):
        """base_value de LinearRegression est bien un float scalaire."""
        model = _make_linear_regression()
        key = _inject_cache(REG_SHAP_LR_MODEL, "1.0.0", model)
        try:
            r = client.post(
                "/explain",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": REG_SHAP_LR_MODEL, "features": {"area": 100.0, "rooms": 4.0}},
            )
            assert r.status_code == 200
            assert isinstance(r.json()["base_value"], float)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_predict_explain_true_linear_regression(self):
        """POST /predict?explain=true sur LinearRegression → shap_values inline non-null."""
        model = _make_linear_regression()
        key = _inject_cache(REG_SHAP_LR_MODEL, "1.0.0", model)
        try:
            r = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                params={"explain": "true"},
                json={"model_name": REG_SHAP_LR_MODEL, "features": {"area": 80.0, "rooms": 3.0}},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["shap_values"] is not None
            assert isinstance(data["shap_values"], dict)
            assert set(data["shap_values"].keys()) == {"area", "rooms"}
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# GET /models/{name}/feature-importance — régresseur linéaire
# ---------------------------------------------------------------------------


class TestRegressionFeatureImportance:
    def test_feature_importance_linear_regression_returns_200(self):
        """GET /feature-importance sur LinearRegression avec prédictions seedées → 200."""
        model = _make_linear_regression()
        key = _inject_cache(REG_FI_MODEL, "1.0.0", model)
        try:
            r = client.get(
                f"/models/{REG_FI_MODEL}/feature-importance",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                params={"last_n": 10, "days": 30},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["model_name"] == REG_FI_MODEL
            assert data["sample_size"] > 0
            fi = data["feature_importance"]
            assert set(fi.keys()) == {"area", "rooms"}
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_feature_importance_linear_regression_ranks_valid(self):
        """Rangs 1 et 2 présents, rank 1 a le mean_abs_shap le plus élevé."""
        model = _make_linear_regression()
        key = _inject_cache(REG_FI_MODEL, "1.0.0", model)
        try:
            r = client.get(
                f"/models/{REG_FI_MODEL}/feature-importance",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                params={"last_n": 10, "days": 30},
            )
            assert r.status_code == 200
            fi = r.json()["feature_importance"]
            ranks = sorted(v["rank"] for v in fi.values())
            assert ranks == list(range(1, len(fi) + 1))
            rank1 = next(v for v in fi.values() if v["rank"] == 1)
            rank2 = next(v for v in fi.values() if v["rank"] == 2)
            assert rank1["mean_abs_shap"] >= rank2["mean_abs_shap"]
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# POST /models/{name}/{version}/validate-input — régresseur
# ---------------------------------------------------------------------------


class TestRegressionInputValidation:
    def test_validate_input_all_features_present_returns_valid(self):
        """validate-input avec toutes les features du régresseur → valid=true."""
        model = _make_linear_regression()
        key = _inject_cache(REG_IV_MODEL, "1.0.0", model)
        try:
            r = client.post(
                f"/models/{REG_IV_MODEL}/1.0.0/validate-input",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"area": 80.0, "rooms": 3.0},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["valid"] is True
            assert data["errors"] == []
            assert set(data["expected_features"]) == {"area", "rooms"}
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_validate_input_missing_feature_returns_invalid(self):
        """validate-input avec une feature manquante → valid=false, erreur missing_feature."""
        model = _make_linear_regression()
        key = _inject_cache(REG_IV_MODEL, "1.0.0", model)
        try:
            r = client.post(
                f"/models/{REG_IV_MODEL}/1.0.0/validate-input",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"area": 80.0},  # 'rooms' manquant
            )
            assert r.status_code == 200
            data = r.json()
            assert data["valid"] is False
            error_types = [e["type"] for e in data["errors"]]
            assert "missing_feature" in error_types
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_validate_input_unexpected_feature_reported(self):
        """validate-input avec une feature non attendue → unexpected_feature dans errors."""
        model = _make_linear_regression()
        key = _inject_cache(REG_IV_MODEL, "1.0.0", model)
        try:
            r = client.post(
                f"/models/{REG_IV_MODEL}/1.0.0/validate-input",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"area": 80.0, "rooms": 3.0, "extra_col": 99.0},
            )
            assert r.status_code == 200
            data = r.json()
            error_types = [e["type"] for e in data["errors"]]
            assert "unexpected_feature" in error_types
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))
