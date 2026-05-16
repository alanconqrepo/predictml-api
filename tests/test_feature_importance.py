"""
Tests pour GET /models/{name}/feature-importance — SHAP agrégé global.

Stratégie :
  - Modèles injectés directement dans le cache Redis via model_service._redis
  - Prédictions créées en DB pour alimenter l'agrégation SHAP
  - Pas de Docker requis (SQLite in-memory)
"""

import asyncio
import io
import joblib
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-fi-q7r2"
FI_RF_MODEL = "fi_rf_model"
FI_LR_MODEL = "fi_lr_model"
FI_SVM_MODEL = "fi_svm_unsupported"
FI_NOFEAT_MODEL = "fi_nofeat_model"
FI_EMPTY_MODEL = "fi_empty_model"
MODEL_VERSION = "1.0.0"

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _make_rf_model() -> RandomForestClassifier:
    X = pd.DataFrame({"petal_length": [1.0, 2.0, 3.0, 4.0], "petal_width": [0.3, 0.5, 0.7, 0.9]})
    y = [0, 1, 0, 1]
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)


def _make_lr_model() -> LogisticRegression:
    X = pd.DataFrame({"petal_length": [1.0, 2.0, 3.0, 4.0], "petal_width": [0.3, 0.5, 0.7, 0.9]})
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _make_svm_model() -> SVC:
    X = pd.DataFrame({"petal_length": [1.0, 2.0, 3.0, 4.0], "petal_width": [0.3, 0.5, 0.7, 0.9]})
    y = [0, 1, 0, 1]
    return SVC().fit(X, y)


def _make_model_no_feature_names() -> LogisticRegression:
    X = np.array([[1.0, 0.3], [2.0, 0.5], [3.0, 0.7], [4.0, 0.9]])
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _inject_cache(model_name: str, version: str, model, feature_baseline=None) -> str:
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            feature_baseline=feature_baseline,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            user = await DBService.create_user(
                db,
                username="test_fi_user",
                email="test_fi@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )
        else:
            user = await DBService.get_user_by_token(db, TEST_TOKEN)

        for name in [FI_RF_MODEL, FI_LR_MODEL, FI_SVM_MODEL, FI_NOFEAT_MODEL, FI_EMPTY_MODEL]:
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

        # Seed predictions for FI_RF_MODEL and FI_LR_MODEL
        for _ in range(5):
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=FI_RF_MODEL,
                model_version=MODEL_VERSION,
                input_features={"petal_length": 2.5, "petal_width": 0.6},
                prediction_result=1,
                probabilities=[0.3, 0.7],
                response_time_ms=10.0,
            )
        for _ in range(5):
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=FI_LR_MODEL,
                model_version=MODEL_VERSION,
                input_features={"petal_length": 1.5, "petal_width": 0.4},
                prediction_result=0,
                probabilities=[0.8, 0.2],
                response_time_ms=5.0,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_feature_importance_no_auth():
    """GET without Authorization header → 401/403."""
    response = client.get(f"/models/{FI_RF_MODEL}/feature-importance")
    assert response.status_code in [401, 403]


def test_feature_importance_invalid_token():
    """GET with invalid token → 401."""
    response = client.get(
        f"/models/{FI_RF_MODEL}/feature-importance",
        headers={"Authorization": "Bearer invalid-xyz"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 404 — unknown model
# ---------------------------------------------------------------------------


def test_feature_importance_unknown_model():
    """GET on non-existent model → 404."""
    response = client.get(
        "/models/totally_unknown_model_xyz/feature-importance",
        headers=AUTH,
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 422 — model without feature_names_in_
# ---------------------------------------------------------------------------


def test_feature_importance_no_feature_names():
    """Model trained on numpy array → no feature_names_in_ → 422."""
    model = _make_model_no_feature_names()
    key = _inject_cache(FI_NOFEAT_MODEL, MODEL_VERSION, model)
    try:
        response = client.get(
            f"/models/{FI_NOFEAT_MODEL}/feature-importance",
            headers=AUTH,
        )
        assert response.status_code == 422
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Empty predictions window → sample_size=0
# ---------------------------------------------------------------------------


def test_feature_importance_no_predictions():
    """Model exists but no predictions in the window → sample_size=0, empty dict."""
    model = _make_rf_model()
    key = _inject_cache(FI_EMPTY_MODEL, MODEL_VERSION, model)
    try:
        response = client.get(
            f"/models/{FI_EMPTY_MODEL}/feature-importance",
            headers=AUTH,
            params={"days": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == FI_EMPTY_MODEL
        assert data["sample_size"] == 0
        assert data["feature_importance"] == {}
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Happy path — RandomForestClassifier
# ---------------------------------------------------------------------------


def test_feature_importance_rf_success():
    """RF model with seeded predictions → returns ranked feature_importance."""
    model = _make_rf_model()
    key = _inject_cache(FI_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.get(
            f"/models/{FI_RF_MODEL}/feature-importance",
            headers=AUTH,
            params={"last_n": 10, "days": 30},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["model_name"] == FI_RF_MODEL
        assert data["version"] == MODEL_VERSION
        assert data["sample_size"] > 0

        fi = data["feature_importance"]
        assert set(fi.keys()) == {"petal_length", "petal_width"}

        for feat_data in fi.values():
            assert "mean_abs_shap" in feat_data
            assert "rank" in feat_data
            assert feat_data["mean_abs_shap"] >= 0.0

        ranks = [fi[f]["rank"] for f in fi]
        assert sorted(ranks) == list(range(1, len(fi) + 1))
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Happy path — LogisticRegression (linear explainer)
# ---------------------------------------------------------------------------


def test_feature_importance_lr_success():
    """LogisticRegression → linear SHAP explainer → valid ranked importance."""
    model = _make_lr_model()
    feature_baseline = {
        "petal_length": {"mean": 2.5, "std": 1.0, "min": 1.0, "max": 4.0},
        "petal_width": {"mean": 0.6, "std": 0.2, "min": 0.3, "max": 0.9},
    }
    key = _inject_cache(FI_LR_MODEL, MODEL_VERSION, model, feature_baseline=feature_baseline)
    try:
        response = client.get(
            f"/models/{FI_LR_MODEL}/feature-importance",
            headers=AUTH,
            params={"last_n": 10, "days": 30},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["model_name"] == FI_LR_MODEL
        assert data["sample_size"] > 0

        fi = data["feature_importance"]
        assert set(fi.keys()) == {"petal_length", "petal_width"}

        ranks = sorted(fi[f]["rank"] for f in fi)
        assert ranks == list(range(1, len(fi) + 1))

        top = min(fi.items(), key=lambda kv: kv[1]["rank"])
        assert (
            fi[top[0]]["mean_abs_shap"]
            >= fi[max(fi.items(), key=lambda kv: kv[1]["rank"])[0]]["mean_abs_shap"]
        )
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Ranking invariant — top feature has highest mean_abs_shap
# ---------------------------------------------------------------------------


def test_feature_importance_ranking_order():
    """Rank 1 always has the highest mean_abs_shap."""
    model = _make_rf_model()
    key = _inject_cache(FI_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.get(
            f"/models/{FI_RF_MODEL}/feature-importance",
            headers=AUTH,
            params={"last_n": 10, "days": 30},
        )
        assert response.status_code == 200
        fi = response.json()["feature_importance"]
        if len(fi) > 1:
            rank1 = next(v for v in fi.values() if v["rank"] == 1)
            rank2 = next(v for v in fi.values() if v["rank"] == 2)
            assert rank1["mean_abs_shap"] >= rank2["mean_abs_shap"]
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Query param — last_n limits sample
# ---------------------------------------------------------------------------


def test_feature_importance_last_n_respected():
    """sample_size ≤ last_n."""
    model = _make_rf_model()
    key = _inject_cache(FI_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.get(
            f"/models/{FI_RF_MODEL}/feature-importance",
            headers=AUTH,
            params={"last_n": 2, "days": 30},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sample_size"] <= 2
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Query param validation — out-of-range values
# ---------------------------------------------------------------------------


def test_feature_importance_last_n_above_max():
    """last_n > 500 → 422 validation error."""
    model = _make_rf_model()
    key = _inject_cache(FI_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.get(
            f"/models/{FI_RF_MODEL}/feature-importance",
            headers=AUTH,
            params={"last_n": 501},
        )
        assert response.status_code == 422
    finally:
        asyncio.run(model_service.clear_cache(key))
