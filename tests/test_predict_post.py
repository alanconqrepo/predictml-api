"""
Tests for the POST /predict endpoint — real execution with sklearn models in cache.

Mock strategy:
  - Inject the model directly into the Redis cache via model_service._redis (key "model:name:version")
  - Create ModelMetadata entries in DB in _setup()
  - Each test cleans up the cache with try/finally to avoid interference
  - No Docker required (SQLite in-memory + models created on the fly)
"""

import asyncio
import io
import joblib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

# Tokens and model names unique to this test file
TEST_TOKEN = "test-token-predict-post-xq7z"
PP_IRIS_MODEL = "pp_iris_model"  # LogisticRegression, 3 features, predict_proba
PP_REGRESSOR_MODEL = "pp_reg_model"  # DecisionTreeRegressor, 2 features, no predict_proba
PP_NOFEAT_MODEL = "pp_nofeat_model"  # LogisticRegression without feature_names_in_
PP_VERSIONED_MODEL = "pp_versioned"  # For testing explicit model_version
MODEL_VERSION = "1.0.0"
MODEL_VERSION_V2 = "2.0.0"


# ---------------------------------------------------------------------------
# Helpers — model construction
# ---------------------------------------------------------------------------


def _make_iris_model() -> LogisticRegression:
    """LogisticRegression on DataFrame → feature_names_in_ + predict_proba."""
    x = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0], "f3": [0.1, 0.2, 0.3, 0.4]}
    )
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(x, y)


def _make_regressor_model() -> DecisionTreeRegressor:
    """DecisionTreeRegressor on DataFrame → feature_names_in_ but NO predict_proba."""
    x = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    y = [0.5, 1.5, 2.5]
    return DecisionTreeRegressor().fit(x, y)


def _make_model_no_feature_names() -> LogisticRegression:
    """LogisticRegression on numpy array → NO feature_names_in_."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(x, y)


def _inject_cache(model_name: str, version: str, model) -> str:
    """Inject a model into the Redis cache; return the key for cleanup."""
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name, version=version, confidence_threshold=None, webhook_url=None
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


# ---------------------------------------------------------------------------
# Module-level setup: user + ModelMetadata entries in DB
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        # Create the test user
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_predict_post_user",
                email="test_predict_post@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )

        # Create ModelMetadata entries (one record per model name
        # to avoid MultipleResultsFound in get_model_metadata without version)
        models_to_create = [
            (PP_IRIS_MODEL, MODEL_VERSION, True),
            (PP_REGRESSOR_MODEL, MODEL_VERSION, True),
            (PP_NOFEAT_MODEL, MODEL_VERSION, True),
            (PP_VERSIONED_MODEL, MODEL_VERSION_V2, True),
        ]
        for name, version, is_prod in models_to_create:
            existing = await DBService.get_model_metadata(db, name, version)
            if not existing:
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=version,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{version}.joblib",
                    is_active=True,
                    is_production=is_prod,
                )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_predict_post_without_auth():
    """POST /predict without Authorization header → 401/403."""
    response = client.post(
        "/predict",
        json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
    )
    assert response.status_code in [401, 403]


def test_predict_post_with_invalid_token():
    """POST /predict with invalid token → 401."""
    response = client.post(
        "/predict",
        headers={"Authorization": "Bearer invalid-token-xyz"},
        json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_predict_success_happy_path():
    """POST /predict — successful prediction: 200 with all expected fields."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.5, "f2": 2.5, "f3": 0.2}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == PP_IRIS_MODEL
        assert data["model_version"] == MODEL_VERSION
        assert data["prediction"] is not None
        assert data["id_obs"] is None
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_response_structure():
    """POST /predict — the response contains all fields from PredictionOutput."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "prediction" in data
        assert "probability" in data
        assert "id_obs" in data
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_with_id_obs():
    """POST /predict with id_obs — the id_obs field is returned in the response."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "id_obs": "patient-42",
                "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1},
            },
        )
        assert response.status_code == 200
        assert response.json()["id_obs"] == "patient-42"
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_with_explicit_version():
    """POST /predict with explicit model_version — the returned version matches."""
    model = _make_iris_model()
    key = _inject_cache(PP_VERSIONED_MODEL, MODEL_VERSION_V2, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_VERSIONED_MODEL,
                "model_version": MODEL_VERSION_V2,
                "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1},
            },
        )
        assert response.status_code == 200
        assert response.json()["model_version"] == MODEL_VERSION_V2
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_saves_to_db():
    """POST /predict — the prediction is persisted to the database."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    id_obs_value = "db-save-test-obs"
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "id_obs": id_obs_value,
                "features": {"f1": 3.0, "f2": 4.0, "f3": 0.3},
            },
        )
        assert response.status_code == 200
    finally:
        asyncio.run(model_service.clear_cache(key))

    # Verify persistence in DB via GET /predictions
    # Use params= to correctly encode special characters (+, :) in datetimes
    start = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    end = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    history = client.get(
        "/predictions",
        params={"name": PP_IRIS_MODEL, "start": start, "end": end},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert history.status_code == 200
    data = history.json()
    assert data["total"] >= 1
    ids_obs = [p["id_obs"] for p in data["predictions"]]
    assert id_obs_value in ids_obs


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


def test_predict_with_predict_proba():
    """Model with predict_proba → probability is a list of floats."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 200
        probability = response.json()["probability"]
        assert isinstance(probability, list)
        assert len(probability) > 0
        assert all(isinstance(p, float) for p in probability)
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_without_predict_proba():
    """Model without predict_proba (regressor) → probability is None."""
    model = _make_regressor_model()
    key = _inject_cache(PP_REGRESSOR_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": PP_REGRESSOR_MODEL, "features": {"f1": 1.5, "f2": 4.5}},
        )
        assert response.status_code == 200
        assert response.json()["probability"] is None
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Business errors (422 / 404)
# ---------------------------------------------------------------------------


def test_predict_missing_features_returns_422():
    """Missing features in the request → 422 with an explanatory message."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        # Only sends f1, missing f2 and f3
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.0}},
        )
        assert response.status_code == 422
        assert (
            "manquantes" in response.json()["detail"].lower()
            or "missing" in response.json()["detail"].lower()
        )
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_model_without_feature_names_in_returns_422():
    """Model without feature_names_in_ (trained on numpy) → 422 with an explanatory message."""
    model = _make_model_no_feature_names()
    key = _inject_cache(PP_NOFEAT_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": PP_NOFEAT_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
        )
        assert response.status_code == 422
        assert "feature_names_in_" in response.json()["detail"]
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_model_not_found_returns_404():
    """Non-existent model in DB → 404."""
    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"model_name": "nonexistent_model_xyzabc", "features": {"f1": 1.0}},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /predict-batch
# ---------------------------------------------------------------------------


def test_predict_batch_success_happy_path():
    """POST /predict-batch — batch of 2 observations → 200 with a list of results."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "inputs": [
                    {"features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
                    {"features": {"f1": 3.0, "f2": 4.0, "f3": 0.3}},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == PP_IRIS_MODEL
        assert data["model_version"] == MODEL_VERSION
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "probability" in pred
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_batch_with_id_obs():
    """POST /predict-batch — id_obs values are returned in the same order."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "inputs": [
                    {"features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}, "id_obs": "obs-a"},
                    {"features": {"f1": 3.0, "f2": 4.0, "f3": 0.3}, "id_obs": "obs-b"},
                ],
            },
        )
        assert response.status_code == 200
        preds = response.json()["predictions"]
        assert preds[0]["id_obs"] == "obs-a"
        assert preds[1]["id_obs"] == "obs-b"
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_batch_without_auth():
    """POST /predict-batch without Authorization header → 401/403."""
    response = client.post(
        "/predict-batch",
        json={
            "model_name": PP_IRIS_MODEL,
            "inputs": [{"features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}}],
        },
    )
    assert response.status_code in [401, 403]


def test_predict_batch_missing_features_returns_422():
    """POST /predict-batch — missing features on one item → 422."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "inputs": [
                    {"features": {"f1": 1.0}},  # missing f2 and f3
                ],
            },
        )
        assert response.status_code == 422
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_batch_model_not_found_returns_404():
    """POST /predict-batch — non-existent model → 404."""
    response = client.post(
        "/predict-batch",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={
            "model_name": "nonexistent_batch_model_xyz",
            "inputs": [{"features": {"f1": 1.0}}],
        },
    )
    assert response.status_code == 404


def test_predict_batch_without_predict_proba():
    """POST /predict-batch — regressor without predict_proba → probability=None for each item."""
    model = _make_regressor_model()
    key = _inject_cache(PP_REGRESSOR_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_REGRESSOR_MODEL,
                "inputs": [
                    {"features": {"f1": 1.0, "f2": 4.0}},
                    {"features": {"f1": 2.0, "f2": 5.0}},
                ],
            },
        )
        assert response.status_code == 200
        for pred in response.json()["predictions"]:
            assert pred["probability"] is None
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_batch_saves_to_db():
    """POST /predict-batch — all predictions are persisted to the database."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    id_obs_a = "batch-db-obs-001"
    id_obs_b = "batch-db-obs-002"
    try:
        response = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "inputs": [
                    {"features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}, "id_obs": id_obs_a},
                    {"features": {"f1": 3.0, "f2": 4.0, "f3": 0.3}, "id_obs": id_obs_b},
                ],
            },
        )
        assert response.status_code == 200
    finally:
        asyncio.run(model_service.clear_cache(key))

    start = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    end = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    history = client.get(
        "/predictions",
        params={"name": PP_IRIS_MODEL, "start": start, "end": end},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert history.status_code == 200
    data = history.json()
    ids_obs = [p["id_obs"] for p in data["predictions"]]
    assert id_obs_a in ids_obs
    assert id_obs_b in ids_obs


def test_predict_batch_empty_inputs_returns_422():
    """POST /predict-batch with empty list → 422 (min_length=1)."""
    response = client.post(
        "/predict-batch",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"model_name": PP_IRIS_MODEL, "inputs": []},
    )
    assert response.status_code == 422


def test_predict_batch_strict_validation_rejects_unexpected_features():
    """POST /predict-batch?strict_validation=true — unexpected feature on one item → 422."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict-batch?strict_validation=true",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "inputs": [
                    {"features": {"f1": 1.0, "f2": 2.0, "f3": 0.1, "extra_col": 99.0}},
                ],
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["valid"] is False
        assert any(e["type"] == "unexpected_feature" for e in detail["errors"])
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_batch_strict_validation_false_allows_unexpected_features():
    """POST /predict-batch without strict_validation — unexpected feature silently ignored → 200."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "model_name": PP_IRIS_MODEL,
                "inputs": [
                    {"features": {"f1": 1.0, "f2": 2.0, "f3": 0.1, "extra_col": 99.0}},
                ],
            },
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 1
    finally:
        asyncio.run(model_service.clear_cache(key))
