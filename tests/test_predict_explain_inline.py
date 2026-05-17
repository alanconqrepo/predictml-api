"""
Tests pour POST /predict?explain=true — explication SHAP inline dans la réponse.

Stratégie de mock :
  - Injecter les modèles directement dans le cache Redis via model_service._redis
  - Créer les entrées ModelMetadata en DB dans _setup()
  - try/finally pour nettoyer le cache après chaque test
"""

import asyncio
import io
import joblib
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-pred-inline-shap-z3k9"
RF_MODEL = "inline_rf_model"
SVM_MODEL = "inline_svm_model"
MODEL_VERSION = "1.0.0"
FEATURES = {"f1": 1.5, "f2": 2.5, "f3": 0.2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rf_model() -> RandomForestClassifier:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0], "f3": [0.1, 0.2, 0.3, 0.4]})
    y = [0, 1, 0, 1]
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)


def _make_svm_model() -> SVC:
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0], "f3": [0.1, 0.2, 0.3, 0.4]})
    y = [0, 1, 0, 1]
    return SVC().fit(X, y)


def _inject_cache(model_name: str, version: str, model, feature_baseline=None) -> str:
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            feature_baseline=feature_baseline,
            webhook_url=None,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="inline_shap_user",
                email="inline_shap@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )
        for name in [RF_MODEL, SVM_MODEL]:
            if not await DBService.get_model_metadata(db, name, MODEL_VERSION):
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=MODEL_VERSION,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{MODEL_VERSION}.joblib",
                    is_active=True,
                    is_production=True,
                )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# explain=false (default) — champs shap absents ou null
# ---------------------------------------------------------------------------


def test_predict_without_explain_shap_fields_are_null():
    """POST /predict sans ?explain → shap_values et shap_base_value sont null."""
    model = _make_rf_model()
    key = _inject_cache(RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": RF_MODEL, "features": FEATURES},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["shap_values"] is None
        assert data["shap_base_value"] is None
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# explain=true — modèle arbre supporté (RandomForest)
# ---------------------------------------------------------------------------


def test_predict_explain_tree_returns_shap_values():
    """POST /predict?explain=true — RandomForest → shap_values et shap_base_value renseignés."""
    model = _make_rf_model()
    key = _inject_cache(RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict?explain=true",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": RF_MODEL, "features": FEATURES},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["shap_values"], dict)
        assert isinstance(data["shap_base_value"], float)
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_explain_shap_keys_match_features():
    """shap_values contient exactement les mêmes clés que les features envoyées."""
    model = _make_rf_model()
    key = _inject_cache(RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict?explain=true",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": RF_MODEL, "features": FEATURES},
        )
        assert response.status_code == 200
        shap_values = response.json()["shap_values"]
        assert set(shap_values.keys()) == set(FEATURES.keys())
        for v in shap_values.values():
            assert isinstance(v, float)
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_predict_explain_response_still_contains_prediction():
    """Avec ?explain=true, les champs prédiction/probabilité sont toujours présents."""
    model = _make_rf_model()
    key = _inject_cache(RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict?explain=true",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": RF_MODEL, "features": FEATURES},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_name" in data
        assert "model_version" in data
        assert data["model_name"] == RF_MODEL
        assert data["model_version"] == MODEL_VERSION
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# explain=true — modèle non supporté (SVC) → silencieux, pas d'erreur
# ---------------------------------------------------------------------------


def test_predict_explain_unsupported_model_returns_null_shap():
    """POST /predict?explain=true — SVC non supporté → shap_values null, pas d'erreur 4xx/5xx."""
    model = _make_svm_model()
    key = _inject_cache(SVM_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict?explain=true",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": SVM_MODEL, "features": FEATURES},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["shap_values"] is None
        assert data["shap_base_value"] is None
        assert "prediction" in data
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# explain=true préservé avec strict_validation
# ---------------------------------------------------------------------------


def test_predict_explain_combined_with_strict_validation():
    """?explain=true et ?strict_validation=true peuvent être combinés."""
    model = _make_rf_model()
    key = _inject_cache(RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict?explain=true&strict_validation=true",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": RF_MODEL, "features": FEATURES},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["shap_values"], dict)
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# explain=false explicite équivalent à défaut
# ---------------------------------------------------------------------------


def test_predict_explain_false_explicit_gives_null_shap():
    """?explain=false explicite → même comportement que sans le paramètre."""
    model = _make_rf_model()
    key = _inject_cache(RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/predict?explain=false",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": RF_MODEL, "features": FEATURES},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["shap_values"] is None
        assert data["shap_base_value"] is None
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Prédiction persistée en DB même avec explain=true
# ---------------------------------------------------------------------------


def test_predict_explain_true_still_logs_to_db():
    """Avec ?explain=true, la prédiction est toujours persistée en base."""
    from datetime import datetime, timedelta, timezone

    model = _make_rf_model()
    key = _inject_cache(RF_MODEL, MODEL_VERSION, model)
    try:
        start = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        end = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()

        before = client.get(
            "/predictions",
            params={"name": RF_MODEL, "start": start, "end": end},
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        count_before = before.json()["total"]

        client.post(
            "/predict?explain=true",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": RF_MODEL, "features": FEATURES},
        )

        after = client.get(
            "/predictions",
            params={"name": RF_MODEL, "start": start, "end": end},
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert after.json()["total"] == count_before + 1
    finally:
        asyncio.run(model_service.clear_cache(key))
