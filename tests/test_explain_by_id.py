"""
Tests pour l'endpoint GET /predictions/{id}/explain — explication SHAP post-hoc.

Stratégie de mock :
  - Injecter le modèle directement dans le cache Redis via model_service._redis
  - Créer les prédictions directement en DB via DBService
  - try/finally pour nettoyer le cache après chaque test
  - Pas de Docker requis (SQLite in-memory + modèles créés à la volée)
"""

import asyncio
import io
import joblib
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TOKEN_OWNER = "test-token-exid-owner-k7m2"
TOKEN_OTHER = "test-token-exid-other-k7m2"
TOKEN_ADMIN = "test-token-exid-admin-k7m2"
MODEL_NAME = "exid_model"
MODEL_VERSION = "1.0.0"
FEATURES = {"petal_length": 5.1, "petal_width": 1.8, "sepal_length": 6.3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rf_model() -> RandomForestClassifier:
    X = pd.DataFrame(
        {
            "petal_length": [1.0, 2.0, 3.0, 4.0],
            "petal_width": [0.5, 1.0, 1.5, 2.0],
            "sepal_length": [4.0, 5.0, 6.0, 7.0],
        }
    )
    y = [0, 1, 0, 1]
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)


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


async def _setup():
    async with _TestSessionLocal() as db:
        for token, name, email, role in [
            (TOKEN_OWNER, "exid_owner", "exid_owner@test.com", "user"),
            (TOKEN_OTHER, "exid_other", "exid_other@test.com", "user"),
            (TOKEN_ADMIN, "exid_admin", "exid_admin@test.com", "admin"),
        ]:
            if not await DBService.get_user_by_token(db, token):
                await DBService.create_user(
                    db,
                    username=name,
                    email=email,
                    api_token=token,
                    role=role,
                    rate_limit=10000,
                )
        if not await DBService.get_model_metadata(db, MODEL_NAME, MODEL_VERSION):
            await DBService.create_model_metadata(
                db,
                name=MODEL_NAME,
                version=MODEL_VERSION,
                minio_bucket="models",
                minio_object_key=f"{MODEL_NAME}/v{MODEL_VERSION}.pkl",
                is_active=True,
                is_production=True,
            )


asyncio.run(_setup())


async def _create_prediction(token: str, pred_status: str = "success") -> int:
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, token)
        p = await DBService.create_prediction(
            db=db,
            user_id=user.id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            input_features=FEATURES,
            prediction_result=0 if pred_status == "success" else None,
            probabilities=[0.7, 0.3] if pred_status == "success" else None,
            response_time_ms=15.0,
            status=pred_status,
            error_message=None if pred_status == "success" else "some error",
        )
        return p.id


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_explain_by_id_without_auth():
    """GET /predictions/{id}/explain sans Authorization → 401."""
    response = client.get("/predictions/1/explain")
    assert response.status_code == 401


def test_explain_by_id_invalid_token():
    """GET /predictions/{id}/explain avec token invalide → 401."""
    response = client.get(
        "/predictions/1/explain",
        headers={"Authorization": "Bearer invalid-token-xyz"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 404 / access control
# ---------------------------------------------------------------------------


def test_explain_by_id_not_found():
    """GET /predictions/{id}/explain — prédiction inexistante → 404."""
    response = client.get(
        "/predictions/999999999/explain",
        headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
    )
    assert response.status_code == 404


def test_explain_by_id_other_user_returns_403():
    """Un utilisateur ne peut pas expliquer la prédiction d'un autre → 403."""
    pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
    response = client.get(
        f"/predictions/{pred_id}/explain",
        headers={"Authorization": f"Bearer {TOKEN_OTHER}"},
    )
    assert response.status_code == 403


def test_explain_by_id_admin_can_explain_any():
    """Un admin peut expliquer n'importe quelle prédiction → 200."""
    model = _make_rf_model()
    key = _inject_cache(MODEL_NAME, MODEL_VERSION, model)
    try:
        pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
        response = client.get(
            f"/predictions/{pred_id}/explain",
            headers={"Authorization": f"Bearer {TOKEN_ADMIN}"},
        )
        assert response.status_code == 200
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Guards — status != success
# ---------------------------------------------------------------------------


def test_explain_by_id_error_status_returns_422():
    """Prédiction en erreur (status='error') → 422."""
    pred_id = asyncio.run(_create_prediction(TOKEN_OWNER, pred_status="error"))
    response = client.get(
        f"/predictions/{pred_id}/explain",
        headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
    )
    assert response.status_code == 422
    assert "success" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Happy path — RandomForestClassifier (tree)
# ---------------------------------------------------------------------------


def test_explain_by_id_success():
    """GET /predictions/{id}/explain — RandomForest → 200 avec tous les champs."""
    model = _make_rf_model()
    key = _inject_cache(MODEL_NAME, MODEL_VERSION, model)
    try:
        pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
        response = client.get(
            f"/predictions/{pred_id}/explain",
            headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == MODEL_NAME
        assert data["model_version"] == MODEL_VERSION
        assert data["model_type"] == "tree"
        assert isinstance(data["shap_values"], dict)
        assert isinstance(data["base_value"], float)
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_explain_by_id_shap_values_match_stored_features():
    """shap_values contient exactement les features stockées en base."""
    model = _make_rf_model()
    key = _inject_cache(MODEL_NAME, MODEL_VERSION, model)
    try:
        pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
        response = client.get(
            f"/predictions/{pred_id}/explain",
            headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
        )
        assert response.status_code == 200
        shap_values = response.json()["shap_values"]
        assert set(shap_values.keys()) == set(FEATURES.keys())
        for v in shap_values.values():
            assert isinstance(v, float)
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_explain_by_id_response_structure():
    """Tous les champs de ExplainOutput sont présents dans la réponse."""
    model = _make_rf_model()
    key = _inject_cache(MODEL_NAME, MODEL_VERSION, model)
    try:
        pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
        response = client.get(
            f"/predictions/{pred_id}/explain",
            headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
        )
        assert response.status_code == 200
        data = response.json()
        for field in ("model_name", "model_version", "prediction", "shap_values", "base_value", "model_type"):
            assert field in data, f"Champ manquant : {field}"
    finally:
        asyncio.run(model_service.clear_cache(key))


def test_explain_by_id_prediction_value_in_response():
    """Le champ 'prediction' correspond à la valeur stockée en base."""
    model = _make_rf_model()
    key = _inject_cache(MODEL_NAME, MODEL_VERSION, model)
    try:
        pred_id = asyncio.run(_create_prediction(TOKEN_OWNER))
        response = client.get(
            f"/predictions/{pred_id}/explain",
            headers={"Authorization": f"Bearer {TOKEN_OWNER}"},
        )
        assert response.status_code == 200
        assert response.json()["prediction"] == 0
    finally:
        asyncio.run(model_service.clear_cache(key))
