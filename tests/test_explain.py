"""
Tests pour l'endpoint POST /explain — explication SHAP locale.

Stratégie de mock :
  - Injecter le modèle directement dans model_service.models_cache (clé "name:version")
  - Créer les entrées ModelMetadata en DB dans _setup()
  - try/finally pour nettoyer le cache après chaque test
  - Pas de Docker requis (SQLite in-memory + modèles créés à la volée)
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-explain-xq9z"
EX_RF_MODEL = "ex_rf_model"          # RandomForestClassifier (tree)
EX_LR_MODEL = "ex_lr_model"          # LogisticRegression (linear)
EX_REG_MODEL = "ex_reg_model"        # DecisionTreeRegressor (tree, régression)
EX_SVM_MODEL = "ex_svm_model"        # SVC — non supporté par SHAP
EX_NOFEAT_MODEL = "ex_nofeat_model"  # LogisticRegression sans feature_names_in_
MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers — construction de modèles
# ---------------------------------------------------------------------------

def _make_rf_model() -> RandomForestClassifier:
    """RandomForestClassifier sur DataFrame → feature_names_in_ + predict_proba."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0], "f3": [0.1, 0.2, 0.3, 0.4]})
    y = [0, 1, 0, 1]
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)


def _make_lr_model() -> LogisticRegression:
    """LogisticRegression sur DataFrame → feature_names_in_ + predict_proba."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0], "f3": [0.1, 0.2, 0.3, 0.4]})
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _make_regressor_model() -> DecisionTreeRegressor:
    """DecisionTreeRegressor sur DataFrame → feature_names_in_, pas de predict_proba."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    y = [0.5, 1.5, 2.5]
    return DecisionTreeRegressor(random_state=42).fit(X, y)


def _make_svm_model() -> SVC:
    """SVC sur DataFrame → type non supporté par SHAP dans ce projet."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0], "f3": [0.1, 0.2, 0.3, 0.4]})
    y = [0, 1, 0, 1]
    return SVC().fit(X, y)


def _make_model_no_feature_names() -> LogisticRegression:
    """LogisticRegression sur numpy array → PAS de feature_names_in_."""
    X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0]])
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_cache(model_name: str, version: str, model, feature_baseline=None) -> str:
    key = f"{model_name}:{version}"
    model_service.models_cache[key] = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            feature_baseline=feature_baseline,
        ),
    }
    return key


# ---------------------------------------------------------------------------
# Setup module-level
# ---------------------------------------------------------------------------

async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_explain_user",
                email="test_explain@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )
        for name in [EX_RF_MODEL, EX_LR_MODEL, EX_REG_MODEL, EX_SVM_MODEL, EX_NOFEAT_MODEL]:
            existing = await DBService.get_model_metadata(db, name, MODEL_VERSION)
            if not existing:
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=MODEL_VERSION,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{MODEL_VERSION}.pkl",
                    is_active=True,
                    is_production=True,
                )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_explain_without_auth():
    """POST /explain sans header Authorization → 401/403."""
    response = client.post(
        "/explain",
        json={"model_name": EX_RF_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
    )
    assert response.status_code in [401, 403]


def test_explain_with_invalid_token():
    """POST /explain avec token invalide → 401."""
    response = client.post(
        "/explain",
        headers={"Authorization": "Bearer invalid-token-xyz"},
        json={"model_name": EX_RF_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Happy path — modèle arbre (RandomForest)
# ---------------------------------------------------------------------------

def test_explain_tree_model_success():
    """POST /explain — RandomForestClassifier → 200 avec shap_values, base_value, model_type=tree."""
    model = _make_rf_model()
    key = _inject_cache(EX_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_RF_MODEL, "features": {"f1": 1.5, "f2": 2.5, "f3": 0.2}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "tree"
        assert isinstance(data["shap_values"], dict)
        assert isinstance(data["base_value"], float)
    finally:
        model_service.models_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Happy path — modèle linéaire (LogisticRegression)
# ---------------------------------------------------------------------------

def test_explain_linear_model_success():
    """POST /explain — LogisticRegression → 200 avec model_type=linear."""
    model = _make_lr_model()
    key = _inject_cache(EX_LR_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_LR_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "linear"
        assert isinstance(data["shap_values"], dict)
    finally:
        model_service.models_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Structure de la réponse
# ---------------------------------------------------------------------------

def test_explain_response_structure():
    """POST /explain — tous les champs de ExplainOutput sont présents."""
    model = _make_rf_model()
    key = _inject_cache(EX_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_RF_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "prediction" in data
        assert "shap_values" in data
        assert "base_value" in data
        assert "model_type" in data
        assert data["model_name"] == EX_RF_MODEL
        assert data["model_version"] == MODEL_VERSION
    finally:
        model_service.models_cache.pop(key, None)


def test_explain_shap_values_have_all_features():
    """POST /explain — shap_values contient exactement toutes les features du modèle."""
    model = _make_rf_model()
    key = _inject_cache(EX_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_RF_MODEL, "features": {"f1": 2.0, "f2": 3.0, "f3": 0.3}},
        )
        assert response.status_code == 200
        shap_values = response.json()["shap_values"]
        assert set(shap_values.keys()) == {"f1", "f2", "f3"}
        for v in shap_values.values():
            assert isinstance(v, float)
    finally:
        model_service.models_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Régresseur (DecisionTreeRegressor)
# ---------------------------------------------------------------------------

def test_explain_regressor():
    """POST /explain — DecisionTreeRegressor → 200 avec model_type=tree."""
    model = _make_regressor_model()
    key = _inject_cache(EX_REG_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_REG_MODEL, "features": {"f1": 1.5, "f2": 4.5}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "tree"
        assert set(data["shap_values"].keys()) == {"f1", "f2"}
    finally:
        model_service.models_cache.pop(key, None)


# ---------------------------------------------------------------------------
# LinearExplainer avec feature_baseline
# ---------------------------------------------------------------------------

def test_explain_linear_with_feature_baseline():
    """POST /explain — LogisticRegression avec feature_baseline → 200, pas de zéros par défaut."""
    model = _make_lr_model()
    baseline = {"f1": {"mean": 2.5}, "f2": {"mean": 3.5}, "f3": {"mean": 0.25}}
    key = _inject_cache(EX_LR_MODEL, MODEL_VERSION, model, feature_baseline=baseline)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_LR_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 200
        assert response.json()["model_type"] == "linear"
    finally:
        model_service.models_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Erreurs métier
# ---------------------------------------------------------------------------

def test_explain_missing_features_returns_422():
    """POST /explain — features manquantes → 422 avec message explicatif."""
    model = _make_rf_model()
    key = _inject_cache(EX_RF_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_RF_MODEL, "features": {"f1": 1.0}},  # manque f2, f3
        )
        assert response.status_code == 422
        assert "manquantes" in response.json()["detail"].lower()
    finally:
        model_service.models_cache.pop(key, None)


def test_explain_model_not_found_returns_404():
    """POST /explain — modèle inexistant en DB → 404."""
    response = client.post(
        "/explain",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"model_name": "nonexistent_explain_model_xyz", "features": {"f1": 1.0}},
    )
    assert response.status_code == 404


def test_explain_unsupported_model_type_returns_422():
    """POST /explain — SVC non supporté par SHAP → 422 avec message clair."""
    model = _make_svm_model()
    key = _inject_cache(EX_SVM_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_SVM_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 422
        assert "SVC" in response.json()["detail"]
    finally:
        model_service.models_cache.pop(key, None)


def test_explain_model_without_feature_names_in_returns_422():
    """POST /explain — modèle sans feature_names_in_ → 422."""
    model = _make_model_no_feature_names()
    key = _inject_cache(EX_NOFEAT_MODEL, MODEL_VERSION, model)
    try:
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_NOFEAT_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 422
        assert "feature_names_in_" in response.json()["detail"]
    finally:
        model_service.models_cache.pop(key, None)


# ---------------------------------------------------------------------------
# /explain ne logue PAS en DB
# ---------------------------------------------------------------------------

def test_explain_does_not_save_to_db():
    """POST /explain — aucune prédiction loggée en DB (contraire à /predict)."""
    from datetime import datetime, timezone, timedelta

    model = _make_rf_model()
    key = _inject_cache(EX_RF_MODEL, MODEL_VERSION, model)
    try:
        # Compter les prédictions existantes
        start = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        end = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
        before = client.get(
            "/predictions",
            params={"name": EX_RF_MODEL, "start": start, "end": end},
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        count_before = before.json()["total"]

        # Appeler /explain
        response = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": EX_RF_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
        )
        assert response.status_code == 200

        # Vérifier que le compteur n'a pas bougé
        end2 = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
        after = client.get(
            "/predictions",
            params={"name": EX_RF_MODEL, "start": start, "end": end2},
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert after.json()["total"] == count_before
    finally:
        model_service.models_cache.pop(key, None)
