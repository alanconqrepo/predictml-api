"""
Tests pour l'endpoint POST /predict — exécution réelle avec modèles sklearn en cache.

Stratégie de mock :
  - Injecter le modèle directement dans model_service.models_cache (clé "name:version")
  - Créer les entrées ModelMetadata en DB dans _setup()
  - Chaque test nettoie le cache avec try/finally pour éviter les interférences
  - Pas de Docker requis (SQLite in-memory + modèles créés à la volée)
"""
import asyncio
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

# Tokens et noms de modèles uniques à ce fichier de test
TEST_TOKEN = "test-token-predict-post-xq7z"
PP_IRIS_MODEL = "pp_iris_model"       # LogisticRegression, 3 features, predict_proba
PP_REGRESSOR_MODEL = "pp_reg_model"   # DecisionTreeRegressor, 2 features, pas de predict_proba
PP_NOFEAT_MODEL = "pp_nofeat_model"   # LogisticRegression sans feature_names_in_
PP_VERSIONED_MODEL = "pp_versioned"   # Pour tester model_version explicite
MODEL_VERSION = "1.0.0"
MODEL_VERSION_V2 = "2.0.0"


# ---------------------------------------------------------------------------
# Helpers — construction de modèles
# ---------------------------------------------------------------------------

def _make_iris_model() -> LogisticRegression:
    """LogisticRegression sur DataFrame → feature_names_in_ + predict_proba."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0], "f3": [0.1, 0.2, 0.3, 0.4]})
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _make_regressor_model() -> DecisionTreeRegressor:
    """DecisionTreeRegressor sur DataFrame → feature_names_in_ mais PAS predict_proba."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    y = [0.5, 1.5, 2.5]
    return DecisionTreeRegressor().fit(X, y)


def _make_model_no_feature_names() -> LogisticRegression:
    """LogisticRegression sur numpy array → PAS de feature_names_in_."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_cache(model_name: str, version: str, model) -> str:
    """Injecte un modèle dans le cache ; retourne la clé pour le nettoyage."""
    key = f"{model_name}:{version}"
    model_service.models_cache[key] = {
        "model": model,
        "metadata": SimpleNamespace(name=model_name, version=version),
    }
    return key


# ---------------------------------------------------------------------------
# Setup module-level : user + entrées ModelMetadata en DB
# ---------------------------------------------------------------------------

async def _setup():
    async with _TestSessionLocal() as db:
        # Créer l'utilisateur de test
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_predict_post_user",
                email="test_predict_post@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )

        # Créer les entrées ModelMetadata (un seul enregistrement par nom de modèle
        # pour éviter MultipleResultsFound dans get_model_metadata sans version)
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
                    minio_object_key=f"{name}/v{version}.pkl",
                    is_active=True,
                    is_production=is_prod,
                )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_predict_post_without_auth():
    """POST /predict sans header Authorization → 401/403."""
    response = client.post(
        "/predict",
        json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.0, "f2": 2.0, "f3": 0.1}},
    )
    assert response.status_code in [401, 403]


def test_predict_post_with_invalid_token():
    """POST /predict avec token invalide → 401."""
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
    """POST /predict — prédiction réussie : 200 avec tous les champs attendus."""
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
        model_service.models_cache.pop(key, None)


def test_predict_response_structure():
    """POST /predict — la réponse contient tous les champs du PredictionOutput."""
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
        model_service.models_cache.pop(key, None)


def test_predict_with_id_obs():
    """POST /predict avec id_obs — le champ id_obs est retourné dans la réponse."""
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
        model_service.models_cache.pop(key, None)


def test_predict_with_explicit_version():
    """POST /predict avec model_version explicite — la version retournée correspond."""
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
        model_service.models_cache.pop(key, None)


def test_predict_saves_to_db():
    """POST /predict — la prédiction est persistée en base de données."""
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
        model_service.models_cache.pop(key, None)

    # Vérifier la persistance en DB via GET /predictions
    # Utiliser params= pour encoder correctement les caractères spéciaux (+, :) dans les datetimes
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
    """Modèle avec predict_proba → probability est une liste de floats."""
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
        model_service.models_cache.pop(key, None)


def test_predict_without_predict_proba():
    """Modèle sans predict_proba (régresseur) → probability est None."""
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
        model_service.models_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Erreurs métier (422 / 404)
# ---------------------------------------------------------------------------

def test_predict_missing_features_returns_422():
    """Features manquantes dans la requête → 422 avec message explicatif."""
    model = _make_iris_model()
    key = _inject_cache(PP_IRIS_MODEL, MODEL_VERSION, model)
    try:
        # Envoie seulement f1, manque f2 et f3
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"model_name": PP_IRIS_MODEL, "features": {"f1": 1.0}},
        )
        assert response.status_code == 422
        assert "manquantes" in response.json()["detail"].lower() or "missing" in response.json()["detail"].lower()
    finally:
        model_service.models_cache.pop(key, None)


def test_predict_model_without_feature_names_in_returns_422():
    """Modèle sans feature_names_in_ (entraîné sur numpy) → 422 avec message explicatif."""
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
        model_service.models_cache.pop(key, None)


def test_predict_model_not_found_returns_404():
    """Modèle inexistant en DB → 404."""
    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"model_name": "nonexistent_model_xyzabc", "features": {"f1": 1.0}},
    )
    assert response.status_code == 404
