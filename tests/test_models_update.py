"""
Tests pour l'endpoint PATCH /models/{name}/{version}
"""
import asyncio
import io
import pickle

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-patch-models"
MODEL_A = "patch_model_a"
MODEL_B = "patch_model_b"


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_patch_models",
                email="test_patch@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


def _create_model(name: str, version: str = "1.0.0", **extra_data) -> dict:
    """Crée un modèle via POST /models et retourne la réponse JSON."""
    data = {"name": name, "version": version, **extra_data}
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data=data,
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_update_model_without_auth():
    """PATCH /models sans auth → 401/403"""
    _create_model(f"{MODEL_A}_noauth")
    response = client.patch(
        f"/models/{MODEL_A}_noauth/1.0.0",
        json={"description": "test"},
    )
    assert response.status_code in [401, 403]


def test_update_model_with_invalid_token():
    """PATCH /models avec token invalide → 401"""
    _create_model(f"{MODEL_A}_badtoken")
    response = client.patch(
        f"/models/{MODEL_A}_badtoken/1.0.0",
        headers={"Authorization": "Bearer invalid"},
        json={"description": "test"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Cas nominaux
# ---------------------------------------------------------------------------

def test_update_description():
    """PATCH → description mise à jour"""
    _create_model(f"{MODEL_A}_desc")
    response = client.patch(
        f"/models/{MODEL_A}_desc/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"description": "nouvelle description"},
    )
    assert response.status_code == 200
    assert response.json()["description"] == "nouvelle description"


def test_update_accuracy_and_features_count():
    """PATCH → accuracy et features_count mis à jour"""
    _create_model(f"{MODEL_A}_metrics")
    response = client.patch(
        f"/models/{MODEL_A}_metrics/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"accuracy": 0.95, "features_count": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accuracy"] == pytest.approx(0.95)
    assert data["features_count"] == 4


def test_update_classes():
    """PATCH → classes mis à jour"""
    _create_model(f"{MODEL_A}_classes")
    response = client.patch(
        f"/models/{MODEL_A}_classes/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"classes": [0, 1, 2]},
    )
    assert response.status_code == 200
    assert response.json()["classes"] == [0, 1, 2]


def test_update_partial_fields_only():
    """PATCH → seuls les champs fournis sont modifiés, les autres restent inchangés"""
    _create_model(f"{MODEL_A}_partial", description="originale", accuracy="0.80")
    response = client.patch(
        f"/models/{MODEL_A}_partial/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"accuracy": 0.99},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accuracy"] == pytest.approx(0.99)
    assert data["description"] == "originale"  # inchangé


# ---------------------------------------------------------------------------
# is_production — exclusivité par modèle
# ---------------------------------------------------------------------------

def test_set_is_production_true():
    """PATCH is_production=true → modèle marqué production"""
    _create_model(f"{MODEL_B}_prod_single")
    response = client.patch(
        f"/models/{MODEL_B}_prod_single/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )
    assert response.status_code == 200
    assert response.json()["is_production"] is True


def test_is_production_exclusive_across_versions():
    """
    Quand v2.0.0 passe is_production=True,
    v1.0.0 (qui était production) doit passer à False automatiquement.
    """
    model_name = f"{MODEL_B}_exclusive"

    # Créer v1 et la passer en production
    _create_model(model_name, version="1.0.0")
    client.patch(
        f"/models/{model_name}/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )

    # Créer v2 et la passer en production
    _create_model(model_name, version="2.0.0")
    r2 = client.patch(
        f"/models/{model_name}/2.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )
    assert r2.status_code == 200
    assert r2.json()["is_production"] is True

    # Vérifier que v1 n'est plus en production
    r1 = client.get("/models")
    models = {m["name"] + "_" + m["version"]: m for m in r1.json()}
    assert models[f"{model_name}_1.0.0"]["is_production"] is False
    assert models[f"{model_name}_2.0.0"]["is_production"] is True


def test_set_is_production_false():
    """PATCH is_production=false → modèle retiré de la production"""
    _create_model(f"{MODEL_B}_unprod")
    client.patch(
        f"/models/{MODEL_B}_unprod/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": True},
    )
    response = client.patch(
        f"/models/{MODEL_B}_unprod/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"is_production": False},
    )
    assert response.status_code == 200
    assert response.json()["is_production"] is False


# ---------------------------------------------------------------------------
# Cas d'erreur
# ---------------------------------------------------------------------------

def test_update_model_not_found():
    """PATCH sur un modèle inexistant → 404"""
    response = client.patch(
        "/models/inexistant_model/9.9.9",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"description": "test"},
    )
    assert response.status_code == 404


def test_update_model_empty_body():
    """PATCH avec body vide → 200, aucun champ modifié"""
    _create_model(f"{MODEL_A}_emptybody", description="stable")
    response = client.patch(
        f"/models/{MODEL_A}_emptybody/1.0.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={},
    )
    assert response.status_code == 200
    assert response.json()["description"] == "stable"
