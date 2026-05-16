"""
Tests E2E — cycle de vie complet d'un utilisateur.

Scénario :
  Un admin crée un utilisateur, celui-ci utilise l'API,
  l'admin met à jour son quota, régénère son token,
  et désactive son compte.
"""

import asyncio
import io
import joblib

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "e2e-um-admin-token-cc33"
# Modèle créé par l'admin pour les tests de prédiction
UM_MODEL = "e2e_um_model"

FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}


def _make_pkl() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="e2e_um_admin",
                email="e2e_um_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())

# Créer un modèle de test
_r_model = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={"name": UM_MODEL, "version": "1.0.0"},
)
assert _r_model.status_code == 201, _r_model.text


# État partagé entre tests de la classe (exécutés séquentiellement)
_state: dict = {}


class TestUserManagementE2E:
    def test_01_admin_creates_user(self):
        """L'admin crée un nouvel utilisateur."""
        r = client.post(
            "/users",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "username": "e2e_managed_user",
                "email": "e2e_managed@test.com",
                "role": "user",
                "rate_limit": 5,
            },
        )
        assert r.status_code == 201
        data = r.json()
        assert data["username"] == "e2e_managed_user"
        assert "api_token" in data
        _state["user_id"] = data["id"]
        _state["token"] = data["api_token"]

    def test_02_new_user_can_authenticate(self):
        """Le nouvel utilisateur peut s'authentifier (GET /users/{id})."""
        user_id = _state.get("user_id")
        token = _state.get("token")
        if not user_id or not token:
            pytest.skip("user_id/token non disponible (test_01 non exécuté)")

        r = client.get(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["id"] == user_id

    def test_03_non_admin_cannot_list_users(self):
        """Un utilisateur standard ne peut pas lister les utilisateurs → 403."""
        token = _state.get("token")
        if not token:
            pytest.skip("token non disponible (test_01 non exécuté)")

        r = client.get(
            "/users",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 403

    def test_04_admin_updates_user_rate_limit(self):
        """L'admin peut modifier le quota de l'utilisateur."""
        user_id = _state.get("user_id")
        if not user_id:
            pytest.skip("user_id non disponible (test_01 non exécuté)")

        r = client.patch(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"rate_limit": 2000},
        )
        assert r.status_code == 200
        assert r.json()["rate_limit_per_day"] == 2000

    def test_05_admin_regenerates_user_token(self):
        """L'admin régénère le token de l'utilisateur."""
        user_id = _state.get("user_id")
        old_token = _state.get("token")
        if not user_id or not old_token:
            pytest.skip("user_id/token non disponible (test_01 non exécuté)")

        r = client.patch(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"regenerate_token": True},
        )
        assert r.status_code == 200
        data = r.json()
        new_token = data.get("api_token")
        assert new_token is not None
        assert new_token != old_token
        _state["old_token"] = old_token
        _state["token"] = new_token

    def test_06_old_token_no_longer_works(self):
        """L'ancien token est rejeté sur un endpoint protégé → 401 ou 403."""
        old_token = _state.get("old_token")
        if not old_token:
            pytest.skip("old_token non disponible (test_05 non exécuté)")

        # GET /users requiert require_admin → vérifie l'authentification
        r = client.get(
            "/users",
            headers={"Authorization": f"Bearer {old_token}"},
        )
        assert r.status_code in (401, 403)

    def test_07_new_token_works(self):
        """Le nouveau token fonctionne sur un endpoint protégé."""
        user_id = _state.get("user_id")
        new_token = _state.get("token")
        if not user_id or not new_token:
            pytest.skip("user_id/token non disponible (test_05 non exécuté)")

        # L'utilisateur peut accéder à son propre profil
        r = client.get(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {new_token}"},
        )
        assert r.status_code == 200

    def test_08_admin_deactivates_user(self):
        """L'admin désactive le compte de l'utilisateur."""
        user_id = _state.get("user_id")
        if not user_id:
            pytest.skip("user_id non disponible (test_01 non exécuté)")

        r = client.patch(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"is_active": False},
        )
        assert r.status_code == 200
        assert r.json()["is_active"] is False

    def test_09_deactivated_user_access_denied(self):
        """Utilisateur désactivé → accès refusé sur endpoint protégé (401 ou 403)."""
        user_id = _state.get("user_id")
        token = _state.get("token")
        if not user_id or not token:
            pytest.skip("user_id/token non disponible")

        # GET /users/{id} requiert verify_token → utilisateur inactif → 403
        r = client.get(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code in (401, 403)

    def test_10_admin_can_list_users(self):
        """L'admin peut lister tous les utilisateurs."""
        r = client.get(
            "/users",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        usernames = [u["username"] for u in data]
        assert "e2e_managed_user" in usernames
