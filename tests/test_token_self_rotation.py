"""
Tests pour POST /users/me/regenerate-token (rotation de token self-service)
"""

import asyncio

from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

# Token admin stable (jamais rotaté dans ces tests)
ADMIN_TOKEN = "test-token-admin-selfregen"
PREFIX = "selfregen"

# Tokens pour les tests de rotation — chaque test a son propre compte
TOKEN_USER = "test-sr-user-token"
TOKEN_READONLY = "test-sr-readonly-token"
TOKEN_ISOLATE = "test-sr-isolate-token"
TOKEN_UNIQUE = "test-sr-unique-token"


async def _setup():
    async with _TestSessionLocal() as db:
        accounts = [
            (ADMIN_TOKEN, f"{PREFIX}_admin", "admin"),
            (TOKEN_USER, f"{PREFIX}_user_rot", "user"),
            (TOKEN_READONLY, f"{PREFIX}_readonly_rot", "readonly"),
            (TOKEN_ISOLATE, f"{PREFIX}_isolate", "user"),
            (TOKEN_UNIQUE, f"{PREFIX}_unique", "user"),
        ]
        for token, username, role in accounts:
            if not await DBService.get_user_by_token(db, token):
                await DBService.create_user(
                    db,
                    username=username,
                    email=f"{username}@test.com",
                    api_token=token,
                    role=role,
                    rate_limit=10000,
                )


asyncio.run(_setup())

# ---------------------------------------------------------------------------
# POST /users/me/regenerate-token
# ---------------------------------------------------------------------------

def test_regenerate_token_no_auth():
    """POST /users/me/regenerate-token sans token → 401/403"""
    r = client.post("/users/me/regenerate-token")
    assert r.status_code in [401, 403]


def test_regenerate_token_as_user():
    """POST /users/me/regenerate-token utilisateur → 200 nouveau token différent"""
    r = client.post(
        "/users/me/regenerate-token",
        headers={"Authorization": f"Bearer {TOKEN_USER}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "api_token" in data
    assert data["api_token"] != TOKEN_USER
    assert len(data["api_token"]) > 10


def test_regenerate_token_as_readonly():
    """POST /users/me/regenerate-token readonly → 200 (tous les rôles autorisés)"""
    r = client.post(
        "/users/me/regenerate-token",
        headers={"Authorization": f"Bearer {TOKEN_READONLY}"},
    )
    assert r.status_code == 200
    assert r.json()["api_token"] != TOKEN_READONLY


def test_old_token_invalidated_after_rotation():
    """Après rotation, l'ancien token n'authentifie plus"""
    # TOKEN_ISOLATE n'a pas encore été rotaté dans cette session de test
    rotate_r = client.post(
        "/users/me/regenerate-token",
        headers={"Authorization": f"Bearer {TOKEN_ISOLATE}"},
    )
    assert rotate_r.status_code == 200
    new_token = rotate_r.json()["api_token"]

    # L'ancien token ne fonctionne plus
    me_with_old = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {TOKEN_ISOLATE}"},
    )
    assert me_with_old.status_code in [401, 403]

    # Le nouveau token fonctionne
    me_with_new = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {new_token}"},
    )
    assert me_with_new.status_code == 200


def test_new_token_different_each_rotation():
    """Chaque rotation produit un token différent"""
    r1 = client.post(
        "/users/me/regenerate-token",
        headers={"Authorization": f"Bearer {TOKEN_UNIQUE}"},
    )
    assert r1.status_code == 200
    token_b = r1.json()["api_token"]

    r2 = client.post(
        "/users/me/regenerate-token",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert r2.status_code == 200
    token_c = r2.json()["api_token"]

    assert TOKEN_UNIQUE != token_b
    assert token_b != token_c
    assert TOKEN_UNIQUE != token_c
