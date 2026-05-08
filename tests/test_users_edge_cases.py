"""
Tests pour les cas limites des endpoints /users non couverts par test_users.py.

Couvre :
- GET /users/{id}/usage avec days=0  → fallback 30 (code : if days < 1: days = 30)
- GET /users/{id}/usage avec days=-1 → fallback 30
- GET /users/{id}/usage avec days=365 → valeur conservée (pas de limite maximale)
- POST /users avec rate_limit=0      → doit être accepté (pas de contrainte min)
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-users-edge-cc55"
USER_TOKEN = "test-token-users-edge-user-cc55"
PREFIX = "users_edge"

_user_id: int = 0


async def _setup():
    global _user_id
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username=f"{PREFIX}_admin",
                email=f"{PREFIX}_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        u = await DBService.get_user_by_token(db, USER_TOKEN)
        if not u:
            u = await DBService.create_user(
                db,
                username=f"{PREFIX}_user",
                email=f"{PREFIX}_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )
        _user_id = u.id


asyncio.run(_setup())

_ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


# ---------------------------------------------------------------------------
# GET /users/{id}/usage — jours invalides → fallback à 30
# ---------------------------------------------------------------------------


def test_usage_days_zero_falls_back_to_30():
    """`days=0` → le code applique le fallback : period_days retourné vaut 30."""
    r = client.get(f"/users/{_user_id}/usage?days=0", headers=_ADMIN_HEADERS)
    assert r.status_code == 200
    assert r.json()["period_days"] == 30


def test_usage_negative_days_falls_back_to_30():
    """`days=-5` → fallback identique à days=0."""
    r = client.get(f"/users/{_user_id}/usage?days=-5", headers=_ADMIN_HEADERS)
    assert r.status_code == 200
    assert r.json()["period_days"] == 30


# ---------------------------------------------------------------------------
# GET /users/{id}/usage — grande valeur acceptée
# ---------------------------------------------------------------------------


def test_usage_large_days_accepted():
    """`days=365` → conservé tel quel (aucune limite maximale côté endpoint)."""
    r = client.get(f"/users/{_user_id}/usage?days=365", headers=_ADMIN_HEADERS)
    assert r.status_code == 200
    assert r.json()["period_days"] == 365


# ---------------------------------------------------------------------------
# GET /users/{id}/usage — accès propre
# ---------------------------------------------------------------------------


def test_usage_user_accesses_own_stats():
    """Un utilisateur regular peut consulter ses propres stats → 200."""
    r = client.get(
        f"/users/{_user_id}/usage",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == _user_id


def test_usage_response_has_by_model_and_by_day():
    """Réponse contient by_model et by_day même si aucune prédiction."""
    r = client.get(f"/users/{_user_id}/usage", headers=_ADMIN_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["by_model"], list)
    assert isinstance(data["by_day"], list)


# ---------------------------------------------------------------------------
# POST /users — rate_limit minimal
# ---------------------------------------------------------------------------


def test_create_user_with_rate_limit_1():
    """rate_limit=1 est une valeur limite acceptable → 201."""
    r = client.post(
        "/users",
        headers=_ADMIN_HEADERS,
        json={
            "username": f"{PREFIX}_ratelimit1",
            "email": f"{PREFIX}_ratelimit1@test.com",
            "rate_limit": 1,
        },
    )
    assert r.status_code == 201
    assert r.json()["rate_limit_per_day"] == 1


def test_create_user_email_must_be_unique():
    """Deux utilisateurs avec le même email → 409 au second."""
    email = f"{PREFIX}_unique_email@test.com"
    r1 = client.post(
        "/users",
        headers=_ADMIN_HEADERS,
        json={"username": f"{PREFIX}_emailu1", "email": email},
    )
    assert r1.status_code == 201

    r2 = client.post(
        "/users",
        headers=_ADMIN_HEADERS,
        json={"username": f"{PREFIX}_emailu2", "email": email},
    )
    assert r2.status_code == 409
