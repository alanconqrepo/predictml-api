"""
Tests pour les endpoints /account-requests
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-admin-accrequests"
USER_TOKEN = "test-token-user-accrequests"
PREFIX = "accreq"


async def _setup():
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
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username=f"{PREFIX}_user",
                email=f"{PREFIX}_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())

# ---------------------------------------------------------------------------
# POST /account-requests (public)
# ---------------------------------------------------------------------------

def test_submit_request_no_auth():
    """POST /account-requests sans token → 201 (endpoint public)"""
    r = client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_req1", "email": f"{PREFIX}_req1@example.com"},
    )
    assert r.status_code == 201
    data = r.json()
    assert data["username"] == f"{PREFIX}_req1"
    assert data["status"] == "pending"
    assert data["id"] > 0


def test_submit_request_default_role():
    """POST /account-requests sans role_requested → défaut 'user'"""
    r = client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_req_role", "email": f"{PREFIX}_req_role@example.com"},
    )
    assert r.status_code == 201
    assert r.json()["role_requested"] == "user"


def test_submit_request_readonly_role():
    """POST /account-requests avec role_requested=readonly → 201"""
    r = client.post(
        "/account-requests",
        json={
            "username": f"{PREFIX}_readonly_req",
            "email": f"{PREFIX}_readonly_req@example.com",
            "role_requested": "readonly",
        },
    )
    assert r.status_code == 201
    assert r.json()["role_requested"] == "readonly"


def test_submit_request_with_message():
    """POST /account-requests avec message → message conservé"""
    r = client.post(
        "/account-requests",
        json={
            "username": f"{PREFIX}_with_msg",
            "email": f"{PREFIX}_with_msg@example.com",
            "message": "Je fais partie de l'équipe Data.",
        },
    )
    assert r.status_code == 201
    assert r.json()["message"] == "Je fais partie de l'équipe Data."


def test_submit_request_duplicate_pending_email():
    """POST /account-requests avec même email déjà pending → 409"""
    email = f"{PREFIX}_dup@example.com"
    client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_dup_a", "email": email},
    )
    r = client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_dup_b", "email": email},
    )
    assert r.status_code == 409
    assert "en attente" in r.json()["detail"]


def test_submit_request_existing_user_email():
    """POST /account-requests avec email d'un compte existant → 409"""
    r = client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_conflict", "email": f"{PREFIX}_admin@test.com"},
    )
    assert r.status_code == 409


def test_submit_request_invalid_email():
    """POST /account-requests avec email invalide → 422"""
    r = client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_bademail", "email": "not-an-email"},
    )
    assert r.status_code == 422


def test_submit_request_username_too_short():
    """POST /account-requests avec username < 3 → 422"""
    r = client.post(
        "/account-requests",
        json={"username": "ab", "email": f"{PREFIX}_short@example.com"},
    )
    assert r.status_code == 422


def test_submit_request_admin_role_forbidden():
    """POST /account-requests avec role_requested=admin → 422 (non autorisé)"""
    r = client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_wantadmin", "email": f"{PREFIX}_wantadmin@example.com", "role_requested": "admin"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /account-requests
# ---------------------------------------------------------------------------

def test_list_requests_no_auth():
    """GET /account-requests sans token → 401/403"""
    r = client.get("/account-requests")
    assert r.status_code in [401, 403]


def test_list_requests_non_admin():
    """GET /account-requests token non-admin → 403"""
    r = client.get(
        "/account-requests",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


def test_list_requests_admin():
    """GET /account-requests admin → 200 liste"""
    r = client.get(
        "/account-requests",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_list_requests_filter_pending():
    """GET /account-requests?status=pending → uniquement les pending"""
    r = client.get(
        "/account-requests",
        params={"status": "pending"},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    for req in r.json():
        assert req["status"] == "pending"


# ---------------------------------------------------------------------------
# GET /account-requests/pending-count
# ---------------------------------------------------------------------------

def test_pending_count_no_auth():
    """GET /account-requests/pending-count sans token → 401/403"""
    r = client.get("/account-requests/pending-count")
    assert r.status_code in [401, 403]


def test_pending_count_admin():
    """GET /account-requests/pending-count admin → 200 avec pending_count"""
    r = client.get(
        "/account-requests/pending-count",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    assert "pending_count" in r.json()
    assert isinstance(r.json()["pending_count"], int)


# ---------------------------------------------------------------------------
# PATCH /account-requests/{id}/approve
# ---------------------------------------------------------------------------

def _create_pending_request(suffix: str) -> int:
    r = client.post(
        "/account-requests",
        json={"username": f"{PREFIX}_{suffix}", "email": f"{PREFIX}_{suffix}@example.com"},
    )
    assert r.status_code == 201
    return r.json()["id"]


def test_approve_non_admin():
    """PATCH /account-requests/{id}/approve token non-admin → 403"""
    req_id = _create_pending_request("approve_nonauth")
    r = client.patch(
        f"/account-requests/{req_id}/approve",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


def test_approve_not_found():
    """PATCH /account-requests/99999/approve → 404"""
    r = client.patch(
        "/account-requests/99999/approve",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 404


def test_approve_success():
    """PATCH /account-requests/{id}/approve admin → user créé + token retourné"""
    req_id = _create_pending_request("approve_ok")
    r = client.patch(
        f"/account-requests/{req_id}/approve",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["request_id"] == req_id
    assert "created_user" in data
    assert "api_token" in data["created_user"]
    assert len(data["created_user"]["api_token"]) > 10


def test_approved_user_can_authenticate():
    """L'utilisateur créé par approbation peut s'authentifier"""
    req_id = _create_pending_request("approve_auth")
    approve_r = client.patch(
        f"/account-requests/{req_id}/approve",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert approve_r.status_code == 200
    new_token = approve_r.json()["created_user"]["api_token"]

    me_r = client.get("/users/me", headers={"Authorization": f"Bearer {new_token}"})
    assert me_r.status_code == 200
    assert me_r.json()["username"] == f"{PREFIX}_approve_auth"


def test_approve_twice():
    """PATCH /account-requests/{id}/approve deux fois → 409 la seconde"""
    req_id = _create_pending_request("approve_twice")
    client.patch(
        f"/account-requests/{req_id}/approve",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    r = client.patch(
        f"/account-requests/{req_id}/approve",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# PATCH /account-requests/{id}/reject
# ---------------------------------------------------------------------------

def test_reject_non_admin():
    """PATCH /account-requests/{id}/reject token non-admin → 403"""
    req_id = _create_pending_request("reject_nonauth")
    r = client.patch(
        f"/account-requests/{req_id}/reject",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
        json={},
    )
    assert r.status_code == 403


def test_reject_success_no_reason():
    """PATCH /account-requests/{id}/reject sans raison → 200 status=rejected"""
    req_id = _create_pending_request("reject_noreason")
    r = client.patch(
        f"/account-requests/{req_id}/reject",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "rejected"
    assert data["rejection_reason"] is None


def test_reject_success_with_reason():
    """PATCH /account-requests/{id}/reject avec raison → raison stockée"""
    req_id = _create_pending_request("reject_reason")
    r = client.patch(
        f"/account-requests/{req_id}/reject",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"reason": "Équipe non identifiée."},
    )
    assert r.status_code == 200
    assert r.json()["rejection_reason"] == "Équipe non identifiée."


def test_reject_already_approved():
    """PATCH /account-requests/{id}/reject sur une demande déjà approuvée → 409"""
    req_id = _create_pending_request("reject_approved")
    client.patch(
        f"/account-requests/{req_id}/approve",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    r = client.patch(
        f"/account-requests/{req_id}/reject",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={},
    )
    assert r.status_code == 409


def test_reject_not_found():
    """PATCH /account-requests/99998/reject → 404"""
    r = client.patch(
        "/account-requests/99998/reject",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={},
    )
    assert r.status_code == 404
