"""
Tests for the /users endpoints.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-admin-users"
USER_TOKEN = "test-token-regular-users"
USERNAME_PREFIX = "test_user_route"


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username=f"{USERNAME_PREFIX}_admin",
                email=f"{USERNAME_PREFIX}_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username=f"{USERNAME_PREFIX}_regular",
                email=f"{USERNAME_PREFIX}_regular@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# POST /users
# ---------------------------------------------------------------------------

def test_create_user_without_auth():
    """POST /users without auth → 401/403."""
    r = client.post("/users", json={"username": "new", "email": "new@test.com"})
    assert r.status_code in [401, 403]


def test_create_user_as_non_admin():
    """POST /users with non-admin token → 403."""
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
        json={"username": "new_user", "email": "new_user@test.com"},
    )
    assert r.status_code == 403


def test_create_user_success():
    """POST /users admin → 201 with generated token."""
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": f"{USERNAME_PREFIX}_created", "email": f"{USERNAME_PREFIX}_created@test.com"},
    )
    assert r.status_code == 201
    data = r.json()
    assert data["username"] == f"{USERNAME_PREFIX}_created"
    assert data["role"] == "user"
    assert data["is_active"] is True
    assert "api_token" in data
    assert len(data["api_token"]) > 10
    assert "id" in data


def test_create_user_with_role_and_rate_limit():
    """POST /users with custom role and rate_limit → 201."""
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "username": f"{USERNAME_PREFIX}_readonly",
            "email": f"{USERNAME_PREFIX}_readonly@test.com",
            "role": "readonly",
            "rate_limit": 500,
        },
    )
    assert r.status_code == 201
    data = r.json()
    assert data["role"] == "readonly"
    assert data["rate_limit_per_day"] == 500


def test_create_user_duplicate():
    """POST /users with existing username/email → 409."""
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": f"{USERNAME_PREFIX}_admin", "email": "other@test.com"},
    )
    assert r.status_code == 409


def test_create_user_invalid_role():
    """POST /users with invalid role → 422."""
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": "bad_role", "email": "bad@test.com", "role": "superuser"},
    )
    assert r.status_code == 422


def test_create_user_invalid_email():
    """POST /users with invalid email → 422."""
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": "bademail", "email": "not-an-email"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /users
# ---------------------------------------------------------------------------

def test_list_users_without_auth():
    """GET /users without auth → 401/403."""
    r = client.get("/users")
    assert r.status_code in [401, 403]


def test_list_users_as_non_admin():
    """GET /users with non-admin token → 403."""
    r = client.get("/users", headers={"Authorization": f"Bearer {USER_TOKEN}"})
    assert r.status_code == 403


def test_list_users_success():
    """GET /users admin → 200 with list of users."""
    r = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    usernames = [u["username"] for u in data]
    assert f"{USERNAME_PREFIX}_admin" in usernames
    assert f"{USERNAME_PREFIX}_regular" in usernames


def test_list_users_limit():
    """GET /users?limit=1 → 200 with exactly 1 user."""
    r = client.get("/users?limit=1", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_list_users_skip():
    """GET /users?skip=N → returns fewer users than a request without skip."""
    total = client.get("/users?limit=500", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    skipped = client.get(
        f"/users?skip={len(total)}&limit=500", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    ).json()
    assert len(skipped) == 0


def test_list_users_limit_max_enforced():
    """GET /users?limit=501 → 422 (exceeds the allowed maximum of 500)."""
    r = client.get("/users?limit=501", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 422


def test_list_users_skip_negative_rejected():
    """GET /users?skip=-1 → 422."""
    r = client.get("/users?skip=-1", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 422


def test_list_users_limit_zero_rejected():
    """GET /users?limit=0 → 422."""
    r = client.get("/users?limit=0", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /users/{user_id}
# ---------------------------------------------------------------------------

def test_get_user_without_auth():
    """GET /users/{id} without auth → 401/403."""
    r = client.get("/users/1")
    assert r.status_code in [401, 403]


def test_get_user_self():
    """A user can retrieve their own profile."""
    # Get the regular user's id via the admin list
    users = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    regular = next(u for u in users if u["username"] == f"{USERNAME_PREFIX}_regular")

    r = client.get(
        f"/users/{regular['id']}",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["username"] == f"{USERNAME_PREFIX}_regular"


def test_get_user_other_as_non_admin():
    """A non-admin user cannot view another user's profile → 403."""
    users = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    admin = next(u for u in users if u["username"] == f"{USERNAME_PREFIX}_admin")

    r = client.get(
        f"/users/{admin['id']}",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


def test_get_user_as_admin():
    """Admin can view any profile."""
    users = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    regular = next(u for u in users if u["username"] == f"{USERNAME_PREFIX}_regular")

    r = client.get(
        f"/users/{regular['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.json()["id"] == regular["id"]


def test_get_user_not_found():
    """GET /users/{id} with non-existent id → 404."""
    r = client.get("/users/999999", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /users/{user_id}
# ---------------------------------------------------------------------------

def test_delete_user_without_auth():
    """DELETE /users/{id} without auth → 401/403."""
    r = client.delete("/users/1")
    assert r.status_code in [401, 403]


def test_delete_user_as_non_admin():
    """DELETE /users/{id} non-admin → 403."""
    users = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    regular = next(u for u in users if u["username"] == f"{USERNAME_PREFIX}_regular")

    r = client.delete(
        f"/users/{regular['id']}",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


def test_delete_user_self():
    """An admin cannot delete themselves → 400."""
    users = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    admin = next(u for u in users if u["username"] == f"{USERNAME_PREFIX}_admin")

    r = client.delete(
        f"/users/{admin['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 400


def test_delete_user_not_found():
    """DELETE /users/{id} with non-existent id → 404."""
    r = client.delete("/users/999999", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 404


def test_delete_user_success():
    """DELETE /users/{id} admin → 204, user not accessible afterwards."""
    # Create a user to delete
    create_r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": f"{USERNAME_PREFIX}_to_delete", "email": f"{USERNAME_PREFIX}_to_delete@test.com"},
    )
    assert create_r.status_code == 201
    user_id = create_r.json()["id"]

    r = client.delete(f"/users/{user_id}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 204

    # No longer accessible
    r2 = client.get(f"/users/{user_id}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r2.status_code == 404


def test_delete_user_twice():
    """DELETE the same user twice → 204 then 404."""
    create_r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": f"{USERNAME_PREFIX}_to_delete2", "email": f"{USERNAME_PREFIX}_to_delete2@test.com"},
    )
    user_id = create_r.json()["id"]

    r1 = client.delete(f"/users/{user_id}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r1.status_code == 204

    r2 = client.delete(f"/users/{user_id}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r2.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /users/{user_id}
# ---------------------------------------------------------------------------

def _create_patchable_user(suffix: str) -> dict:
    """Create a temporary user for PATCH tests."""
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": f"{USERNAME_PREFIX}_patch_{suffix}", "email": f"{USERNAME_PREFIX}_patch_{suffix}@test.com"},
    )
    assert r.status_code == 201
    return r.json()


def test_patch_user_without_auth():
    """PATCH /users/{id} without auth → 401/403."""
    r = client.patch("/users/1", json={"is_active": False})
    assert r.status_code in [401, 403]


def test_patch_user_as_non_admin():
    """PATCH /users/{id} non-admin → 403."""
    user = _create_patchable_user("nonAdmin")
    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
        json={"is_active": False},
    )
    assert r.status_code == 403


def test_patch_user_not_found():
    """PATCH /users/{id} with non-existent id → 404."""
    r = client.patch(
        "/users/999999",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_active": False},
    )
    assert r.status_code == 404


def test_patch_user_toggle_inactive():
    """PATCH /users/{id} → deactivate a user."""
    user = _create_patchable_user("toggleOff")
    assert user["is_active"] is True

    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_active": False},
    )
    assert r.status_code == 200
    assert r.json()["is_active"] is False


def test_patch_user_toggle_active():
    """PATCH /users/{id} → reactivate a deactivated user."""
    user = _create_patchable_user("toggleOn")

    # Deactivate first
    client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_active": False},
    )

    # Reactivate
    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_active": True},
    )
    assert r.status_code == 200
    assert r.json()["is_active"] is True


def test_patch_user_change_role():
    """PATCH /users/{id} → change role."""
    user = _create_patchable_user("changeRole")
    assert user["role"] == "user"

    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"role": "readonly"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "readonly"


def test_patch_user_invalid_role():
    """PATCH /users/{id} with invalid role → 422."""
    user = _create_patchable_user("badRole")
    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"role": "superuser"},
    )
    assert r.status_code == 422


def test_patch_user_change_rate_limit():
    """PATCH /users/{id} → update rate limit."""
    user = _create_patchable_user("rateLimit")

    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"rate_limit": 250},
    )
    assert r.status_code == 200
    assert r.json()["rate_limit_per_day"] == 250


def test_patch_user_regenerate_token():
    """PATCH /users/{id} regenerate_token=True → new different token."""
    user = _create_patchable_user("regenToken")
    original_token = user["api_token"]

    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"regenerate_token": True},
    )
    assert r.status_code == 200
    new_token = r.json()["api_token"]
    assert new_token != original_token
    assert len(new_token) > 10


def test_patch_user_partial_update():
    """PATCH /users/{id} → partial update (only rate_limit changes)."""
    user = _create_patchable_user("partial")
    original_role = user["role"]

    r = client.patch(
        f"/users/{user['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"rate_limit": 42},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["rate_limit_per_day"] == 42
    assert data["role"] == original_role  # unchanged


def test_patch_user_cannot_deactivate_self():
    """An admin cannot deactivate themselves → 400."""
    users = client.get("/users", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).json()
    admin = next(u for u in users if u["username"] == f"{USERNAME_PREFIX}_admin")

    r = client.patch(
        f"/users/{admin['id']}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_active": False},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /users/me
# ---------------------------------------------------------------------------

def test_get_me_without_auth():
    """GET /users/me without auth → 401/403."""
    r = client.get("/users/me")
    assert r.status_code in [401, 403]


def test_get_me_as_user():
    """GET /users/me with user token → 200 with correct profile."""
    r = client.get("/users/me", headers={"Authorization": f"Bearer {USER_TOKEN}"})
    assert r.status_code == 200
    data = r.json()
    assert data["username"] == f"{USERNAME_PREFIX}_regular"
    assert data["role"] == "user"
    assert "api_token" in data
    assert "id" in data
    assert "email" in data


def test_get_me_as_admin():
    """GET /users/me with admin token → 200 with admin profile."""
    r = client.get("/users/me", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 200
    data = r.json()
    assert data["username"] == f"{USERNAME_PREFIX}_admin"
    assert data["role"] == "admin"


def test_get_me_returns_own_token():
    """GET /users/me returns the authenticated user's token."""
    r = client.get("/users/me", headers={"Authorization": f"Bearer {USER_TOKEN}"})
    assert r.status_code == 200
    assert r.json()["api_token"] == USER_TOKEN


# ---------------------------------------------------------------------------
# GET /users/me/quota
# ---------------------------------------------------------------------------

def test_get_my_quota_without_auth():
    """GET /users/me/quota without auth → 401/403."""
    r = client.get("/users/me/quota")
    assert r.status_code in [401, 403]


def test_get_my_quota_structure():
    """GET /users/me/quota → 200 with the 4 expected fields."""
    r = client.get("/users/me/quota", headers={"Authorization": f"Bearer {USER_TOKEN}"})
    assert r.status_code == 200
    data = r.json()
    assert "rate_limit_per_day" in data
    assert "used_today" in data
    assert "remaining_today" in data
    assert "reset_at" in data


def test_get_my_quota_values():
    """GET /users/me/quota → consistent values."""
    r = client.get("/users/me/quota", headers={"Authorization": f"Bearer {USER_TOKEN}"})
    assert r.status_code == 200
    data = r.json()
    assert data["rate_limit_per_day"] == 10000
    assert data["used_today"] >= 0
    assert data["remaining_today"] == max(0, data["rate_limit_per_day"] - data["used_today"])


def test_get_my_quota_reset_at_is_tomorrow():
    """GET /users/me/quota → reset_at is tomorrow at midnight UTC."""
    from datetime import date, timedelta, timezone
    import dateutil.parser

    r = client.get("/users/me/quota", headers={"Authorization": f"Bearer {USER_TOKEN}"})
    assert r.status_code == 200
    reset_at = dateutil.parser.isoparse(r.json()["reset_at"])
    tomorrow = date.today() + timedelta(days=1)
    assert reset_at.date() == tomorrow


def test_get_my_quota_as_admin():
    """GET /users/me/quota also works for an admin."""
    r = client.get("/users/me/quota", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 200
    data = r.json()
    assert data["rate_limit_per_day"] == 10000
