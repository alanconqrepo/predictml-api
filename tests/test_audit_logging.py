"""
Tests for audit logging on sensitive admin operations.

Verifies that audit_log() is called with the correct action, actor_id,
and resource string at each instrumented endpoint.
"""

import asyncio
import io
import pickle
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

ADMIN_TOKEN = "test-token-audit-admin-x9q3"
USER_TOKEN = "test-token-audit-user-m7k2"
MODEL_PREFIX = "audit_model"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="audit_admin",
                email="audit_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="audit_user",
                email="audit_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": name, "version": version},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _get_admin_id() -> int:
    r = client.get("/users/me", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    return r.json()["id"]


# ---------------------------------------------------------------------------
# model.upload
# ---------------------------------------------------------------------------


def test_audit_model_upload():
    name = f"{MODEL_PREFIX}_upload_a1"
    admin_id = _get_admin_id()
    with patch("src.api.models.audit_log") as mock_audit:
        _create_model(name)
    mock_audit.assert_called_once_with(
        "model.upload", actor_id=admin_id, resource=f"{name}:1.0.0"
    )


def test_audit_model_upload_not_called_on_failure():
    """Duplicate upload (409) must not emit an audit log."""
    name = f"{MODEL_PREFIX}_upload_dup"
    _create_model(name)
    with patch("src.api.models.audit_log") as mock_audit:
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
            data={"name": name, "version": "1.0.0"},
        )
    assert r.status_code == 409
    mock_audit.assert_not_called()


# ---------------------------------------------------------------------------
# model.delete (single version)
# ---------------------------------------------------------------------------


def test_audit_model_delete_version():
    name = f"{MODEL_PREFIX}_del_v1"
    admin_id = _get_admin_id()
    _create_model(name)
    with patch("src.api.models.audit_log") as mock_audit:
        r = client.delete(
            f"/models/{name}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 204
    mock_audit.assert_called_once_with(
        "model.delete", actor_id=admin_id, resource=f"{name}:1.0.0"
    )


def test_audit_model_delete_version_not_called_on_404():
    with patch("src.api.models.audit_log") as mock_audit:
        r = client.delete(
            f"/models/{MODEL_PREFIX}_nonexistent/9.9.9",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 404
    mock_audit.assert_not_called()


# ---------------------------------------------------------------------------
# model.delete_all
# ---------------------------------------------------------------------------


def test_audit_model_delete_all_versions():
    name = f"{MODEL_PREFIX}_del_all"
    admin_id = _get_admin_id()
    _create_model(name, "1.0.0")
    _create_model(name, "2.0.0")
    with patch("src.api.models.audit_log") as mock_audit:
        r = client.delete(
            f"/models/{name}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 200
    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args
    assert call_kwargs[0][0] == "model.delete_all"
    assert call_kwargs[1]["actor_id"] == admin_id
    assert call_kwargs[1]["resource"] == name
    assert set(call_kwargs[1]["details"]["versions"]) == {"1.0.0", "2.0.0"}


# ---------------------------------------------------------------------------
# model.deprecate
# ---------------------------------------------------------------------------


def test_audit_model_deprecate():
    name = f"{MODEL_PREFIX}_deprecate"
    admin_id = _get_admin_id()
    _create_model(name)
    with patch("src.api.models.audit_log") as mock_audit:
        r = client.patch(
            f"/models/{name}/1.0.0/deprecate",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 200
    mock_audit.assert_called_once_with(
        "model.deprecate", actor_id=admin_id, resource=f"{name}:1.0.0"
    )


# ---------------------------------------------------------------------------
# model.policy_update
# ---------------------------------------------------------------------------


def test_audit_model_policy_update():
    name = f"{MODEL_PREFIX}_policy"
    admin_id = _get_admin_id()
    _create_model(name)
    with patch("src.api.models.audit_log") as mock_audit:
        r = client.patch(
            f"/models/{name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"auto_promote": True, "min_accuracy": 0.9, "min_sample_validation": 10},
        )
    assert r.status_code == 200
    mock_audit.assert_called_once_with(
        "model.policy_update", actor_id=admin_id, resource=name
    )


# ---------------------------------------------------------------------------
# model.schedule_update
# ---------------------------------------------------------------------------


def test_audit_model_schedule_update():
    name = f"{MODEL_PREFIX}_sched"
    admin_id = _get_admin_id()
    _create_model(name)
    with patch("src.api.models.audit_log") as mock_audit:
        r = client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "0 3 * * 1", "lookback_days": 30, "auto_promote": False, "enabled": True},
        )
    assert r.status_code == 200
    mock_audit.assert_called_once()
    assert mock_audit.call_args[0][0] == "model.schedule_update"
    assert mock_audit.call_args[1]["actor_id"] == admin_id
    assert mock_audit.call_args[1]["resource"] == f"{name}:1.0.0"
    assert mock_audit.call_args[1]["details"]["cron"] == "0 3 * * 1"


# ---------------------------------------------------------------------------
# retrain.trigger
# ---------------------------------------------------------------------------

VALID_TRAIN_SCRIPT = """\
import os
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)
with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(json.dumps({"accuracy": 0.97, "f1_score": 0.96}))
"""


async def _mock_exec_success(*args, **kwargs):
    """Subprocess mock: writes a dummy model file and returns exit code 0."""
    from unittest.mock import AsyncMock

    env = kwargs.get("env", {})
    output_path = env.get("OUTPUT_MODEL_PATH", "")
    if output_path:
        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=200).fit(X, y)
        with open(output_path, "wb") as f:
            pickle.dump(model, f)

    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(
        return_value=(b'{"accuracy": 0.97, "f1_score": 0.96}\n', b"")
    )
    proc.kill = MagicMock()
    return proc


def test_audit_retrain_trigger():
    name = f"{MODEL_PREFIX}_retrain"
    admin_id = _get_admin_id()
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={
            "file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
            "train_file": ("train.py", io.BytesIO(VALID_TRAIN_SCRIPT.encode()), "text/x-python"),
        },
        data={"name": name, "version": "1.0.0"},
    )
    assert r.status_code == 201

    _minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()

    with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec_success), \
         patch("src.api.models.audit_log") as mock_audit:
        r = client.post(
            f"/models/{name}/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31", "new_version": "1.1.0"},
        )
    assert r.status_code == 200
    assert r.json()["success"] is True
    mock_audit.assert_called_once()
    assert mock_audit.call_args[0][0] == "retrain.trigger"
    assert mock_audit.call_args[1]["actor_id"] == admin_id
    assert mock_audit.call_args[1]["resource"] == f"{name}:1.0.0"
    assert mock_audit.call_args[1]["details"]["new_version"] == "1.1.0"


# ---------------------------------------------------------------------------
# user.create
# ---------------------------------------------------------------------------


def test_audit_user_create():
    admin_id = _get_admin_id()
    with patch("src.api.users.audit_log") as mock_audit:
        r = client.post(
            "/users",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"username": "audit_new_user_q7", "email": "audit_new_user_q7@test.com"},
        )
    assert r.status_code == 201
    new_id = r.json()["id"]
    mock_audit.assert_called_once_with(
        "user.create",
        actor_id=admin_id,
        resource=f"user:{new_id}",
        details={"username": "audit_new_user_q7"},
    )


def test_audit_user_create_not_called_on_conflict():
    """Duplicate username → 409 must not emit an audit log."""
    client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": "audit_dup_user", "email": "audit_dup1@test.com"},
    )
    with patch("src.api.users.audit_log") as mock_audit:
        r = client.post(
            "/users",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"username": "audit_dup_user", "email": "audit_dup2@test.com"},
        )
    assert r.status_code == 409
    mock_audit.assert_not_called()


# ---------------------------------------------------------------------------
# user.delete
# ---------------------------------------------------------------------------


def test_audit_user_delete():
    admin_id = _get_admin_id()
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": "audit_to_delete", "email": "audit_to_delete@test.com"},
    )
    target_id = r.json()["id"]
    with patch("src.api.users.audit_log") as mock_audit:
        r = client.delete(
            f"/users/{target_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 204
    mock_audit.assert_called_once_with(
        "user.delete", actor_id=admin_id, resource=f"user:{target_id}"
    )


def test_audit_user_delete_not_called_on_404():
    with patch("src.api.users.audit_log") as mock_audit:
        r = client.delete(
            "/users/999999",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 404
    mock_audit.assert_not_called()


# ---------------------------------------------------------------------------
# user.token_regen
# ---------------------------------------------------------------------------


def test_audit_user_token_regen():
    admin_id = _get_admin_id()
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": "audit_regen_user", "email": "audit_regen@test.com"},
    )
    target_id = r.json()["id"]
    with patch("src.api.users.audit_log") as mock_audit:
        r = client.patch(
            f"/users/{target_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"regenerate_token": True},
        )
    assert r.status_code == 200
    mock_audit.assert_called_once_with(
        "user.token_regen", actor_id=admin_id, resource=f"user:{target_id}"
    )


# ---------------------------------------------------------------------------
# user.update
# ---------------------------------------------------------------------------


def test_audit_user_update():
    admin_id = _get_admin_id()
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": "audit_update_user", "email": "audit_update@test.com"},
    )
    target_id = r.json()["id"]
    with patch("src.api.users.audit_log") as mock_audit:
        r = client.patch(
            f"/users/{target_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"rate_limit": 500},
        )
    assert r.status_code == 200
    mock_audit.assert_called_once_with(
        "user.update", actor_id=admin_id, resource=f"user:{target_id}"
    )


def test_audit_user_update_not_token_regen_when_flag_false():
    """regenerate_token=False → user.update, not user.token_regen."""
    admin_id = _get_admin_id()
    r = client.post(
        "/users",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"username": "audit_update_no_regen", "email": "audit_no_regen@test.com"},
    )
    target_id = r.json()["id"]
    with patch("src.api.users.audit_log") as mock_audit:
        r = client.patch(
            f"/users/{target_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"regenerate_token": False, "rate_limit": 200},
        )
    assert r.status_code == 200
    assert mock_audit.call_args[0][0] == "user.update"
