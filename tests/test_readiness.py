"""
Tests pour GET /models/{name}/readiness
"""
import asyncio
import io
import joblib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

TEST_TOKEN = "test-token-readiness"
TEST_MODEL_NAME = "test_readiness_model"


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    _jbuf = io.BytesIO()
    joblib.dump(model, _jbuf)
    return _jbuf.getvalue()


async def _setup_user():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_readiness_user",
                email="test_readiness@test.com",
                api_token=TEST_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup_user())

_AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}

_BASELINE = {
    "sepal_length": {"mean": 5.8, "std": 0.83, "min": 4.3, "max": 7.9, "null_rate": 0.0},
    "sepal_width": {"mean": 3.1, "std": 0.44, "min": 2.0, "max": 4.4, "null_rate": 0.0},
}


def _create_model(name: str, version: str = "1.0.0") -> dict:
    resp = client.post(
        "/models",
        data={"name": name, "version": version},
        files={"file": (f"{name}.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        headers=_AUTH,
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _patch_model(name: str, version: str, payload: dict) -> dict:
    resp = client.patch(f"/models/{name}/{version}", json=payload, headers=_AUTH)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_readiness_requires_auth():
    response = client.get(f"/models/{TEST_MODEL_NAME}/readiness?version=1.0.0")
    assert response.status_code in (401, 403)


def test_readiness_model_not_found():
    response = client.get(
        "/models/inexistant_model_xyz/readiness?version=1.0.0",
        headers=_AUTH,
    )
    assert response.status_code == 404


def test_readiness_version_not_found():
    _create_model(TEST_MODEL_NAME, "1.0.0")
    response = client.get(
        f"/models/{TEST_MODEL_NAME}/readiness?version=99.0.0",
        headers=_AUTH,
    )
    assert response.status_code == 404


def test_readiness_response_shape():
    """La réponse contient tous les champs attendus."""
    _create_model(TEST_MODEL_NAME, "2.0.0")
    response = client.get(
        f"/models/{TEST_MODEL_NAME}/readiness?version=2.0.0",
        headers=_AUTH,
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "ready" in data
    assert "checked_at" in data
    assert "checks" in data
    checks = data["checks"]
    assert "is_production" in checks
    assert "file_accessible" in checks
    assert "baseline_computed" in checks
    assert "no_critical_drift" in checks
    for check in checks.values():
        assert "pass" in check


def test_readiness_not_production():
    """Un modèle non mis en production échoue le check is_production."""
    _create_model(TEST_MODEL_NAME, "3.0.0")
    response = client.get(
        f"/models/{TEST_MODEL_NAME}/readiness?version=3.0.0",
        headers=_AUTH,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False
    assert data["checks"]["is_production"]["pass"] is False
    assert "is_production=False" in data["checks"]["is_production"]["detail"]


def test_readiness_no_baseline():
    """Un modèle sans baseline échoue le check baseline_computed."""
    _create_model(TEST_MODEL_NAME, "4.0.0")
    response = client.get(
        f"/models/{TEST_MODEL_NAME}/readiness?version=4.0.0",
        headers=_AUTH,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["baseline_computed"]["pass"] is False
    assert data["checks"]["baseline_computed"]["detail"] == "feature_baseline is null"


def test_readiness_file_not_accessible():
    """MinIO indisponible → file_accessible échoue."""
    _create_model(TEST_MODEL_NAME, "5.0.0")
    _minio_mock.get_object_info.return_value = None
    try:
        response = client.get(
            f"/models/{TEST_MODEL_NAME}/readiness?version=5.0.0",
            headers=_AUTH,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["checks"]["file_accessible"]["pass"] is False
        assert "MinIO" in data["checks"]["file_accessible"]["detail"]
        assert data["ready"] is False
    finally:
        _minio_mock.get_object_info.return_value = {"size": 100, "etag": "abc"}


def test_readiness_baseline_computed_pass():
    """Un modèle avec baseline passe le check baseline_computed."""
    _create_model(TEST_MODEL_NAME, "6.0.0")
    _patch_model(TEST_MODEL_NAME, "6.0.0", {"feature_baseline": _BASELINE})
    response = client.get(
        f"/models/{TEST_MODEL_NAME}/readiness?version=6.0.0",
        headers=_AUTH,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["baseline_computed"]["pass"] is True
    assert data["checks"]["baseline_computed"]["detail"] is None


def test_readiness_critical_drift():
    """Un drift critique → no_critical_drift échoue."""
    _create_model(TEST_MODEL_NAME, "7.0.0")
    _patch_model(TEST_MODEL_NAME, "7.0.0", {"feature_baseline": _BASELINE})
    _minio_mock.get_object_info.return_value = {"size": 100}
    try:
        with patch("src.api.models.drift_service.summarize_drift", return_value="critical"):
            response = client.get(
                f"/models/{TEST_MODEL_NAME}/readiness?version=7.0.0",
                headers=_AUTH,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["checks"]["no_critical_drift"]["pass"] is False
        assert "critical" in data["checks"]["no_critical_drift"]["detail"]
        assert data["ready"] is False
    finally:
        _minio_mock.get_object_info.return_value = {"size": 100, "etag": "abc"}


def test_readiness_all_checks_pass():
    """Toutes les conditions satisfaites → ready=True."""
    _create_model(TEST_MODEL_NAME, "8.0.0")
    _patch_model(TEST_MODEL_NAME, "8.0.0", {
        "is_production": True,
        "feature_baseline": _BASELINE,
    })
    _minio_mock.get_object_info.return_value = {"size": 100, "etag": "abc"}
    try:
        with patch("src.api.models.drift_service.summarize_drift", return_value="ok"):
            response = client.get(
                f"/models/{TEST_MODEL_NAME}/readiness?version=8.0.0",
                headers=_AUTH,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["model_name"] == TEST_MODEL_NAME
        assert data["version"] == "8.0.0"
        for check in data["checks"].values():
            assert check["pass"] is True
            assert check["detail"] is None
    finally:
        _minio_mock.get_object_info.return_value = {"size": 100, "etag": "abc"}


def test_readiness_no_drift_when_no_baseline():
    """Sans baseline, no_critical_drift passe toujours (rien à mesurer)."""
    _create_model(TEST_MODEL_NAME, "9.0.0")
    response = client.get(
        f"/models/{TEST_MODEL_NAME}/readiness?version=9.0.0",
        headers=_AUTH,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["no_critical_drift"]["pass"] is True
