"""
Tests pour les endpoints POST /models/{name}/{version}/warmup
et GET /models/{name}/{version}/download.

Couvre :
- Warmup succès (modèle déjà en cache / à charger)
- Warmup modèle inexistant → 404
- Warmup sans auth → 401/403
- Warmup erreur model_service → 500
- Download succès → bytes + Content-Disposition
- Download modèle inexistant → 404
- Download sans auth → 401/403
- Download erreur MinIO → 500
"""

import asyncio
import io
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-warmup-admin-xx01"
USER_TOKEN = "test-token-warmup-user-xx02"
WM_MODEL = "warmup_dl_model"


def _make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="warmup_admin",
                email="warmup_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="warmup_user",
                email="warmup_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


def _create_model(name=WM_MODEL, version="1.0.0") -> dict:
    resp = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={
            "file": ("model.pkl", io.BytesIO(_make_pkl_bytes()), "application/octet-stream")
        },
        data={"name": name, "version": version, "description": "warmup test"},
    )
    return resp


def _inject_cache(name: str, version: str):
    """Injecte un modèle sklearn dans le cache FakeRedis."""
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
        ),
    }
    asyncio.run(model_service._redis.set(f"model:{name}:{version}", pickle.dumps(data)))


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmupEndpoint:
    def test_warmup_without_auth_returns_401(self):
        """POST /models/{name}/{version}/warmup sans auth → 401/403."""
        resp = client.post(f"/models/{WM_MODEL}/1.0.0/warmup")
        assert resp.status_code in [401, 403]

    def test_warmup_non_admin_returns_403(self):
        """POST /models/{name}/{version}/warmup avec token user → 403."""
        resp = client.post(
            f"/models/{WM_MODEL}/1.0.0/warmup",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert resp.status_code in [401, 403]

    def test_warmup_model_not_found_returns_404(self):
        """POST /models/nonexistent/9.9.9/warmup → 404."""
        with patch.object(
            model_service,
            "load_model",
            side_effect=__import__(
                "fastapi", fromlist=["HTTPException"]
            ).HTTPException(status_code=404, detail="Not found"),
        ):
            resp = client.post(
                "/models/nonexistent_wm/9.9.9/warmup",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )
        assert resp.status_code == 404

    def test_warmup_already_cached_returns_flag(self):
        """Modèle déjà en cache → already_cached=True dans la réponse."""
        _create_model(name=f"{WM_MODEL}_cached", version="1.0.0")
        _inject_cache(f"{WM_MODEL}_cached", "1.0.0")

        with patch.object(
            model_service,
            "load_model",
            new_callable=AsyncMock,
        ):
            resp = client.post(
                f"/models/{WM_MODEL}_cached/1.0.0/warmup",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == f"{WM_MODEL}_cached"
        assert data["version"] == "1.0.0"
        assert data["already_cached"] is True

    def test_warmup_new_model_loads_successfully(self):
        """Modèle pas encore en cache → load_model appelé, already_cached=False."""
        _create_model(name=f"{WM_MODEL}_new", version="1.0.0")

        with patch.object(
            model_service,
            "load_model",
            new_callable=AsyncMock,
        ):
            resp = client.post(
                f"/models/{WM_MODEL}_new/1.0.0/warmup",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "load_time_ms" in data
        assert data["already_cached"] is False

    def test_warmup_load_error_returns_500(self):
        """Erreur lors du chargement (non-HTTP) → 500."""
        _create_model(name=f"{WM_MODEL}_err", version="1.0.0")

        with patch.object(
            model_service,
            "load_model",
            side_effect=RuntimeError("pickle error"),
        ):
            resp = client.post(
                f"/models/{WM_MODEL}_err/1.0.0/warmup",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class TestDownloadEndpoint:
    def test_download_without_auth_returns_401(self):
        """GET /models/{name}/{version}/download sans auth → 401/403."""
        resp = client.get(f"/models/{WM_MODEL}/1.0.0/download")
        assert resp.status_code in [401, 403]

    def test_download_non_admin_returns_403(self):
        """GET /models/{name}/{version}/download avec token user → 403."""
        resp = client.get(
            f"/models/{WM_MODEL}/1.0.0/download",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert resp.status_code in [401, 403]

    def test_download_model_not_found_returns_404(self):
        """GET /models/nonexistent/9.9.9/download → 404."""
        resp = client.get(
            "/models/nonexistent_dl/9.9.9/download",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert resp.status_code == 404

    def test_download_success_returns_bytes(self):
        """Download d'un modèle existant → 200 + Content-Disposition."""
        _create_model(name=f"{WM_MODEL}_dl", version="1.0.0")
        pkl_bytes = _make_pkl_bytes()

        from tests.conftest import _minio_mock

        with patch.object(_minio_mock, "download_file_bytes", return_value=pkl_bytes):
            resp = client.get(
                f"/models/{WM_MODEL}_dl/1.0.0/download",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )

        assert resp.status_code == 200
        assert resp.content == pkl_bytes
        assert "attachment" in resp.headers.get("content-disposition", "")

    def test_download_minio_error_returns_500(self):
        """Erreur MinIO lors du téléchargement → 500."""
        _create_model(name=f"{WM_MODEL}_minerr", version="1.0.0")

        from tests.conftest import _minio_mock

        with patch.object(
            _minio_mock,
            "download_file_bytes",
            side_effect=Exception("MinIO connexion perdue"),
        ):
            resp = client.get(
                f"/models/{WM_MODEL}_minerr/1.0.0/download",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            )

        assert resp.status_code == 500
