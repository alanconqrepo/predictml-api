"""
Tests pour les fonctionnalités #19 (tags), #20 (taille upload), #23 (webhook)

#19 — Tags sur les modèles :
  - Créer un modèle avec tags → vérifier le champ dans la réponse
  - PATCH pour modifier les tags
  - GET /models?tag= filtre correctement

#20 — Validation taille upload :
  - POST /models avec fichier trop grand → 413

#23 — Webhook sortant :
  - POST /predict avec webhook_url configuré → send_webhook appelé en background
"""
import asyncio
import io
import joblib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TW_ADMIN_TOKEN = "test-token-tags-wh-admin-a9bz"
TW_USER_TOKEN = "test-token-tags-wh-user-b8cy"
TW_MODEL_TAGGED = "tw_tagged_model"
TW_MODEL_WEBHOOK = "tw_webhook_model"
TW_MODEL_UPLOAD = "tw_upload_model"
MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lr_model():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.1, 0.2, 0.3, 0.4]})
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_model(model_name: str, version: str, model, webhook_url=None):
    """Injecte un modèle dans le cache Redis avec métadonnées."""
    key = f"model:{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            webhook_url=webhook_url,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(key, _jbuf.getvalue()))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

async def _setup():
    async with _TestSessionLocal() as db:
        # Admin
        if not await DBService.get_user_by_token(db, TW_ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="tw_admin_user",
                email="tw_admin@test.com",
                api_token=TW_ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        # User
        if not await DBService.get_user_by_token(db, TW_USER_TOKEN):
            await DBService.create_user(
                db,
                username="tw_regular_user",
                email="tw_user@test.com",
                api_token=TW_USER_TOKEN,
                role="user",
                rate_limit=10000,
            )

        # Modèle avec tags
        if not await DBService.get_model_metadata(db, TW_MODEL_TAGGED, MODEL_VERSION):
            await DBService.create_model_metadata(
                db,
                name=TW_MODEL_TAGGED,
                version=MODEL_VERSION,
                minio_bucket="models",
                minio_object_key=f"{TW_MODEL_TAGGED}/v{MODEL_VERSION}.pkl",
                is_active=True,
                is_production=True,
                tags=["production", "finance"],
            )

        # Modèle avec webhook
        if not await DBService.get_model_metadata(db, TW_MODEL_WEBHOOK, MODEL_VERSION):
            await DBService.create_model_metadata(
                db,
                name=TW_MODEL_WEBHOOK,
                version=MODEL_VERSION,
                minio_bucket="models",
                minio_object_key=f"{TW_MODEL_WEBHOOK}/v{MODEL_VERSION}.pkl",
                is_active=True,
                is_production=True,
                webhook_url="https://example.com/webhook",
            )


asyncio.run(_setup())

# Injecter les modèles dans le cache
_inject_model(TW_MODEL_TAGGED, MODEL_VERSION, _make_lr_model())
_inject_model(
    TW_MODEL_WEBHOOK,
    MODEL_VERSION,
    _make_lr_model(),
    webhook_url="https://example.com/webhook",
)


# ===========================================================================
# #19 — Tags
# ===========================================================================

class TestTags:
    def test_model_created_with_tags_via_db(self):
        """Un modèle créé avec tags les expose dans GET /models."""
        r = client.get("/models", headers={"Authorization": f"Bearer {TW_ADMIN_TOKEN}"})
        assert r.status_code == 200
        models = r.json()
        tagged = next((m for m in models if m["name"] == TW_MODEL_TAGGED), None)
        assert tagged is not None
        assert tagged["tags"] == ["production", "finance"]

    def test_get_models_tag_filter_match(self):
        """GET /models?tag=finance ne retourne que les modèles avec ce tag."""
        r = client.get(
            "/models",
            params={"tag": "finance"},
            headers={"Authorization": f"Bearer {TW_ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        models = r.json()
        assert all("finance" in (m.get("tags") or []) for m in models)
        names = [m["name"] for m in models]
        assert TW_MODEL_TAGGED in names

    def test_get_models_tag_filter_no_match(self):
        """GET /models?tag=inexistant retourne liste vide."""
        r = client.get(
            "/models",
            params={"tag": "tag_qui_nexiste_pas_xyz"},
        )
        assert r.status_code == 200
        assert r.json() == []

    def test_patch_model_updates_tags(self):
        """PATCH /models/{name}/{version} met à jour les tags."""
        r = client.patch(
            f"/models/{TW_MODEL_TAGGED}/{MODEL_VERSION}",
            headers={"Authorization": f"Bearer {TW_ADMIN_TOKEN}"},
            json={"tags": ["production", "v2", "updated"]},
        )
        assert r.status_code == 200
        assert r.json()["tags"] == ["production", "v2", "updated"]

    def test_model_without_tags_returns_none(self):
        """Un modèle sans tags expose tags=null."""
        r = client.get("/models", headers={"Authorization": f"Bearer {TW_ADMIN_TOKEN}"})
        assert r.status_code == 200
        webhook_model = next(
            (m for m in r.json() if m["name"] == TW_MODEL_WEBHOOK), None
        )
        assert webhook_model is not None
        assert webhook_model.get("tags") is None


# ===========================================================================
# #20 — Validation taille upload
# ===========================================================================

class TestUploadSize:
    def test_upload_oversized_file_returns_413(self):
        """POST /models avec fichier > MAX_MODEL_SIZE_MB → 413."""
        big_bytes = b"x" * 100

        with patch("src.api.models.settings") as mock_settings:
            mock_settings.MAX_MODEL_SIZE_MB = 0  # 0 MB → tout fichier non-vide est trop grand
            r = client.post(
                "/models",
                headers={"Authorization": f"Bearer {TW_ADMIN_TOKEN}"},
                data={
                    "name": TW_MODEL_UPLOAD,
                    "version": "9.9.9",
                },
                files={"file": ("model.pkl", io.BytesIO(big_bytes), "application/octet-stream")},
            )
        assert r.status_code == 413
        assert "taille maximale" in r.json()["detail"].lower()

    def test_upload_valid_size_succeeds(self):
        """POST /models avec fichier dans la limite → pas de 413 (201 ou autre erreur)."""
        tiny_bytes = b"x" * 10

        with patch("src.api.models.settings") as mock_settings:
            mock_settings.MAX_MODEL_SIZE_MB = 500  # 500 MB → 10 octets passent
            r = client.post(
                "/models",
                headers={"Authorization": f"Bearer {TW_ADMIN_TOKEN}"},
                data={
                    "name": TW_MODEL_UPLOAD,
                    "version": "1.1.1",
                },
                files={"file": ("model.pkl", io.BytesIO(tiny_bytes), "application/octet-stream")},
            )
        # Pas un 413 (le fichier passera la vérification de taille)
        assert r.status_code != 413


# ===========================================================================
# #23 — Webhook sortant
# ===========================================================================

class TestWebhook:
    def test_predict_triggers_webhook(self):
        """POST /predict avec un modèle ayant webhook_url → send_webhook appelé."""
        with patch(
            "src.api.predict.send_webhook", new_callable=AsyncMock
        ) as mock_send:
            r = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {TW_USER_TOKEN}"},
                json={
                    "model_name": TW_MODEL_WEBHOOK,
                    "features": {"f1": 1.0, "f2": 0.5},
                },
            )
            assert r.status_code == 200
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "https://example.com/webhook"
            payload = call_args[0][1]
            assert payload["model_name"] == TW_MODEL_WEBHOOK
            assert "prediction" in payload

    def test_predict_no_webhook_not_called(self):
        """POST /predict avec un modèle sans webhook_url → send_webhook non appelé."""
        with patch(
            "src.api.predict.send_webhook", new_callable=AsyncMock
        ) as mock_send:
            r = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {TW_USER_TOKEN}"},
                json={
                    "model_name": TW_MODEL_TAGGED,
                    "features": {"f1": 1.0, "f2": 0.5},
                },
            )
            assert r.status_code == 200
            mock_send.assert_not_called()

    def test_patch_model_sets_webhook_url(self):
        """PATCH /models/{name}/{version} permet de définir un webhook_url."""
        r = client.patch(
            f"/models/{TW_MODEL_TAGGED}/{MODEL_VERSION}",
            headers={"Authorization": f"Bearer {TW_ADMIN_TOKEN}"},
            json={"webhook_url": "https://newwebhook.example.com/cb"},
        )
        assert r.status_code == 200
        assert r.json()["webhook_url"] == "https://newwebhook.example.com/cb"
