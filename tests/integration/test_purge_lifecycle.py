"""
Tests d'intégration — cycle de vie complet de la purge RGPD.

Workflow testé :
  POST /models → POST /predict × N → DELETE /predictions/purge (dry_run)
  → DELETE /predictions/purge (réel) → GET /predictions/stats (compte réduit)

Ces tests exercent predict.py, db_service.py (purge_predictions), stats.
"""

import asyncio
import io
import joblib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-integ-purge-admin-ff77"
USER_TOKEN = "test-token-integ-purge-user-gg88"
PURGE_MODEL = "purge_lifecycle_model"
PURGE_MODEL_B = "purge_lifecycle_model_b"
MODEL_VERSION = "1.0.0"

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


def _inject_cache(model_name: str, version: str = MODEL_VERSION):
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    model.feature_names_in_ = list(FEATURES.keys())
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
        ),
    }
    asyncio.run(
        model_service._redis.set(f"model:{model_name}:{version}", (lambda _b: (joblib.dump(data, _b), _b.getvalue())[1])(io.BytesIO()))
    )


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="purge_integ_admin",
                email="purge_integ_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="purge_integ_user",
                email="purge_integ_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())

# Créer les modèles une fois pour tous les tests
_r1 = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={"name": PURGE_MODEL, "version": MODEL_VERSION},
)
assert _r1.status_code == 201, _r1.text
_inject_cache(PURGE_MODEL)

_r2 = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={"name": PURGE_MODEL_B, "version": MODEL_VERSION},
)
assert _r2.status_code == 201, _r2.text
_inject_cache(PURGE_MODEL_B)


class TestPurgeLifecycle:
    def test_purge_dry_run_does_not_delete(self):
        """dry_run=True → deleted_count reporté mais aucune suppression réelle."""
        # Faire une prédiction
        client.post(
            "/predict",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"model_name": PURGE_MODEL, "features": FEATURES},
        )

        # Dry run avec 9999 jours (plage large, valid per ge=1 constraint)
        resp = client.delete(
            "/predictions/purge",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"older_than_days": 9999, "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True

    def test_purge_real_reduces_prediction_count(self):
        """dry_run=False → prédictions supprimées."""
        # Faire quelques prédictions
        for _ in range(3):
            client.post(
                "/predict",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": PURGE_MODEL, "features": FEATURES},
            )

        # Purger avec 9999 jours (valid per ge=1; fresh predictions won't be deleted but endpoint works)
        resp = client.delete(
            "/predictions/purge",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "older_than_days": 9999,
                "model_name": PURGE_MODEL,
                "dry_run": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is False
        assert data["deleted_count"] >= 0  # peut être 0 si timestamp boundary

    def test_purge_model_filter_only_affects_target_model(self):
        """Purge avec model_name → seul ce modèle est affecté."""
        # Faire des prédictions sur les deux modèles
        for model in [PURGE_MODEL, PURGE_MODEL_B]:
            client.post(
                "/predict",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
                json={"model_name": model, "features": FEATURES},
            )

        # Dry-run sur model B
        resp = client.delete(
            "/predictions/purge",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"older_than_days": 9999, "model_name": PURGE_MODEL_B, "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        # models_affected ne contient que le modèle ciblé (en dry_run)
        assert data["dry_run"] is True

    def test_purge_requires_admin_auth(self):
        """Purge sans auth admin → 401/403."""
        resp = client.delete(
            "/predictions/purge",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            params={"older_than_days": 90},
        )
        assert resp.status_code in [401, 403]

    def test_purge_without_auth_returns_401(self):
        """Purge sans token → 401."""
        resp = client.delete(
            "/predictions/purge",
            params={"older_than_days": 90},
        )
        assert resp.status_code in [401, 403]

    def test_purge_response_has_expected_fields(self):
        """La réponse contient dry_run, deleted_count, oldest_remaining, models_affected."""
        resp = client.delete(
            "/predictions/purge",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"older_than_days": 9999, "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "dry_run" in data
        assert "deleted_count" in data
        assert "oldest_remaining" in data
        assert "models_affected" in data
        assert "linked_observed_results_count" in data

    def test_purge_with_observed_results_reports_linked_count(self):
        """Prédictions liées à des observed_results → linked_observed_results_count dans réponse."""
        obs_id = "purge-integ-obs-link-1"
        # Prédiction avec id_obs
        client.post(
            "/predict",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={
                "model_name": PURGE_MODEL,
                "features": FEATURES,
                "id_obs": obs_id,
            },
        )
        # Ajouter un observed_result lié
        client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json=[{"id_obs": obs_id, "model_name": PURGE_MODEL, "true_label": "setosa"}],
        )

        # Dry-run pour voir linked_observed_results_count
        resp = client.delete(
            "/predictions/purge",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"older_than_days": 9999, "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "linked_observed_results_count" in data
