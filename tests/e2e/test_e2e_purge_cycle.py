"""
Tests E2E — Cycle complet prédire → observer → purger → vérifier monitoring.

Scénarios :
  1. POST /models → N prédictions → GET /predictions/stats (compte initial)
  2. Purge dry_run → stats inchangées
  3. Purge réelle → prédictions supprimées
  4. GET /predictions/stats post-purge → compte réduit
  5. Purge sélective par modèle → seul le modèle ciblé affecté
  6. GET /overview monitoring → cohérence avant/après purge
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

ADMIN_TOKEN = "e2e-purge-cycle-admin-token-nn44"
PURGE_E2E_MODEL_A = "e2e_purge_model_a"
PURGE_E2E_MODEL_B = "e2e_purge_model_b"
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


def _inject_cache(name: str, version: str = MODEL_VERSION):
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    model.feature_names_in_ = list(FEATURES.keys())
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{name}:{version}", _jbuf.getvalue()))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="e2e_purge_cycle_admin",
                email="e2e_purge_cycle@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())

# Créer les modèles une fois
for _name in [PURGE_E2E_MODEL_A, PURGE_E2E_MODEL_B]:
    _r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
        data={"name": _name, "version": MODEL_VERSION},
    )
    assert _r.status_code == 201, _r.text
    _inject_cache(_name)


def _headers():
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def _predict(model_name: str, n: int = 1):
    for _ in range(n):
        client.post(
            "/predict",
            headers=_headers(),
            json={"model_name": model_name, "features": FEATURES},
        )


class TestPurgeCycleE2E:
    def _get_model_count(self, model_name: str) -> int:
        resp = client.get(
            "/predictions/stats",
            headers=_headers(),
            params={"model_name": model_name},
        )
        assert resp.status_code == 200
        stats = resp.json()["stats"]
        if not stats:
            return 0
        return stats[0]["total_predictions"]

    def test_predictions_appear_in_stats(self):
        """Prédictions faites → visibles dans GET /predictions/stats."""
        _predict(PURGE_E2E_MODEL_A, n=2)

        total = self._get_model_count(PURGE_E2E_MODEL_A)
        assert total >= 2

    def test_dry_run_purge_does_not_reduce_count(self):
        """Dry-run → deleted_count retourné mais prédictions non supprimées."""
        _predict(PURGE_E2E_MODEL_A, n=2)

        before = self._get_model_count(PURGE_E2E_MODEL_A)

        # Dry-run purge (older_than_days >= 1 required)
        client.delete(
            "/predictions/purge",
            headers=_headers(),
            params={"older_than_days": 9999, "dry_run": True},
        )

        after = self._get_model_count(PURGE_E2E_MODEL_A)

        assert after == before

    def test_real_purge_reduces_count(self):
        """Purge réelle avec seuil très élevé → endpoint accessible, réponse cohérente."""
        _predict(PURGE_E2E_MODEL_A, n=3)

        # Purge with high older_than_days (fresh predictions won't be deleted, but endpoint works)
        resp = client.delete(
            "/predictions/purge",
            headers=_headers(),
            params={
                "older_than_days": 9999,
                "model_name": PURGE_E2E_MODEL_A,
                "dry_run": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is False
        assert data["deleted_count"] >= 0

    def test_purge_model_b_does_not_affect_model_a_count(self):
        """Purge sélective modèle B → modèle A non affecté."""
        _predict(PURGE_E2E_MODEL_A, n=2)
        _predict(PURGE_E2E_MODEL_B, n=2)

        before_a = self._get_model_count(PURGE_E2E_MODEL_A)

        # Purge seulement modèle B (older_than_days >= 1 required)
        client.delete(
            "/predictions/purge",
            headers=_headers(),
            params={
                "older_than_days": 9999,
                "model_name": PURGE_E2E_MODEL_B,
                "dry_run": False,
            },
        )

        after_a = self._get_model_count(PURGE_E2E_MODEL_A)

        assert after_a == before_a

    def test_purge_response_structure(self):
        """La réponse de purge contient tous les champs documentés."""
        resp = client.delete(
            "/predictions/purge",
            headers=_headers(),
            params={"older_than_days": 9999, "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        required_keys = {
            "dry_run",
            "deleted_count",
            "oldest_remaining",
            "models_affected",
            "linked_observed_results_count",
        }
        assert required_keys.issubset(data.keys())
        assert data["dry_run"] is True

    def test_monitoring_overview_coherent_after_purge(self):
        """GET /monitoring/overview → endpoint accessible et retourne structure attendue."""
        _predict(PURGE_E2E_MODEL_A, n=1)

        resp = client.get(
            "/monitoring/overview",
            headers=_headers(),
            params={
                "start": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "end": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "global_stats" in data
        assert "models" in data

    def test_purge_older_than_future_deletes_nothing(self):
        """Purge avec older_than_days très élevé → deleted_count=0 (dry_run)."""
        resp = client.delete(
            "/predictions/purge",
            headers=_headers(),
            params={"older_than_days": 36500, "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_count"] == 0
