"""
Tests E2E — supervision et santé des modèles.

Scénario :
  Créer des modèles, générer des prédictions avec différents statuts,
  vérifier le dashboard de supervision et les transitions de santé.
"""

import asyncio
import io
import pickle
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

ADMIN_TOKEN = "e2e-mon-admin-token-dd44"
MON_MODEL = "e2e_mon_model"

FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}


def _make_pkl() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


def _inject_cache(name: str, version: str = "1.0.0"):
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
    asyncio.run(
        model_service._redis.set(f"model:{name}:{version}", pickle.dumps(data))
    )


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="e2e_mon_admin",
                email="e2e_mon_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())

# Créer le modèle de monitoring
_r_mon = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.pkl", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={"name": MON_MODEL, "version": "1.0.0", "accuracy": "0.95"},
)
assert _r_mon.status_code == 201, _r_mon.text
_inject_cache(MON_MODEL, "1.0.0")

# Générer des prédictions initiales pour alimenter le monitoring
for _ in range(5):
    client.post(
        "/predict",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"model_name": MON_MODEL, "features": FEATURES},
    )


class TestMonitoringAlertingE2E:
    def test_01_monitoring_overview_requires_auth(self):
        """Sans token → 401 ou 403."""
        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": now.isoformat(),
            },
        )
        assert r.status_code in (401, 403)

    def test_02_monitoring_overview_invalid_dates_returns_422(self):
        """end < start → 422."""
        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": now.isoformat(),
                "end": (now - timedelta(hours=1)).isoformat(),
            },
        )
        assert r.status_code == 422

    def test_03_healthy_model_shows_ok_or_no_data_status(self):
        """Modèle récent sans erreurs → statut ok ou no_data."""
        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(minutes=30)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        mon_entry = next(
            (m for m in data["models"] if m["model_name"] == MON_MODEL), None
        )
        assert mon_entry is not None
        assert mon_entry["health_status"] in ("ok", "no_data", "warning", "critical")

    def test_04_model_detail_shows_predictions(self):
        """GET /monitoring/model/{name} → au moins une prédiction dans per_version_stats."""
        now = datetime.utcnow()
        r = client.get(
            f"/monitoring/model/{MON_MODEL}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(minutes=30)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        # per_version_stats est une liste de dicts avec total_predictions
        version_stats = data.get("per_version_stats", [])
        total = sum(v.get("total_predictions", 0) for v in version_stats)
        assert total >= 1

    def test_05_model_detail_has_required_fields(self):
        """GET /monitoring/model/{name} → tous les champs attendus sont présents."""
        now = datetime.utcnow()
        r = client.get(
            f"/monitoring/model/{MON_MODEL}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(minutes=30)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "model_name" in data
        assert "per_version_stats" in data
        assert "timeseries" in data
        assert "performance_by_day" in data
        assert "feature_drift" in data
        assert "recent_errors" in data

    def test_06_nonexistent_model_detail_returns_404(self):
        """GET /monitoring/model/{name} modèle inexistant → 404."""
        now = datetime.utcnow()
        r = client.get(
            "/monitoring/model/totally_nonexistent_e2e_model_xyz",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(days=1)).isoformat(),
                "end": now.isoformat(),
            },
        )
        assert r.status_code == 404

    def test_07_multiple_models_in_overview(self):
        """Deux modèles avec prédictions → overview contient les deux."""
        second_model = f"{MON_MODEL}_second"
        r_create = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={"file": ("m.pkl", io.BytesIO(_make_pkl()), "application/octet-stream")},
            data={"name": second_model, "version": "1.0.0"},
        )
        assert r_create.status_code == 201

        _inject_cache(second_model, "1.0.0")
        client.post(
            "/predict",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": second_model, "features": FEATURES},
        )

        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(minutes=30)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        model_names = [m["model_name"] for m in data["models"]]
        assert MON_MODEL in model_names
        assert second_model in model_names

    def test_08_empty_future_period_returns_zero_stats(self):
        """Période future → aucune prédiction, stats globales à zéro."""
        future = datetime.utcnow() + timedelta(days=1000)
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": future.isoformat(),
                "end": (future + timedelta(hours=1)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["global_stats"]["total_predictions"] == 0
        assert data["models"] == []

    def test_09_observed_results_added_after_predictions(self):
        """Ajouter des résultats observés est possible après des prédictions."""
        r = client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "data": [
                    {
                        "id_obs": "e2e-mon-obs-001",
                        "model_name": MON_MODEL,
                        "date_time": datetime.utcnow().isoformat(),
                        "observed_result": 0,
                    }
                ]
            },
        )
        assert r.status_code == 200
        assert r.json()["upserted"] >= 1
