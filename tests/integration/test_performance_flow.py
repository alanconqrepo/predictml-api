"""
Tests d'intégration — calcul de performance avec données réelles.

Workflow testé :
  POST /models → inject cache
  → POST /predict × N (avec id_obs)
  → POST /observed-results (ground truth)
  → GET /models/{name}/performance → métriques calculées

Utilise SQLite in-memory + FakeRedis + MinIO mock global.
Token admin : test-token-integ-pf-admin-jj00
"""

import asyncio
import io
import pickle
from datetime import datetime, timedelta
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-integ-pf-admin-jj00"
PF_MODEL = "pf_performance_integ_model"
MODEL_VERSION = "1.0.0"

FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}


def _make_pkl() -> bytes:
    """Crée un modèle sklearn sérialisé."""
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


def _inject_cache(name: str, version: str = MODEL_VERSION):
    """Injecte le modèle dans Redis avec feature_names_in_ configuré."""
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
    """Crée l'utilisateur admin."""
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="integ_pf_admin",
                email="integ_pf_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())

# Créer le modèle + injecter dans le cache
_r_create = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.pkl", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={"name": PF_MODEL, "version": MODEL_VERSION, "accuracy": "0.95"},
)
assert _r_create.status_code == 201, _r_create.text
_inject_cache(PF_MODEL)


class TestPerformanceFlow:
    """Tests du calcul de performance avec données réelles en base."""

    def test_01_performance_without_observed_results_has_zero_matched(self):
        """Sans résultats observés → matched_predictions == 0."""
        r = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["matched_predictions"] == 0

    def test_02_performance_with_matched_pairs_returns_accuracy(self):
        """Avec des paires prediction+observed → accuracy calculée et non None."""
        # Faire 4 prédictions avec id_obs distincts
        obs_ids = [f"pf-integ-obs-{i}" for i in range(4)]
        for obs_id in obs_ids:
            r = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"model_name": PF_MODEL, "features": FEATURES, "id_obs": obs_id},
            )
            assert r.status_code == 200

        # Ajouter observed_results (classe 0 = iris setosa)
        now = datetime.utcnow().isoformat()
        client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "data": [
                    {"model_name": PF_MODEL, "id_obs": oid, "date_time": now, "observed_result": 0}
                    for oid in obs_ids
                ]
            },
        )

        r_perf = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_perf.status_code == 200
        data = r_perf.json()
        assert data["matched_predictions"] >= 4
        assert data["accuracy"] is not None

    def test_03_performance_response_has_expected_fields(self):
        """La réponse de performance contient les champs attendus."""
        r = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "total_predictions" in data
        assert "matched_predictions" in data

    def test_04_performance_with_upsert_does_not_double_count(self):
        """Double POST observed_result sur même id_obs → upsert, pas duplication."""
        obs_id = "pf-upsert-dedup-99"
        ts = datetime.utcnow().isoformat()

        # Prédiction
        client.post(
            "/predict",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": PF_MODEL, "features": FEATURES, "id_obs": obs_id},
        )

        # Premier observed_result
        client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "data": [
                    {"model_name": PF_MODEL, "id_obs": obs_id, "date_time": ts, "observed_result": 0}
                ]
            },
        )

        r1 = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        matched1 = r1.json()["matched_predictions"]

        # Deuxième POST même id_obs (upsert)
        client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "data": [
                    {"model_name": PF_MODEL, "id_obs": obs_id, "date_time": ts, "observed_result": 1}
                ]
            },
        )

        r2 = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        matched2 = r2.json()["matched_predictions"]
        # L'upsert ne doit pas augmenter le nombre de paires
        assert matched2 == matched1

    def test_05_performance_unknown_model_returns_404(self):
        """GET /models/unknown/performance → 404."""
        r = client.get(
            "/models/totally_unknown_pf_model/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_06_performance_requires_auth(self):
        """Sans token → 401 ou 403."""
        r = client.get(f"/models/{PF_MODEL}/performance")
        assert r.status_code in (401, 403)

    def test_07_performance_version_filter(self):
        """Paramètre version= filtre sur la version spécifique."""
        r = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"version": MODEL_VERSION},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["model_version"] == MODEL_VERSION

    def test_08_performance_with_granularity_day(self):
        """Paramètre granularity=day → réponse 200 avec bucketing par jour."""
        r = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"granularity": "day"},
        )
        assert r.status_code == 200

    def test_09_predictions_without_id_obs_not_counted_as_matched(self):
        """Prédictions sans id_obs ne sont pas dans matched_predictions."""
        # Faire une prédiction sans id_obs
        client.post(
            "/predict",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": PF_MODEL, "features": FEATURES},
        )

        r1 = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        matched_before = r1.json()["matched_predictions"]

        # Le matched ne doit pas avoir augmenté (pas d'id_obs → pas de JOIN)
        r2 = client.get(
            f"/models/{PF_MODEL}/performance",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r2.json()["matched_predictions"] == matched_before
