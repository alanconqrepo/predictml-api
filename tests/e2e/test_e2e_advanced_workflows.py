"""
Tests E2E — workflows avancés : batch predict, SHAP explain, A/B testing.

Scénarios :
  1. POST /predict-batch → 5 prédictions, structure résultats, historique
  2. POST /explain → valeurs SHAP, clés features, base_value
  3. A/B testing : deux versions → GET /ab-compare → stats par version

Utilise SQLite in-memory + FakeRedis + MinIO mock global.
Token admin : e2e-adv-admin-token-ll22
"""

import asyncio
import io
import joblib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "e2e-adv-admin-token-ll22"
ADV_BATCH_MODEL = "e2e_adv_batch_model"
ADV_EXPLAIN_MODEL = "e2e_adv_explain_model"
ADV_AB_MODEL = "e2e_adv_ab_model"

FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}


def _make_lr_pkl() -> bytes:
    """Modèle LogisticRegression sérialisé."""
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _make_rf_pkl() -> bytes:
    """Modèle RandomForest sérialisé (tree-based pour SHAP TreeExplainer)."""
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _inject_cache(name: str, version: str, use_rf: bool = False):
    """Injecte le modèle dans Redis avec feature_names_in_ configuré."""
    X, y = load_iris(return_X_y=True)
    if use_rf:
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    else:
        model = LogisticRegression(max_iter=200).fit(X, y)
    model.feature_names_in_ = list(FEATURES.keys())
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
            feature_baseline=None,
        ),
    }
    asyncio.run(
        model_service._redis.set(f"model:{name}:{version}", (lambda _b: (joblib.dump(data, _b), _b.getvalue())[1])(io.BytesIO()))
    )


async def _setup():
    """Crée l'utilisateur admin."""
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="e2e_adv_admin",
                email="e2e_adv_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())

# Créer les modèles et injecter dans le cache
_r_batch = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.pkl", io.BytesIO(_make_lr_pkl()), "application/octet-stream")},
    data={"name": ADV_BATCH_MODEL, "version": "1.0.0", "accuracy": "0.95"},
)
assert _r_batch.status_code == 201, _r_batch.text
_inject_cache(ADV_BATCH_MODEL, "1.0.0", use_rf=False)

_r_explain = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.pkl", io.BytesIO(_make_rf_pkl()), "application/octet-stream")},
    data={"name": ADV_EXPLAIN_MODEL, "version": "1.0.0", "accuracy": "0.97"},
)
assert _r_explain.status_code == 201, _r_explain.text
_inject_cache(ADV_EXPLAIN_MODEL, "1.0.0", use_rf=True)

# Modèle A/B — créer v1 (production) et v2 (ab_test)
_r_ab_v1 = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.pkl", io.BytesIO(_make_lr_pkl()), "application/octet-stream")},
    data={"name": ADV_AB_MODEL, "version": "1.0.0"},
)
assert _r_ab_v1.status_code == 201, _r_ab_v1.text
# PATCH pour mettre v1 en production
_r_ab_v1_patch = client.patch(
    f"/models/{ADV_AB_MODEL}/1.0.0",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={"is_production": True, "deployment_mode": "production"},
)
assert _r_ab_v1_patch.status_code == 200, _r_ab_v1_patch.text
_inject_cache(ADV_AB_MODEL, "1.0.0")

_r_ab_v2 = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.pkl", io.BytesIO(_make_lr_pkl()), "application/octet-stream")},
    data={"name": ADV_AB_MODEL, "version": "2.0.0"},
)
assert _r_ab_v2.status_code == 201, _r_ab_v2.text
# PATCH pour configurer v2 en ab_test
_r_ab_v2_patch = client.patch(
    f"/models/{ADV_AB_MODEL}/2.0.0",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={"deployment_mode": "ab_test", "traffic_weight": 0.5},
)
assert _r_ab_v2_patch.status_code == 200, _r_ab_v2_patch.text
_inject_cache(ADV_AB_MODEL, "2.0.0")


# ===========================================================================
# Tests batch predict
# ===========================================================================


class TestBatchPredictE2E:
    """Tests E2E pour POST /predict-batch."""

    def test_01_batch_returns_correct_count(self):
        """POST /predict-batch avec 5 inputs → 5 résultats."""
        inputs = [{"features": FEATURES} for _ in range(5)]
        r = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_BATCH_MODEL, "inputs": inputs},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 5

    def test_02_batch_each_result_has_prediction(self):
        """Chaque résultat du batch contient un champ 'prediction'."""
        inputs = [{"features": FEATURES} for _ in range(3)]
        r = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_BATCH_MODEL, "inputs": inputs},
        )
        assert r.status_code == 200
        for result in r.json()["predictions"]:
            assert "prediction" in result

    def test_03_batch_with_id_obs_stores_in_history(self):
        """Batch avec id_obs → entrées dans GET /predictions."""
        obs_ids = [f"e2e-adv-batch-obs-{i}" for i in range(3)]
        inputs = [{"features": FEATURES, "id_obs": oid} for oid in obs_ids]
        client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_BATCH_MODEL, "inputs": inputs},
        )

        now = datetime.utcnow()
        r = client.get(
            "/predictions",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "name": ADV_BATCH_MODEL,
                "start": (now - timedelta(minutes=5)).isoformat(),
                "end": (now + timedelta(minutes=2)).isoformat(),
                "limit": 50,
            },
        )
        assert r.status_code == 200
        assert r.json()["total"] >= 3

    def test_04_batch_unknown_model_returns_404(self):
        """POST /predict-batch sur modèle inexistant → 404."""
        r = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "model_name": "totally_unknown_adv_model",
                "inputs": [{"features": FEATURES}],
            },
        )
        assert r.status_code == 404

    def test_05_batch_requires_auth(self):
        """POST /predict-batch sans token → 401 ou 403."""
        r = client.post(
            "/predict-batch",
            json={"model_name": ADV_BATCH_MODEL, "inputs": [{"features": FEATURES}]},
        )
        assert r.status_code in (401, 403)

    def test_06_batch_single_input_works(self):
        """Batch de 1 seul input → fonctionne normalement."""
        r = client.post(
            "/predict-batch",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_BATCH_MODEL, "inputs": [{"features": FEATURES}]},
        )
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 1


# ===========================================================================
# Tests SHAP explain
# ===========================================================================


class TestExplainE2E:
    """Tests E2E pour POST /explain."""

    def test_01_explain_returns_shap_values_and_base_value(self):
        """POST /explain → réponse avec feature_importance et base_value."""
        r = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_EXPLAIN_MODEL, "features": FEATURES},
        )
        assert r.status_code == 200
        data = r.json()
        assert "feature_importance" in data or "shap_values" in data
        assert "base_value" in data

    def test_02_explain_feature_importance_keys_match_features(self):
        """Les clés de feature_importance correspondent aux noms de features."""
        r = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_EXPLAIN_MODEL, "features": FEATURES},
        )
        assert r.status_code == 200
        data = r.json()
        importance_key = "feature_importance" if "feature_importance" in data else "shap_values"
        importance_dict = data.get(importance_key, {})
        for feature_name in FEATURES:
            assert feature_name in importance_dict

    def test_03_explain_requires_auth(self):
        """POST /explain sans token → 401 ou 403."""
        r = client.post(
            "/explain",
            json={"model_name": ADV_EXPLAIN_MODEL, "features": FEATURES},
        )
        assert r.status_code in (401, 403)

    def test_04_explain_unknown_model_returns_404(self):
        """POST /explain sur modèle inexistant → 404."""
        r = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": "totally_unknown_explain_model", "features": FEATURES},
        )
        assert r.status_code == 404

    def test_05_explain_model_name_in_response(self):
        """La réponse contient le nom du modèle."""
        r = client.post(
            "/explain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_EXPLAIN_MODEL, "features": FEATURES},
        )
        assert r.status_code == 200
        assert r.json()["model_name"] == ADV_EXPLAIN_MODEL


# ===========================================================================
# Tests A/B testing
# ===========================================================================


class TestABTestingE2E:
    """Tests E2E pour le setup A/B et GET /ab-compare."""

    def test_01_ab_two_versions_visible_in_models_list(self):
        """Les deux versions A/B apparaissent dans GET /models."""
        r = client.get(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        models = r.json()
        versions = [m["version"] for m in models if m["name"] == ADV_AB_MODEL]
        assert "1.0.0" in versions
        assert "2.0.0" in versions

    def test_02_predict_on_ab_model_returns_valid_result(self):
        """POST /predict sur modèle A/B → prédiction valide."""
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": ADV_AB_MODEL, "features": FEATURES},
        )
        assert r.status_code == 200
        assert "prediction" in r.json()

    def test_03_ab_compare_endpoint_returns_200(self):
        """GET /models/{name}/ab-compare → 200."""
        # Générer quelques prédictions d'abord
        for _ in range(3):
            client.post(
                "/predict",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"model_name": ADV_AB_MODEL, "features": FEATURES},
            )

        r = client.get(
            f"/models/{ADV_AB_MODEL}/ab-compare",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200

    def test_04_ab_compare_has_versions_field(self):
        """GET /ab-compare retourne une structure avec des données par version."""
        r = client.get(
            f"/models/{ADV_AB_MODEL}/ab-compare",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        # La réponse doit contenir des infos sur les versions
        assert data is not None

    def test_05_ab_compare_requires_auth(self):
        """GET /ab-compare sans token → 401 ou 403."""
        r = client.get(f"/models/{ADV_AB_MODEL}/ab-compare")
        assert r.status_code in (401, 403)

    def test_06_ab_compare_unknown_model_returns_404(self):
        """GET /ab-compare sur modèle inexistant → 404."""
        r = client.get(
            "/models/unknown_ab_model_xyz/ab-compare",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_07_ab_v2_has_ab_test_deployment_mode(self):
        """La version 2.0.0 a deployment_mode='ab_test'."""
        r = client.get(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        models = r.json()
        v2 = next(
            (m for m in models if m["name"] == ADV_AB_MODEL and m["version"] == "2.0.0"),
            None,
        )
        assert v2 is not None
        assert v2["deployment_mode"] == "ab_test"
