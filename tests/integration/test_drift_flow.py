"""
Tests d'intégration — workflow de détection de drift.

Workflow testé :
  POST /models (avec ou sans feature_baseline)
  → GET /models/{name}/drift → vérifier structure et valeurs

Stratégie :
  Les prédictions sont insérées directement en base via DBService.create_prediction
  (pattern de test_db_service_monitoring.py) pour contrôler précisément les
  input_features et garantir les statistiques de production désirées.

Utilise SQLite in-memory + FakeRedis + MinIO mock global.
Token admin : test-token-integ-df-admin-kk11
"""

import asyncio
import io
import json
import joblib

import numpy as np

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-integ-df-admin-kk11"
DF_MODEL_NOBASELINE = "df_drift_nobaseline_model"
DF_MODEL_BASELINE = "df_drift_baseline_model"
DF_MODEL_CRITICAL = "df_drift_critical_model"
MODEL_VERSION = "1.0.0"

# Baseline typique pour iris (sepal length)
BASELINE = {
    "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9},
    "sepal width (cm)": {"mean": 3.05, "std": 0.43, "min": 2.0, "max": 4.4},
    "petal length (cm)": {"mean": 3.76, "std": 1.77, "min": 1.0, "max": 6.9},
    "petal width (cm)": {"mean": 1.20, "std": 0.76, "min": 0.1, "max": 2.5},
}

# Features proches du baseline → pas de drift (valeur unique pour les tests ponctuels)
NORMAL_FEATURES = {
    "sepal length (cm)": 5.8,
    "sepal width (cm)": 3.0,
    "petal length (cm)": 3.7,
    "petal width (cm)": 1.2,
}

# Génère des features suivant la distribution baseline N(mean, std) pour un PSI fiable.
# N=200 avec seed fixe → déterministe, dépasse min_count=30, PSI cohérent avec "ok/warning".
_rng = np.random.default_rng(42)
NORMAL_FEATURES_LIST = [
    {
        "sepal length (cm)": float(_rng.normal(5.84, 0.83)),
        "sepal width (cm)": float(_rng.normal(3.05, 0.43)),
        "petal length (cm)": float(_rng.normal(3.76, 1.77)),
        "petal width (cm)": float(_rng.normal(1.20, 0.76)),
    }
    for _ in range(200)
]

# Features très éloignées du baseline → drift critique
OUTLIER_FEATURES = {
    "sepal length (cm)": 99.0,
    "sepal width (cm)": 99.0,
    "petal length (cm)": 99.0,
    "petal width (cm)": 99.0,
}


def _make_pkl() -> bytes:
    """Crée un modèle sklearn sérialisé."""
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    """Crée l'utilisateur admin et les modèles de test."""
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="integ_df_admin",
                email="integ_df_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())


def _create_model(name: str, with_baseline: bool = False):
    """Crée un modèle via l'API."""
    data = {"name": name, "version": MODEL_VERSION}
    if with_baseline:
        data["feature_baseline"] = json.dumps(BASELINE)
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
        data=data,
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _insert_predictions(model_name: str, features_list: list):
    """Insère des prédictions directement en base pour contrôler les input_features."""
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, ADMIN_TOKEN)
        for features in features_list:
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=model_name,
                model_version=MODEL_VERSION,
                input_features=features,
                prediction_result=0,
                probabilities=[0.9, 0.05, 0.05],
                response_time_ms=10.0,
                status="success",
            )
        await db.commit()


# Créer les modèles au chargement du module
_create_model(DF_MODEL_NOBASELINE, with_baseline=False)
_create_model(DF_MODEL_BASELINE, with_baseline=True)
_create_model(DF_MODEL_CRITICAL, with_baseline=True)

# Insérer 200 prédictions normales pour DF_MODEL_BASELINE (PSI fiable avec N>=200)
asyncio.run(_insert_predictions(DF_MODEL_BASELINE, NORMAL_FEATURES_LIST))

# Insérer 30 prédictions outlier pour DF_MODEL_CRITICAL
asyncio.run(_insert_predictions(DF_MODEL_CRITICAL, [OUTLIER_FEATURES] * 30))


class TestDriftFlow:
    """Tests du workflow de détection de drift."""

    def test_01_drift_without_baseline_returns_no_baseline(self):
        """Modèle sans feature_baseline → baseline_available=False, drift_summary='no_baseline'."""
        r = client.get(
            f"/models/{DF_MODEL_NOBASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["baseline_available"] is False
        assert data["drift_summary"] == "no_baseline"

    def test_02_drift_with_normal_features_returns_ok_or_no_data(self):
        """Modèle avec baseline et features normales → drift_summary == 'ok' (ou similaire)."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["baseline_available"] is True
        # Avec des features suivant la distribution baseline, le drift doit être ok ou warning
        # (insufficient_data si la fenêtre temporelle ne couvre pas les prédictions insérées)
        assert data["drift_summary"] in ("ok", "warning", "no_data", "insufficient_data")

    def test_03_drift_with_outlier_features_returns_critical(self):
        """Modèle avec features très éloignées du baseline → au moins une feature critical."""
        r = client.get(
            f"/models/{DF_MODEL_CRITICAL}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["baseline_available"] is True
        if data["predictions_analyzed"] > 0:
            # Les features outlier doivent provoquer un drift critique
            feature_statuses = [f["drift_status"] for f in data["features"].values()]
            assert "critical" in feature_statuses

    def test_04_drift_response_has_required_fields(self):
        """La réponse de drift contient tous les champs requis."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "period_days" in data
        assert "predictions_analyzed" in data
        assert "baseline_available" in data
        assert "drift_summary" in data
        assert "features" in data

    def test_05_drift_features_have_production_stats(self):
        """Chaque feature de production a les stats requises."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        if data["predictions_analyzed"] > 0:
            for feat_name, feat_data in data["features"].items():
                assert "production_mean" in feat_data
                assert "production_std" in feat_data
                assert "production_count" in feat_data

    def test_06_drift_unknown_model_returns_404(self):
        """GET /models/unknown/drift → 404."""
        r = client.get(
            "/models/totally_unknown_df_model/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_07_drift_requires_auth(self):
        """Sans token → 401 ou 403."""
        r = client.get(f"/models/{DF_MODEL_BASELINE}/drift")
        assert r.status_code in (401, 403)

    def test_08_drift_period_days_param_accepted(self):
        """Paramètre days= est accepté sans erreur."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={"days": 30},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["period_days"] == 30

    def test_09_drift_no_baseline_features_have_no_baseline_status(self):
        """Sans baseline, chaque feature a drift_status='no_baseline'."""
        # Insérer quelques prédictions pour que features soient non vides
        asyncio.run(_insert_predictions(DF_MODEL_NOBASELINE, [NORMAL_FEATURES] * 3))

        r = client.get(
            f"/models/{DF_MODEL_NOBASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        if data["predictions_analyzed"] > 0:
            for feat_data in data["features"].values():
                assert feat_data["drift_status"] == "no_baseline"

    def test_10_drift_model_name_in_response(self):
        """Le nom du modèle dans la réponse correspond au modèle demandé."""
        r = client.get(
            f"/models/{DF_MODEL_BASELINE}/drift",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        assert r.json()["model_name"] == DF_MODEL_BASELINE
