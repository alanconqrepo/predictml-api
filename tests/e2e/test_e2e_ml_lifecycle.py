"""
Tests E2E — cycle de vie ML complet.

Scénario :
  Un admin déploie un modèle (avec script d'entraînement),
  un utilisateur fait des prédictions,
  l'admin ajoute des résultats observés, vérifie le monitoring,
  déclenche un retrain et met la nouvelle version en production.
"""

import asyncio
import io
import joblib
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

ADMIN_TOKEN = "e2e-ml-admin-token-aa11"
USER_TOKEN = "e2e-ml-user-token-bb22"
E2E_MODEL = "e2e_ml_model"

FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}

VALID_TRAIN_SCRIPT = """\
import os
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)
with open(OUTPUT_MODEL_PATH, "wb") as f:
    joblib.dump(model, f)

print(json.dumps({"accuracy": 0.96, "f1_score": 0.95}))
"""


def _make_pkl() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


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
        model_service._redis.set(f"model:{name}:{version}", (lambda _b: (joblib.dump(data, _b), _b.getvalue())[1])(io.BytesIO()))
    )


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="e2e_ml_admin",
                email="e2e_ml_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="e2e_ml_user",
                email="e2e_ml_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=1000,
            )
        await db.commit()


asyncio.run(_setup())

_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


async def _mock_exec_success(*args, **kwargs):
    env = kwargs.get("env", {})
    output_path = env.get("OUTPUT_MODEL_PATH", "")
    if output_path:
        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=200).fit(X, y)
        with open(output_path, "wb") as f:
            joblib.dump(model, f)
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(
        return_value=(
            b'{"accuracy": 0.96, "f1_score": 0.95}\n',
            b"",
        )
    )
    proc.kill = MagicMock()
    return proc


# Créer le modèle une seule fois à l'initialisation du module
_r_create = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={
        "file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream"),
        "train_file": (
            "train.py",
            io.BytesIO(VALID_TRAIN_SCRIPT.encode()),
            "text/x-python",
        ),
    },
    data={"name": E2E_MODEL, "version": "1.0.0", "accuracy": "0.90"},
)
assert _r_create.status_code == 201, _r_create.text
_inject_cache(E2E_MODEL, "1.0.0")


class TestMLLifecycleE2E:
    def test_01_admin_creates_model(self):
        """Le modèle créé par l'admin est visible dans GET /models."""
        r = client.get(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        names = [m["name"] for m in r.json()]
        assert E2E_MODEL in names

    def test_02_user_cannot_list_users(self):
        """Un utilisateur standard ne peut pas lister les utilisateurs → 403."""
        r = client.get(
            "/users",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert r.status_code == 403

    def test_03_admin_predicts_on_model(self):
        """Prédiction sur le modèle par l'admin → succès."""
        # Utilise ADMIN_TOKEN pour predict + cohérence avec GET /predictions
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"model_name": E2E_MODEL, "features": FEATURES},
        )
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data

    def test_04_admin_sees_predictions_in_history(self):
        """Prédictions visibles dans GET /predictions."""
        now = datetime.utcnow()
        r = client.get(
            "/predictions",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "name": E2E_MODEL,
                "start": (now - timedelta(minutes=10)).isoformat(),
                "end": (now + timedelta(minutes=5)).isoformat(),
                "limit": 50,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1

    def test_05_admin_adds_observed_results(self):
        """L'admin peut ajouter des résultats observés."""
        r = client.post(
            "/observed-results",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "data": [
                    {
                        "id_obs": "e2e-obs-lifecycle-01",
                        "model_name": E2E_MODEL,
                        "date_time": datetime.utcnow().isoformat(),
                        "observed_result": 0,
                    }
                ]
            },
        )
        assert r.status_code == 200
        assert r.json()["upserted"] >= 1

    def test_06_admin_checks_model_history(self):
        """Historique du modèle visible → au moins une entrée."""
        r = client.get(
            f"/models/{E2E_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1

    def test_07_admin_retrains_model(self):
        """Retrain → nouvelle version 2.0.0 créée."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{E2E_MODEL}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2025-01-01",
                    "end_date": "2025-12-31",
                    "new_version": "2.0.0",
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["new_version"] == "2.0.0"

    def test_08_new_version_visible_in_models_list(self):
        """Après retrain, version 2.0.0 visible dans GET /models."""
        r = client.get(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        versions = [
            m["version"] for m in r.json() if m.get("name") == E2E_MODEL
        ]
        assert "2.0.0" in versions

    def test_09_admin_sets_new_version_as_production(self):
        """Retrain avec set_production=True → nouvelle version est en production."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{E2E_MODEL}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "new_version": "3.0.0",
                    "set_production": True,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["new_model_metadata"]["is_production"] is True

    def test_10_monitoring_overview_includes_model(self):
        """Le monitoring global inclut le modèle E2E après prédictions."""
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
        assert E2E_MODEL in model_names
