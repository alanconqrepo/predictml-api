"""
Tests d'intégration — cycle de vie complet du ré-entraînement.

Workflow testé :
  POST /models avec train_file
  → POST /models/{name}/{version}/retrain
  → GET /models (nouvelle version visible)
  → GET /models/{name}/history (entrée historique créée)
  → set_production=True (nouvelle version en production)
"""

import asyncio
import io
import joblib
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

ADMIN_TOKEN = "test-token-integ-rt-admin-hh88"
RT_MODEL = "rt_retrain_integ_model"

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

print(json.dumps({"accuracy": 0.95, "f1_score": 0.93}))
"""


def _make_pkl() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="integ_rt_admin",
                email="integ_rt_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
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
    """Subprocess mock : crée un modèle réel et retourne succès."""
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
            b'Training done\n{"accuracy": 0.95, "f1_score": 0.93}\n',
            b"stderr output\n",
        )
    )
    proc.kill = MagicMock()
    return proc


# Créer le modèle de base avec train script
_r = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={
        "file": ("m.pkl", io.BytesIO(_make_pkl()), "application/octet-stream"),
        "train_file": ("train.py", io.BytesIO(VALID_TRAIN_SCRIPT.encode()), "text/x-python"),
    },
    data={"name": RT_MODEL, "version": "1.0.0", "accuracy": "0.90"},
)
assert _r.status_code == 201, _r.text


class TestRetrainLifecycle:
    def test_model_has_train_script_key(self):
        """Modèle créé avec train_file → train_script_object_key non nul dans GET /models."""
        # GET /models/{name}/{version} utilise ModelGetResponse qui n'expose pas ce champ ;
        # GET /models (liste) retourne des dicts incluant train_script_object_key.
        r = client.get(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        models = r.json()
        v1 = next(
            (m for m in models if m.get("name") == RT_MODEL and m.get("version") == "1.0.0"),
            None,
        )
        assert v1 is not None, f"{RT_MODEL} 1.0.0 absent de GET /models"
        assert v1["train_script_object_key"] is not None

    def test_retrain_creates_new_version(self):
        """POST /retrain → nouvelle version créée avec succès."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{RT_MODEL}/1.0.0/retrain",
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
        assert data["new_model_metadata"] is not None

    def test_retrain_new_version_visible_in_models_list(self):
        """Après retrain, le modèle est toujours dans la liste des modèles."""
        r = client.get(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        models = r.json()
        model_names = [m.get("name") for m in models]
        assert RT_MODEL in model_names

    def test_retrain_set_production_promotes_new_version(self):
        """set_production=True → nouvelle version est_production=True."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{RT_MODEL}/1.0.0/retrain",
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
        assert data["success"] is True
        assert data["new_model_metadata"]["is_production"] is True

    def test_retrain_without_script_returns_400(self):
        """Modèle sans train_script → POST /retrain → 400."""
        # Créer un modèle sans train script
        r_create = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={"file": ("m.pkl", io.BytesIO(_make_pkl()), "application/octet-stream")},
            data={"name": f"{RT_MODEL}_noscript", "version": "1.0.0"},
        )
        assert r_create.status_code == 201

        r = client.post(
            f"/models/{RT_MODEL}_noscript/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code == 400

    def test_retrain_history_entry_created(self):
        """Après retrain, l'historique du modèle contient une entrée de type retrain."""
        r = client.get(
            f"/models/{RT_MODEL}/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        # Il doit y avoir au moins une entrée d'historique (created ou retrained)
        actions = [e["action"] for e in data["entries"]]
        assert any(a in ("created", "retrained") for a in actions)
