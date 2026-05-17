"""
Tests pour la fonctionnalité de ré-entraînement.

Couvre :
- _validate_train_script() : validation statique (unit tests)
- POST /models avec train_file  : upload + validation à la création
- POST /models/{name}/{version}/retrain : ré-entraînement (subprocess mocké)
  - Auth / permissions
  - Modèle inexistant ou sans script
  - Succès : nouvelle version créée, métriques extraites depuis stdout
  - set_production=True : nouvelle version promue en production
  - Échec script (returncode != 0)
  - Script qui ne produit pas le fichier modèle
  - Version en conflit (409)
"""

import asyncio
import io
import json
import joblib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.api.models import _validate_train_script
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

ADMIN_TOKEN = "test-token-retrain-admin-zz77"
USER_TOKEN = "test-token-retrain-user-ww66"
MODEL_PREFIX = "retrain_model"

# ---------------------------------------------------------------------------
# Script de référence respectant le contrat
# ---------------------------------------------------------------------------

VALID_TRAIN_SCRIPT = """\
import os
import joblib
import json
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    joblib.dump(model, f)

print(json.dumps({"accuracy": 0.97, "f1_score": 0.96}))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _create_model(name: str, version: str = "1.0.0", with_train_script: bool = False) -> dict:
    """Crée un modèle en base via POST /models."""
    files: dict = {
        "file": ("model.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
    }
    if with_train_script:
        files["train_file"] = (
            "train.py",
            io.BytesIO(VALID_TRAIN_SCRIPT.encode()),
            "text/x-python",
        )
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files=files,
        data={"name": name, "version": version, "accuracy": "0.90", "f1_score": "0.89"},
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Setup utilisateurs
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="retrain_admin",
                email="retrain_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="retrain_user",
                email="retrain_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())

# Configurer le mock MinIO pour download_file_bytes (utilisé par le endpoint retrain)
_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


# ---------------------------------------------------------------------------
# Mocks du sous-processus asyncio
# ---------------------------------------------------------------------------


async def _mock_exec_success(*args, **kwargs):
    """Subprocess mock : écrit un modèle factice et retourne succès (code 0)."""
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
            b'Training 80 samples\n{"accuracy": 0.95, "f1_score": 0.93}\n',
            b"[train.py] Done\n",
        )
    )
    proc.kill = MagicMock()
    return proc


async def _mock_exec_failure(*args, **kwargs):
    """Subprocess mock : returncode != 0 → échec du script."""
    proc = MagicMock()
    proc.returncode = 1
    proc.communicate = AsyncMock(return_value=(b"", b"Error: fichier introuvable\n"))
    proc.kill = MagicMock()
    return proc


async def _mock_exec_no_output_file(*args, **kwargs):
    """Subprocess mock : returncode 0 mais n'écrit PAS le fichier output."""
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"{}", b"Finished (sans sauvegarder)\n"))
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Tests unitaires — _validate_train_script()
# ---------------------------------------------------------------------------


class TestValidateTrainScript:
    def test_valid_script_returns_none(self):
        """Un script respectant toutes les contraintes → None (valide)."""
        assert _validate_train_script(VALID_TRAIN_SCRIPT) is None

    def test_invalid_syntax_returns_error(self):
        bad_script = "def broken(\n    pass\n"
        error = _validate_train_script(bad_script)
        assert error is not None
        assert "syntaxe" in error.lower() or "invalid" in error.lower()

    def test_missing_train_start_date_returns_error(self):
        script = VALID_TRAIN_SCRIPT.replace("TRAIN_START_DATE", "START_DATE")
        error = _validate_train_script(script)
        assert error is not None
        assert "TRAIN_START_DATE" in error

    def test_missing_train_end_date_returns_error(self):
        script = VALID_TRAIN_SCRIPT.replace("TRAIN_END_DATE", "END_DATE")
        error = _validate_train_script(script)
        assert error is not None
        assert "TRAIN_END_DATE" in error

    def test_missing_output_model_path_returns_error(self):
        script = VALID_TRAIN_SCRIPT.replace("OUTPUT_MODEL_PATH", "OUTPUT_PATH")
        error = _validate_train_script(script)
        assert error is not None
        assert "OUTPUT_MODEL_PATH" in error

    def test_missing_save_call_returns_error(self):
        """Script sans aucun appel de sauvegarde → erreur (syntaxe valide mais sans save)."""
        # Script syntaxiquement valide avec tous les tokens requis, mais sans save call
        script_no_save = (
            "import os\n"
            "import json\n"
            "TRAIN_START_DATE = os.environ['TRAIN_START_DATE']\n"
            "TRAIN_END_DATE = os.environ['TRAIN_END_DATE']\n"
            "OUTPUT_MODEL_PATH = os.environ['OUTPUT_MODEL_PATH']\n"
            "print(json.dumps({'accuracy': 0.9}))\n"
        )
        error = _validate_train_script(script_no_save)
        assert error is not None
        assert "joblib.dump" in error or "sauvegarder" in error.lower() or "save" in error.lower()

    def test_joblib_dump_accepted(self):
        """joblib.dump est accepté par le validateur."""
        # VALID_TRAIN_SCRIPT utilise déjà joblib.dump — la validation doit passer
        assert _validate_train_script(VALID_TRAIN_SCRIPT) is None

    def test_save_model_accepted(self):
        """save_model(...) est une alternative valide à joblib.dump."""
        script = VALID_TRAIN_SCRIPT.replace("joblib.dump(model, f)", "save_model(model, f)")
        assert _validate_train_script(script) is None

    def test_empty_string_returns_error(self):
        """Script vide → erreur (aucun token requis présent)."""
        assert _validate_train_script("") is not None

    def test_all_tokens_present_but_no_save(self):
        """Tous les tokens d'env présents mais pas de sauvegarde → erreur."""
        script = (
            "import os\n"
            "TRAIN_START_DATE = os.environ['TRAIN_START_DATE']\n"
            "TRAIN_END_DATE = os.environ['TRAIN_END_DATE']\n"
            "OUTPUT_MODEL_PATH = os.environ['OUTPUT_MODEL_PATH']\n"
            "print('done')\n"
        )
        error = _validate_train_script(script)
        assert error is not None

    def test_disallowed_import_subprocess_rejected(self):
        """import subprocess → rejeté (module hors allowlist)."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport subprocess\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "subprocess" in error

    def test_disallowed_import_requests_rejected(self):
        """import requests → rejeté (module hors allowlist)."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport requests\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "requests" in error

    def test_disallowed_from_import_socket_rejected(self):
        """from socket import create_connection → rejeté."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nfrom socket import create_connection\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "socket" in error

    def test_disallowed_import_submodule_rejected(self):
        """import urllib.request → rejeté (urllib hors allowlist)."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport urllib.request\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "urllib" in error

    def test_allowed_sklearn_submodule_accepted(self):
        """from sklearn.linear_model import LogisticRegression → autorisé."""
        assert _validate_train_script(VALID_TRAIN_SCRIPT) is None

    def test_allowed_numpy_import_accepted(self):
        """import numpy as np → autorisé."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport numpy as np\n"
        )
        assert _validate_train_script(script) is None

    def test_disallowed_import_ctypes_rejected(self):
        """import ctypes → rejeté."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport ctypes\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "ctypes" in error


# ---------------------------------------------------------------------------
# Tests — POST /models avec train_file
# ---------------------------------------------------------------------------


class TestCreateModelWithTrainFile:
    def test_upload_with_valid_train_file_sets_key(self):
        """POST /models avec un train.py valide → train_script_object_key non null."""
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={
                "file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
                "train_file": ("train.py", io.BytesIO(VALID_TRAIN_SCRIPT.encode()), "text/x-python"),
            },
            data={"name": f"{MODEL_PREFIX}_with_script", "version": "1.0.0"},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["train_script_object_key"] is not None
        expected_key = f"{MODEL_PREFIX}_with_script/v1.0.0_train.py"
        assert data["train_script_object_key"] == expected_key

    def test_upload_without_train_file_key_is_null(self):
        """POST /models sans train.py → train_script_object_key est null."""
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={"file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
            data={"name": f"{MODEL_PREFIX}_no_script", "version": "1.0.0"},
        )
        assert r.status_code == 201
        assert r.json()["train_script_object_key"] is None

    def test_upload_with_invalid_syntax_returns_422(self):
        """train.py avec syntaxe invalide → 422."""
        bad_script = b"def broken(\n    pass\n"
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={
                "file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
                "train_file": ("train.py", io.BytesIO(bad_script), "text/x-python"),
            },
            data={"name": f"{MODEL_PREFIX}_bad_syntax", "version": "1.0.0"},
        )
        assert r.status_code == 422
        assert "train.py" in r.json()["detail"].lower() or "syntaxe" in r.json()["detail"].lower()

    def test_upload_missing_train_start_date_returns_422(self):
        """train.py sans TRAIN_START_DATE → 422."""
        script = VALID_TRAIN_SCRIPT.replace("TRAIN_START_DATE", "START_DATE")
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={
                "file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
                "train_file": ("train.py", io.BytesIO(script.encode()), "text/x-python"),
            },
            data={"name": f"{MODEL_PREFIX}_no_start", "version": "1.0.0"},
        )
        assert r.status_code == 422
        assert "TRAIN_START_DATE" in r.json()["detail"]

    def test_upload_missing_output_model_path_returns_422(self):
        """train.py sans OUTPUT_MODEL_PATH → 422."""
        script = VALID_TRAIN_SCRIPT.replace("OUTPUT_MODEL_PATH", "OUTPUT_PATH")
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={
                "file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
                "train_file": ("train.py", io.BytesIO(script.encode()), "text/x-python"),
            },
            data={"name": f"{MODEL_PREFIX}_no_out", "version": "1.0.0"},
        )
        assert r.status_code == 422
        assert "OUTPUT_MODEL_PATH" in r.json()["detail"]

    def test_upload_train_file_without_save_call_returns_422(self):
        """train.py sans joblib.dump/save_model → 422."""
        script = VALID_TRAIN_SCRIPT.replace("joblib.dump(model, f)", "# joblib.dump removed")
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={
                "file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
                "train_file": ("train.py", io.BytesIO(script.encode()), "text/x-python"),
            },
            data={"name": f"{MODEL_PREFIX}_no_save", "version": "1.0.0"},
        )
        assert r.status_code == 422

    def test_upload_empty_train_file_returns_400(self):
        """train.py vide → 400."""
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={
                "file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
                "train_file": ("train.py", io.BytesIO(b""), "text/x-python"),
            },
            data={"name": f"{MODEL_PREFIX}_empty_script", "version": "1.0.0"},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Tests — POST /models/{name}/{version}/retrain
# ---------------------------------------------------------------------------


class TestRetrainEndpoint:

    @classmethod
    def setup_class(cls):
        """Crée les modèles nécessaires aux tests de retrain."""
        # Modèle avec script train.py
        cls.model_with_script = _create_model(
            f"{MODEL_PREFIX}_has_script", "1.0.0", with_train_script=True
        )
        # Modèle sans script
        cls.model_no_script = _create_model(
            f"{MODEL_PREFIX}_no_script_rt", "1.0.0", with_train_script=False
        )

    # --- Auth / permissions ---

    def test_retrain_without_auth_returns_401(self):
        """POST /retrain sans header Authorization → 401."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code in [401, 403]

    def test_retrain_with_non_admin_token_returns_403(self):
        """POST /retrain avec un token user (non-admin) → 403."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code == 403

    # --- Erreurs métier ---

    def test_retrain_model_not_found_returns_404(self):
        """POST /retrain sur un modèle inexistant → 404."""
        r = client.post(
            "/models/inexistant_model_xyz/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code == 404

    def test_retrain_without_train_script_returns_400(self):
        """POST /retrain sur un modèle sans train_script_object_key → 400."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_no_script_rt/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code == 400
        assert "script" in r.json()["detail"].lower() or "train_script" in r.json()["detail"]

    def test_retrain_conflict_if_version_exists(self):
        """POST /retrain avec new_version déjà existante → 409."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2025-01-01",
                    "end_date": "2025-12-31",
                    "new_version": "1.0.0",  # version déjà existante
                },
            )
        assert r.status_code == 409
        assert "existe déjà" in r.json()["detail"]

    # --- Succès ---

    def test_retrain_success_creates_new_version(self):
        """POST /retrain succès → nouvelle version créée dans la DB."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
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
        assert data["model_name"] == f"{MODEL_PREFIX}_has_script"
        assert data["source_version"] == "1.0.0"
        assert data["new_version"] == "2.0.0"
        assert data["error"] is None
        assert data["new_model_metadata"] is not None
        assert data["new_model_metadata"]["version"] == "2.0.0"
        assert data["new_model_metadata"]["is_production"] is False

    def test_retrain_success_extracts_metrics_from_stdout(self):
        """POST /retrain : les métriques JSON sur stdout sont propagées à la nouvelle version."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "new_version": "2.1.0",
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        # Le mock retourne {"accuracy": 0.95, "f1_score": 0.93} dans stdout
        new_meta = data["new_model_metadata"]
        assert new_meta["accuracy"] == pytest.approx(0.95, abs=1e-6)
        assert new_meta["f1_score"] == pytest.approx(0.93, abs=1e-6)

    def test_retrain_success_with_set_production(self):
        """POST /retrain avec set_production=True → nouvelle version en production."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "new_version": "3.0.0",
                    "set_production": True,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["new_model_metadata"]["is_production"] is True

    def test_retrain_success_new_version_has_train_script_key(self):
        """La nouvelle version hérite du train_script_object_key (pour enchaîner les retrains)."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2022-01-01",
                    "end_date": "2022-12-31",
                    "new_version": "4.0.0",
                },
            )
        assert r.status_code == 200
        new_meta = r.json()["new_model_metadata"]
        assert new_meta["train_script_object_key"] is not None

    def test_retrain_success_returns_stdout_stderr(self):
        """POST /retrain : stdout et stderr sont retournés dans la réponse."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2021-01-01",
                    "end_date": "2021-12-31",
                    "new_version": "5.0.0",
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert "stdout" in data
        assert "stderr" in data
        assert len(data["stdout"]) > 0
        assert len(data["stderr"]) > 0

    def test_retrain_auto_generates_version_if_not_provided(self):
        """POST /retrain sans new_version → version auto-générée non vide."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2020-01-01", "end_date": "2020-12-31"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        # La version auto-générée ne doit pas être vide et doit être différente de 1.0.0
        assert data["new_version"]
        assert data["new_version"] != "1.0.0"

    # --- Échecs du script ---

    def test_retrain_script_failure_returns_success_false(self):
        """Script qui retourne code != 0 → success=False dans la réponse (pas d'exception)."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_failure),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2019-01-01",
                    "end_date": "2019-12-31",
                    "new_version": "99.0.0",
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert data["error"] is not None
        assert data["new_model_metadata"] is None
        assert "fichier" in data["stderr"].lower() or len(data["stderr"]) > 0

    def test_retrain_script_no_output_file_returns_success_false(self):
        """Script qui retourne code 0 mais n'écrit pas le .joblib → success=False."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_no_output_file),
        ):
            r = client.post(
                f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2018-01-01",
                    "end_date": "2018-12-31",
                    "new_version": "98.0.0",
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert data["error"] is not None
        assert "OUTPUT_MODEL_PATH" in data["error"] or "modèle" in data["error"].lower()
