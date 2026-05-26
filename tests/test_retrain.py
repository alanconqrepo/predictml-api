"""
Tests for the retraining feature.

Covers:
- _validate_train_script(): static validation (unit tests)
- POST /models with train_file: upload + validation at creation
- POST /models/{name}/{version}/retrain: retraining (mocked subprocess)
  - Auth / permissions
  - Non-existent model or missing script
  - Success: new version created, metrics extracted from stdout
  - set_production=True: new version promoted to production
  - Script failure (returncode != 0)
  - Script that does not produce the model file
  - Version conflict (409)
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
# Reference script respecting the contract
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
    """Create a model in the database via POST /models."""
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
# User setup
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

# Configure the MinIO mock for download_file_bytes (used by the retrain endpoint)
_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


# ---------------------------------------------------------------------------
# asyncio subprocess mocks
# ---------------------------------------------------------------------------


async def _mock_exec_success(*args, **kwargs):
    """Subprocess mock (legacy format — kept for compatibility).
    Returns stdout/stderr via communicate() for tests that need it."""
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
    """Subprocess mock: returncode != 0 → script failure."""
    proc = MagicMock()
    proc.returncode = 1
    proc.communicate = AsyncMock(return_value=(b"", b"Error: fichier introuvable\n"))
    proc.kill = MagicMock()
    return proc


async def _mock_exec_no_output_file(*args, **kwargs):
    """Subprocess mock: returncode 0 but does NOT write the output file."""
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"{}", b"Finished (sans sauvegarder)\n"))
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Subprocess mocks for retrain_service (line-by-line reading via readline)
# ---------------------------------------------------------------------------


async def _mock_exec_success_service(*args, **kwargs):
    """Mock for retrain_service.do_retrain() — uses readline() + stderr.read()."""
    env = kwargs.get("env", {})
    output_path = env.get("OUTPUT_MODEL_PATH", "")
    if output_path:
        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=200).fit(X, y)
        with open(output_path, "wb") as f:
            joblib.dump(model, f)

    stdout_lines = [
        b"Training 80 samples\n",
        b'{"accuracy": 0.95, "f1_score": 0.93}\n',
        b"",  # EOF
    ]
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = AsyncMock()
    proc.stdout.readline = AsyncMock(side_effect=stdout_lines)
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"[train.py] Done\n")
    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    return proc


async def _mock_exec_failure_service(*args, **kwargs):
    """Mock retrain_service — returncode != 0."""
    proc = MagicMock()
    proc.returncode = 1
    proc.stdout = AsyncMock()
    proc.stdout.readline = AsyncMock(side_effect=[b""])  # immediate EOF
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"Error: fichier introuvable\n")
    proc.wait = AsyncMock(return_value=1)
    proc.kill = MagicMock()
    return proc


async def _mock_exec_no_output_service(*args, **kwargs):
    """Mock retrain_service — returncode 0 but no .joblib produced."""
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = AsyncMock()
    proc.stdout.readline = AsyncMock(side_effect=[b"{}\\n", b""])
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"Finished (sans sauvegarder)\n")
    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Unit tests — _validate_train_script()
# ---------------------------------------------------------------------------


class TestValidateTrainScript:
    def test_valid_script_returns_none(self):
        """A script respecting all constraints → None (valid)."""
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
        """Script without any save call → error (valid syntax but no save call)."""
        # Syntactically valid script with all required tokens, but no save call
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
        """joblib.dump is accepted by the validator."""
        # VALID_TRAIN_SCRIPT already uses joblib.dump — validation must pass
        assert _validate_train_script(VALID_TRAIN_SCRIPT) is None

    def test_save_model_accepted(self):
        """save_model(...) is a valid alternative to joblib.dump."""
        script = VALID_TRAIN_SCRIPT.replace("joblib.dump(model, f)", "save_model(model, f)")
        assert _validate_train_script(script) is None

    def test_empty_string_returns_error(self):
        """Empty script → error (no required tokens present)."""
        assert _validate_train_script("") is not None

    def test_all_tokens_present_but_no_save(self):
        """All env tokens present but no save call → error."""
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
        """import subprocess → rejected (module not in allowlist)."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport subprocess\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "subprocess" in error

    def test_disallowed_import_requests_rejected(self):
        """import requests → rejected (module not in allowlist)."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport requests\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "requests" in error

    def test_disallowed_from_import_socket_rejected(self):
        """from socket import create_connection → rejected."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nfrom socket import create_connection\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "socket" in error

    def test_disallowed_import_submodule_rejected(self):
        """import urllib.request → rejected (urllib not in allowlist)."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport urllib.request\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "urllib" in error

    def test_allowed_sklearn_submodule_accepted(self):
        """from sklearn.linear_model import LogisticRegression → allowed."""
        assert _validate_train_script(VALID_TRAIN_SCRIPT) is None

    def test_allowed_numpy_import_accepted(self):
        """import numpy as np → allowed."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport numpy as np\n"
        )
        assert _validate_train_script(script) is None

    def test_disallowed_import_ctypes_rejected(self):
        """import ctypes → rejected."""
        script = VALID_TRAIN_SCRIPT.replace(
            "import os\n", "import os\nimport ctypes\n"
        )
        error = _validate_train_script(script)
        assert error is not None
        assert "ctypes" in error


# ---------------------------------------------------------------------------
# Tests — POST /models with train_file
# ---------------------------------------------------------------------------


class TestCreateModelWithTrainFile:
    def test_upload_with_valid_train_file_sets_key(self):
        """POST /models with a valid train.py → train_script_object_key not null."""
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
        expected_key = f"{MODEL_PREFIX}_with_script/1.0.0/train.py"
        assert data["train_script_object_key"] == expected_key

    def test_upload_without_train_file_key_is_null(self):
        """POST /models without train.py → train_script_object_key is null."""
        r = client.post(
            "/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            files={"file": ("m.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
            data={"name": f"{MODEL_PREFIX}_no_script", "version": "1.0.0"},
        )
        assert r.status_code == 201
        assert r.json()["train_script_object_key"] is None

    def test_upload_with_invalid_syntax_returns_422(self):
        """train.py with invalid syntax → 422."""
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
        """train.py without TRAIN_START_DATE → 422."""
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
        """train.py without OUTPUT_MODEL_PATH → 422."""
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
        """train.py without joblib.dump/save_model → 422."""
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
        """Empty train.py → 400."""
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
        """Create the models needed for retrain tests."""
        # Model with train.py script
        cls.model_with_script = _create_model(
            f"{MODEL_PREFIX}_has_script", "1.0.0", with_train_script=True
        )
        # Model without script
        cls.model_no_script = _create_model(
            f"{MODEL_PREFIX}_no_script_rt", "1.0.0", with_train_script=False
        )

    # --- Auth / permissions ---

    def test_retrain_without_auth_returns_401(self):
        """POST /retrain without Authorization header → 401."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code in [401, 403]

    def test_retrain_with_non_admin_token_returns_403(self):
        """POST /retrain with a non-admin user token → 403."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code == 403

    # --- Business errors ---

    def test_retrain_model_not_found_returns_404(self):
        """POST /retrain on a non-existent model → 404."""
        r = client.post(
            "/models/inexistant_model_xyz/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code == 404

    def test_retrain_without_train_script_returns_400(self):
        """POST /retrain on a model without train_script_object_key → 400."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_no_script_rt/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"start_date": "2025-01-01", "end_date": "2025-12-31"},
        )
        assert r.status_code == 400
        assert "script" in r.json()["detail"].lower() or "train_script" in r.json()["detail"]

    def test_retrain_conflict_if_version_exists(self):
        """POST /retrain with an already existing new_version → 409 (checked before enqueue)."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "new_version": "1.0.0",  # already existing version
            },
        )
        assert r.status_code == 409
        assert "existe déjà" in r.json()["detail"]

    # --- Success (202 + job_id with ARQ) ---

    def test_retrain_enqueues_job_returns_202(self):
        """POST /retrain → 202 Accepted with job_id and queued status."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "new_version": "2.0.0",
            },
        )
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["model_name"] == f"{MODEL_PREFIX}_has_script"
        assert data["model_version"] == "1.0.0"
        assert data["new_version"] == "2.0.0"
        assert data["triggered_by"] == "retrain_admin"

    def test_retrain_enqueues_job_with_set_production(self):
        """POST /retrain with set_production=True → 202, job will be promoted after execution."""
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
        assert r.status_code == 202
        data = r.json()
        assert data["job_id"] is not None
        assert data["status"] == "queued"
        assert data["new_version"] == "3.0.0"

    def test_retrain_auto_generates_version_if_not_provided(self):
        """POST /retrain without new_version → 202 with auto-generated version in job_id."""
        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"start_date": "2020-01-01", "end_date": "2020-12-31"},
        )
        assert r.status_code == 202
        data = r.json()
        assert data["status"] == "queued"
        assert data["new_version"]
        assert data["new_version"] != "1.0.0"
        # The auto-generated version contains the timestamp in format "1.0.0-retrain-YYYYMMDDHHMMSS"
        assert "retrain" in data["new_version"]

    def test_retrain_creates_task_run_in_db(self):
        """POST /retrain → TaskRun created in DB with status=queued."""
        import asyncio
        from src.db.models.task_run import TaskRun
        from tests.conftest import _TestSessionLocal

        r = client.post(
            f"/models/{MODEL_PREFIX}_has_script/1.0.0/retrain",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
                "new_version": "7.0.0",
            },
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        async def _check():
            from sqlalchemy import select
            import uuid
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(TaskRun).where(TaskRun.id == uuid.UUID(job_id))
                )
                row = result.scalar_one_or_none()
            return row

        row = asyncio.run(_check())
        assert row is not None
        assert row.status == "queued"
        assert row.model_name == f"{MODEL_PREFIX}_has_script"
        assert row.new_version == "7.0.0"

    # --- Unit tests for retrain logic (via retrain_service) ---

    def test_retrain_service_success(self):
        """retrain_service.do_retrain() success → new version created + metrics."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.services.retrain_service import do_retrain

        async def _run():
            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_mock_exec_success_service),
            ):
                return await do_retrain(
                    model_name=f"{MODEL_PREFIX}_has_script",
                    source_version="1.0.0",
                    new_version="service_2.0.0",
                    start_date="2025-01-01",
                    end_date="2025-12-31",
                    set_production=False,
                    triggered_by="test_admin",
                )

        result = asyncio.run(_run())
        assert result["success"] is True
        assert result["new_version"] == "service_2.0.0"
        assert result["accuracy"] == pytest.approx(0.95, abs=1e-6)
        assert result["f1_score"] == pytest.approx(0.93, abs=1e-6)

    def test_retrain_service_script_failure(self):
        """retrain_service.do_retrain() with failed script → success=False."""
        import asyncio
        from unittest.mock import AsyncMock, patch
        from src.services.retrain_service import do_retrain

        async def _run():
            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_mock_exec_failure_service),
            ):
                return await do_retrain(
                    model_name=f"{MODEL_PREFIX}_has_script",
                    source_version="1.0.0",
                    new_version="service_99.0.0",
                    start_date="2019-01-01",
                    end_date="2019-12-31",
                )

        result = asyncio.run(_run())
        assert result["success"] is False
        assert result["error"] is not None
        assert "code 1" in result["error"]

    def test_retrain_service_no_output_file(self):
        """retrain_service.do_retrain() with no .joblib produced → success=False."""
        import asyncio
        from unittest.mock import AsyncMock, patch
        from src.services.retrain_service import do_retrain

        async def _run():
            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_mock_exec_no_output_service),
            ):
                return await do_retrain(
                    model_name=f"{MODEL_PREFIX}_has_script",
                    source_version="1.0.0",
                    new_version="service_98.0.0",
                    start_date="2018-01-01",
                    end_date="2018-12-31",
                )

        result = asyncio.run(_run())
        assert result["success"] is False
        assert "OUTPUT_MODEL_PATH" in result["error"] or "joblib" in result["error"].lower()
