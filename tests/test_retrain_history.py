"""
Tests pour GET /models/{name}/retrain-history

Couvre :
- Auth : token manquant → 401, token invalide → 401
- Modèle inconnu → 404
- Modèle sans retrain → liste vide, total=0
- Retrain manuel : entrée correctement construite (metrics, source_version, trained_by)
- auto_promoted persisté dans training_stats
- Pagination (limit/offset)
- Plusieurs retrains : ordre DESC par timestamp
"""

import asyncio
import io
import joblib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.db.models import ModelMetadata
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

ADMIN_TOKEN = "test-token-rh-admin-zz99"
USER_TOKEN = "test-token-rh-user-xx88"
MODEL_PREFIX = "rh_model"


# ---------------------------------------------------------------------------
# Script train.py valide (pour retrain mock)
# ---------------------------------------------------------------------------

VALID_TRAIN_SCRIPT = """\
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    joblib.dump(model, f)
"""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="rh_admin",
                email="rh_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="rh_user",
                email="rh_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())

_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={
            "file": ("model.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
            "train_file": ("train.py", io.BytesIO(VALID_TRAIN_SCRIPT.encode()), "text/x-python"),
        },
        data={"name": name, "version": version, "accuracy": "0.90", "f1_score": "0.88"},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _insert_retrained_version(
    name: str,
    version: str,
    parent_version: str,
    training_stats: dict,
    trained_by: str = "admin",
) -> None:
    """Insère directement une version avec parent_version en base pour simuler un retrain."""
    async with _TestSessionLocal() as db:
        meta = ModelMetadata(
            name=name,
            version=version,
            minio_bucket="models",
            minio_object_key=f"{name}/v{version}.joblib",
            file_size_bytes=512,
            accuracy=training_stats.get("accuracy"),
            f1_score=training_stats.get("f1_score"),
            trained_by=trained_by,
            parent_version=parent_version,
            training_stats=training_stats,
            is_active=True,
            is_production=False,
        )
        db.add(meta)
        await db.commit()


async def _make_subprocess_mock(*args, **kwargs):
    """Subprocess mock : écrit un fichier modèle et retourne succès."""
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
        return_value=(b'{"accuracy": 0.95, "f1_score": 0.93}\n', b"")
    )
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestRetrainHistoryAuth:
    def test_no_token_returns_401(self):
        r = client.get(f"/models/{MODEL_PREFIX}_auth/retrain-history")
        assert r.status_code == 401

    def test_invalid_token_returns_401(self):
        r = client.get(
            f"/models/{MODEL_PREFIX}_auth/retrain-history",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert r.status_code == 401

    def test_user_token_is_accepted(self):
        """Le endpoint est accessible à tout utilisateur authentifié (pas seulement admin)."""
        name = f"{MODEL_PREFIX}_auth_user"
        _create_model(name)
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 404 tests
# ---------------------------------------------------------------------------


class TestRetrainHistoryNotFound:
    def test_unknown_model_returns_404(self):
        r = client.get(
            "/models/unknown_model_xyzabc/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_404_message_contains_model_name(self):
        r = client.get(
            "/models/ghost_model_xyz/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert "ghost_model_xyz" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Empty history
# ---------------------------------------------------------------------------


class TestRetrainHistoryEmpty:
    def test_model_without_retrains_returns_empty_list(self):
        name = f"{MODEL_PREFIX}_no_retrain"
        _create_model(name)
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == name
        assert data["history"] == []
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# Entry content
# ---------------------------------------------------------------------------


class TestRetrainHistoryEntryContent:
    def test_retrained_version_appears_in_history(self):
        name = f"{MODEL_PREFIX}_content"
        _create_model(name, "1.0.0")
        asyncio.run(
            _insert_retrained_version(
                name,
                version="1.1.0",
                parent_version="1.0.0",
                training_stats={
                    "train_start_date": "2026-01-01",
                    "train_end_date": "2026-01-31",
                    "trained_at": "2026-02-01T03:00:00",
                    "n_rows": 5000,
                    "accuracy": 0.95,
                    "f1_score": 0.93,
                },
                trained_by="test_admin",
            )
        )
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        entry = data["history"][0]
        assert entry["new_version"] == "1.1.0"
        assert entry["source_version"] == "1.0.0"
        assert entry["trained_by"] == "test_admin"
        assert entry["accuracy"] == pytest.approx(0.95)
        assert entry["f1_score"] == pytest.approx(0.93)
        assert entry["n_rows"] == 5000
        assert entry["train_start_date"] == "2026-01-01"
        assert entry["train_end_date"] == "2026-01-31"

    def test_auto_promoted_true_appears_in_entry(self):
        name = f"{MODEL_PREFIX}_autopromoted"
        _create_model(name, "1.0.0")
        asyncio.run(
            _insert_retrained_version(
                name,
                version="1.1.0",
                parent_version="1.0.0",
                training_stats={
                    "train_start_date": "2026-01-01",
                    "train_end_date": "2026-01-31",
                    "trained_at": "2026-02-01T03:00:00",
                    "auto_promoted": True,
                    "auto_promote_reason": "all criteria met",
                },
            )
        )
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        entry = r.json()["history"][0]
        assert entry["auto_promoted"] is True
        assert entry["auto_promote_reason"] == "all criteria met"

    def test_auto_promoted_false_appears_in_entry(self):
        name = f"{MODEL_PREFIX}_notpromoted"
        _create_model(name, "1.0.0")
        asyncio.run(
            _insert_retrained_version(
                name,
                version="1.1.0",
                parent_version="1.0.0",
                training_stats={
                    "train_start_date": "2026-01-01",
                    "train_end_date": "2026-01-31",
                    "trained_at": "2026-02-01T03:00:00",
                    "auto_promoted": False,
                    "auto_promote_reason": "insufficient samples",
                },
            )
        )
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        entry = r.json()["history"][0]
        assert entry["auto_promoted"] is False
        assert entry["auto_promote_reason"] == "insufficient samples"

    def test_no_auto_promote_field_is_null(self):
        """Versions sans policy (champ absent de training_stats) → auto_promoted null."""
        name = f"{MODEL_PREFIX}_nopolicy"
        _create_model(name, "1.0.0")
        asyncio.run(
            _insert_retrained_version(
                name,
                version="1.1.0",
                parent_version="1.0.0",
                training_stats={
                    "train_start_date": "2026-01-01",
                    "train_end_date": "2026-01-31",
                    "trained_at": "2026-02-01T03:00:00",
                },
            )
        )
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        entry = r.json()["history"][0]
        assert entry["auto_promoted"] is None

    def test_original_version_not_in_history(self):
        """La version initiale (sans parent_version) n'apparaît pas dans l'historique."""
        name = f"{MODEL_PREFIX}_original_excluded"
        _create_model(name, "1.0.0")
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.json()["total"] == 0


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestRetrainHistoryPagination:
    def test_limit_is_respected(self):
        name = f"{MODEL_PREFIX}_pagination"
        _create_model(name, "1.0.0")
        for i in range(1, 6):
            asyncio.run(
                _insert_retrained_version(
                    name,
                    version=f"1.{i}.0",
                    parent_version="1.0.0",
                    training_stats={"train_start_date": "2026-01-01", "train_end_date": "2026-01-31"},
                )
            )
        r = client.get(
            f"/models/{name}/retrain-history",
            params={"limit": 3},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        data = r.json()
        assert data["total"] == 5
        assert len(data["history"]) == 3

    def test_offset_skips_entries(self):
        name = f"{MODEL_PREFIX}_offset"
        _create_model(name, "1.0.0")
        for i in range(1, 4):
            asyncio.run(
                _insert_retrained_version(
                    name,
                    version=f"2.{i}.0",
                    parent_version="1.0.0",
                    training_stats={"train_start_date": "2026-01-01", "train_end_date": "2026-01-31"},
                )
            )
        r_first = client.get(
            f"/models/{name}/retrain-history",
            params={"limit": 2, "offset": 0},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        r_second = client.get(
            f"/models/{name}/retrain-history",
            params={"limit": 2, "offset": 2},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        first_versions = {e["new_version"] for e in r_first.json()["history"]}
        second_versions = {e["new_version"] for e in r_second.json()["history"]}
        assert first_versions.isdisjoint(second_versions)

    def test_total_reflects_full_count(self):
        name = f"{MODEL_PREFIX}_total"
        _create_model(name, "1.0.0")
        for i in range(1, 5):
            asyncio.run(
                _insert_retrained_version(
                    name,
                    version=f"3.{i}.0",
                    parent_version="1.0.0",
                    training_stats={"train_start_date": "2026-01-01", "train_end_date": "2026-01-31"},
                )
            )
        r = client.get(
            f"/models/{name}/retrain-history",
            params={"limit": 1},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.json()["total"] == 4
        assert len(r.json()["history"]) == 1


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestRetrainHistoryOrdering:
    def test_most_recent_retrain_is_first(self):
        """Les entrées sont triées par created_at DESC — la plus récente est en tête."""
        name = f"{MODEL_PREFIX}_order"
        _create_model(name, "1.0.0")
        for i in range(1, 4):
            asyncio.run(
                _insert_retrained_version(
                    name,
                    version=f"4.{i}.0",
                    parent_version="1.0.0",
                    training_stats={"train_start_date": "2026-01-01", "train_end_date": "2026-01-31"},
                )
            )
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        timestamps = [e["timestamp"] for e in r.json()["history"]]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class TestRetrainHistorySchema:
    def test_response_contains_required_top_level_fields(self):
        name = f"{MODEL_PREFIX}_schema"
        _create_model(name)
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        data = r.json()
        assert "model_name" in data
        assert "history" in data
        assert "total" in data
        assert data["model_name"] == name

    def test_entry_has_all_expected_fields(self):
        name = f"{MODEL_PREFIX}_fields"
        _create_model(name, "1.0.0")
        asyncio.run(
            _insert_retrained_version(
                name,
                version="5.1.0",
                parent_version="1.0.0",
                training_stats={
                    "train_start_date": "2026-03-01",
                    "train_end_date": "2026-03-31",
                    "n_rows": 1234,
                    "auto_promoted": True,
                    "auto_promote_reason": "ok",
                },
            )
        )
        r = client.get(
            f"/models/{name}/retrain-history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        entry = r.json()["history"][0]
        for field in [
            "timestamp", "source_version", "new_version", "trained_by",
            "accuracy", "f1_score", "auto_promoted", "auto_promote_reason",
            "n_rows", "train_start_date", "train_end_date",
        ]:
            assert field in entry, f"Missing field: {field}"
