"""
Tests pour l'endpoint GET /observed-results/export
"""

import asyncio
import csv
import io
import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-obs-export-admin"
USER_TOKEN = "test-token-obs-export-user"
TEST_MODEL = "obs_export_model"
NOW = datetime.now(timezone.utc).replace(tzinfo=None)
START = (NOW - timedelta(hours=1)).isoformat()
END = (NOW + timedelta(hours=1)).isoformat()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            admin = await DBService.create_user(
                db,
                username="obs_export_admin",
                email="obs_export_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        else:
            admin = await DBService.get_user_by_token(db, ADMIN_TOKEN)

        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="obs_export_user",
                email="obs_export_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )

        # Seed 3 observed results for TEST_MODEL
        records = [
            {
                "id_obs": f"exp_obs_{i}",
                "model_name": TEST_MODEL,
                "observed_result": i,
                "date_time": NOW,
                "user_id": admin.id,
            }
            for i in range(3)
        ]
        await DBService.upsert_observed_results(db, records)


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_export_requires_auth():
    r = client.get(f"/observed-results/export?start={START}&end={END}")
    assert r.status_code == 401


def test_export_requires_admin():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Validation des paramètres
# ---------------------------------------------------------------------------


def test_export_start_after_end():
    r = client.get(
        f"/observed-results/export?start={END}&end={START}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 422


def test_export_invalid_format():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&format=xml",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Format CSV (défaut)
# ---------------------------------------------------------------------------


def test_export_csv_default_format():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert 'attachment; filename="observed_results_export.csv"' in r.headers["content-disposition"]


def test_export_csv_columns():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    reader = csv.DictReader(io.StringIO(r.text))
    assert set(reader.fieldnames) == {"id_obs", "model_name", "observed_result", "date_time"}


def test_export_csv_row_count():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 3


def test_export_csv_values():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    rows = list(csv.DictReader(io.StringIO(r.text)))
    for row in rows:
        assert row["model_name"] == TEST_MODEL
        assert row["id_obs"].startswith("exp_obs_")


# ---------------------------------------------------------------------------
# Format JSONL
# ---------------------------------------------------------------------------


def test_export_jsonl_format():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}&format=jsonl",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    assert "ndjson" in r.headers["content-type"]
    assert 'attachment; filename="observed_results_export.jsonl"' in r.headers["content-disposition"]


def test_export_jsonl_row_count():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}&format=jsonl",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    assert len(lines) == 3


def test_export_jsonl_fields():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}&format=jsonl",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    record = json.loads(lines[0])
    assert set(record.keys()) == {"id_obs", "model_name", "observed_result", "date_time"}


def test_export_jsonl_values():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name={TEST_MODEL}&format=jsonl",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    for line in lines:
        record = json.loads(line)
        assert record["model_name"] == TEST_MODEL


# ---------------------------------------------------------------------------
# model_name optionnel (tous les modèles)
# ---------------------------------------------------------------------------


def test_export_without_model_name_returns_all():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) >= 3


# ---------------------------------------------------------------------------
# Résultat vide — CSV avec seulement l'en-tête
# ---------------------------------------------------------------------------


def test_export_empty_result_csv():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name=model_inexistant",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    assert len(lines) == 1
    assert lines[0] == "id_obs,model_name,observed_result,date_time"


def test_export_empty_result_jsonl():
    r = client.get(
        f"/observed-results/export?start={START}&end={END}&model_name=model_inexistant&format=jsonl",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.text.strip() == ""
