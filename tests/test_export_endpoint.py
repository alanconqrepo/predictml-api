"""
Tests pour l'endpoint GET /predictions/export
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

TEST_TOKEN = "test-token-export-endpoint"
TEST_MODEL = "export_test_model"
NOW = datetime.now(timezone.utc).replace(tzinfo=None)
START = (NOW - timedelta(hours=1)).isoformat()
END = (NOW + timedelta(hours=1)).isoformat()


async def _setup():
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, TEST_TOKEN)
        if not user:
            user = await DBService.create_user(
                db,
                username="test_export_user",
                email="test_export_user@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )
        # Seed 3 success + 1 error predictions
        for i in range(3):
            await DBService.create_prediction(
                db=db,
                user_id=user.id,
                model_name=TEST_MODEL,
                model_version="1.0.0",
                input_features={"sepal_length": 5.0 + i, "sepal_width": 3.5},
                prediction_result=i % 3,
                probabilities=[0.1, 0.8, 0.1],
                response_time_ms=10.0 + i,
                client_ip="127.0.0.1",
                user_agent="test",
                status="success",
                id_obs=f"export_obs_{i}",
            )
        await DBService.create_prediction(
            db=db,
            user_id=user.id,
            model_name=TEST_MODEL,
            model_version="1.0.0",
            input_features={"sepal_length": 9.0},
            prediction_result=None,
            probabilities=None,
            response_time_ms=5.0,
            client_ip="127.0.0.1",
            user_agent="test",
            status="error",
            error_message="feature manquante",
            id_obs="export_obs_err",
        )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_export_requires_auth():
    r = client.get(f"/predictions/export?start={START}&end={END}")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Validation des paramètres
# ---------------------------------------------------------------------------


def test_export_start_after_end():
    r = client.get(
        f"/predictions/export?start={END}&end={START}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 422


def test_export_invalid_format():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&format=xml",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 400


def test_export_invalid_status():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&status=pending",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Format CSV (défaut)
# ---------------------------------------------------------------------------


def test_export_csv_default_format():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert 'attachment; filename="predictions_export.csv"' in r.headers["content-disposition"]


def test_export_csv_columns():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    reader = csv.DictReader(io.StringIO(r.text))
    expected = {
        "id", "timestamp", "model_name", "model_version", "username",
        "id_obs", "prediction_result", "probabilities", "response_time_ms",
        "status", "error_message", "is_shadow", "input_features",
    }
    assert expected == set(reader.fieldnames)


def test_export_csv_row_count():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 4  # 3 success + 1 error


def test_export_csv_values():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    rows = list(csv.DictReader(io.StringIO(r.text)))
    for row in rows:
        assert row["model_name"] == TEST_MODEL
        assert row["model_version"] == "1.0.0"
        assert row["username"] == "test_export_user"


# ---------------------------------------------------------------------------
# Format JSONL
# ---------------------------------------------------------------------------


def test_export_jsonl_format():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}&format=jsonl",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    assert "ndjson" in r.headers["content-type"]
    assert 'attachment; filename="predictions_export.jsonl"' in r.headers["content-disposition"]


def test_export_jsonl_row_count():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}&format=jsonl",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    assert len(lines) == 4


def test_export_jsonl_fields():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}&format=jsonl",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    record = json.loads(lines[0])
    expected = {
        "id", "timestamp", "model_name", "model_version", "username",
        "id_obs", "prediction_result", "probabilities", "response_time_ms",
        "status", "error_message", "is_shadow", "input_features",
    }
    assert expected == set(record.keys())


# ---------------------------------------------------------------------------
# include_features=false
# ---------------------------------------------------------------------------


def test_export_csv_without_features():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}&include_features=false",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    reader = csv.DictReader(io.StringIO(r.text))
    assert "input_features" not in reader.fieldnames


def test_export_jsonl_without_features():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}"
        "&format=jsonl&include_features=false",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    record = json.loads(lines[0])
    assert "input_features" not in record


# ---------------------------------------------------------------------------
# Filtre status
# ---------------------------------------------------------------------------


def test_export_status_filter_success():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}&status=success",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 3
    assert all(row["status"] == "success" for row in rows)


def test_export_status_filter_error():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name={TEST_MODEL}&status=error",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error_message"] == "feature manquante"


# ---------------------------------------------------------------------------
# model_name optionnel (tous les modèles)
# ---------------------------------------------------------------------------


def test_export_without_model_name_returns_all():
    r = client.get(
        f"/predictions/export?start={START}&end={END}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.text)))
    # Au moins nos 4 prédictions de test
    assert len(rows) >= 4


# ---------------------------------------------------------------------------
# Résultat vide — CSV avec seulement l'en-tête
# ---------------------------------------------------------------------------


def test_export_empty_result_csv():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name=model_inexistant",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    lines = [ln for ln in r.text.strip().split("\n") if ln]
    assert len(lines) == 1  # seulement l'en-tête
    assert lines[0].startswith("id,")


def test_export_empty_result_jsonl():
    r = client.get(
        f"/predictions/export?start={START}&end={END}&model_name=model_inexistant&format=jsonl",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    assert r.text.strip() == ""
