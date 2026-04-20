"""
Tests pour POST /observed-results/upload-csv
"""
import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-csv-upload"


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_csv_upload",
                email="test_csv_upload@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())

HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}

VALID_CSV = (
    "id_obs,model_name,observed_result,date_time\n"
    "obs-csv-1,iris_model,1,2024-06-01T10:00:00\n"
    "obs-csv-2,iris_model,0,2024-06-01 11:00:00\n"
)


def _upload(csv_text: str, model_name: str | None = None, token: str = TEST_TOKEN, filename: str = "test.csv"):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = {}
    if model_name is not None:
        data["model_name"] = model_name
    return client.post(
        "/observed-results/upload-csv",
        headers=headers,
        files={"file": (filename, csv_text.encode(), "text/csv")},
        data=data,
    )


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_upload_csv_without_auth():
    r = client.post(
        "/observed-results/upload-csv",
        files={"file": ("t.csv", VALID_CSV.encode(), "text/csv")},
    )
    assert r.status_code in (401, 403)


def test_upload_csv_invalid_token():
    r = _upload(VALID_CSV, token="bad-token-xyz")
    assert r.status_code == 401


# ── Happy path ────────────────────────────────────────────────────────────────

def test_upload_valid_csv_all_rows_imported():
    r = _upload(VALID_CSV)
    assert r.status_code == 200
    body = r.json()
    assert body["upserted"] == 2
    assert body["skipped_rows"] == 0
    assert body["parse_errors"] == []
    assert body["filename"] == "test.csv"


def test_upload_csv_upserts_on_conflict():
    csv_first = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-upsert-csv,iris_model,0,2024-06-01T12:00:00\n"
    )
    csv_second = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-upsert-csv,iris_model,42,2024-06-01T12:00:00\n"
    )
    _upload(csv_first)
    _upload(csv_second)

    r = client.get(
        "/observed-results",
        headers=HEADERS,
        params={"id_obs": "obs-upsert-csv", "model_name": "iris_model"},
    )
    assert r.status_code == 200
    assert r.json()["results"][0]["observed_result"] == 42


def test_upload_csv_model_name_override():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-override-csv,original_model,1,2024-06-01T10:00:00\n"
    )
    r = _upload(csv_text, model_name="override_model")
    assert r.status_code == 200
    assert r.json()["upserted"] == 1

    r2 = client.get(
        "/observed-results",
        headers=HEADERS,
        params={"id_obs": "obs-override-csv", "model_name": "override_model"},
    )
    assert r2.json()["total"] == 1


def test_upload_csv_date_formats():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-date-iso,m,1,2024-06-01T10:00:00\n"
        "obs-date-space,m,2,2024-06-01 10:00:00\n"
        "obs-date-ms,m,3,2024-06-01T10:00:00.123456\n"
        "obs-date-ymd,m,4,2024-06-01\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    assert r.json()["upserted"] == 4
    assert r.json()["skipped_rows"] == 0


def test_upload_csv_observed_result_types():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-type-int,m,5,2024-06-01T10:00:00\n"
        "obs-type-float,m,3.14,2024-06-01T10:00:00\n"
        "obs-type-str,m,approved,2024-06-01T10:00:00\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    assert r.json()["upserted"] == 3


def test_upload_csv_header_only_returns_zero():
    r = _upload("id_obs,model_name,observed_result,date_time\n")
    assert r.status_code == 200
    body = r.json()
    assert body["upserted"] == 0
    assert body["skipped_rows"] == 0


# ── Partial success / parse errors ────────────────────────────────────────────

def test_upload_csv_missing_id_obs_skipped():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        ",iris_model,1,2024-06-01T10:00:00\n"
        "obs-valid-1,iris_model,2,2024-06-01T10:00:00\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    body = r.json()
    assert body["upserted"] == 1
    assert body["skipped_rows"] == 1
    assert body["parse_errors"][0]["row"] == 2
    assert "id_obs" in body["parse_errors"][0]["reason"]


def test_upload_csv_missing_model_name_skipped():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-nomodel,,1,2024-06-01T10:00:00\n"
        "obs-hasmodel,iris_model,1,2024-06-01T10:00:00\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    body = r.json()
    assert body["skipped_rows"] == 1
    assert "model_name" in body["parse_errors"][0]["reason"]


def test_upload_csv_missing_observed_result_skipped():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-nores,iris_model,,2024-06-01T10:00:00\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    body = r.json()
    assert body["skipped_rows"] == 1
    assert "observed_result" in body["parse_errors"][0]["reason"]


def test_upload_csv_invalid_date_skipped():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-baddate,iris_model,1,not-a-date\n"
        "obs-gooddate,iris_model,2,2024-06-01T10:00:00\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    body = r.json()
    assert body["upserted"] == 1
    assert body["skipped_rows"] == 1
    assert "date" in body["parse_errors"][0]["reason"]


def test_upload_csv_mixed_valid_and_invalid():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "obs-mix-ok-1,iris_model,1,2024-06-01T10:00:00\n"
        ",iris_model,1,2024-06-01T10:00:00\n"
        "obs-mix-ok-2,iris_model,2,2024-06-02T10:00:00\n"
        "obs-mix-bad,,3,2024-06-03T10:00:00\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    body = r.json()
    assert body["upserted"] == 2
    assert body["skipped_rows"] == 2
    assert len(body["parse_errors"]) == 2


def test_upload_csv_parse_errors_contain_row_numbers():
    csv_text = (
        "id_obs,model_name,observed_result,date_time\n"
        "ok-1,m,1,2024-06-01T10:00:00\n"
        ",m,1,2024-06-01T10:00:00\n"
    )
    r = _upload(csv_text)
    assert r.status_code == 200
    errors = r.json()["parse_errors"]
    assert errors[0]["row"] == 3  # header=1, first data row=2, second=3


# ── Size limit ────────────────────────────────────────────────────────────────

def test_upload_csv_too_large_rejected():
    large_content = b"id_obs,model_name,observed_result,date_time\n" + b"x" * (10 * 1024 * 1024 + 1)
    r = client.post(
        "/observed-results/upload-csv",
        headers=HEADERS,
        files={"file": ("big.csv", large_content, "text/csv")},
    )
    assert r.status_code == 422
    assert "10 MB" in r.json()["detail"]
