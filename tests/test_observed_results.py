"""
Tests for the POST /observed-results and GET /observed-results endpoints.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-observed-results"
NOW = datetime.now(timezone.utc).replace(tzinfo=None)
DT = NOW.isoformat()
START = (NOW - timedelta(hours=1)).isoformat()
END = (NOW + timedelta(hours=1)).isoformat()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_observed",
                email="test_observed@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


# ── POST /observed-results ────────────────────────────────────────────────────

def test_upsert_without_auth():
    response = client.post(
        "/observed-results",
        json={"data": [{"id_obs": "obs-1", "model_name": "m", "date_time": DT, "observed_result": 1}]},
    )
    assert response.status_code in [401, 403]


def test_upsert_with_invalid_token():
    response = client.post(
        "/observed-results",
        headers={"Authorization": "Bearer bad-token"},
        json={"data": [{"id_obs": "obs-1", "model_name": "m", "date_time": DT, "observed_result": 1}]},
    )
    assert response.status_code == 401


def test_upsert_insert_single():
    """Insert a single row — returns upserted=1."""
    response = client.post(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={
            "data": [
                {
                    "id_obs": "obs-single",
                    "model_name": "iris_model",
                    "date_time": DT,
                    "observed_result": 0,
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["upserted"] >= 1


def test_upsert_insert_multiple():
    """Insert multiple rows."""
    response = client.post(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={
            "data": [
                {"id_obs": "obs-multi-1", "model_name": "iris_model", "date_time": DT, "observed_result": 1},
                {"id_obs": "obs-multi-2", "model_name": "iris_model", "date_time": DT, "observed_result": 2},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["upserted"] >= 2


def test_upsert_overwrites_existing():
    """
    Submitting the same (id_obs, model_name) twice with different values
    must overwrite the row — the second value is retained.
    """
    payload_first = {
        "data": [
            {"id_obs": "obs-overwrite", "model_name": "iris_model", "date_time": DT, "observed_result": 0}
        ]
    }
    payload_second = {
        "data": [
            {"id_obs": "obs-overwrite", "model_name": "iris_model", "date_time": DT, "observed_result": 99}
        ]
    }
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

    client.post("/observed-results", headers=headers, json=payload_first)
    client.post("/observed-results", headers=headers, json=payload_second)

    # Verify value via GET
    r = client.get(
        "/observed-results",
        headers=headers,
        params={"id_obs": "obs-overwrite", "model_name": "iris_model"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert data["results"][0]["observed_result"] == 99


def test_upsert_empty_data_rejected():
    """An empty list must raise a 422."""
    response = client.post(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={"data": []},
    )
    assert response.status_code == 422


def test_upsert_string_result():
    """observed_result can be a string."""
    response = client.post(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        json={
            "data": [
                {"id_obs": "obs-str", "model_name": "loan_model", "date_time": DT, "observed_result": "approved"}
            ]
        },
    )
    assert response.status_code == 200


# ── GET /observed-results ─────────────────────────────────────────────────────

def test_get_without_auth():
    response = client.get("/observed-results")
    assert response.status_code in [401, 403]


def test_get_no_filters_returns_results():
    """Without filters, returns all rows with pagination."""
    response = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "results" in data
    assert data["limit"] == 100
    assert data["offset"] == 0


def test_get_filter_by_model_name():
    response = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"model_name": "iris_model"},
    )
    assert response.status_code == 200
    data = response.json()
    for r in data["results"]:
        assert r["model_name"] == "iris_model"


def test_get_filter_by_id_obs():
    response = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"id_obs": "obs-single"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["id_obs"] == "obs-single"


def test_get_filter_by_date_range():
    response = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"start": START, "end": END},
    )
    assert response.status_code == 200
    assert response.json()["total"] >= 0


def test_get_start_after_end_rejected():
    response = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"start": END, "end": START},
    )
    assert response.status_code == 422


def test_get_unknown_model_returns_empty():
    response = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"model_name": "modele_inexistant_xyz"},
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_get_pagination_fields():
    response = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"limit": 5, "offset": 0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 5
    assert data["offset"] == 0
    assert len(data["results"]) <= 5


def test_get_response_contains_username():
    """Each result exposes the username of the submitter."""
    r = client.get(
        "/observed-results",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"id_obs": "obs-single"},
    )
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["username"] == "test_observed"


# ── GET /observed-results/stats ───────────────────────────────────────────────

def test_stats_requires_auth():
    r = client.get("/observed-results/stats")
    assert r.status_code in [401, 403]


def test_stats_global_structure():
    """Without model_name: returns global fields and by_model."""
    r = client.get(
        "/observed-results/stats",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "total_predictions" in data
    assert "labeled_count" in data
    assert "coverage_rate" in data
    assert data["model_name"] is None
    assert "by_model" in data
    assert data["by_version"] is None


def test_stats_model_filter_structure():
    """With model_name: returns model fields and by_version."""
    r = client.get(
        "/observed-results/stats",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"model_name": "iris_model"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "iris_model"
    assert "total_predictions" in data
    assert "labeled_count" in data
    assert "coverage_rate" in data
    assert "by_version" in data
    assert data["by_model"] is None


def test_stats_coverage_rate_range():
    """coverage_rate must be in [0, 1]."""
    r = client.get(
        "/observed-results/stats",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["coverage_rate"] <= 1.0


def test_stats_unknown_model_returns_zeros():
    """An unknown model returns 0 everywhere without error."""
    r = client.get(
        "/observed-results/stats",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        params={"model_name": "modele_inexistant_xyz"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_predictions"] == 0
    assert data["labeled_count"] == 0
    assert data["coverage_rate"] == 0.0
    assert data["by_version"] == []


def test_stats_labeled_lte_total():
    """labeled_count cannot exceed total_predictions."""
    r = client.get(
        "/observed-results/stats",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["labeled_count"] <= data["total_predictions"]


def test_stats_by_model_coverage_consistent():
    """For each model in by_model, coverage = labeled / predictions."""
    r = client.get(
        "/observed-results/stats",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    for m in r.json().get("by_model") or []:
        if m["predictions"] > 0:
            expected = round(m["labeled"] / m["predictions"], 3)
            assert abs(m["coverage"] - expected) < 0.001
