"""
Tests pour GET /models — filtres avancés (is_production, algorithm, min_accuracy, deployment_mode)
"""
import asyncio
import io
import pickle

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TOKEN = "test-token-list-filters"
ADMIN_TOKEN = "test-admin-token-list-filters"


def _make_pkl() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TOKEN):
            await DBService.create_user(
                db,
                username="lf_user",
                email="lf_user@test.com",
                api_token=TOKEN,
                role="user",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="lf_admin",
                email="lf_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())

HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def _create(name: str, version: str = "1.0.0", **extra_form) -> dict:
    data = {"name": name, "version": version, **extra_form}
    r = client.post(
        "/models",
        data=data,
        files={"file": (f"{name}.pkl", io.BytesIO(_make_pkl()), "application/octet-stream")},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _set_production(name: str, version: str):
    r = client.patch(
        f"/models/{name}/{version}",
        json={"is_production": True},
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text


def _set_deployment_mode(name: str, version: str, mode: str, weight: float = 0.5):
    r = client.patch(
        f"/models/{name}/{version}",
        json={"deployment_mode": mode, "traffic_weight": weight},
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# is_production filter
# ---------------------------------------------------------------------------

def test_filter_is_production_true():
    _create("lf_rf_prod", "1.0.0", algorithm="RandomForest")
    _create("lf_rf_notprod", "1.0.0", algorithm="RandomForest")
    _set_production("lf_rf_prod", "1.0.0")

    r = client.get("/models", params={"is_production": "true"})
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "lf_rf_prod" in names
    for m in r.json():
        assert m["is_production"] is True


def test_filter_is_production_false():
    _create("lf_notprod_only", "1.0.0")

    r = client.get("/models", params={"is_production": "false"})
    assert r.status_code == 200
    for m in r.json():
        assert m["is_production"] is False


# ---------------------------------------------------------------------------
# algorithm filter
# ---------------------------------------------------------------------------

def test_filter_algorithm_exact_match():
    _create("lf_xgb_model", "1.0.0", algorithm="XGBoost")
    _create("lf_lr_model", "1.0.0", algorithm="LogisticRegression")

    r = client.get("/models", params={"algorithm": "XGBoost"})
    assert r.status_code == 200
    results = r.json()
    xgb_names = [m["name"] for m in results if m["algorithm"] == "XGBoost"]
    lr_names = [m["name"] for m in results if m["algorithm"] == "LogisticRegression"]
    assert "lf_xgb_model" in xgb_names
    assert "lf_lr_model" not in [m["name"] for m in results]
    for m in results:
        assert m["algorithm"] == "XGBoost"


def test_filter_algorithm_no_match():
    r = client.get("/models", params={"algorithm": "NonExistentAlgo999"})
    assert r.status_code == 200
    assert r.json() == []


# ---------------------------------------------------------------------------
# min_accuracy filter
# ---------------------------------------------------------------------------

def test_filter_min_accuracy():
    _create("lf_high_acc", "1.0.0", accuracy="0.95")
    _create("lf_low_acc", "1.0.0", accuracy="0.60")

    r = client.get("/models", params={"min_accuracy": "0.90"})
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "lf_high_acc" in names
    assert "lf_low_acc" not in names
    for m in r.json():
        if m["accuracy"] is not None:
            assert m["accuracy"] >= 0.90


def test_filter_min_accuracy_excludes_null():
    _create("lf_null_acc", "1.0.0")

    r = client.get("/models", params={"min_accuracy": "0.50"})
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "lf_null_acc" not in names


# ---------------------------------------------------------------------------
# deployment_mode filter
# ---------------------------------------------------------------------------

def test_filter_deployment_mode_ab_test():
    _create("lf_ab_model", "1.0.0")
    _create("lf_shadow_model", "1.0.0")
    _set_deployment_mode("lf_ab_model", "1.0.0", "ab_test", 0.5)
    _set_deployment_mode("lf_shadow_model", "1.0.0", "shadow", 0.0)

    r = client.get("/models", params={"deployment_mode": "ab_test"})
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "lf_ab_model" in names
    assert "lf_shadow_model" not in names
    for m in r.json():
        assert m["deployment_mode"] == "ab_test"


def test_filter_deployment_mode_shadow():
    r = client.get("/models", params={"deployment_mode": "shadow"})
    assert r.status_code == 200
    for m in r.json():
        assert m["deployment_mode"] == "shadow"


def test_filter_deployment_mode_invalid_returns_422():
    r = client.get("/models", params={"deployment_mode": "invalid_mode"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Combined filters (AND logic)
# ---------------------------------------------------------------------------

def test_combined_is_production_and_algorithm():
    _create("lf_combo_rf_prod", "1.0.0", algorithm="RandomForest", accuracy="0.92")
    _create("lf_combo_rf_notprod", "1.0.0", algorithm="RandomForest", accuracy="0.88")
    _set_production("lf_combo_rf_prod", "1.0.0")

    r = client.get("/models", params={"is_production": "true", "algorithm": "RandomForest"})
    assert r.status_code == 200
    results = r.json()
    names = [m["name"] for m in results]
    assert "lf_combo_rf_prod" in names
    assert "lf_combo_rf_notprod" not in names
    for m in results:
        assert m["is_production"] is True
        assert m["algorithm"] == "RandomForest"


def test_combined_algorithm_and_min_accuracy():
    _create("lf_combo2_match", "1.0.0", algorithm="GradientBoosting", accuracy="0.93")
    _create("lf_combo2_low", "1.0.0", algorithm="GradientBoosting", accuracy="0.72")
    _create("lf_combo2_other_algo", "1.0.0", algorithm="SVM", accuracy="0.95")

    r = client.get(
        "/models", params={"algorithm": "GradientBoosting", "min_accuracy": "0.90"}
    )
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "lf_combo2_match" in names
    assert "lf_combo2_low" not in names
    assert "lf_combo2_other_algo" not in names


# ---------------------------------------------------------------------------
# tag filter still works alongside new filters
# ---------------------------------------------------------------------------

def test_tag_filter_still_works():
    _create("lf_tagged", "1.0.0", tags='["finance"]', algorithm="DecisionTree")

    r = client.get("/models", params={"tag": "finance", "algorithm": "DecisionTree"})
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "lf_tagged" in names
    for m in r.json():
        assert m["algorithm"] == "DecisionTree"
        assert "finance" in (m.get("tags") or [])


# ---------------------------------------------------------------------------
# No filters returns all models
# ---------------------------------------------------------------------------

def test_no_filters_returns_all_active():
    r = client.get("/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
