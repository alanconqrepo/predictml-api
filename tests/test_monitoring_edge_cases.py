"""
Tests pour les cas limites de src/api/monitoring.py.

Couvre :
- _worst_health avec statuts "no_baseline" et "insufficient_data"
- GET /monitoring/model/{name} : feature_drift.baseline_available=False (pas de baseline)
- GET /monitoring/model/{name} : structure de feature_drift quand pas de baseline
- GET /monitoring/overview : champs global_stats (active_models, models_ok/warning/critical)
- GET /monitoring/overview : end == start → 422
"""

import asyncio
import io
import joblib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.api.monitoring import _worst_health
from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-monitoring-edge-001"
MODEL_PREFIX = "mon_edge"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


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
                username="mon_edge_admin",
                email="mon_edge_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
        data={"name": name, "version": version},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _inject_model_cache(name: str, version: str = "1.0.0"):
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    model.feature_names_in_ = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=name, version=version,
            confidence_threshold=None, webhook_url=None,
        ),
    }
    asyncio.run(
        model_service._redis.set(f"model:{name}:{version}", (lambda _b: (joblib.dump(data, _b), _b.getvalue())[1])(io.BytesIO()))
    )


def _predict(model_name: str):
    _inject_model_cache(model_name)
    return client.post(
        "/predict",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "model_name": model_name,
            "features": {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            },
        },
    )


def _period(hours_back: int = 1, hours_forward: int = 1) -> dict:
    now = datetime.utcnow()
    return {
        "start": (now - timedelta(hours=hours_back)).isoformat(),
        "end": (now + timedelta(hours=hours_forward)).isoformat(),
    }


# ---------------------------------------------------------------------------
# _worst_health — statuts "no_baseline" et "insufficient_data"
# ---------------------------------------------------------------------------


def test_worst_health_no_baseline_loses_to_warning():
    """'no_baseline' (niveau 0) < 'warning' (niveau 1) → 'warning' est retourné."""
    assert _worst_health("no_baseline", "warning") == "warning"


def test_worst_health_insufficient_data_loses_to_warning():
    """'insufficient_data' (niveau 0) < 'warning' → 'warning' est retourné."""
    assert _worst_health("insufficient_data", "warning") == "warning"


def test_worst_health_no_baseline_and_insufficient_data_same_level():
    """'no_baseline' et 'insufficient_data' sont au même niveau (0)."""
    result = _worst_health("no_baseline", "insufficient_data")
    assert result in ("no_baseline", "insufficient_data")


def test_worst_health_all_benign_statuses_no_critical():
    """no_baseline + no_data + ok → aucun n'est warning ou critical."""
    result = _worst_health("no_baseline", "no_data", "ok")
    assert result not in ("warning", "critical")


def test_worst_health_no_baseline_loses_to_critical():
    """'no_baseline' (niveau 0) < 'critical' (niveau 2)."""
    assert _worst_health("no_baseline", "critical") == "critical"


# ---------------------------------------------------------------------------
# GET /monitoring/overview — end == start → 422
# ---------------------------------------------------------------------------


def test_monitoring_overview_end_equal_start_returns_422():
    """end == start → 422 (end doit être strictement postérieur à start)."""
    now = datetime.utcnow()
    r = client.get(
        "/monitoring/overview",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params={"start": now.isoformat(), "end": now.isoformat()},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /monitoring/overview — champs global_stats
# ---------------------------------------------------------------------------


def test_monitoring_overview_global_stats_has_models_counts():
    """global_stats contient active_models, models_ok, models_warning, models_critical."""
    model_name = f"{MODEL_PREFIX}_overview_gs"
    _create_model(model_name)
    _predict(model_name)

    r = client.get(
        "/monitoring/overview",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params=_period(),
    )
    assert r.status_code == 200
    gs = r.json()["global_stats"]
    for field in ("active_models", "models_ok", "models_warning", "models_critical"):
        assert field in gs, f"Champ manquant dans global_stats : {field}"
    assert gs["active_models"] >= 1
    assert gs["models_ok"] + gs["models_warning"] + gs["models_critical"] == gs["active_models"]


def test_monitoring_overview_global_stats_total_predictions_positive():
    """global_stats.total_predictions >= 1 après une prédiction."""
    model_name = f"{MODEL_PREFIX}_overview_tp"
    _create_model(model_name)
    _predict(model_name)

    r = client.get(
        "/monitoring/overview",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params=_period(),
    )
    assert r.status_code == 200
    assert r.json()["global_stats"]["total_predictions"] >= 1


# ---------------------------------------------------------------------------
# GET /monitoring/model/{name} — feature_drift sans baseline
# ---------------------------------------------------------------------------


def test_monitoring_model_detail_no_baseline_returns_false():
    """Sans feature_baseline → feature_drift.baseline_available == False."""
    model_name = f"{MODEL_PREFIX}_detail_nobl"
    _create_model(model_name)
    _predict(model_name)

    r = client.get(
        f"/monitoring/model/{model_name}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params=_period(),
    )
    assert r.status_code == 200
    fd = r.json()["feature_drift"]
    assert fd["baseline_available"] is False


def test_monitoring_model_detail_no_baseline_drift_summary_is_no_baseline():
    """Sans baseline → drift_summary == 'no_baseline'."""
    model_name = f"{MODEL_PREFIX}_detail_nobl2"
    _create_model(model_name)
    _predict(model_name)

    r = client.get(
        f"/monitoring/model/{model_name}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params=_period(),
    )
    assert r.status_code == 200
    assert r.json()["feature_drift"]["drift_summary"] == "no_baseline"


def test_monitoring_model_detail_structure_complete():
    """Réponse de monitoring/model/{name} contient tous les champs attendus."""
    model_name = f"{MODEL_PREFIX}_detail_struct"
    _create_model(model_name)
    _predict(model_name)

    r = client.get(
        f"/monitoring/model/{model_name}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params=_period(),
    )
    assert r.status_code == 200
    data = r.json()
    for field in (
        "model_name", "period", "per_version_stats", "timeseries",
        "performance_by_day", "feature_drift", "ab_comparison", "recent_errors",
    ):
        assert field in data, f"Champ manquant : {field}"
    assert data["model_name"] == model_name


def test_monitoring_model_detail_end_equal_start_returns_422():
    """end == start → 422."""
    now = datetime.utcnow()
    r = client.get(
        f"/monitoring/model/some_model",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        params={"start": now.isoformat(), "end": now.isoformat()},
    )
    assert r.status_code == 422
