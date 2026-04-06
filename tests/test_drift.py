"""
Tests pour la détection de data drift.

Couvre :
- compute_feature_drift (logique pure, pas de DB)
- summarize_drift
- GET /models/{name}/drift (endpoint)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.core.security import verify_token
from src.services.drift_service import compute_feature_drift, summarize_drift
from src.schemas.model import FeatureDriftResult

# Utilisateur admin fictif injecté via dependency_overrides
_FAKE_USER = SimpleNamespace(id=1, username="admin", role="admin", is_active=True)
AUTH_HEADERS = {"Authorization": "Bearer test-token"}

client = TestClient(app)


async def _fake_verify_token():
    return _FAKE_USER


class _AuthOverride:
    """Context manager qui injecte un faux utilisateur dans les dépendances FastAPI."""

    def __enter__(self):
        app.dependency_overrides[verify_token] = _fake_verify_token
        return self

    def __exit__(self, *_):
        app.dependency_overrides.pop(verify_token, None)


def _patch_auth():
    return _AuthOverride()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_baseline(mean=5.84, std=0.83, mn=4.3, mx=7.9):
    return {"sepal_length": {"mean": mean, "std": std, "min": mn, "max": mx}}


def _make_prod_stats(mean=5.84, std=0.83, mn=4.3, mx=7.9, count=100):
    import numpy as np

    rng = np.random.default_rng(42)
    values = rng.normal(loc=mean, scale=std, size=count).tolist()
    return {
        "sepal_length": {
            "mean": mean,
            "std": std,
            "min": mn,
            "max": mx,
            "count": count,
            "values": values,
        }
    }


def _fake_metadata(name="iris_model", version="1.0.0", feature_baseline=None):
    return SimpleNamespace(
        name=name,
        version=version,
        feature_baseline=feature_baseline,
    )


# ---------------------------------------------------------------------------
# Tests logique pure — compute_feature_drift
# ---------------------------------------------------------------------------


def test_compute_feature_drift_ok():
    """Distribution de production proche du baseline → statut ok (z-score)."""
    baseline = _make_baseline(mean=5.84, std=0.83)
    # Même mean → z_score = 0 → "ok" garanti
    prod = _make_prod_stats(mean=5.84, std=0.83, count=2000)

    results = compute_feature_drift(baseline, prod, min_count=30)

    assert "sepal_length" in results
    feat = results["sepal_length"]
    # Avec mean identique, z_score = 0 donc statut ok (PSI peut varier mais z prévaut)
    assert feat.z_score is not None
    assert feat.z_score < 2.0
    assert feat.production_count == 2000
    # Statut basé sur z → au maximum "ok" pour z=0
    assert feat.drift_status in ("ok", "warning")  # PSI peut être légèrement > 0.1


def test_compute_feature_drift_critical():
    """Production très éloignée du baseline → statut critical."""
    baseline = _make_baseline(mean=5.84, std=0.83)
    # Moyenne de production à +5 écarts-types
    prod = _make_prod_stats(mean=10.0, std=0.83, count=100)

    results = compute_feature_drift(baseline, prod, min_count=30)

    feat = results["sepal_length"]
    assert feat.drift_status == "critical"
    assert feat.z_score is not None
    assert feat.z_score >= 3.0


def test_compute_feature_drift_warning():
    """Production modérément éloignée → statut warning."""
    baseline = _make_baseline(mean=5.84, std=0.83)
    # z ≈ 2.5 (entre 2 et 3)
    prod = _make_prod_stats(mean=5.84 + 2.5 * 0.83, std=0.83, count=100)

    results = compute_feature_drift(baseline, prod, min_count=30)

    feat = results["sepal_length"]
    assert feat.drift_status in ("warning", "critical")


def test_compute_feature_drift_insufficient_data():
    """Moins de min_count prédictions → statut insufficient_data."""
    baseline = _make_baseline()
    prod = _make_prod_stats(count=5)  # 5 < min_count=30

    results = compute_feature_drift(baseline, prod, min_count=30)

    assert results["sepal_length"].drift_status == "insufficient_data"
    assert results["sepal_length"].z_score is None


def test_compute_feature_drift_no_baseline_for_feature():
    """Feature présente en production mais absente du baseline → no_baseline."""
    baseline = {}  # pas de baseline pour sepal_length
    prod = _make_prod_stats(count=100)

    results = compute_feature_drift(baseline, prod, min_count=30)

    assert results["sepal_length"].drift_status == "no_baseline"
    assert results["sepal_length"].baseline_mean is None


def test_compute_feature_drift_zero_std_in_baseline():
    """std=0 dans le baseline (feature constante) — pas d'erreur."""
    baseline = {"feat": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}
    prod = {
        "feat": {
            "mean": 1.5,
            "std": 0.1,
            "min": 1.0,
            "max": 2.0,
            "count": 50,
            "values": [1.5] * 50,
        }
    }

    results = compute_feature_drift(baseline, prod, min_count=30)
    # Pas d'erreur, z_score=None car std=0
    assert results["feat"].z_score is None


# ---------------------------------------------------------------------------
# Tests summarize_drift
# ---------------------------------------------------------------------------


def test_summarize_drift_ok():
    features = {
        "f1": FeatureDriftResult(production_count=100, drift_status="ok"),
        "f2": FeatureDriftResult(production_count=80, drift_status="ok"),
    }
    assert summarize_drift(features, baseline_available=True) == "ok"


def test_summarize_drift_worst_wins():
    features = {
        "f1": FeatureDriftResult(production_count=100, drift_status="ok"),
        "f2": FeatureDriftResult(production_count=100, drift_status="critical"),
    }
    assert summarize_drift(features, baseline_available=True) == "critical"


def test_summarize_drift_no_baseline():
    features = {}
    assert summarize_drift(features, baseline_available=False) == "no_baseline"


def test_summarize_drift_all_insufficient():
    features = {
        "f1": FeatureDriftResult(production_count=5, drift_status="insufficient_data"),
    }
    assert summarize_drift(features, baseline_available=True) == "insufficient_data"


# ---------------------------------------------------------------------------
# Tests endpoint GET /models/{name}/drift
# ---------------------------------------------------------------------------


def _patch_db_service(metadata, prod_stats):
    """Helper pour patcher DBService dans les tests endpoint."""
    get_meta = AsyncMock(return_value=metadata)
    get_stats = AsyncMock(return_value=prod_stats)
    return (
        patch("src.api.models.DBService.get_model_metadata", get_meta),
        patch("src.api.models.DBService.get_feature_production_stats", get_stats),
    )


def test_drift_endpoint_model_not_found():
    """404 si le modèle n'existe pas."""
    with _patch_auth(), patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=None)):
        response = client.get("/models/unknown_model/drift", headers=AUTH_HEADERS)
    assert response.status_code == 404


def test_drift_endpoint_no_baseline():
    """drift_summary=no_baseline si feature_baseline est None."""
    meta = _fake_metadata(feature_baseline=None)
    prod = _make_prod_stats(count=50)

    p1, p2 = _patch_db_service(meta, prod)
    with _patch_auth(), p1, p2:
        response = client.get("/models/iris_model/drift", headers=AUTH_HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert data["drift_summary"] == "no_baseline"
    assert data["baseline_available"] is False
    assert "sepal_length" in data["features"]
    assert data["features"]["sepal_length"]["drift_status"] == "no_baseline"


def test_drift_endpoint_with_baseline_ok():
    """Résultat avec baseline chargé et distribution identique → z_score=0, pas critical."""
    baseline = {"sepal_length": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}}
    meta = _fake_metadata(feature_baseline=baseline)
    # mean identique → z_score=0 garanti; PSI peut être légèrement > 0 avec 100 samples
    prod = _make_prod_stats(mean=5.84, std=0.83, count=100)

    p1, p2 = _patch_db_service(meta, prod)
    with _patch_auth(), p1, p2:
        response = client.get("/models/iris_model/drift?days=7", headers=AUTH_HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert data["baseline_available"] is True
    assert data["model_name"] == "iris_model"
    assert data["period_days"] == 7
    feat = data["features"]["sepal_length"]
    # z_score=0 → pas critical
    assert feat["drift_status"] in ("ok", "warning")
    assert feat["baseline_mean"] == pytest.approx(5.84, abs=1e-4)
    assert feat["z_score"] == pytest.approx(0.0, abs=1e-4)


def test_drift_endpoint_with_baseline_critical():
    """Résultat avec forte déviation → critical."""
    baseline = {"sepal_length": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}}
    meta = _fake_metadata(feature_baseline=baseline)
    prod = _make_prod_stats(mean=10.0, std=0.83, count=100)

    p1, p2 = _patch_db_service(meta, prod)
    with _patch_auth(), p1, p2:
        response = client.get("/models/iris_model/drift", headers=AUTH_HEADERS)

    data = response.json()
    assert data["drift_summary"] == "critical"
    assert data["features"]["sepal_length"]["drift_status"] == "critical"


def test_drift_endpoint_insufficient_data():
    """Moins de min_predictions → insufficient_data."""
    baseline = {"sepal_length": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}}
    meta = _fake_metadata(feature_baseline=baseline)
    # 3 prédictions, bien en dessous du min_predictions=30 par défaut
    prod = _make_prod_stats(count=3)

    p1, p2 = _patch_db_service(meta, prod)
    with _patch_auth(), p1, p2:
        response = client.get("/models/iris_model/drift", headers=AUTH_HEADERS)

    data = response.json()
    assert data["features"]["sepal_length"]["drift_status"] == "insufficient_data"


def test_drift_endpoint_no_auth():
    """Sans token → 401/403."""
    response = client.get("/models/iris_model/drift")
    assert response.status_code in (401, 403)


def test_drift_endpoint_structure():
    """Vérifie la structure complète de la réponse."""
    baseline = {"f1": {"mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0}}
    meta = _fake_metadata(feature_baseline=baseline)
    prod = {
        "f1": {
            "mean": 1.1,
            "std": 0.5,
            "min": 0.0,
            "max": 2.0,
            "count": 100,
            "values": [1.1] * 100,
        }
    }

    p1, p2 = _patch_db_service(meta, prod)
    with _patch_auth(), p1, p2:
        response = client.get("/models/iris_model/drift?days=14&min_predictions=50", headers=AUTH_HEADERS)

    assert response.status_code == 200
    data = response.json()

    # Champs obligatoires
    for field in ("model_name", "model_version", "period_days", "predictions_analyzed",
                  "baseline_available", "drift_summary", "features"):
        assert field in data, f"Champ manquant : {field}"

    assert data["period_days"] == 14

    feat = data["features"]["f1"]
    for field in ("baseline_mean", "baseline_std", "production_mean", "production_count",
                  "z_score", "drift_status"):
        assert field in feat, f"Champ feature manquant : {field}"
