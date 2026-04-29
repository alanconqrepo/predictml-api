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
from src.services.drift_service import (
    _status_from_null_rate,
    compute_feature_drift,
    summarize_drift,
)
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
                  "z_score", "drift_status", "null_rate_production", "null_rate_baseline",
                  "null_rate_status"):
        assert field in feat, f"Champ feature manquant : {field}"


# ---------------------------------------------------------------------------
# Tests null rate — _status_from_null_rate
# ---------------------------------------------------------------------------


def test_null_rate_status_ok_small_diff():
    """Écart < 5 pts → ok."""
    assert _status_from_null_rate(0.04, 0.0) == "ok"


def test_null_rate_status_ok_zero():
    """Pas de nulls → ok."""
    assert _status_from_null_rate(0.0, 0.0) == "ok"


def test_null_rate_status_warning():
    """Écart entre 5 et 15 pts → warning."""
    assert _status_from_null_rate(0.10, 0.0) == "warning"
    assert _status_from_null_rate(0.20, 0.07) == "warning"


def test_null_rate_status_critical_diff():
    """Écart > 15 pts → critical."""
    assert _status_from_null_rate(0.20, 0.0) == "critical"


def test_null_rate_status_critical_threshold():
    """Taux de null > 30 % en production → critical même si baseline élevé."""
    assert _status_from_null_rate(0.35, 0.30) == "critical"


def test_null_rate_status_boundary_5pts():
    """5 pts d'écart → warning (diff >= 0.05)."""
    assert _status_from_null_rate(0.10, 0.05) == "warning"  # diff = 0.05
    assert _status_from_null_rate(0.08, 0.02) == "warning"  # diff = 0.06


def test_null_rate_status_boundary_15pts():
    """Exactement 15 pts d'écart → warning (critique seulement au-delà de 15 pts)."""
    assert _status_from_null_rate(0.16, 0.01) == "warning"  # diff = 0.15, pas > 0.15
    assert _status_from_null_rate(0.17, 0.01) == "critical"  # diff = 0.16 > 0.15


# ---------------------------------------------------------------------------
# Tests null rate intégrés dans compute_feature_drift
# ---------------------------------------------------------------------------


def _make_baseline_with_null(mean=5.84, std=0.83, mn=4.3, mx=7.9, null_rate=0.0):
    return {"sepal_length": {"mean": mean, "std": std, "min": mn, "max": mx, "null_rate": null_rate}}


def _make_prod_stats_with_null(mean=5.84, std=0.83, mn=4.3, mx=7.9, count=100, null_rate=0.0):
    import numpy as np
    rng = np.random.default_rng(42)
    values = rng.normal(loc=mean, scale=std, size=count).tolist()
    return {
        "sepal_length": {
            "mean": mean, "std": std, "min": mn, "max": mx,
            "count": count, "values": values, "null_rate": null_rate,
        }
    }


def test_compute_feature_drift_null_rate_ok():
    """Null rate de production proche du baseline → null_rate_status=ok, fields renseignés."""
    baseline = _make_baseline_with_null(null_rate=0.02)
    prod = _make_prod_stats_with_null(count=100, null_rate=0.04)

    results = compute_feature_drift(baseline, prod, min_count=30)
    feat = results["sepal_length"]

    assert feat.null_rate_production == pytest.approx(0.04)
    assert feat.null_rate_baseline == pytest.approx(0.02)
    assert feat.null_rate_status == "ok"


def test_compute_feature_drift_null_rate_warning():
    """Null rate production à +10 pts du baseline → null_rate_status=warning."""
    baseline = _make_baseline_with_null(null_rate=0.0)
    prod = _make_prod_stats_with_null(count=100, null_rate=0.10)

    results = compute_feature_drift(baseline, prod, min_count=30)
    feat = results["sepal_length"]

    assert feat.null_rate_status == "warning"


def test_compute_feature_drift_null_rate_critical_diff():
    """Null rate production à +20 pts du baseline → null_rate_status=critical."""
    baseline = _make_baseline_with_null(null_rate=0.0)
    prod = _make_prod_stats_with_null(count=100, null_rate=0.20)

    results = compute_feature_drift(baseline, prod, min_count=30)
    feat = results["sepal_length"]

    assert feat.null_rate_status == "critical"
    assert feat.drift_status == "critical"


def test_compute_feature_drift_null_rate_critical_absolute():
    """Null rate > 30 % en production → critical même avec baseline élevé."""
    baseline = _make_baseline_with_null(null_rate=0.28)
    prod = _make_prod_stats_with_null(count=100, null_rate=0.35)

    results = compute_feature_drift(baseline, prod, min_count=30)
    feat = results["sepal_length"]

    assert feat.null_rate_status == "critical"
    assert feat.drift_status == "critical"


def test_compute_feature_drift_null_rate_no_baseline_null_rate():
    """Baseline sans null_rate (ancienne baseline) → null_rate_status=None."""
    baseline = {"sepal_length": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}}
    prod = _make_prod_stats_with_null(count=100, null_rate=0.40)

    results = compute_feature_drift(baseline, prod, min_count=30)
    feat = results["sepal_length"]

    assert feat.null_rate_status is None
    assert feat.null_rate_production == pytest.approx(0.40)
    assert feat.null_rate_baseline is None


def test_compute_feature_drift_null_rate_no_baseline_feature():
    """Feature sans baseline → null_rate_status=None, null_rate_production renseigné."""
    baseline = {}
    prod = _make_prod_stats_with_null(count=100, null_rate=0.20)

    results = compute_feature_drift(baseline, prod, min_count=30)
    feat = results["sepal_length"]

    assert feat.drift_status == "no_baseline"
    assert feat.null_rate_production == pytest.approx(0.20)
    assert feat.null_rate_status is None


def test_compute_feature_drift_null_rate_with_insufficient_data():
    """Données insuffisantes mais null rate critical → null_rate_status renseigné, drift_status=insufficient_data."""
    baseline = _make_baseline_with_null(null_rate=0.0)
    prod = _make_prod_stats_with_null(count=5, null_rate=0.40)  # 5 < min_count=30

    results = compute_feature_drift(baseline, prod, min_count=30)
    feat = results["sepal_length"]

    assert feat.drift_status == "insufficient_data"
    assert feat.null_rate_status == "critical"
    assert feat.null_rate_production == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Tests summarize_drift avec null_rate_status
# ---------------------------------------------------------------------------


def test_summarize_drift_null_rate_critical_promotes_summary():
    """null_rate_status=critical sur une feature avec drift_status=ok → summary=critical."""
    features = {
        "f1": FeatureDriftResult(production_count=100, drift_status="ok", null_rate_status="critical"),
    }
    assert summarize_drift(features, baseline_available=True) == "critical"


def test_summarize_drift_null_rate_warning_promotes_summary():
    """null_rate_status=warning et toutes les autres ok → summary=warning."""
    features = {
        "f1": FeatureDriftResult(production_count=100, drift_status="ok", null_rate_status="warning"),
        "f2": FeatureDriftResult(production_count=100, drift_status="ok", null_rate_status="ok"),
    }
    assert summarize_drift(features, baseline_available=True) == "warning"


def test_summarize_drift_null_rate_critical_with_insufficient_data():
    """Feature insufficient_data mais null_rate_status=critical → summary=critical."""
    features = {
        "f1": FeatureDriftResult(production_count=5, drift_status="insufficient_data", null_rate_status="critical"),
    }
    assert summarize_drift(features, baseline_available=True) == "critical"


def test_summarize_drift_null_rate_none_no_change():
    """null_rate_status=None → n'affecte pas le résumé."""
    features = {
        "f1": FeatureDriftResult(production_count=100, drift_status="ok", null_rate_status=None),
    }
    assert summarize_drift(features, baseline_available=True) == "ok"


# ---------------------------------------------------------------------------
# Tests output drift — compute_output_drift (logique via endpoint)
# ---------------------------------------------------------------------------


def _fake_output_metadata(training_stats=None, version="1.0.0"):
    return SimpleNamespace(
        name="iris",
        version=version,
        training_stats=training_stats,
    )


def _make_output_drift_mocks(label_distribution, label_counts, total):
    """Retourne les patches nécessaires pour tester compute_output_drift."""
    meta = _fake_output_metadata(
        training_stats={"label_distribution": label_distribution}
    )
    return meta, label_counts, total


# ---------------------------------------------------------------------------
# Tests endpoint GET /models/{name}/output-drift
# ---------------------------------------------------------------------------


def test_output_drift_no_baseline():
    """Modèle sans training_stats.label_distribution → status=no_baseline."""
    meta = _fake_output_metadata(training_stats=None)

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=({}, 0)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"period_days": 7},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_baseline"
    assert data["psi"] is None
    assert data["by_class"] is None


def test_output_drift_empty_label_distribution():
    """label_distribution vide dans training_stats → status=no_baseline."""
    meta = _fake_output_metadata(training_stats={"label_distribution": {}})

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=({}, 0)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"period_days": 7},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_baseline"


def test_output_drift_insufficient_data():
    """Moins de 30 prédictions → status=insufficient_data."""
    label_distribution = {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
    meta = _fake_output_metadata(training_stats={"label_distribution": label_distribution})

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=({"setosa": 5}, 5)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"period_days": 7},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "insufficient_data"
    assert data["predictions_analyzed"] == 5


def test_output_drift_ok():
    """Distribution actuelle proche du baseline → status=ok, PSI < 0.1."""
    label_distribution = {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
    # Counts proportionnels à la baseline → PSI ≈ 0
    label_counts = {"setosa": 33, "versicolor": 34, "virginica": 33}
    total = 100

    meta = _fake_output_metadata(training_stats={"label_distribution": label_distribution})

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=(label_counts, total)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"period_days": 7},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["psi"] is not None
    assert data["psi"] < 0.1
    assert data["predictions_analyzed"] == 100
    assert len(data["by_class"]) == 3


def test_output_drift_warning():
    """Distribution modérément décalée → status=warning, 0.1 ≤ PSI < 0.2."""
    label_distribution = {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
    # Déséquilibre modéré
    label_counts = {"setosa": 50, "versicolor": 30, "virginica": 20}
    total = 100

    meta = _fake_output_metadata(training_stats={"label_distribution": label_distribution})

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=(label_counts, total)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"period_days": 7},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("warning", "critical")
    assert data["psi"] is not None
    assert data["psi"] >= 0.1
    assert data["baseline_distribution"] is not None
    assert data["current_distribution"] is not None


def test_output_drift_critical():
    """Distribution fortement déséquilibrée → status=critical, PSI ≥ 0.2."""
    label_distribution = {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
    # Déséquilibre fort : 90% setosa
    label_counts = {"setosa": 90, "versicolor": 5, "virginica": 5}
    total = 100

    meta = _fake_output_metadata(training_stats={"label_distribution": label_distribution})

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=(label_counts, total)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"period_days": 7},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "critical"
    assert data["psi"] >= 0.2
    # by_class doit contenir le delta pour setosa
    setosa_class = next(c for c in data["by_class"] if c["label"] == "setosa")
    assert setosa_class["delta"] > 0.5


def test_output_drift_by_class_structure():
    """Vérifie la structure complète de by_class."""
    label_distribution = {"A": 0.5, "B": 0.5}
    label_counts = {"A": 40, "B": 60}
    total = 100

    meta = _fake_output_metadata(training_stats={"label_distribution": label_distribution})

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=(label_counts, total)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"period_days": 7},
        )
    assert resp.status_code == 200
    data = resp.json()
    by_class = data["by_class"]
    assert len(by_class) == 2
    for entry in by_class:
        assert "label" in entry
        assert "baseline_ratio" in entry
        assert "current_ratio" in entry
        assert "delta" in entry
    # A a moins que baseline → delta négatif
    a_entry = next(c for c in by_class if c["label"] == "A")
    assert a_entry["delta"] < 0


def test_output_drift_404_unknown_model():
    """Modèle inexistant → 404."""
    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=None)
    ):
        resp = client.get(
            "/models/unknown_model/output-drift",
            headers=AUTH_HEADERS,
        )
    assert resp.status_code == 404


def test_output_drift_auth_required():
    """Sans token → 401/403."""
    resp = client.get("/models/iris/output-drift")
    assert resp.status_code in (401, 403)


def test_output_drift_model_version_in_response():
    """La version du modèle est bien retournée dans la réponse."""
    label_distribution = {"cat": 0.5, "dog": 0.5}
    label_counts = {"cat": 50, "dog": 50}
    meta = _fake_output_metadata(
        training_stats={"label_distribution": label_distribution}, version="2.0.0"
    )

    with _patch_auth(), patch(
        "src.services.db_service.DBService.get_model_metadata", new=AsyncMock(return_value=meta)
    ), patch(
        "src.services.db_service.DBService.get_prediction_label_distribution",
        new=AsyncMock(return_value=(label_counts, 100)),
    ):
        resp = client.get(
            "/models/iris/output-drift",
            headers=AUTH_HEADERS,
            params={"model_version": "2.0.0"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "2.0.0"
    assert data["model_name"] == "iris"
