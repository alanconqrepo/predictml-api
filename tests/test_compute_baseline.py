"""
Tests pour POST /models/{name}/{version}/compute-baseline

Couvre :
- Auth : 401 sans token, 403 pour non-admin
- 404 si modèle introuvable
- 422 si prédictions insuffisantes (< 100) ou aucune prédiction
- dry_run=True (défaut) : retourne le baseline sans sauvegarder
- dry_run=False : sauvegarde le baseline et active le drift
- Format de réponse : {mean, std, min, max} par feature
- Param days transmis à get_feature_production_stats
"""

import asyncio
import io
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.core.security import require_admin, verify_token
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-baseline-admin-x99"
USER_TOKEN = "test-token-baseline-user-y88"
MODEL_PREFIX = "baseline_model"

# ---------------------------------------------------------------------------
# Fake user injected via dependency_overrides
# ---------------------------------------------------------------------------

_FAKE_ADMIN = SimpleNamespace(id=1, username="baseline_admin", role="admin", is_active=True)
_FAKE_USER = SimpleNamespace(id=2, username="baseline_user", role="user", is_active=True)
AUTH_HEADERS = {"Authorization": "Bearer test-token"}


# ---------------------------------------------------------------------------
# Setup DB users
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="baseline_admin",
                email="baseline_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="baseline_user",
                email="baseline_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": name, "version": version},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _fake_metadata(name="iris", version="1.0.0"):
    return SimpleNamespace(
        name=name,
        version=version,
        feature_baseline=None,
        description=None,
        algorithm=None,
        features_count=None,
        classes=None,
        accuracy=None,
        precision=None,
        recall=None,
        f1_score=None,
        training_metrics=None,
        confidence_threshold=None,
        trained_by=None,
        training_date=None,
        training_dataset=None,
        training_params=None,
        tags=None,
        webhook_url=None,
        is_production=False,
        is_active=True,
        deprecated_at=None,
        traffic_weight=None,
        deployment_mode=None,
    )


def _make_prod_stats(count: int = 200):
    import numpy as np
    rng = np.random.default_rng(42)
    values = rng.normal(5.84, 0.83, count).tolist()
    return {
        "sepal_length": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9, "count": count, "values": values},
        "petal_length": {"mean": 3.76, "std": 1.77, "min": 1.0, "max": 6.9, "count": count, "values": values},
    }


class _AdminOverride:
    def __enter__(self):
        app.dependency_overrides[verify_token] = lambda: _FAKE_ADMIN
        return self

    def __exit__(self, *_):
        app.dependency_overrides.pop(verify_token, None)


# ---------------------------------------------------------------------------
# Tests auth
# ---------------------------------------------------------------------------


def test_compute_baseline_no_auth():
    """Sans token → 403 (HTTPBearer renvoie 403 si absent)"""
    r = client.post("/models/iris/1.0.0/compute-baseline")
    assert r.status_code in (401, 403)


def test_compute_baseline_invalid_token():
    """Token invalide → 401"""
    r = client.post(
        "/models/iris/1.0.0/compute-baseline",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert r.status_code == 401


def test_compute_baseline_non_admin_forbidden():
    """Token non-admin → 403"""
    r = client.post(
        "/models/iris/1.0.0/compute-baseline",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Tests 404
# ---------------------------------------------------------------------------


def test_compute_baseline_model_not_found():
    """Modèle introuvable → 404"""
    with _AdminOverride(), patch(
        "src.api.models.DBService.get_model_metadata", AsyncMock(return_value=None)
    ):
        r = client.post("/models/nonexistent/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tests 422 — données insuffisantes
# ---------------------------------------------------------------------------


def test_compute_baseline_no_predictions():
    """Aucune prédiction disponible → 422"""
    meta = _fake_metadata()
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value={})):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert "insuffisantes" in r.json()["detail"].lower()


def test_compute_baseline_insufficient_predictions():
    """Moins de 100 prédictions → 422 avec message explicite"""
    meta = _fake_metadata()
    stats = _make_prod_stats(count=50)
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "50" in detail
    assert "100" in detail


def test_compute_baseline_exactly_99_predictions():
    """99 prédictions → 422 (limite stricte < 100)"""
    meta = _fake_metadata()
    stats = _make_prod_stats(count=99)
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 422


def test_compute_baseline_exactly_100_predictions():
    """100 prédictions → 200 (seuil atteint)"""
    meta = _fake_metadata()
    stats = _make_prod_stats(count=100)
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Tests dry_run=True (défaut)
# ---------------------------------------------------------------------------


def test_compute_baseline_dry_run_default_is_true():
    """dry_run=True par défaut — réponse valide, dry_run=true dans réponse"""
    meta = _fake_metadata()
    stats = _make_prod_stats(count=200)
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["dry_run"] is True


def test_compute_baseline_dry_run_response_format():
    """Vérifie le format complet de la réponse"""
    meta = _fake_metadata(name="iris", version="1.0.0")
    stats = _make_prod_stats(count=200)
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)):
        r = client.post("/models/iris/1.0.0/compute-baseline?dry_run=true", headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "iris"
    assert data["version"] == "1.0.0"
    assert data["predictions_used"] == 200
    assert data["dry_run"] is True
    assert "sepal_length" in data["baseline"]
    assert "petal_length" in data["baseline"]


def test_compute_baseline_feature_stats_keys():
    """Chaque feature du baseline a exactement {mean, std, min, max}"""
    meta = _fake_metadata()
    stats = _make_prod_stats(count=150)
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 200
    for feat_stats in r.json()["baseline"].values():
        assert set(feat_stats.keys()) == {"mean", "std", "min", "max"}
        assert "count" not in feat_stats
        assert "values" not in feat_stats


def test_compute_baseline_values_match_stats():
    """Les valeurs retournées correspondent aux stats calculées"""
    meta = _fake_metadata()
    stats = {
        "feat_a": {"mean": 3.5, "std": 1.2, "min": 1.0, "max": 6.0, "count": 200, "values": []},
    }
    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 200
    feat = r.json()["baseline"]["feat_a"]
    assert feat["mean"] == pytest.approx(3.5)
    assert feat["std"] == pytest.approx(1.2)
    assert feat["min"] == pytest.approx(1.0)
    assert feat["max"] == pytest.approx(6.0)


def test_compute_baseline_dry_run_does_not_save():
    """dry_run=True ne doit pas modifier le modèle en base"""
    model = _create_model(f"{MODEL_PREFIX}_no_save")
    name = model["name"]
    version = model["version"]
    stats = _make_prod_stats(count=200)

    with patch(
        "src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)
    ):
        r = client.post(
            f"/models/{name}/{version}/compute-baseline?dry_run=true",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 200
    assert r.json()["dry_run"] is True

    # Vérifier que le modèle en base n'a PAS de feature_baseline
    r2 = client.get(f"/models/{name}/{version}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r2.status_code == 200
    assert r2.json()["feature_baseline"] is None


# ---------------------------------------------------------------------------
# Tests dry_run=False
# ---------------------------------------------------------------------------


def test_compute_baseline_dry_run_false_saves_baseline():
    """dry_run=False sauvegarde le baseline dans la base"""
    model = _create_model(f"{MODEL_PREFIX}_save")
    name = model["name"]
    version = model["version"]
    stats = _make_prod_stats(count=300)

    with patch(
        "src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)
    ):
        r = client.post(
            f"/models/{name}/{version}/compute-baseline?dry_run=false",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["dry_run"] is False
    assert data["predictions_used"] == 300
    assert "sepal_length" in data["baseline"]

    # Vérifier que le baseline est maintenant sauvegardé
    r2 = client.get(f"/models/{name}/{version}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r2.status_code == 200
    saved_baseline = r2.json()["feature_baseline"]
    assert saved_baseline is not None
    assert "sepal_length" in saved_baseline
    assert set(saved_baseline["sepal_length"].keys()) == {"mean", "std", "min", "max"}


def test_compute_baseline_dry_run_false_logs_history():
    """dry_run=False crée une entrée d'historique avec changed_fields=['feature_baseline']"""
    model = _create_model(f"{MODEL_PREFIX}_history")
    name = model["name"]
    version = model["version"]
    stats = _make_prod_stats(count=200)

    with patch(
        "src.api.models.DBService.get_feature_production_stats", AsyncMock(return_value=stats)
    ):
        r = client.post(
            f"/models/{name}/{version}/compute-baseline?dry_run=false",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
    assert r.status_code == 200

    # Vérifier l'historique
    r_hist = client.get(
        f"/models/{name}/{version}/history",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r_hist.status_code == 200
    entries = r_hist.json()["entries"]
    # Chercher une entrée avec feature_baseline dans changed_fields
    baseline_entries = [e for e in entries if "feature_baseline" in (e.get("changed_fields") or [])]
    assert len(baseline_entries) >= 1
    assert baseline_entries[0]["action"] == "updated"


# ---------------------------------------------------------------------------
# Tests paramètre days
# ---------------------------------------------------------------------------


def test_compute_baseline_days_param_forwarded():
    """Le paramètre days est bien transmis à get_feature_production_stats"""
    meta = _fake_metadata()
    stats = _make_prod_stats(count=200)
    mock_get_stats = AsyncMock(return_value=stats)

    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", mock_get_stats):
        r = client.post("/models/iris/1.0.0/compute-baseline?days=60", headers=AUTH_HEADERS)
    assert r.status_code == 200
    # Vérifier que days=60 a été passé
    call_args = mock_get_stats.call_args
    assert call_args[0][3] == 60 or call_args.kwargs.get("days") == 60


def test_compute_baseline_days_default_30():
    """days=30 par défaut"""
    meta = _fake_metadata()
    stats = _make_prod_stats(count=200)
    mock_get_stats = AsyncMock(return_value=stats)

    with _AdminOverride(), \
         patch("src.api.models.DBService.get_model_metadata", AsyncMock(return_value=meta)), \
         patch("src.api.models.DBService.get_feature_production_stats", mock_get_stats):
        r = client.post("/models/iris/1.0.0/compute-baseline", headers=AUTH_HEADERS)
    assert r.status_code == 200
    call_args = mock_get_stats.call_args
    assert call_args[0][3] == 30 or call_args.kwargs.get("days") == 30
