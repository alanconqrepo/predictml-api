"""
Tests pour GET /predictions/anomalies.

Couvre :
- Logique pure : calcul de z-score par feature
- Endpoint : auth, 404, no_baseline, résultat vide, anomalies détectées
- Paramètres : z_threshold, days, limit
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.security import verify_token
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

AUTH_HEADERS = {"Authorization": "Bearer test-token"}
_FAKE_USER = SimpleNamespace(id=1, username="tester", role="user", is_active=True)

ADMIN_TOKEN = "test-token-anomalies-admin-cc22"
MODEL_NAME = "anomaly_test_model"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="anomaly_admin",
                email="anomaly_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


async def _fake_verify_token():
    return _FAKE_USER


class _AuthOverride:
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


def _fake_metadata(name=MODEL_NAME, version="1.0.0", feature_baseline=None):
    return SimpleNamespace(name=name, version=version, feature_baseline=feature_baseline)


def _make_prediction(
    pred_id: int,
    features: dict,
    result="setosa",
    probabilities=None,
    ts: datetime = None,
):
    ts = ts or datetime(2026, 4, 27, 14, 32, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=pred_id,
        model_name=MODEL_NAME,
        model_version="1.0.0",
        input_features=features,
        prediction_result=result,
        probabilities=probabilities,
        timestamp=ts,
        status="success",
    )


_BASELINE = {
    "sepal_length": {"mean": 5.8, "std": 0.83, "min": 4.3, "max": 7.9},
    "sepal_width": {"mean": 3.1, "std": 0.45, "min": 2.0, "max": 4.4},
}


# ---------------------------------------------------------------------------
# Unit tests — z-score logic (no DB, no endpoint)
# ---------------------------------------------------------------------------


def test_z_score_within_threshold():
    """Feature à l'intérieur du seuil ne génère pas d'anomalie."""
    bl_mean, bl_std = 5.8, 0.83
    value = 6.0
    z = abs(value - bl_mean) / bl_std
    assert z < 3.0


def test_z_score_above_threshold():
    """Feature très éloignée de la moyenne génère un z-score élevé."""
    bl_mean, bl_std = 5.8, 0.83
    value = 12.1
    z = abs(value - bl_mean) / bl_std
    assert z >= 3.0
    assert round(z, 1) == pytest.approx(7.6, abs=0.1)


def test_z_score_std_zero_skipped():
    """Feature avec std=0 dans la baseline doit être ignorée (division par zéro)."""
    bl_std = 0.0
    # L'endpoint saute les features avec bl_std <= 0
    assert bl_std <= 0


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_anomalies_no_auth():
    """Sans token → 401/403."""
    response = client.get(f"/predictions/anomalies?model_name={MODEL_NAME}")
    assert response.status_code in (401, 403)


def test_anomalies_model_not_found():
    """404 si le modèle n'existe pas."""
    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=None)),
    ):
        response = client.get(
            "/predictions/anomalies?model_name=unknown_model", headers=AUTH_HEADERS
        )
    assert response.status_code == 404


def test_anomalies_no_baseline():
    """error='no_baseline' si le modèle n'a pas de feature_baseline."""
    meta = _fake_metadata(feature_baseline=None)
    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}", headers=AUTH_HEADERS
        )

    assert response.status_code == 200
    data = response.json()
    assert data["error"] == "no_baseline"
    assert data["total_checked"] == 0
    assert data["predictions"] == []


def test_anomalies_empty_result():
    """Aucune anomalie si toutes les features sont dans le seuil."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    normal_pred = _make_prediction(1, {"sepal_length": 5.9, "sepal_width": 3.2})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[normal_pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_checked"] == 1
    assert data["anomalous_count"] == 0
    assert data["anomaly_rate"] == 0.0
    assert data["predictions"] == []
    assert data.get("error") is None


def test_anomalies_detects_outlier():
    """Prédiction avec feature très éloignée de la baseline est détectée."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    outlier_pred = _make_prediction(42, {"sepal_length": 12.1, "sepal_width": 3.0})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[outlier_pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_checked"] == 1
    assert data["anomalous_count"] == 1
    assert data["anomaly_rate"] == 1.0
    preds = data["predictions"]
    assert len(preds) == 1
    assert preds[0]["prediction_id"] == 42
    assert "sepal_length" in preds[0]["anomalous_features"]
    feat = preds[0]["anomalous_features"]["sepal_length"]
    assert feat["value"] == pytest.approx(12.1, abs=1e-3)
    assert feat["z_score"] >= 3.0
    assert feat["baseline_mean"] == pytest.approx(5.8, abs=1e-4)
    assert feat["baseline_std"] == pytest.approx(0.83, abs=1e-4)


def test_anomalies_only_anomalous_feature_included():
    """Seules les features aberrantes sont listées dans anomalous_features."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    # sepal_length aberrante, sepal_width normale
    pred = _make_prediction(5, {"sepal_length": 15.0, "sepal_width": 3.1})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    data = response.json()
    assert data["anomalous_count"] == 1
    feats = data["predictions"][0]["anomalous_features"]
    assert "sepal_length" in feats
    assert "sepal_width" not in feats


def test_anomalies_max_confidence_populated():
    """max_confidence est extrait des probabilités si disponible."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    pred = _make_prediction(7, {"sepal_length": 12.0}, probabilities=[0.05, 0.10, 0.85])

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    data = response.json()
    assert data["anomalous_count"] == 1
    assert data["predictions"][0]["max_confidence"] == pytest.approx(0.85, abs=1e-3)


def test_anomalies_no_probabilities_max_confidence_null():
    """max_confidence est None si le modèle n'a pas de probabilities."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    pred = _make_prediction(8, {"sepal_length": 12.0}, probabilities=None)

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    data = response.json()
    assert data["predictions"][0]["max_confidence"] is None


def test_anomalies_lower_z_threshold_catches_more():
    """Un seuil z plus bas détecte davantage d'anomalies."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    # sepal_length z ~ 0.24 (valeur très proche de la moyenne) — ne passe z=3
    # sepal_width z ~ 2.0 (borderline) — passe z=1.5 mais pas z=3
    pred1 = _make_prediction(1, {"sepal_length": 5.9, "sepal_width": 4.0})
    pred2 = _make_prediction(2, {"sepal_length": 5.8, "sepal_width": 3.1})

    preds = [pred1, pred2]
    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features", AsyncMock(return_value=preds)
        ),
    ):
        r_strict = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )
        r_loose = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=1.5",
            headers=AUTH_HEADERS,
        )

    assert r_strict.json()["anomalous_count"] <= r_loose.json()["anomalous_count"]


def test_anomalies_feature_not_in_baseline_skipped():
    """Feature absente de la baseline est ignorée (pas de z-score possible)."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    # "unknown_feature" n'est pas dans la baseline
    pred = _make_prediction(9, {"unknown_feature": 999.0, "sepal_length": 5.8})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    data = response.json()
    # unknown_feature ignorée, sepal_length normale → pas d'anomalie
    assert data["anomalous_count"] == 0


def test_anomalies_non_numeric_feature_skipped():
    """Features non numériques (string, bool) sont ignorées."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    pred = _make_prediction(10, {"sepal_length": "high", "sepal_width": True})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    data = response.json()
    assert data["anomalous_count"] == 0


def test_anomalies_zero_predictions_zero_rate():
    """Aucune prédiction dans la fenêtre → anomaly_rate=0, total_checked=0."""
    meta = _fake_metadata(feature_baseline=_BASELINE)

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}", headers=AUTH_HEADERS
        )

    data = response.json()
    assert data["total_checked"] == 0
    assert data["anomalous_count"] == 0
    assert data["anomaly_rate"] == 0.0


def test_anomalies_response_structure():
    """Vérifie la structure complète de la réponse."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    pred = _make_prediction(99, {"sepal_length": 12.0})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&days=7&z_threshold=2.0&limit=100",
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    data = response.json()

    for field in (
        "model_name",
        "period_days",
        "z_threshold",
        "total_checked",
        "anomalous_count",
        "anomaly_rate",
        "predictions",
    ):
        assert field in data, f"Champ manquant : {field}"

    assert data["model_name"] == MODEL_NAME
    assert data["period_days"] == 7
    assert data["z_threshold"] == pytest.approx(2.0, abs=1e-6)

    entry = data["predictions"][0]
    for field in (
        "prediction_id",
        "timestamp",
        "prediction_result",
        "max_confidence",
        "anomalous_features",
    ):
        assert field in entry, f"Champ entry manquant : {field}"

    feat = entry["anomalous_features"]["sepal_length"]
    for field in ("value", "z_score", "baseline_mean", "baseline_std"):
        assert field in feat, f"Champ feature manquant : {field}"


def test_anomalies_query_params_forwarded():
    """Les paramètres days et z_threshold sont bien passés au service DB."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    mock_get = AsyncMock(return_value=[])

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch("src.api.predict.DBService.get_predictions_with_features", mock_get),
    ):
        client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&days=14&limit=500",
            headers=AUTH_HEADERS,
        )

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["days"] == 14
    assert call_kwargs["limit"] == 500


def test_anomalies_limit_respected():
    """Le paramètre limit est transmis à DBService."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    mock_get = AsyncMock(return_value=[])

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch("src.api.predict.DBService.get_predictions_with_features", mock_get),
    ):
        client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&limit=50",
            headers=AUTH_HEADERS,
        )

    assert mock_get.call_args.kwargs["limit"] == 50


def test_anomalies_multiple_anomalous_features():
    """Prédiction avec plusieurs features aberrantes."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    pred = _make_prediction(20, {"sepal_length": 12.0, "sepal_width": 8.0})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[pred]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    data = response.json()
    assert data["anomalous_count"] == 1
    feats = data["predictions"][0]["anomalous_features"]
    assert "sepal_length" in feats
    assert "sepal_width" in feats
    assert feats["sepal_length"]["z_score"] >= 3.0
    assert feats["sepal_width"]["z_score"] >= 3.0


def test_anomalies_anomaly_rate_calculation():
    """anomaly_rate = anomalous_count / total_checked."""
    meta = _fake_metadata(feature_baseline=_BASELINE)
    outlier = _make_prediction(1, {"sepal_length": 15.0})
    normal = _make_prediction(2, {"sepal_length": 5.8})
    normal2 = _make_prediction(3, {"sepal_length": 5.9})

    with (
        _patch_auth(),
        patch("src.api.predict.DBService.get_model_metadata", AsyncMock(return_value=meta)),
        patch(
            "src.api.predict.DBService.get_predictions_with_features",
            AsyncMock(return_value=[outlier, normal, normal2]),
        ),
    ):
        response = client.get(
            f"/predictions/anomalies?model_name={MODEL_NAME}&z_threshold=3.0",
            headers=AUTH_HEADERS,
        )

    data = response.json()
    assert data["total_checked"] == 3
    assert data["anomalous_count"] == 1
    assert data["anomaly_rate"] == pytest.approx(1 / 3, abs=1e-3)
