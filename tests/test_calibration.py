"""
Tests pour GET /models/{name}/calibration.

Stratégie :
  - Endpoint auth (401/403)
  - Aucune paire → insufficient_data (sample_size=0)
  - Toutes les probabilités null → 422
  - Moins de 30 paires → insufficient_data
  - ≥ 30 paires bien calibrées → status ok
  - Sur-confiance détectée → overconfident
  - Sous-confiance détectée → underconfident
  - Filtrage par version
  - Structure de réponse (champs obligatoires, reliability bins)
  - n_bins paramètre respecté
"""

import asyncio
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.db.models import ObservedResult, Prediction
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TOKEN = "test-token-calib-xr9p"
MODEL_ENOUGH = "calib_model_enough"
MODEL_NO_PROBA = "calib_model_no_proba"
MODEL_FEW = "calib_model_few"
MODEL_OVERCONF = "calib_model_overconf"
MODEL_UNDERCONF = "calib_model_underconf"
MODEL_V1 = "calib_model_versioned"
VERSION = "1.0.0"
VERSION_B = "2.0.0"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


async def _make_prediction(db, uid, model_name, version, id_obs, prediction_result, probabilities):
    pred = Prediction(
        user_id=uid,
        model_name=model_name,
        model_version=version,
        id_obs=id_obs,
        input_features={"x": 1},
        prediction_result=prediction_result,
        probabilities=probabilities,
        response_time_ms=5.0,
        status="success",
        timestamp=NOW,
    )
    db.add(pred)


async def _make_obs(db, uid, id_obs, model_name, observed_result):
    obs = ObservedResult(
        id_obs=id_obs,
        model_name=model_name,
        observed_result=observed_result,
        date_time=NOW,
        user_id=uid,
    )
    db.add(obs)


async def _setup():
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, TOKEN)
        if not user:
            user = await DBService.create_user(
                db,
                username="calib_user",
                email="calib_user@test.com",
                api_token=TOKEN,
                role="user",
                rate_limit=100000,
            )

        uid = user.id

        # MODEL_ENOUGH — 40 paires avec bonnes probabilités, calibration ~ok
        for i in range(40):
            obs_id = f"calib-enough-{i}"
            correct = i % 2  # 50% correct
            proba = [0.4, 0.6] if correct else [0.6, 0.4]
            await _make_prediction(db, uid, MODEL_ENOUGH, VERSION, obs_id, correct, proba)
            await _make_obs(db, uid, obs_id, MODEL_ENOUGH, correct)

        # MODEL_NO_PROBA — 40 paires sans probabilités
        for i in range(40):
            obs_id = f"calib-noproba-{i}"
            await _make_prediction(db, uid, MODEL_NO_PROBA, VERSION, obs_id, 1, None)
            await _make_obs(db, uid, obs_id, MODEL_NO_PROBA, 1)

        # MODEL_FEW — 10 paires seulement
        for i in range(10):
            obs_id = f"calib-few-{i}"
            await _make_prediction(db, uid, MODEL_FEW, VERSION, obs_id, 1, [0.2, 0.8])
            await _make_obs(db, uid, obs_id, MODEL_FEW, 1)

        # MODEL_OVERCONF — 40 paires sur-confiantes (conf ~0.95, accuracy ~0.50)
        for i in range(40):
            obs_id = f"calib-overconf-{i}"
            pred = 1
            proba = [0.05, 0.95]
            observed = 1 if i < 20 else 0  # 50% précision
            await _make_prediction(db, uid, MODEL_OVERCONF, VERSION, obs_id, pred, proba)
            await _make_obs(db, uid, obs_id, MODEL_OVERCONF, observed)

        # MODEL_UNDERCONF — 40 paires sous-confiantes (conf ~0.51, accuracy ~0.90)
        for i in range(40):
            obs_id = f"calib-underconf-{i}"
            pred = 1
            proba = [0.49, 0.51]
            observed = 1 if i < 36 else 0  # 90% précision
            await _make_prediction(db, uid, MODEL_UNDERCONF, VERSION, obs_id, pred, proba)
            await _make_obs(db, uid, obs_id, MODEL_UNDERCONF, observed)

        # MODEL_V1 — two versions, 30 paires each
        for version in [VERSION, VERSION_B]:
            for i in range(30):
                obs_id = f"calib-versioned-{version}-{i}"
                await _make_prediction(db, uid, MODEL_V1, version, obs_id, 1, [0.2, 0.8])
                await _make_obs(db, uid, obs_id, MODEL_V1, 1)

        await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_calibration_no_auth():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration")
    assert r.status_code in [401, 403]


def test_calibration_invalid_token():
    r = client.get(
        f"/models/{MODEL_ENOUGH}/calibration",
        headers={"Authorization": "Bearer bad-token-xyz"},
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Empty model → insufficient_data with sample_size=0
# ---------------------------------------------------------------------------


def test_calibration_unknown_model_returns_insufficient_data():
    r = client.get("/models/totally_unknown_model_xyz/calibration", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["calibration_status"] == "insufficient_data"
    assert data["sample_size"] == 0
    assert data["reliability"] == []


# ---------------------------------------------------------------------------
# All probabilities null → 422
# ---------------------------------------------------------------------------


def test_calibration_all_null_probabilities_returns_422():
    r = client.get(f"/models/{MODEL_NO_PROBA}/calibration", headers=AUTH)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Fewer than 30 pairs → insufficient_data
# ---------------------------------------------------------------------------


def test_calibration_few_samples_returns_insufficient_data():
    r = client.get(f"/models/{MODEL_FEW}/calibration", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["calibration_status"] == "insufficient_data"
    assert data["sample_size"] < 30
    assert data["brier_score"] is None


# ---------------------------------------------------------------------------
# ≥ 30 pairs — response structure
# ---------------------------------------------------------------------------


def test_calibration_response_fields_present():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == MODEL_ENOUGH
    assert data["sample_size"] >= 30
    assert data["brier_score"] is not None
    assert data["calibration_status"] in ["ok", "overconfident", "underconfident"]
    assert data["mean_confidence"] is not None
    assert data["mean_accuracy"] is not None
    assert data["overconfidence_gap"] is not None
    assert isinstance(data["reliability"], list)


def test_calibration_reliability_bins_structure():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration", headers=AUTH)
    assert r.status_code == 200
    bins = r.json()["reliability"]
    assert len(bins) > 0
    for b in bins:
        assert "confidence_bin" in b
        assert "predicted_rate" in b
        assert "observed_rate" in b
        assert "count" in b
        assert b["count"] > 0


def test_calibration_brier_score_range():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration", headers=AUTH)
    assert r.status_code == 200
    bs = r.json()["brier_score"]
    assert 0.0 <= bs <= 1.0


# ---------------------------------------------------------------------------
# Overconfident detection
# ---------------------------------------------------------------------------


def test_calibration_overconfident_status():
    r = client.get(f"/models/{MODEL_OVERCONF}/calibration", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["calibration_status"] == "overconfident"
    assert data["overconfidence_gap"] > 0.05


# ---------------------------------------------------------------------------
# Underconfident detection
# ---------------------------------------------------------------------------


def test_calibration_underconfident_status():
    r = client.get(f"/models/{MODEL_UNDERCONF}/calibration", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["calibration_status"] == "underconfident"
    assert data["overconfidence_gap"] < -0.05


# ---------------------------------------------------------------------------
# n_bins parameter
# ---------------------------------------------------------------------------


def test_calibration_n_bins_respected():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration?n_bins=5", headers=AUTH)
    assert r.status_code == 200
    bins = r.json()["reliability"]
    assert len(bins) <= 5


def test_calibration_n_bins_default_10():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration", headers=AUTH)
    assert r.status_code == 200
    bins = r.json()["reliability"]
    assert len(bins) <= 10


def test_calibration_n_bins_too_small_returns_422():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration?n_bins=1", headers=AUTH)
    assert r.status_code == 422


def test_calibration_n_bins_too_large_returns_422():
    r = client.get(f"/models/{MODEL_ENOUGH}/calibration?n_bins=21", headers=AUTH)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Version filtering
# ---------------------------------------------------------------------------


def test_calibration_version_filter():
    r1 = client.get(f"/models/{MODEL_V1}/calibration?version={VERSION}", headers=AUTH)
    r2 = client.get(f"/models/{MODEL_V1}/calibration?version={VERSION_B}", headers=AUTH)
    assert r1.status_code == 200
    assert r2.status_code == 200
    d1 = r1.json()
    d2 = r2.json()
    assert d1["version"] == VERSION
    assert d2["version"] == VERSION_B
    # Each version has 30 pairs — both should report their own sample_size
    assert d1["sample_size"] == 30
    assert d2["sample_size"] == 30


def test_calibration_no_version_filter_aggregates_all():
    r = client.get(f"/models/{MODEL_V1}/calibration", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    # Both versions combined = 60 pairs
    assert data["sample_size"] == 60


# ---------------------------------------------------------------------------
# overconfidence_gap sign
# ---------------------------------------------------------------------------


def test_calibration_gap_sign_overconfident():
    r = client.get(f"/models/{MODEL_OVERCONF}/calibration", headers=AUTH)
    gap = r.json()["overconfidence_gap"]
    # overconfident: mean_confidence > mean_accuracy → gap > 0
    assert gap > 0


def test_calibration_gap_sign_underconfident():
    r = client.get(f"/models/{MODEL_UNDERCONF}/calibration", headers=AUTH)
    gap = r.json()["overconfidence_gap"]
    # underconfident: mean_confidence < mean_accuracy → gap < 0
    assert gap < 0
