"""
Tests pour GET /models/{name}/performance-report — rapport consolidé.

Stratégie :
  - Modèle RF injecté dans le cache Redis pour SHAP
  - Modèle SVC (pas de feature_names_in_) pour tester feature_importance=null
  - Prédictions + ObservedResult seedés pour alimenter performance et calibration
  - Pas de Docker requis (SQLite in-memory)
"""

import asyncio
import io
import joblib
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.db.models import ObservedResult, Prediction
from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-pr-x4k9"
PR_MODEL = "pr_test_model"
PR_MODEL_NOSHAP = "pr_model_noshap"
MODEL_VERSION = "1.0.0"

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}
NOW = datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _make_rf_model() -> RandomForestClassifier:
    x_train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [0.3, 0.5, 0.7, 0.9]})
    y = [0, 1, 0, 1]
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(x_train, y)


def _make_svc_model() -> SVC:
    x_train = np.array([[1.0, 0.3], [2.0, 0.5], [3.0, 0.7], [4.0, 0.9]])
    y = [0, 1, 0, 1]
    return SVC().fit(x_train, y)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _inject_cache(model_name: str, version: str, model, feature_baseline=None) -> str:
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            feature_baseline=feature_baseline,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, TEST_TOKEN)
        if not user:
            user = await DBService.create_user(
                db,
                username="pr_test_user",
                email="pr_test@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=100000,
            )

        uid = user.id

        for model_name in [PR_MODEL, PR_MODEL_NOSHAP]:
            if not await DBService.get_model_metadata(db, model_name, MODEL_VERSION):
                await DBService.create_model_metadata(
                    db,
                    name=model_name,
                    version=MODEL_VERSION,
                    minio_bucket="models",
                    minio_object_key=f"{model_name}/v{MODEL_VERSION}.pkl",
                    is_active=True,
                    is_production=True,
                )

        # Seed predictions + observed results for PR_MODEL (for performance / calibration)
        for i in range(6):
            obs_id = f"pr-obs-{i}"
            correct = i % 2
            proba = [0.4, 0.6] if correct else [0.6, 0.4]
            pred = Prediction(
                user_id=uid,
                model_name=PR_MODEL,
                model_version=MODEL_VERSION,
                id_obs=obs_id,
                input_features={"feat_a": float(i + 1), "feat_b": 0.5},
                prediction_result=correct,
                probabilities=proba,
                response_time_ms=10.0,
                status="success",
                timestamp=NOW,
            )
            db.add(pred)
            obs = ObservedResult(
                id_obs=obs_id,
                model_name=PR_MODEL,
                observed_result=correct,
                date_time=NOW,
                user_id=uid,
            )
            db.add(obs)

        await db.commit()


asyncio.run(_setup())

# Inject model caches
_rf_cache_key = _inject_cache(PR_MODEL, MODEL_VERSION, _make_rf_model())
_svc_cache_key = _inject_cache(PR_MODEL_NOSHAP, MODEL_VERSION, _make_svc_model())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_report_no_auth():
    response = client.get(f"/models/{PR_MODEL}/performance-report")
    assert response.status_code in [401, 403]


def test_report_invalid_token():
    response = client.get(
        f"/models/{PR_MODEL}/performance-report",
        headers={"Authorization": "Bearer invalid-xyz-token"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 404
# ---------------------------------------------------------------------------


def test_report_unknown_model():
    response = client.get(
        "/models/totally_unknown_model_xyz_pr/performance-report",
        headers=AUTH,
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 200 — schema structure
# ---------------------------------------------------------------------------


def test_report_200_top_level_keys():
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    expected_keys = {
        "model_name",
        "generated_at",
        "period_days",
        "performance",
        "drift",
        "feature_importance",
        "calibration",
        "ab_comparison",
    }
    assert expected_keys.issubset(data.keys())


def test_report_model_name_matches():
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["model_name"] == PR_MODEL


def test_report_period_days_default():
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["period_days"] == 30


def test_report_period_days_custom():
    response = client.get(
        f"/models/{PR_MODEL}/performance-report", headers=AUTH, params={"days": 7}
    )
    assert response.status_code == 200
    assert response.json()["period_days"] == 7


def test_report_days_invalid_zero():
    response = client.get(
        f"/models/{PR_MODEL}/performance-report", headers=AUTH, params={"days": 0}
    )
    assert response.status_code == 422


def test_report_generated_at_is_string():
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    assert isinstance(response.json()["generated_at"], str)


# ---------------------------------------------------------------------------
# Sections are nullable
# ---------------------------------------------------------------------------


def test_report_sections_are_nullable():
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    for section in ["performance", "drift", "feature_importance", "calibration", "ab_comparison"]:
        assert section in data
        assert data[section] is None or isinstance(data[section], dict)


# ---------------------------------------------------------------------------
# Performance section
# ---------------------------------------------------------------------------


def test_report_performance_section_structure():
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    perf = data["performance"]
    assert perf is not None
    assert perf["model_name"] == PR_MODEL
    assert "total_predictions" in perf
    assert "model_type" in perf
    assert "matched_predictions" in perf


# ---------------------------------------------------------------------------
# Drift section
# ---------------------------------------------------------------------------


def test_report_drift_section_no_baseline():
    """No feature_baseline set on model → drift_summary == 'no_baseline'."""
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    drift = data["drift"]
    assert drift is not None
    assert drift["drift_summary"] == "no_baseline"
    assert drift["baseline_available"] is False


# ---------------------------------------------------------------------------
# Calibration section
# ---------------------------------------------------------------------------


def test_report_calibration_section_present_or_null():
    """Key exists, value is None or a dict with expected fields."""
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert "calibration" in data
    cal = data["calibration"]
    if cal is not None:
        assert "calibration_status" in cal
        assert "sample_size" in cal


# ---------------------------------------------------------------------------
# A/B comparison section
# ---------------------------------------------------------------------------


def test_report_ab_comparison_section_keys():
    response = client.get(f"/models/{PR_MODEL}/performance-report", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert "ab_comparison" in data
    ab = data["ab_comparison"]
    if ab is not None:
        assert "model_name" in ab
        assert "versions" in ab
        assert isinstance(ab["versions"], list)


# ---------------------------------------------------------------------------
# Feature importance section
# ---------------------------------------------------------------------------


def test_report_feature_importance_null_for_noshap_model():
    """SVC (no feature_names_in_) → feature_importance is null."""
    response = client.get(f"/models/{PR_MODEL_NOSHAP}/performance-report", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["feature_importance"] is None


def test_report_feature_importance_present_for_rf_model():
    """RF model with seeded predictions → feature_importance is not None."""
    response = client.get(
        f"/models/{PR_MODEL}/performance-report", headers=AUTH, params={"days": 30}
    )
    assert response.status_code == 200
    data = response.json()
    fi = data["feature_importance"]
    assert fi is not None
    assert "feature_importance" in fi
    assert "sample_size" in fi


# ---------------------------------------------------------------------------
# format param
# ---------------------------------------------------------------------------


def test_report_format_json_param():
    """?format=json is accepted and returns 200."""
    response = client.get(
        f"/models/{PR_MODEL}/performance-report", headers=AUTH, params={"format": "json"}
    )
    assert response.status_code == 200


def test_report_format_html_falls_back_to_json():
    """?format=html is accepted (phase 2 stub) and still returns JSON."""
    response = client.get(
        f"/models/{PR_MODEL}/performance-report", headers=AUTH, params={"format": "html"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
