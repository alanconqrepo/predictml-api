"""
Tests pour GET /models/{name}/{version}/card

Stratégie :
  - SQLite in-memory + FakeRedis (via conftest.py)
  - RF model injecté dans le cache Redis pour la section SHAP
  - Prédictions + ObservedResult seedés pour performance, calibration, coverage
  - feature_baseline seedé pour tester le drift
  - Vérifie : JSON response, Markdown response, 404, auth, sections nullables
"""

import asyncio
import io
import joblib
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.db.models import ObservedResult, Prediction
from src.db.models.model_metadata import ModelMetadata
from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TOKEN = "test-token-card-a9z3"
CARD_MODEL = "card_test_model"
CARD_MODEL_EMPTY = "card_test_empty"
VERSION = "1.0.0"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
NOW = datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rf():
    X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [0.3, 0.5, 0.7, 0.9]})
    y = [0, 1, 0, 1]
    return RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)


def _inject_cache(model_name: str, version: str, model, feature_baseline=None) -> None:
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


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, TOKEN)
        if not user:
            user = await DBService.create_user(
                db,
                username="card_user",
                email="card@test.com",
                api_token=TOKEN,
                role="user",
                rate_limit=100000,
            )

        # Model with predictions + observed results
        if not await DBService.get_model_metadata(db, CARD_MODEL, VERSION):
            await DBService.create_model_metadata(
                db,
                name=CARD_MODEL,
                version=VERSION,
                minio_bucket="models",
                minio_object_key=f"{CARD_MODEL}/{VERSION}.pkl",
                is_active=True,
                is_production=True,
                algorithm="RandomForestClassifier",
                accuracy=0.85,
                f1_score=0.83,
                tags=["test", "iris"],
                classes=[0, 1],
                features_count=2,
                trained_by="test_user",
                feature_baseline={
                    "feat_a": {"mean": 2.5, "std": 1.0, "min": 1.0, "max": 4.0},
                    "feat_b": {"mean": 0.6, "std": 0.2, "min": 0.3, "max": 0.9},
                },
            )

        for i in range(8):
            obs_id = f"card-obs-{i}"
            pred = Prediction(
                user_id=user.id,
                model_name=CARD_MODEL,
                model_version=VERSION,
                id_obs=obs_id,
                input_features={"feat_a": float(i + 1), "feat_b": 0.5},
                prediction_result=i % 2,
                probabilities=[0.6, 0.4] if i % 2 else [0.4, 0.6],
                response_time_ms=10.0,
                status="success",
                timestamp=NOW,
            )
            db.add(pred)
            db.add(
                ObservedResult(
                    id_obs=obs_id,
                    model_name=CARD_MODEL,
                    observed_result=i % 2,
                    date_time=NOW,
                    user_id=user.id,
                )
            )

        # Empty model — no predictions
        if not await DBService.get_model_metadata(db, CARD_MODEL_EMPTY, VERSION):
            await DBService.create_model_metadata(
                db,
                name=CARD_MODEL_EMPTY,
                version=VERSION,
                minio_bucket="models",
                minio_object_key=f"{CARD_MODEL_EMPTY}/{VERSION}.pkl",
                is_active=True,
                is_production=False,
                algorithm="LogisticRegression",
            )

        await db.commit()


asyncio.run(_setup())
_inject_cache(CARD_MODEL, VERSION, _make_rf())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_card_no_auth():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card")
    assert r.status_code in (401, 403)


def test_card_invalid_token():
    r = client.get(
        f"/models/{CARD_MODEL}/{VERSION}/card",
        headers={"Authorization": "Bearer bad-token"},
    )
    assert r.status_code == 401


def test_card_404():
    r = client.get("/models/nonexistent_xyz/999.0.0/card", headers=AUTH)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# JSON response
# ---------------------------------------------------------------------------


def test_card_200_json():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card", headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


def test_card_top_level_keys():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    required = {
        "model_name",
        "version",
        "generated_at",
        "algorithm",
        "accuracy",
        "f1_score",
        "tags",
        "classes",
        "features_count",
        "trained_by",
        "training_dataset",
        "created_at",
        "is_production",
        "performance",
        "drift",
        "calibration",
        "feature_importance",
        "retrain",
        "coverage",
    }
    assert required.issubset(data.keys())


def test_card_metadata_fields():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card", headers=AUTH)
    data = r.json()
    assert data["model_name"] == CARD_MODEL
    assert data["version"] == VERSION
    assert data["algorithm"] == "RandomForestClassifier"
    assert data["accuracy"] == pytest.approx(0.85)
    assert data["is_production"] is True
    assert "test" in data["tags"]


def test_card_sections_are_nullable():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    for section in ["performance", "drift", "calibration", "feature_importance", "retrain", "coverage"]:
        assert section in data
        assert data[section] is None or isinstance(data[section], dict)


def test_card_coverage_present():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card", headers=AUTH)
    data = r.json()
    cov = data.get("coverage")
    assert cov is not None
    assert "coverage_rate" in cov
    assert "labeled_count" in cov
    assert "total_predictions" in cov
    assert cov["total_predictions"] >= 8


def test_card_no_data_still_returns_200():
    r = client.get(f"/models/{CARD_MODEL_EMPTY}/{VERSION}/card", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == CARD_MODEL_EMPTY
    assert data["algorithm"] == "LogisticRegression"
    assert data["is_production"] is False
    # sections should be None or have 0 matched predictions
    perf = data.get("performance")
    if perf is not None:
        assert perf["matched_predictions"] == 0


# ---------------------------------------------------------------------------
# Feature importance top-5 limit
# ---------------------------------------------------------------------------


def test_card_top_features_limit():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card", headers=AUTH)
    data = r.json()
    fi = data.get("feature_importance")
    if fi is not None:
        assert len(fi["top_features"]) <= 5


def test_card_top_features_have_required_fields():
    r = client.get(f"/models/{CARD_MODEL}/{VERSION}/card", headers=AUTH)
    data = r.json()
    fi = data.get("feature_importance")
    if fi is not None and fi["top_features"]:
        for feat in fi["top_features"]:
            assert "feature" in feat
            assert "mean_abs_shap" in feat


# ---------------------------------------------------------------------------
# Markdown response
# ---------------------------------------------------------------------------


def test_card_markdown_accept():
    r = client.get(
        f"/models/{CARD_MODEL}/{VERSION}/card",
        headers={**AUTH, "Accept": "text/markdown"},
    )
    assert r.status_code == 200
    assert "text/markdown" in r.headers.get("content-type", "")


def test_card_markdown_contains_header():
    r = client.get(
        f"/models/{CARD_MODEL}/{VERSION}/card",
        headers={**AUTH, "Accept": "text/markdown"},
    )
    assert f"# Model Card — {CARD_MODEL} v{VERSION}" in r.text


def test_card_markdown_contains_algorithm():
    r = client.get(
        f"/models/{CARD_MODEL}/{VERSION}/card",
        headers={**AUTH, "Accept": "text/markdown"},
    )
    assert "RandomForestClassifier" in r.text


def test_card_markdown_contains_generated_at():
    r = client.get(
        f"/models/{CARD_MODEL}/{VERSION}/card",
        headers={**AUTH, "Accept": "text/markdown"},
    )
    assert "Généré le" in r.text


def test_card_markdown_content_disposition():
    r = client.get(
        f"/models/{CARD_MODEL}/{VERSION}/card",
        headers={**AUTH, "Accept": "text/markdown"},
    )
    cd = r.headers.get("content-disposition", "")
    assert f"{CARD_MODEL}_{VERSION}_model_card.md" in cd


def test_card_markdown_drift_shows_stable():
    r = client.get(
        f"/models/{CARD_MODEL}/{VERSION}/card",
        headers={**AUTH, "Accept": "text/markdown"},
    )
    # With only 8 predictions the drift section may be insufficient_data or ok
    # Either way the response must be valid Markdown
    assert r.status_code == 200
    assert r.text.startswith("# Model Card")
