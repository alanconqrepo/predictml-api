"""
Tests pour l'endpoint GET /models/{name}/performance-timeline
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.db.models import ModelMetadata, ObservedResult, Prediction
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-timeline"
AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}
NOW = datetime.now(timezone.utc).replace(tzinfo=None)

# Model names (unique to avoid cross-test contamination)
MODEL_CLF = "tl_classification"
MODEL_REG = "tl_regression"
MODEL_NO_OBS = "tl_no_observed"
MODEL_MULTI_VER = "tl_multi_version"
MODEL_TRAINING_STATS = "tl_training_stats"


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            user = await DBService.create_user(
                db,
                username="test_tl_user",
                email="test_tl@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )
        else:
            from sqlalchemy import select

            from src.db.models import User

            result = await db.execute(select(User).where(User.api_token == TEST_TOKEN))
            user = result.scalar_one_or_none()

        user_id = user.id

        # ── Classification model: 2 versions with pairs ───────────────────────
        for ver, pairs, offset_min in [
            (
                "1.0.0",
                [("tl-clf-1a", 0, 0), ("tl-clf-1b", 1, 1), ("tl-clf-1c", 0, 1), ("tl-clf-1d", 1, 0)],
                60,
            ),
            (
                "2.0.0",
                [("tl-clf-2a", 1, 1), ("tl-clf-2b", 1, 1), ("tl-clf-2c", 1, 0)],
                20,
            ),
        ]:
            existing = await DBService.get_model_metadata(db, MODEL_CLF, ver)
            if not existing:
                db.add(
                    ModelMetadata(
                        name=MODEL_CLF,
                        version=ver,
                        minio_bucket="models",
                        minio_object_key=f"{MODEL_CLF}/v{ver}.joblib",
                        classes=[0, 1],
                        is_active=True,
                        is_production=(ver == "2.0.0"),
                        created_at=NOW - timedelta(minutes=offset_min),
                    )
                )
                await db.flush()
                for id_obs, pred, obs in pairs:
                    db.add(
                        Prediction(
                            user_id=user_id,
                            model_name=MODEL_CLF,
                            model_version=ver,
                            id_obs=id_obs,
                            input_features={"x": 1},
                            prediction_result=pred,
                            probabilities=None,
                            response_time_ms=10.0,
                            status="success",
                            timestamp=NOW - timedelta(minutes=offset_min + 5),
                        )
                    )
                    db.add(
                        ObservedResult(
                            id_obs=id_obs,
                            model_name=MODEL_CLF,
                            observed_result=obs,
                            date_time=NOW - timedelta(minutes=offset_min),
                            user_id=user_id,
                        )
                    )
                await db.commit()

        # ── Regression model: 1 version with float pairs ──────────────────────
        existing = await DBService.get_model_metadata(db, MODEL_REG, "1.0.0")
        if not existing:
            db.add(
                ModelMetadata(
                    name=MODEL_REG,
                    version="1.0.0",
                    minio_bucket="models",
                    minio_object_key=f"{MODEL_REG}/v1.0.0.joblib",
                    classes=None,
                    is_active=True,
                    is_production=True,
                )
            )
            await db.flush()
            for id_obs, pred, obs in [
                ("tl-reg-1", 10.3, 10.5),
                ("tl-reg-2", 20.7, 19.8),
                ("tl-reg-3", 30.1, 30.2),
            ]:
                db.add(
                    Prediction(
                        user_id=user_id,
                        model_name=MODEL_REG,
                        model_version="1.0.0",
                        id_obs=id_obs,
                        input_features={"x": pred},
                        prediction_result=pred,
                        probabilities=None,
                        response_time_ms=5.0,
                        status="success",
                        timestamp=NOW - timedelta(minutes=10),
                    )
                )
                db.add(
                    ObservedResult(
                        id_obs=id_obs,
                        model_name=MODEL_REG,
                        observed_result=obs,
                        date_time=NOW - timedelta(minutes=5),
                        user_id=user_id,
                    )
                )
            await db.commit()

        # ── Model with predictions but NO observed results ────────────────────
        existing = await DBService.get_model_metadata(db, MODEL_NO_OBS, "1.0.0")
        if not existing:
            db.add(
                ModelMetadata(
                    name=MODEL_NO_OBS,
                    version="1.0.0",
                    minio_bucket="models",
                    minio_object_key=f"{MODEL_NO_OBS}/v1.0.0.joblib",
                    classes=[0, 1],
                    is_active=True,
                    is_production=True,
                )
            )
            await db.flush()
            db.add(
                Prediction(
                    user_id=user_id,
                    model_name=MODEL_NO_OBS,
                    model_version="1.0.0",
                    id_obs=None,
                    input_features={"x": 1},
                    prediction_result=1,
                    probabilities=None,
                    response_time_ms=10.0,
                    status="success",
                    timestamp=NOW - timedelta(minutes=5),
                )
            )
            await db.commit()

        # ── Multi-version model: 3 versions to verify ordering ────────────────
        for ver, offset_min in [("3.0.0", 90), ("1.0.0", 180), ("2.0.0", 120)]:
            existing = await DBService.get_model_metadata(db, MODEL_MULTI_VER, ver)
            if not existing:
                db.add(
                    ModelMetadata(
                        name=MODEL_MULTI_VER,
                        version=ver,
                        minio_bucket="models",
                        minio_object_key=f"{MODEL_MULTI_VER}/v{ver}.joblib",
                        classes=[0, 1],
                        is_active=True,
                        is_production=(ver == "3.0.0"),
                        created_at=NOW - timedelta(minutes=offset_min),
                    )
                )
        await db.commit()

        # ── Model with training_stats ─────────────────────────────────────────
        existing = await DBService.get_model_metadata(db, MODEL_TRAINING_STATS, "1.0.0")
        if not existing:
            trained_at_str = (NOW - timedelta(hours=2)).isoformat()
            db.add(
                ModelMetadata(
                    name=MODEL_TRAINING_STATS,
                    version="1.0.0",
                    minio_bucket="models",
                    minio_object_key=f"{MODEL_TRAINING_STATS}/v1.0.0.joblib",
                    classes=[0, 1],
                    is_active=True,
                    is_production=True,
                    training_stats={
                        "trained_at": trained_at_str,
                        "n_rows": 12450,
                        "train_start_date": "2026-01-01",
                        "train_end_date": "2026-01-31",
                    },
                )
            )
            await db.commit()


asyncio.run(_setup())


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_timeline_classification_two_versions():
    response = client.get(f"/models/{MODEL_CLF}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == MODEL_CLF
    assert len(data["timeline"]) == 2

    # Ordered by deployed_at ASC → v1.0.0 first
    v1 = data["timeline"][0]
    assert v1["version"] == "1.0.0"
    assert v1["sample_count"] == 4
    assert v1["accuracy"] == pytest.approx(0.5, abs=0.01)
    assert v1["f1_score"] is not None
    assert v1["mae"] is None

    v2 = data["timeline"][1]
    assert v2["version"] == "2.0.0"
    assert v2["sample_count"] == 3
    assert v2["accuracy"] is not None
    assert v2["f1_score"] is not None
    assert v2["mae"] is None


def test_timeline_classification_accuracy_rounding():
    response = client.get(f"/models/{MODEL_CLF}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    for entry in data["timeline"]:
        if entry["accuracy"] is not None:
            # Should be rounded to at most 4 decimal places
            assert entry["accuracy"] == round(entry["accuracy"], 4)
        if entry["f1_score"] is not None:
            assert entry["f1_score"] == round(entry["f1_score"], 4)


def test_timeline_regression():
    response = client.get(f"/models/{MODEL_REG}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == MODEL_REG
    assert len(data["timeline"]) == 1

    entry = data["timeline"][0]
    assert entry["version"] == "1.0.0"
    assert entry["sample_count"] == 3
    assert entry["mae"] is not None
    assert entry["mae"] > 0
    assert entry["accuracy"] is None
    assert entry["f1_score"] is None


def test_timeline_no_observed_results():
    response = client.get(f"/models/{MODEL_NO_OBS}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()

    assert len(data["timeline"]) == 1
    entry = data["timeline"][0]
    assert entry["sample_count"] == 0
    assert entry["accuracy"] is None
    assert entry["f1_score"] is None
    assert entry["mae"] is None


def test_timeline_model_not_found():
    response = client.get("/models/tl_does_not_exist/performance-timeline", headers=AUTH)
    assert response.status_code == 404


def test_timeline_unauthenticated():
    response = client.get(f"/models/{MODEL_CLF}/performance-timeline")
    assert response.status_code in (401, 403)


def test_timeline_version_ordering():
    response = client.get(f"/models/{MODEL_MULTI_VER}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()

    versions = [entry["version"] for entry in data["timeline"]]
    deployed_ats = [entry["deployed_at"] for entry in data["timeline"]]

    # Should be sorted ASC by deployed_at
    assert deployed_ats == sorted(deployed_ats)
    # v1.0.0 was deployed earliest (offset 180 min), v3.0.0 last (offset 90 min)
    assert versions[0] == "1.0.0"
    assert versions[-1] == "3.0.0"


def test_timeline_training_stats_populated():
    response = client.get(f"/models/{MODEL_TRAINING_STATS}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()

    assert len(data["timeline"]) == 1
    entry = data["timeline"][0]
    assert entry["trained_at"] is not None
    assert entry["n_rows_trained"] == 12450


def test_timeline_training_stats_absent():
    # MODEL_REG has no training_stats set
    response = client.get(f"/models/{MODEL_REG}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()

    entry = data["timeline"][0]
    assert entry["trained_at"] is None
    assert entry["n_rows_trained"] is None


def test_timeline_response_structure():
    response = client.get(f"/models/{MODEL_CLF}/performance-timeline", headers=AUTH)
    assert response.status_code == 200
    data = response.json()

    assert "model_name" in data
    assert "timeline" in data
    assert isinstance(data["timeline"], list)

    for entry in data["timeline"]:
        assert "version" in entry
        assert "deployed_at" in entry
        assert "accuracy" in entry
        assert "mae" in entry
        assert "f1_score" in entry
        assert "sample_count" in entry
        assert "trained_at" in entry
        assert "n_rows_trained" in entry
