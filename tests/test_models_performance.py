"""
Tests pour l'endpoint GET /models/{name}/performance
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

TEST_TOKEN = "test-token-perf"
NOW = datetime.now(timezone.utc).replace(tzinfo=None)
MODEL_CLF_BIN = "perf_clf_binary"
MODEL_CLF_MULTI = "perf_clf_multi"
MODEL_REG = "perf_regression"


async def _setup():
    async with _TestSessionLocal() as db:
        # Utilisateur de test
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            user = await DBService.create_user(
                db,
                username="test_perf_user",
                email="test_perf@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )
        else:
            from sqlalchemy import select

            from src.db.models import User

            result = await db.execute(
                select(User).where(User.api_token == TEST_TOKEN)
            )
            user = result.scalar_one_or_none()

        user_id = user.id

        # ── Modèle classification binaire ────────────────────────────────────
        existing = await DBService.get_model_metadata(db, MODEL_CLF_BIN, "1.0")
        if not existing:
            meta_bin = ModelMetadata(
                name=MODEL_CLF_BIN,
                version="1.0",
                minio_bucket="models",
                minio_object_key="perf_clf_binary/v1.0.joblib",
                classes=[0, 1],
                is_active=True,
                is_production=True,
            )
            db.add(meta_bin)
            await db.flush()

            # 4 prédictions avec id_obs, 1 sans
            pairs_bin = [
                ("obs-bin-1", 0, 0),  # correct
                ("obs-bin-2", 1, 1),  # correct
                ("obs-bin-3", 0, 1),  # incorrect
                ("obs-bin-4", 1, 0),  # incorrect
            ]
            for id_obs, pred, obs in pairs_bin:
                p = Prediction(
                    user_id=user_id,
                    model_name=MODEL_CLF_BIN,
                    model_version="1.0",
                    id_obs=id_obs,
                    input_features={"x": 1},
                    prediction_result=pred,
                    probabilities=None,
                    response_time_ms=10.0,
                    status="success",
                    timestamp=NOW - timedelta(minutes=10),
                )
                db.add(p)
                o = ObservedResult(
                    id_obs=id_obs,
                    model_name=MODEL_CLF_BIN,
                    observed_result=obs,
                    date_time=NOW - timedelta(minutes=5),
                    user_id=user_id,
                )
                db.add(o)
            # Prédiction sans observed_result (ne doit pas compter dans matched)
            db.add(
                Prediction(
                    user_id=user_id,
                    model_name=MODEL_CLF_BIN,
                    model_version="1.0",
                    id_obs=None,
                    input_features={"x": 2},
                    prediction_result=1,
                    probabilities=None,
                    response_time_ms=10.0,
                    status="success",
                    timestamp=NOW - timedelta(minutes=10),
                )
            )
            await db.commit()

        # ── Modèle classification multiclasse ────────────────────────────────
        existing = await DBService.get_model_metadata(db, MODEL_CLF_MULTI, "1.0")
        if not existing:
            meta_multi = ModelMetadata(
                name=MODEL_CLF_MULTI,
                version="1.0",
                minio_bucket="models",
                minio_object_key="perf_clf_multi/v1.0.joblib",
                classes=["cat", "dog", "bird"],
                is_active=True,
                is_production=True,
            )
            db.add(meta_multi)
            await db.flush()

            pairs_multi = [
                ("obs-multi-1", "cat", "cat"),
                ("obs-multi-2", "dog", "dog"),
                ("obs-multi-3", "bird", "cat"),
                ("obs-multi-4", "cat", "bird"),
                ("obs-multi-5", "dog", "dog"),
                ("obs-multi-6", "bird", "bird"),
            ]
            for id_obs, pred, obs in pairs_multi:
                db.add(
                    Prediction(
                        user_id=user_id,
                        model_name=MODEL_CLF_MULTI,
                        model_version="1.0",
                        id_obs=id_obs,
                        input_features={"x": 1},
                        prediction_result=pred,
                        probabilities=[0.8, 0.1, 0.1],
                        response_time_ms=10.0,
                        status="success",
                        timestamp=NOW - timedelta(minutes=10),
                    )
                )
                db.add(
                    ObservedResult(
                        id_obs=id_obs,
                        model_name=MODEL_CLF_MULTI,
                        observed_result=obs,
                        date_time=NOW - timedelta(minutes=5),
                        user_id=user_id,
                    )
                )
            await db.commit()

        # ── Modèle régression ────────────────────────────────────────────────
        existing = await DBService.get_model_metadata(db, MODEL_REG, "1.0")
        if not existing:
            meta_reg = ModelMetadata(
                name=MODEL_REG,
                version="1.0",
                minio_bucket="models",
                minio_object_key="perf_regression/v1.0.joblib",
                classes=None,
                is_active=True,
                is_production=True,
            )
            db.add(meta_reg)
            await db.flush()

            # y_pred ≈ y_true (bonnes prédictions) — floats non-entiers pour forcer régression
            reg_pairs = [
                ("obs-reg-1", 10.3, 10.5),
                ("obs-reg-2", 20.7, 19.8),
                ("obs-reg-3", 30.1, 30.2),
                ("obs-reg-4", 40.9, 39.5),
            ]
            for id_obs, pred, obs in reg_pairs:
                db.add(
                    Prediction(
                        user_id=user_id,
                        model_name=MODEL_REG,
                        model_version="1.0",
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


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_performance_without_auth():
    response = client.get(f"/models/{MODEL_CLF_BIN}/performance")
    assert response.status_code == 401


def test_performance_model_not_found():
    response = client.get(
        "/models/modele_inexistant_xyz/performance",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 404


def test_performance_no_data():
    """Modèle sans prédictions jointes → matched_predictions=0, métriques absentes."""
    async def _add_empty_model():
        async with _TestSessionLocal() as db:
            existing = await DBService.get_model_metadata(db, "perf_empty_model", "1.0")
            if not existing:
                db.add(
                    ModelMetadata(
                        name="perf_empty_model",
                        version="1.0",
                        minio_bucket="models",
                        minio_object_key="perf_empty/v1.0.joblib",
                        is_active=True,
                        is_production=True,
                    )
                )
                await db.commit()

    asyncio.run(_add_empty_model())

    response = client.get(
        "/models/perf_empty_model/performance",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["matched_predictions"] == 0
    assert data["accuracy"] is None
    assert data["mae"] is None
    assert data["by_period"] is None


def test_performance_binary_classification():
    """Classification binaire : 2/4 correctes → accuracy=0.5."""
    response = client.get(
        f"/models/{MODEL_CLF_BIN}/performance",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == MODEL_CLF_BIN
    assert data["model_type"] == "classification"
    assert data["matched_predictions"] == 4
    assert data["total_predictions"] >= 4

    assert data["accuracy"] == pytest.approx(0.5, abs=0.01)
    assert data["f1_weighted"] is not None
    assert data["confusion_matrix"] is not None
    assert len(data["confusion_matrix"]) == 2  # 2 classes
    assert data["classes"] is not None

    # Métriques par classe présentes
    assert data["per_class_metrics"] is not None
    assert len(data["per_class_metrics"]) == 2

    # Métriques régression absentes
    assert data["mae"] is None
    assert data["mse"] is None


def test_performance_multiclass_classification():
    """Classification multiclasse (cat/dog/bird) : per_class présent pour chaque classe."""
    response = client.get(
        f"/models/{MODEL_CLF_MULTI}/performance",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["model_type"] == "classification"
    assert data["matched_predictions"] == 6

    assert data["accuracy"] is not None
    assert 0.0 <= data["accuracy"] <= 1.0

    per_class = data["per_class_metrics"]
    assert per_class is not None
    # Les 3 classes doivent être présentes
    for cls in ["cat", "dog", "bird"]:
        assert cls in per_class
        assert "precision" in per_class[cls]
        assert "recall" in per_class[cls]
        assert "f1_score" in per_class[cls]
        assert "support" in per_class[cls]

    assert data["confusion_matrix"] is not None
    assert len(data["confusion_matrix"]) == 3  # 3x3


def test_performance_regression():
    """Régression : mae/mse/rmse/r2 présents, métriques classification absentes."""
    response = client.get(
        f"/models/{MODEL_REG}/performance",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["model_type"] == "regression"
    assert data["matched_predictions"] == 4

    assert data["mae"] is not None
    assert data["mse"] is not None
    assert data["rmse"] is not None
    assert data["r2"] is not None

    # rmse = sqrt(mse)
    import math
    assert data["rmse"] == pytest.approx(math.sqrt(data["mse"]), abs=0.001)

    # Métriques classification absentes
    assert data["accuracy"] is None
    assert data["confusion_matrix"] is None
    assert data["per_class_metrics"] is None


def test_performance_with_version_filter():
    """Filtrage par version — version inexistante → 0 matches."""
    response = client.get(
        f"/models/{MODEL_CLF_BIN}/performance?version=99.0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 404


def test_performance_granularity_day():
    """Avec granularity=day → by_period non vide."""
    response = client.get(
        f"/models/{MODEL_CLF_BIN}/performance?granularity=day",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["by_period"] is not None
    assert len(data["by_period"]) >= 1

    # Chaque bucket a les bons champs
    for bucket in data["by_period"]:
        assert "period" in bucket
        assert "matched_count" in bucket
        assert bucket["matched_count"] > 0
        # Classification → accuracy présente
        assert bucket["accuracy"] is not None
        # Régression fields absents
        assert bucket["mae"] is None
