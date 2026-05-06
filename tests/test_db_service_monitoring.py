"""
Tests unitaires pour les méthodes de monitoring de src/services/db_service.py

Couvre :
- DBService.get_global_monitoring_stats()
- DBService.get_accuracy_drift()
- DBService.get_feature_production_stats()
- DBService.get_model_version_stats_range()
- DBService.get_model_predictions_timeseries()
- DBService.get_model_recent_errors()
"""

import asyncio
import json
from datetime import datetime, timedelta

import pytest

from src.db.models import Prediction, User
from src.db.models.observed_result import ObservedResult
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

# ---------------------------------------------------------------------------
# Tokens et constantes
# ---------------------------------------------------------------------------

ADMIN_TOKEN = "test-token-dbmon-admin-aa55"
MODEL_A = "monitor_model_a"
MODEL_B = "monitor_model_b"

_NOW = datetime(2025, 6, 15, 12, 0, 0)
_YESTERDAY = _NOW - timedelta(days=1)
_LAST_WEEK = _NOW - timedelta(days=7)


# ---------------------------------------------------------------------------
# Setup : créer les utilisateurs et les prédictions de test
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        # Utilisateur admin
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="dbmon_admin",
                email="dbmon_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_user_id(token: str) -> int:
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, token)
        return user.id


async def _insert_prediction(
    model_name: str,
    version: str = "1.0.0",
    status: str = "success",
    response_time_ms: float = 50.0,
    is_shadow: bool = False,
    timestamp: datetime = None,
    id_obs: str = None,
    prediction_result=1,
    input_features: dict = None,
    error_message: str = None,
) -> int:
    """Insère une prédiction directement en base et retourne son id."""
    user_id = await _get_user_id(ADMIN_TOKEN)
    ts = timestamp or _NOW
    feat = input_features or {"feature_a": 1.0, "feature_b": 2.0}
    async with _TestSessionLocal() as db:
        pred = Prediction(
            user_id=user_id,
            model_name=model_name,
            model_version=version,
            status=status,
            response_time_ms=response_time_ms,
            is_shadow=is_shadow,
            timestamp=ts,
            id_obs=id_obs,
            prediction_result=prediction_result,
            input_features=feat,
            error_message=error_message,
        )
        db.add(pred)
        await db.commit()
        await db.refresh(pred)
        return pred.id


async def _insert_observed_result(
    model_name: str,
    id_obs: str,
    observed_result,
    timestamp: datetime = None,
) -> None:
    user_id = await _get_user_id(ADMIN_TOKEN)
    ts = timestamp or _NOW
    async with _TestSessionLocal() as db:
        obs = ObservedResult(
            user_id=user_id,
            model_name=model_name,
            id_obs=id_obs,
            observed_result=observed_result,
            date_time=ts,
        )
        db.add(obs)
        await db.commit()


# ---------------------------------------------------------------------------
# get_global_monitoring_stats()
# ---------------------------------------------------------------------------


class TestGetGlobalMonitoringStats:
    def test_returns_empty_list_when_no_predictions_in_period(self):
        """Aucune prédiction dans la période → liste vide."""

        async def _run():
            start = _NOW + timedelta(days=100)
            end = _NOW + timedelta(days=101)
            async with _TestSessionLocal() as db:
                stats = await DBService.get_global_monitoring_stats(db, start, end)
            assert stats == []

        asyncio.run(_run())

    def test_aggregates_predictions_by_model_name(self):
        """Deux modèles → deux entrées dans le résultat."""

        async def _run():
            ts = _NOW + timedelta(hours=1)
            await _insert_prediction(f"{MODEL_A}_gms", timestamp=ts)
            await _insert_prediction(f"{MODEL_B}_gms", timestamp=ts)

            async with _TestSessionLocal() as db:
                stats = await DBService.get_global_monitoring_stats(
                    db,
                    ts - timedelta(minutes=5),
                    ts + timedelta(minutes=5),
                )
            names = [s["model_name"] for s in stats]
            assert f"{MODEL_A}_gms" in names
            assert f"{MODEL_B}_gms" in names

        asyncio.run(_run())

    def test_calculates_error_rate_correctly(self):
        """2 succès + 1 erreur → error_rate = 0.333."""

        async def _run():
            ts = _NOW + timedelta(hours=2)
            model = f"{MODEL_A}_err_rate"
            await _insert_prediction(model, status="success", timestamp=ts)
            await _insert_prediction(model, status="success", timestamp=ts)
            await _insert_prediction(model, status="error", timestamp=ts, error_message="fail")

            async with _TestSessionLocal() as db:
                stats = await DBService.get_global_monitoring_stats(
                    db,
                    ts - timedelta(minutes=5),
                    ts + timedelta(minutes=5),
                )
            entry = next(s for s in stats if s["model_name"] == model)
            assert entry["total_predictions"] == 3  # 2 succès + 1 erreur (non-shadow)
            assert entry["error_count"] == 1
            assert abs(entry["error_rate"] - 1 / 3) < 0.01

        asyncio.run(_run())

    def test_excludes_shadow_from_production_count(self):
        """Les prédictions shadow ne sont pas comptées dans total_predictions."""

        async def _run():
            ts = _NOW + timedelta(hours=3)
            model = f"{MODEL_A}_shadow_excl"
            await _insert_prediction(model, status="success", timestamp=ts, is_shadow=False)
            await _insert_prediction(model, status="success", timestamp=ts, is_shadow=True)

            async with _TestSessionLocal() as db:
                stats = await DBService.get_global_monitoring_stats(
                    db,
                    ts - timedelta(minutes=5),
                    ts + timedelta(minutes=5),
                )
            entry = next(s for s in stats if s["model_name"] == model)
            assert entry["total_predictions"] == 1
            assert entry["shadow_predictions"] == 1

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_accuracy_drift()
# ---------------------------------------------------------------------------


class TestGetAccuracyDrift:
    def test_returns_empty_when_no_observed_results(self):
        """Pas de résultats observés → liste vide."""

        async def _run():
            ts = _NOW + timedelta(hours=10)
            model = f"{MODEL_A}_drift_empty"
            await _insert_prediction(model, timestamp=ts, id_obs="obs_1")

            async with _TestSessionLocal() as db:
                result = await DBService.get_accuracy_drift(
                    db, model, ts - timedelta(minutes=5), ts + timedelta(minutes=5)
                )
            assert result == []

        asyncio.run(_run())

    def test_returns_daily_accuracy(self):
        """Prédiction + observed_result correspondant → accuracy calculée."""

        async def _run():
            ts = _NOW + timedelta(hours=11)
            model = f"{MODEL_A}_drift_acc"
            await _insert_prediction(
                model, timestamp=ts, id_obs="drift_obs_1", prediction_result="cat"
            )
            await _insert_prediction(
                model, timestamp=ts, id_obs="drift_obs_2", prediction_result="dog"
            )
            await _insert_prediction(
                model, timestamp=ts, id_obs="drift_obs_3", prediction_result="cat"
            )
            await _insert_observed_result(model, "drift_obs_1", "cat", timestamp=ts)
            await _insert_observed_result(model, "drift_obs_2", "cat", timestamp=ts)
            await _insert_observed_result(model, "drift_obs_3", "cat", timestamp=ts)

            async with _TestSessionLocal() as db:
                result = await DBService.get_accuracy_drift(
                    db, model, ts - timedelta(minutes=5), ts + timedelta(minutes=5)
                )
            assert len(result) == 1
            entry = result[0]
            assert entry["matched_count"] == 3
            assert abs(entry["accuracy"] - 2 / 3) < 0.01  # 2 correct / 3 total

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_feature_production_stats()
# ---------------------------------------------------------------------------


class TestGetFeatureProductionStats:
    def test_returns_empty_when_no_predictions(self):
        """Aucune prédiction pour ce modèle → dict vide."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.get_feature_production_stats(
                    db, "nonexistent_model_xyz", "1.0.0", days=7
                )
            assert result == {}

        asyncio.run(_run())

    def test_aggregates_numeric_features(self):
        """Prédictions avec features numériques → stats calculées."""

        async def _run():
            model = f"{MODEL_A}_feat_stats"
            for i in range(3):
                await _insert_prediction(
                    model,
                    timestamp=datetime.utcnow() - timedelta(hours=i),
                    input_features={"feat_x": float(i + 1), "feat_y": 10.0},
                )

            async with _TestSessionLocal() as db:
                result = await DBService.get_feature_production_stats(
                    db, model, "1.0.0", days=7
                )
            assert "feat_x" in result
            assert "feat_y" in result
            assert result["feat_x"]["count"] == 3
            assert result["feat_x"]["mean"] is not None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_model_version_stats_range()
# ---------------------------------------------------------------------------


class TestGetModelVersionStatsRange:
    def test_returns_stats_per_version(self):
        """Deux versions différentes → deux entrées dans le résultat."""

        async def _run():
            ts = _NOW + timedelta(hours=20)
            model = f"{MODEL_A}_ver_stats"
            await _insert_prediction(model, version="1.0.0", timestamp=ts)
            await _insert_prediction(model, version="2.0.0", timestamp=ts)

            async with _TestSessionLocal() as db:
                result = await DBService.get_model_version_stats_range(
                    db, model, ts - timedelta(minutes=5), ts + timedelta(minutes=5)
                )
            versions = [r["version"] for r in result]
            assert "1.0.0" in versions
            assert "2.0.0" in versions

        asyncio.run(_run())

    def test_returns_empty_for_unknown_model(self):
        """Modèle inconnu → liste vide."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.get_model_version_stats_range(
                    db,
                    "totally_unknown_model_xyz",
                    _NOW - timedelta(days=30),
                    _NOW + timedelta(days=30),
                )
            assert result == []

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_model_predictions_timeseries()
# ---------------------------------------------------------------------------


class TestGetModelPredictionsTimeseries:
    def test_returns_daily_timeseries(self):
        """Prédictions sur 2 jours → 2 entrées dans la série temporelle."""

        async def _run():
            model = f"{MODEL_A}_timeseries"
            day1 = datetime(2025, 6, 10, 10, 0, 0)
            day2 = datetime(2025, 6, 11, 10, 0, 0)
            await _insert_prediction(model, timestamp=day1, response_time_ms=30.0)
            await _insert_prediction(model, timestamp=day2, response_time_ms=50.0)

            async with _TestSessionLocal() as db:
                result = await DBService.get_model_predictions_timeseries(
                    db, model, day1 - timedelta(hours=1), day2 + timedelta(hours=1)
                )
            assert len(result) == 2
            dates = [r["date"] for r in result]
            assert "2025-06-10" in dates
            assert "2025-06-11" in dates

        asyncio.run(_run())

    def test_returns_empty_for_unknown_model(self):
        """Modèle inconnu → liste vide."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.get_model_predictions_timeseries(
                    db,
                    "no_such_model_timeseries",
                    _NOW - timedelta(days=7),
                    _NOW,
                )
            assert result == []

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_model_recent_errors()
# ---------------------------------------------------------------------------


class TestGetModelRecentErrors:
    def test_returns_last_n_error_messages(self):
        """3 erreurs → retourne au plus limit messages distincts."""

        async def _run():
            ts = _NOW + timedelta(hours=30)
            model = f"{MODEL_A}_recent_errors"
            await _insert_prediction(
                model,
                status="error",
                timestamp=ts,
                error_message="Model not loaded",
            )
            await _insert_prediction(
                model,
                status="error",
                timestamp=ts + timedelta(seconds=1),
                error_message="Feature mismatch",
            )
            await _insert_prediction(
                model,
                status="success",
                timestamp=ts + timedelta(seconds=2),
            )

            async with _TestSessionLocal() as db:
                result = await DBService.get_model_recent_errors(
                    db, model, ts - timedelta(minutes=1), ts + timedelta(minutes=5), limit=5
                )
            assert "Model not loaded" in result
            assert "Feature mismatch" in result

        asyncio.run(_run())

    def test_deduplicates_error_messages(self):
        """Messages d'erreur identiques → dédupliqués."""

        async def _run():
            ts = _NOW + timedelta(hours=31)
            model = f"{MODEL_A}_dedup_errors"
            for _ in range(3):
                await _insert_prediction(
                    model,
                    status="error",
                    timestamp=ts,
                    error_message="Same error",
                )

            async with _TestSessionLocal() as db:
                result = await DBService.get_model_recent_errors(
                    db, model, ts - timedelta(minutes=1), ts + timedelta(minutes=5)
                )
            assert result.count("Same error") == 1

        asyncio.run(_run())

    def test_returns_empty_when_no_errors(self):
        """Aucune erreur → liste vide."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.get_model_recent_errors(
                    db,
                    "no_error_model_xyz",
                    _NOW - timedelta(days=1),
                    _NOW + timedelta(days=1),
                )
            assert result == []

        asyncio.run(_run())
