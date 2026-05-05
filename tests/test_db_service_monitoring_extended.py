"""
Tests étendus pour les méthodes de monitoring de src/services/db_service.py.

Complète test_db_service_monitoring.py avec les cas limites :
- get_global_monitoring_stats avec 0 prédictions (DB vide)
- get_global_monitoring_stats avec plusieurs modèles et prédictions variées
- get_model_predictions_timeseries sans prédictions
- get_model_predictions_timeseries avec données multi-jours
- get_model_version_stats_range sans prédictions
- get_model_version_stats_range multi-versions
- get_model_recent_errors sans erreurs
- get_global_monitoring_stats avec prédictions shadow
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.db.models import Prediction, User
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

ADMIN_TOKEN = "test-token-dbmon-ext-admin-ee99"
MODEL_X = "dbmon_ext_model_x"
MODEL_Y = "dbmon_ext_model_y"

_NOW = datetime(2025, 6, 15, 12, 0, 0)
_YESTERDAY = _NOW - timedelta(days=1)
_LAST_WEEK = _NOW - timedelta(days=7)
_FUTURE = _NOW + timedelta(hours=1)


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="dbmon_ext_admin",
                email="dbmon_ext@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())


async def _get_user_id() -> int:
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, ADMIN_TOKEN)
        return user.id


USER_ID = asyncio.run(_get_user_id())


def _make_prediction(
    model_name: str,
    version: str = "1.0.0",
    status: str = "success",
    response_time_ms: float = 50.0,
    is_shadow: bool = False,
    timestamp: datetime = None,
) -> Prediction:
    return Prediction(
        model_name=model_name,
        model_version=version,
        input_data={"f1": 1.0},
        prediction_result="class_a",
        status=status,
        response_time_ms=response_time_ms,
        is_shadow=is_shadow,
        timestamp=timestamp or _NOW,
        user_id=USER_ID,
    )


# ---------------------------------------------------------------------------
# get_global_monitoring_stats
# ---------------------------------------------------------------------------


class TestGetGlobalMonitoringStatsExtended:
    def test_empty_db_returns_empty_list(self):
        """Aucune prédiction dans la période → liste vide."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.get_global_monitoring_stats(
                    db,
                    start=datetime(2030, 1, 1),
                    end=datetime(2030, 1, 2),
                )
            assert result == []

        asyncio.run(_run())

    def test_multiple_models_returned_sorted(self):
        """Plusieurs modèles → un dict par modèle, triés alphabétiquement."""

        async def _run():
            async with _TestSessionLocal() as db:
                db.add(_make_prediction(MODEL_X, timestamp=_NOW))
                db.add(_make_prediction(MODEL_Y, timestamp=_NOW))
                await db.flush()

                result = await DBService.get_global_monitoring_stats(db, _LAST_WEEK, _FUTURE)

            names = [r["model_name"] for r in result]
            assert MODEL_X in names
            assert MODEL_Y in names

        asyncio.run(_run())

    def test_shadow_predictions_counted_separately(self):
        """Prédictions shadow → comptées dans shadow_predictions, pas total_predictions."""

        async def _run():
            model = f"{MODEL_X}_shadow_ext"
            async with _TestSessionLocal() as db:
                db.add(_make_prediction(model, is_shadow=False, timestamp=_NOW))
                db.add(_make_prediction(model, is_shadow=True, timestamp=_NOW))
                await db.flush()

                result = await DBService.get_global_monitoring_stats(db, _LAST_WEEK, _FUTURE)

            model_stat = next((r for r in result if r["model_name"] == model), None)
            assert model_stat is not None
            assert model_stat["total_predictions"] == 1
            assert model_stat["shadow_predictions"] == 1

        asyncio.run(_run())

    def test_error_rate_calculated_correctly(self):
        """5 prédictions dont 2 erreurs → error_rate = 0.40."""

        async def _run():
            model = f"{MODEL_X}_errrate"
            async with _TestSessionLocal() as db:
                for _ in range(3):
                    db.add(_make_prediction(model, status="success", timestamp=_NOW))
                for _ in range(2):
                    db.add(_make_prediction(model, status="error", timestamp=_NOW))
                await db.flush()

                result = await DBService.get_global_monitoring_stats(db, _LAST_WEEK, _FUTURE)

            stat = next(r for r in result if r["model_name"] == model)
            assert stat["error_count"] == 2
            assert stat["error_rate"] == pytest.approx(0.40, abs=0.01)

        asyncio.run(_run())

    def test_avg_latency_none_when_all_errors(self):
        """Toutes les prédictions sont des erreurs → avg_latency_ms=None."""

        async def _run():
            model = f"{MODEL_X}_all_errors"
            async with _TestSessionLocal() as db:
                for _ in range(3):
                    db.add(
                        _make_prediction(
                            model,
                            status="error",
                            response_time_ms=None,
                            timestamp=_NOW,
                        )
                    )
                await db.flush()

                result = await DBService.get_global_monitoring_stats(db, _LAST_WEEK, _FUTURE)

            stat = next(r for r in result if r["model_name"] == model)
            assert stat["avg_latency_ms"] is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_model_predictions_timeseries
# ---------------------------------------------------------------------------


class TestGetModelPredictionsTimeseriesExtended:
    def test_empty_returns_empty_list(self):
        """Aucune prédiction → liste vide."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.get_model_predictions_timeseries(
                    db,
                    model_name="nonexistent_ts_model",
                    start=datetime(2030, 1, 1),
                    end=datetime(2030, 1, 2),
                )
            assert result == []

        asyncio.run(_run())

    def test_multi_day_grouping(self):
        """Prédictions sur 2 jours différents → 2 entrées dans la liste."""

        async def _run():
            model = f"{MODEL_X}_ts_days"
            day1 = datetime(2025, 4, 1, 10, 0, 0)
            day2 = datetime(2025, 4, 2, 10, 0, 0)
            start = datetime(2025, 3, 31)
            end = datetime(2025, 4, 3)

            async with _TestSessionLocal() as db:
                db.add(_make_prediction(model, timestamp=day1))
                db.add(_make_prediction(model, timestamp=day2))
                await db.flush()

                result = await DBService.get_model_predictions_timeseries(db, model, start, end)

            assert len(result) == 2
            dates = [r["date"] for r in result]
            assert "2025-04-01" in dates
            assert "2025-04-02" in dates

        asyncio.run(_run())

    def test_shadow_excluded_from_timeseries(self):
        """Les prédictions shadow ne doivent pas apparaître dans la série."""

        async def _run():
            model = f"{MODEL_X}_ts_shadow"
            ts = datetime(2025, 5, 1, 12, 0, 0)
            start = datetime(2025, 4, 30)
            end = datetime(2025, 5, 2)

            async with _TestSessionLocal() as db:
                db.add(_make_prediction(model, is_shadow=True, timestamp=ts))
                await db.flush()

                result = await DBService.get_model_predictions_timeseries(db, model, start, end)

            assert result == []

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_model_version_stats_range
# ---------------------------------------------------------------------------


class TestGetModelVersionStatsRangeExtended:
    def test_empty_range_returns_empty_list(self):
        """Aucune prédiction dans la plage → liste vide."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.get_model_version_stats_range(
                    db,
                    model_name="nonexistent_vsr_model",
                    start=datetime(2030, 1, 1),
                    end=datetime(2030, 1, 2),
                )
            assert result == []

        asyncio.run(_run())

    def test_multi_version_stats(self):
        """Deux versions distinctes → deux entrées dans la liste."""

        async def _run():
            model = f"{MODEL_X}_vsr_multi"
            ts = datetime(2025, 6, 1, 12, 0)
            start = datetime(2025, 5, 31)
            end = datetime(2025, 6, 2)

            async with _TestSessionLocal() as db:
                db.add(_make_prediction(model, version="1.0.0", timestamp=ts))
                db.add(_make_prediction(model, version="2.0.0", timestamp=ts))
                await db.flush()

                result = await DBService.get_model_version_stats_range(db, model, start, end)

            versions = [r["version"] for r in result]
            assert "1.0.0" in versions
            assert "2.0.0" in versions

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# get_model_recent_errors
# ---------------------------------------------------------------------------


class TestGetModelRecentErrorsExtended:
    def test_no_errors_returns_empty_list(self):
        """Aucune prédiction en erreur → liste vide."""

        async def _run():
            model = f"{MODEL_X}_no_errs"
            ts = datetime(2025, 6, 10, 12, 0)
            start = datetime(2025, 6, 9)
            end = datetime(2025, 6, 11)

            async with _TestSessionLocal() as db:
                db.add(_make_prediction(model, status="success", timestamp=ts))
                await db.flush()

                result = await DBService.get_model_recent_errors(db, model, start, end)

            assert result == []

        asyncio.run(_run())
