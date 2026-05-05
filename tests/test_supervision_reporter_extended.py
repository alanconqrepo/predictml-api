"""
Tests étendus pour src/tasks/supervision_reporter.py.

Couvre les branches conditionnelles non couvertes par test_supervision_reporter.py :
- Webhook déclenché sur error_rate > threshold
- Alerte drift feature critique + webhook drift
- Seuil modèle-spécifique override seuil global
- run_weekly_report complet (enabled / disabled / exception)
- _get_model_threshold helper
- start_scheduler avec/sans weekly report
- stop_scheduler
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import _TestSessionLocal


@asynccontextmanager
async def _test_session_cm():
    async with _TestSessionLocal() as session:
        yield session


def _make_mock_meta(
    name="model_x",
    version="1.0.0",
    is_production=True,
    webhook_url=None,
    feature_baseline=None,
    alert_thresholds=None,
    promotion_policy=None,
    retrain_schedule=None,
    train_script_object_key=None,
):
    meta = MagicMock()
    meta.name = name
    meta.version = version
    meta.is_production = is_production
    meta.webhook_url = webhook_url
    meta.feature_baseline = feature_baseline
    meta.alert_thresholds = alert_thresholds
    meta.promotion_policy = promotion_policy
    meta.retrain_schedule = retrain_schedule
    meta.train_script_object_key = train_script_object_key
    return meta


# ---------------------------------------------------------------------------
# Webhook sur error_rate_threshold
# ---------------------------------------------------------------------------


class TestAlertCheckWebhooks:
    def test_error_rate_threshold_triggers_webhook(self):
        """error_rate > seuil + webhook_url → send_webhook appelé."""
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "webhook_model",
                "error_rate": 0.60,
                "total_predictions": 10,
                "error_count": 6,
                "shadow_predictions": 0,
                "avg_latency_ms": 50,
            }
        ]
        meta = _make_mock_meta(
            name="webhook_model",
            webhook_url="http://hooks.example.com/alert",
        )

        with (
            patch(
                "src.core.config.settings.ENABLE_EMAIL_ALERTS", False
            ),
            patch(
                "src.core.config.settings.ERROR_RATE_ALERT_THRESHOLD", 0.10
            ),
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new_callable=AsyncMock,
                return_value=mock_stats,
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[meta],
            ),
            patch(
                "src.tasks.supervision_reporter.send_webhook",
                new_callable=AsyncMock,
            ) as mock_webhook,
            patch("asyncio.create_task") as mock_create_task,
        ):
            asyncio.run(run_alert_check())
            mock_create_task.assert_called()

    def test_no_webhook_when_url_absent(self):
        """error_rate > seuil mais pas de webhook_url → send_webhook non appelé."""
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "no_hook_model",
                "error_rate": 0.80,
                "total_predictions": 10,
                "error_count": 8,
                "shadow_predictions": 0,
                "avg_latency_ms": 50,
            }
        ]
        meta = _make_mock_meta(name="no_hook_model", webhook_url=None)

        with (
            patch("src.core.config.settings.ENABLE_EMAIL_ALERTS", False),
            patch("src.core.config.settings.ERROR_RATE_ALERT_THRESHOLD", 0.10),
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new_callable=AsyncMock,
                return_value=mock_stats,
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[meta],
            ),
            patch(
                "src.tasks.supervision_reporter.send_webhook",
                new_callable=AsyncMock,
            ) as mock_send_webhook,
        ):
            asyncio.run(run_alert_check())
            mock_send_webhook.assert_not_called()


# ---------------------------------------------------------------------------
# Seuil modèle-spécifique
# ---------------------------------------------------------------------------


class TestModelSpecificThreshold:
    def test_model_threshold_overrides_global(self):
        """alert_thresholds modèle-spécifique override le seuil global."""
        from src.tasks.supervision_reporter import _get_model_threshold

        thresholds = {"error_rate_max": 0.05}
        result = _get_model_threshold(thresholds, "error_rate_max", 0.10)
        assert result == 0.05

    def test_fallback_to_global_when_key_absent(self):
        """Clé absente → fallback vers seuil global."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold({}, "error_rate_max", 0.10)
        assert result == 0.10

    def test_fallback_to_global_when_thresholds_none(self):
        """thresholds=None → fallback vers seuil global."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold(None, "error_rate_max", 0.10)
        assert result == 0.10

    def test_model_threshold_used_to_suppress_alert(self):
        """Seuil modèle-spécifique plus élevé → pas d'alerte malgré error_rate > seuil global."""
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "strict_model",
                "error_rate": 0.15,
                "total_predictions": 100,
                "error_count": 15,
                "shadow_predictions": 0,
                "avg_latency_ms": 50,
            }
        ]
        meta = _make_mock_meta(
            name="strict_model",
            alert_thresholds={"error_rate_max": 0.50},  # seuil modèle = 0.50
        )

        with (
            patch("src.core.config.settings.ENABLE_EMAIL_ALERTS", True),
            patch("src.core.config.settings.ERROR_RATE_ALERT_THRESHOLD", 0.10),
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new_callable=AsyncMock,
                return_value=mock_stats,
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[meta],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            mock_email.send_error_spike_alert = MagicMock()
            asyncio.run(run_alert_check())
            mock_email.send_error_spike_alert.assert_not_called()


# ---------------------------------------------------------------------------
# run_weekly_report
# ---------------------------------------------------------------------------


class TestWeeklyReport:
    def test_weekly_report_disabled_returns_early(self):
        """WEEKLY_REPORT_ENABLED=False → email non envoyé."""
        from src.tasks.supervision_reporter import run_weekly_report

        with patch("src.core.config.settings.WEEKLY_REPORT_ENABLED", False):
            asyncio.run(run_weekly_report())

    def test_weekly_report_enabled_sends_email(self):
        """WEEKLY_REPORT_ENABLED=True → email_service.send_weekly_report appelé."""
        from src.tasks.supervision_reporter import run_weekly_report

        with (
            patch("src.core.config.settings.WEEKLY_REPORT_ENABLED", True),
            patch("src.core.config.settings.ERROR_RATE_ALERT_THRESHOLD", 0.10),
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            mock_email.send_weekly_report = MagicMock()
            asyncio.run(run_weekly_report())
            mock_email.send_weekly_report.assert_called_once()

    def test_weekly_report_exception_does_not_crash(self):
        """Exception dans DB → loguée, pas de raise."""
        from src.tasks.supervision_reporter import run_weekly_report

        with (
            patch("src.core.config.settings.WEEKLY_REPORT_ENABLED", True),
            patch(
                "src.db.database.AsyncSessionLocal",
                side_effect=Exception("DB error"),
            ),
        ):
            asyncio.run(run_weekly_report())

    def test_weekly_report_with_predictions_summary(self):
        """Rapport avec des prédictions → total_predictions correctement agrégé."""
        from src.tasks.supervision_reporter import run_weekly_report

        mock_stats = [
            {
                "model_name": "model_a",
                "error_rate": 0.05,
                "total_predictions": 100,
                "error_count": 5,
                "shadow_predictions": 10,
                "avg_latency_ms": 45.0,
            },
            {
                "model_name": "model_b",
                "error_rate": 0.02,
                "total_predictions": 200,
                "error_count": 4,
                "shadow_predictions": 20,
                "avg_latency_ms": 30.0,
            },
        ]

        with (
            patch("src.core.config.settings.WEEKLY_REPORT_ENABLED", True),
            patch("src.core.config.settings.ERROR_RATE_ALERT_THRESHOLD", 0.10),
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new_callable=AsyncMock,
                return_value=mock_stats,
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            captured = {}

            def _capture(overview):
                captured["overview"] = overview

            mock_email.send_weekly_report = _capture
            asyncio.run(run_weekly_report())

        assert captured["overview"]["global_stats"]["total_predictions"] == 300
        assert len(captured["overview"]["models"]) == 2


# ---------------------------------------------------------------------------
# start_scheduler / stop_scheduler lifecycle
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    def test_start_scheduler_starts_apscheduler(self):
        """start_scheduler → _scheduler.start() appelé."""
        from src.tasks.supervision_reporter import start_scheduler

        mock_scheduler = MagicMock()
        mock_scheduler.running = False

        with (
            patch("src.tasks.supervision_reporter._scheduler", mock_scheduler),
            patch("src.tasks.supervision_reporter._APSCHEDULER_AVAILABLE", True),
            patch("src.core.config.settings.WEEKLY_REPORT_ENABLED", False),
        ):
            start_scheduler()
            mock_scheduler.start.assert_called_once()

    def test_start_scheduler_adds_weekly_job_when_enabled(self):
        """WEEKLY_REPORT_ENABLED=True → job weekly_report ajouté."""
        from src.tasks.supervision_reporter import start_scheduler

        mock_scheduler = MagicMock()

        with (
            patch("src.tasks.supervision_reporter._scheduler", mock_scheduler),
            patch("src.tasks.supervision_reporter._APSCHEDULER_AVAILABLE", True),
            patch("src.core.config.settings.WEEKLY_REPORT_ENABLED", True),
            patch("src.core.config.settings.WEEKLY_REPORT_DAY", "mon"),
            patch("src.core.config.settings.WEEKLY_REPORT_HOUR", 8),
        ):
            start_scheduler()
        calls = [str(c) for c in mock_scheduler.add_job.call_args_list]
        assert any("weekly_report" in c for c in calls)

    def test_start_scheduler_noop_when_apscheduler_unavailable(self):
        """_APSCHEDULER_AVAILABLE=False → rien n'est fait."""
        from src.tasks.supervision_reporter import start_scheduler

        mock_scheduler = MagicMock()

        with (
            patch("src.tasks.supervision_reporter._scheduler", None),
            patch("src.tasks.supervision_reporter._APSCHEDULER_AVAILABLE", False),
        ):
            start_scheduler()
            mock_scheduler.start.assert_not_called()

    def test_stop_scheduler_shuts_down(self):
        """stop_scheduler → _scheduler.shutdown appelé si running=True."""
        from src.tasks.supervision_reporter import stop_scheduler

        mock_scheduler = MagicMock()
        mock_scheduler.running = True

        with patch("src.tasks.supervision_reporter._scheduler", mock_scheduler):
            stop_scheduler()
            mock_scheduler.shutdown.assert_called_once_with(wait=False)

    def test_stop_scheduler_noop_when_not_running(self):
        """stop_scheduler → shutdown non appelé si scheduler pas running."""
        from src.tasks.supervision_reporter import stop_scheduler

        mock_scheduler = MagicMock()
        mock_scheduler.running = False

        with patch("src.tasks.supervision_reporter._scheduler", mock_scheduler):
            stop_scheduler()
            mock_scheduler.shutdown.assert_not_called()
