"""
Tests unitaires pour src/tasks/supervision_reporter.py

Couvre :
- run_alert_check() — désactivé par défaut, envoi d'alerte si seuil dépassé, gestion d'exception
- run_weekly_report() — désactivé par défaut, envoi du rapport, gestion d'exception
- start_scheduler() / stop_scheduler() — lifecycle APScheduler
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import _TestSessionLocal


# ---------------------------------------------------------------------------
# Helper : fournit un AsyncSessionLocal de test
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _test_session_cm():
    """Remplace AsyncSessionLocal par la session SQLite de test."""
    async with _TestSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# run_alert_check()
# ---------------------------------------------------------------------------


class TestRunAlertCheck:
    def test_run_alert_check_skips_when_disabled(self):
        """ENABLE_EMAIL_ALERTS=False → aucun e-mail envoyé (webhooks peuvent toujours partir)."""
        from src.tasks.supervision_reporter import run_alert_check
        from src.core.config import settings

        with (
            patch.object(settings, "ENABLE_EMAIL_ALERTS", False),
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
            asyncio.run(run_alert_check())
            mock_email.send_error_spike_alert.assert_not_called()
            mock_email.send_performance_alert.assert_not_called()
            mock_email.send_drift_alert.assert_not_called()

    def test_run_alert_check_sends_error_spike_when_threshold_exceeded(self):
        """Taux d'erreur > seuil → send_error_spike_alert est appelé."""
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "super_model",
                "error_rate": 0.50,  # > seuil de 0.10
                "total_predictions": 10,
                "error_count": 5,
                "shadow_predictions": 0,
                "avg_latency_ms": 50,
            }
        ]

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "ENABLE_EMAIL_ALERTS", True),
            patch(
                "src.db.database.AsyncSessionLocal",
                new=_test_session_cm,
            ),
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
                return_value=[],
            ),
            patch(
                "src.services.email_service.email_service"
            ) as mock_email,
        ):
            mock_email.send_error_spike_alert = MagicMock(return_value=True)
            asyncio.run(run_alert_check())
            mock_email.send_error_spike_alert.assert_called_once_with("super_model", 0.50)

    def test_run_alert_check_no_alert_when_below_threshold(self):
        """Taux d'erreur < seuil → send_error_spike_alert n'est pas appelé."""
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "healthy_model",
                "error_rate": 0.02,  # < seuil de 0.10
                "total_predictions": 50,
                "error_count": 1,
                "shadow_predictions": 0,
                "avg_latency_ms": 30,
            }
        ]

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "ENABLE_EMAIL_ALERTS", True),
            patch(
                "src.db.database.AsyncSessionLocal",
                new=_test_session_cm,
            ),
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
                return_value=[],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            asyncio.run(run_alert_check())
            mock_email.send_error_spike_alert.assert_not_called()

    def test_run_alert_check_handles_exception_gracefully(self):
        """Exception dans la DB → loguée, pas de raise."""
        from src.tasks.supervision_reporter import run_alert_check

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "ENABLE_EMAIL_ALERTS", True),
            patch(
                "src.db.database.AsyncSessionLocal",
                side_effect=Exception("DB connection failed"),
            ),
        ):
            # Ne doit pas lever d'exception
            asyncio.run(run_alert_check())

    def test_run_alert_check_sends_performance_drift_on_drop(self):
        """Baisse d'accuracy > seuil (PERFORMANCE_DRIFT_ALERT_THRESHOLD) → send_performance_alert."""
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "degraded_model",
                "error_rate": 0.01,
                "total_predictions": 100,
                "error_count": 1,
                "shadow_predictions": 0,
                "avg_latency_ms": 40,
            }
        ]
        # 4 points: première moitié accuracy=0.95, deuxième moitié accuracy=0.70 → drop=0.25 > 0.10
        perf_data = [
            {"date": "2025-01-01", "matched_count": 10, "accuracy": 0.95},
            {"date": "2025-01-02", "matched_count": 10, "accuracy": 0.95},
            {"date": "2025-01-03", "matched_count": 10, "accuracy": 0.70},
            {"date": "2025-01-04", "matched_count": 10, "accuracy": 0.70},
        ]

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "ENABLE_EMAIL_ALERTS", True),
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "PERFORMANCE_DRIFT_ALERT_THRESHOLD", 0.10),
            patch(
                "src.db.database.AsyncSessionLocal",
                new=_test_session_cm,
            ),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new_callable=AsyncMock,
                return_value=mock_stats,
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new_callable=AsyncMock,
                return_value=perf_data,
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            mock_email.send_performance_alert = MagicMock(return_value=True)
            mock_email.send_error_spike_alert = MagicMock(return_value=False)
            asyncio.run(run_alert_check())
            mock_email.send_performance_alert.assert_called_once()


# ---------------------------------------------------------------------------
# run_weekly_report()
# ---------------------------------------------------------------------------


class TestRunWeeklyReport:
    def test_run_weekly_report_skips_when_disabled(self):
        """WEEKLY_REPORT_ENABLED=False → fonction retourne immédiatement."""
        from src.tasks.supervision_reporter import run_weekly_report

        with patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "WEEKLY_REPORT_ENABLED", False):
            with patch("src.db.database.AsyncSessionLocal") as mock_sess:
                asyncio.run(run_weekly_report())
                mock_sess.assert_not_called()

    def test_run_weekly_report_calls_email_service(self):
        """WEEKLY_REPORT_ENABLED=True + données → send_weekly_report est appelé."""
        from src.tasks.supervision_reporter import run_weekly_report

        mock_stats = [
            {
                "model_name": "weekly_model",
                "error_rate": 0.05,
                "total_predictions": 200,
                "error_count": 10,
                "shadow_predictions": 5,
                "avg_latency_ms": 45,
            }
        ]

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "WEEKLY_REPORT_ENABLED", True),
            patch(
                "src.db.database.AsyncSessionLocal",
                new=_test_session_cm,
            ),
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
            mock_email.send_weekly_report = MagicMock(return_value=True)
            asyncio.run(run_weekly_report())
            mock_email.send_weekly_report.assert_called_once()
            call_arg = mock_email.send_weekly_report.call_args.args[0]
            assert "period" in call_arg
            assert "global_stats" in call_arg
            assert "models" in call_arg

    def test_run_weekly_report_handles_exception_gracefully(self):
        """Exception DB → loguée, pas de raise."""
        from src.tasks.supervision_reporter import run_weekly_report

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "WEEKLY_REPORT_ENABLED", True),
            patch(
                "src.db.database.AsyncSessionLocal",
                side_effect=Exception("DB error"),
            ),
        ):
            asyncio.run(run_weekly_report())


# ---------------------------------------------------------------------------
# start_scheduler() / stop_scheduler()
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    def test_start_scheduler_adds_alert_check_job(self):
        """ENABLE_EMAIL_ALERTS=True → add_job avec id='alert_check' est appelé."""
        from src.tasks.supervision_reporter import start_scheduler

        mock_sched = MagicMock()
        mock_sched.running = False

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "ENABLE_EMAIL_ALERTS", True),
            patch("src.core.config.settings.WEEKLY_REPORT_ENABLED", False),
            patch("src.tasks.supervision_reporter._scheduler", mock_sched),
            patch("src.tasks.supervision_reporter._APSCHEDULER_AVAILABLE", True),
        ):
            start_scheduler()
            mock_sched.add_job.assert_called()
            job_ids = [call.kwargs.get("id") for call in mock_sched.add_job.call_args_list]
            assert "alert_check" in job_ids
            mock_sched.start.assert_called_once()

    def test_start_scheduler_adds_weekly_report_job(self):
        """WEEKLY_REPORT_ENABLED=True → add_job avec id='weekly_report' est appelé."""
        from src.tasks.supervision_reporter import start_scheduler

        mock_sched = MagicMock()

        with (
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "ENABLE_EMAIL_ALERTS", False),
            patch.object(__import__("src.core.config", fromlist=["settings"]).settings, "WEEKLY_REPORT_ENABLED", True),
            patch("src.tasks.supervision_reporter._scheduler", mock_sched),
            patch("src.tasks.supervision_reporter._APSCHEDULER_AVAILABLE", True),
        ):
            start_scheduler()
            job_ids = [call.kwargs.get("id") for call in mock_sched.add_job.call_args_list]
            assert "weekly_report" in job_ids

    def test_stop_scheduler_calls_shutdown_when_running(self):
        """_scheduler.running=True → shutdown(wait=False) est appelé."""
        from src.tasks.supervision_reporter import stop_scheduler

        mock_sched = MagicMock()
        mock_sched.running = True

        with patch("src.tasks.supervision_reporter._scheduler", mock_sched):
            stop_scheduler()
            mock_sched.shutdown.assert_called_once_with(wait=False)

    def test_stop_scheduler_noop_when_not_running(self):
        """_scheduler.running=False → shutdown n'est pas appelé."""
        from src.tasks.supervision_reporter import stop_scheduler

        mock_sched = MagicMock()
        mock_sched.running = False

        with patch("src.tasks.supervision_reporter._scheduler", mock_sched):
            stop_scheduler()
            mock_sched.shutdown.assert_not_called()
