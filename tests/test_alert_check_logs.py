"""
Tests for:
- DBService.create_alert_check_log / get_alert_check_logs / get_last_alert_check_at / count_predictions_since
- GET /monitoring/alert-checks endpoint (auth, filters, pagination)
- supervision_reporter: email suppressed when no new predictions since last check
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "alert-check-admin-token-xx99"
USER_TOKEN = "alert-check-user-token-xx88"


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="alert_check_admin",
                email="alert_check_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="alert_check_user",
                email="alert_check_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# DBService — unit tests
# ---------------------------------------------------------------------------


class TestAlertCheckLogDBService:
    def test_create_and_retrieve_log(self):
        async def _run():
            async with _TestSessionLocal() as db:
                log = await DBService.create_alert_check_log(
                    db,
                    check_type="error_spike",
                    model_name="iris",
                    model_version="1.0.0",
                    result="alert_triggered",
                    alert_sent=True,
                    webhook_sent=False,
                    new_predictions_count=150,
                    details={"error_rate": 0.08, "threshold": 0.05},
                )
                assert log.id is not None
                assert log.check_type == "error_spike"
                assert log.model_name == "iris"
                assert log.result == "alert_triggered"
                assert log.alert_sent is True
                assert log.new_predictions_count == 150
                assert log.details["error_rate"] == 0.08

        asyncio.run(_run())

    def test_get_alert_check_logs_no_filter(self):
        async def _run():
            async with _TestSessionLocal() as db:
                await DBService.create_alert_check_log(
                    db,
                    check_type="auc",
                    model_name="wine",
                    result="ok",
                )
                rows, total = await DBService.get_alert_check_logs(db, limit=100)
                assert total >= 1
                assert len(rows) >= 1

        asyncio.run(_run())

    def test_get_alert_check_logs_filter_model(self):
        async def _run():
            async with _TestSessionLocal() as db:
                await DBService.create_alert_check_log(
                    db,
                    check_type="feature_drift",
                    model_name="unique_model_xyz",
                    result="alert_triggered",
                )
                rows, total = await DBService.get_alert_check_logs(
                    db, model_name="unique_model_xyz"
                )
                assert total >= 1
                assert all(r.model_name == "unique_model_xyz" for r in rows)

        asyncio.run(_run())

    def test_get_alert_check_logs_filter_check_type(self):
        async def _run():
            async with _TestSessionLocal() as db:
                await DBService.create_alert_check_log(
                    db,
                    check_type="output_drift",
                    model_name="test_model_type_filter",
                    result="ok",
                )
                rows, total = await DBService.get_alert_check_logs(
                    db,
                    model_name="test_model_type_filter",
                    check_type="output_drift",
                )
                assert total >= 1
                assert all(r.check_type == "output_drift" for r in rows)

        asyncio.run(_run())

    def test_get_last_alert_check_at_no_logs(self):
        async def _run():
            async with _TestSessionLocal() as db:
                last = await DBService.get_last_alert_check_at(db, "nonexistent_model_zzz")
                assert last is None

        asyncio.run(_run())

    def test_get_last_alert_check_at_returns_max(self):
        async def _run():
            async with _TestSessionLocal() as db:
                model = "last_check_model_test"
                await DBService.create_alert_check_log(
                    db, check_type="error_spike", model_name=model, result="ok"
                )
                await DBService.create_alert_check_log(
                    db, check_type="auc", model_name=model, result="ok"
                )
                last = await DBService.get_last_alert_check_at(db, model)
                assert isinstance(last, datetime)

        asyncio.run(_run())

    def test_count_predictions_since_no_predictions(self):
        async def _run():
            async with _TestSessionLocal() as db:
                since = datetime.utcnow() - timedelta(hours=1)
                count = await DBService.count_predictions_since(
                    db, "model_never_predicted", since
                )
                assert count == 0

        asyncio.run(_run())

    def test_get_alert_check_logs_pagination(self):
        async def _run():
            async with _TestSessionLocal() as db:
                model = "pagination_model_test"
                for i in range(5):
                    await DBService.create_alert_check_log(
                        db, check_type="error_spike", model_name=model, result="ok"
                    )
                rows, total = await DBService.get_alert_check_logs(
                    db, model_name=model, limit=3, offset=0
                )
                assert total >= 5
                assert len(rows) == 3

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# API endpoint — GET /monitoring/alert-checks
# ---------------------------------------------------------------------------


class TestAlertChecksEndpoint:
    def test_requires_auth(self):
        r = client.get("/monitoring/alert-checks")
        assert r.status_code == 401

    def test_requires_admin(self):
        r = client.get(
            "/monitoring/alert-checks",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert r.status_code == 403

    def test_returns_list(self):
        r = client.get(
            "/monitoring/alert-checks",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_filter_by_model(self):
        async def _insert():
            async with _TestSessionLocal() as db:
                await DBService.create_alert_check_log(
                    db,
                    check_type="error_spike",
                    model_name="endpoint_filter_model",
                    result="alert_triggered",
                    alert_sent=True,
                )

        asyncio.run(_insert())

        r = client.get(
            "/monitoring/alert-checks",
            params={"model_name": "endpoint_filter_model"},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        assert all(item["model_name"] == "endpoint_filter_model" for item in data["items"])

    def test_filter_by_check_type(self):
        r = client.get(
            "/monitoring/alert-checks",
            params={"check_type": "performance_drift"},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert all(item["check_type"] == "performance_drift" for item in data["items"])

    def test_response_schema_fields(self):
        async def _insert():
            async with _TestSessionLocal() as db:
                await DBService.create_alert_check_log(
                    db,
                    check_type="auc",
                    model_name="schema_check_model",
                    model_version="2.0.0",
                    result="ok",
                    alert_sent=False,
                    webhook_sent=True,
                    new_predictions_count=42,
                    details={"auc": 0.85, "auc_min": 0.90},
                )

        asyncio.run(_insert())

        r = client.get(
            "/monitoring/alert-checks",
            params={"model_name": "schema_check_model"},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        item = r.json()["items"][0]
        assert "checked_at" in item
        assert "check_type" in item
        assert "model_name" in item
        assert "result" in item
        assert "alert_sent" in item
        assert "webhook_sent" in item
        assert "new_predictions_count" in item
        assert "details" in item


# ---------------------------------------------------------------------------
# supervision_reporter — guard "no new predictions"
# ---------------------------------------------------------------------------


class TestSupervisionReporterGuard:
    """
    Tests that emails are suppressed when no new predictions exist since last check,
    but that alert_check_logs are still written.
    """

    def _make_mock_stat(self, model_name: str, error_rate: float = 0.20):
        return {
            "model_name": model_name,
            "error_rate": error_rate,
            "total_predictions": 100,
            "error_count": int(error_rate * 100),
            "shadow_predictions": 0,
            "avg_latency_ms": 50.0,
        }

    def test_email_suppressed_when_no_new_predictions(self):
        """When count_predictions_since returns 0, email must NOT be sent."""
        from src.tasks.supervision_reporter import run_alert_check

        send_mock = MagicMock(return_value=True)
        with (
            patch("src.tasks.supervision_reporter.settings") as mock_settings,
            patch("src.db.database.AsyncSessionLocal") as mock_session_factory,
            patch("src.services.db_service.DBService.get_global_monitoring_stats", new_callable=AsyncMock) as mock_stats,
            patch("src.services.db_service.DBService.get_all_active_models", new_callable=AsyncMock) as mock_metas,
            patch("src.services.db_service.DBService.get_last_alert_check_at", new_callable=AsyncMock) as mock_last,
            patch("src.services.db_service.DBService.count_predictions_since", new_callable=AsyncMock) as mock_count,
            patch("src.services.db_service.DBService.create_alert_check_log", new_callable=AsyncMock) as mock_log,
            patch("src.services.db_service.DBService.get_accuracy_drift", new_callable=AsyncMock) as mock_drift,
            patch("src.services.email_service.email_service") as mock_email,
            patch("src.core.ml_metrics.drift_detected_total") as mock_metrics,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = True
            mock_settings.ERROR_RATE_ALERT_THRESHOLD = 0.05
            mock_settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD = 0.10

            # Simulate error rate above threshold
            mock_stats.return_value = [self._make_mock_stat("guard_model", error_rate=0.20)]
            mock_metas.return_value = []

            # last_check_at is set → model has been checked before
            mock_last.return_value = datetime.utcnow() - timedelta(hours=6)
            # No new predictions since last check
            mock_count.return_value = 0

            mock_drift.return_value = []
            mock_log.return_value = MagicMock()

            # Mock session context manager
            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_db

            mock_email.send_error_spike_alert = send_mock
            mock_metrics.labels.return_value.inc = MagicMock()

            asyncio.run(run_alert_check())

            # Email must NOT have been sent
            send_mock.assert_not_called()

            # But the check log must still be written
            assert mock_log.called

    def test_email_sent_on_first_check(self):
        """When last_check_at is None (first check), email MUST be sent if threshold exceeded."""
        from src.tasks.supervision_reporter import run_alert_check

        send_mock = MagicMock(return_value=True)
        with (
            patch("src.tasks.supervision_reporter.settings") as mock_settings,
            patch("src.db.database.AsyncSessionLocal") as mock_session_factory,
            patch("src.services.db_service.DBService.get_global_monitoring_stats", new_callable=AsyncMock) as mock_stats,
            patch("src.services.db_service.DBService.get_all_active_models", new_callable=AsyncMock) as mock_metas,
            patch("src.services.db_service.DBService.get_last_alert_check_at", new_callable=AsyncMock) as mock_last,
            patch("src.services.db_service.DBService.count_predictions_since", new_callable=AsyncMock) as mock_count,
            patch("src.services.db_service.DBService.create_alert_check_log", new_callable=AsyncMock) as mock_log,
            patch("src.services.db_service.DBService.get_accuracy_drift", new_callable=AsyncMock) as mock_drift,
            patch("src.services.email_service.email_service") as mock_email,
            patch("src.core.ml_metrics.drift_detected_total") as mock_metrics,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = True
            mock_settings.ERROR_RATE_ALERT_THRESHOLD = 0.05
            mock_settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD = 0.10

            mock_stats.return_value = [self._make_mock_stat("first_check_model", error_rate=0.20)]
            mock_metas.return_value = []

            # No previous check → first run
            mock_last.return_value = None
            mock_count.return_value = 0  # not called when last_check_at is None

            mock_drift.return_value = []
            mock_log.return_value = MagicMock()

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_db

            mock_email.send_error_spike_alert = send_mock
            mock_metrics.labels.return_value.inc = MagicMock()

            asyncio.run(run_alert_check())

            # Email MUST have been sent on first check
            send_mock.assert_called_once()

    def test_email_sent_when_new_predictions_exist(self):
        """When new predictions exist since last check, email IS sent if threshold exceeded."""
        from src.tasks.supervision_reporter import run_alert_check

        send_mock = MagicMock(return_value=True)
        with (
            patch("src.tasks.supervision_reporter.settings") as mock_settings,
            patch("src.db.database.AsyncSessionLocal") as mock_session_factory,
            patch("src.services.db_service.DBService.get_global_monitoring_stats", new_callable=AsyncMock) as mock_stats,
            patch("src.services.db_service.DBService.get_all_active_models", new_callable=AsyncMock) as mock_metas,
            patch("src.services.db_service.DBService.get_last_alert_check_at", new_callable=AsyncMock) as mock_last,
            patch("src.services.db_service.DBService.count_predictions_since", new_callable=AsyncMock) as mock_count,
            patch("src.services.db_service.DBService.create_alert_check_log", new_callable=AsyncMock) as mock_log,
            patch("src.services.db_service.DBService.get_accuracy_drift", new_callable=AsyncMock) as mock_drift,
            patch("src.services.email_service.email_service") as mock_email,
            patch("src.core.ml_metrics.drift_detected_total") as mock_metrics,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = True
            mock_settings.ERROR_RATE_ALERT_THRESHOLD = 0.05
            mock_settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD = 0.10

            mock_stats.return_value = [self._make_mock_stat("active_model", error_rate=0.20)]
            mock_metas.return_value = []

            mock_last.return_value = datetime.utcnow() - timedelta(hours=6)
            # New predictions exist
            mock_count.return_value = 75

            mock_drift.return_value = []
            mock_log.return_value = MagicMock()

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_db

            mock_email.send_error_spike_alert = send_mock
            mock_metrics.labels.return_value.inc = MagicMock()

            asyncio.run(run_alert_check())

            # Email MUST have been sent
            send_mock.assert_called_once()
