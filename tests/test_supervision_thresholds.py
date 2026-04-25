"""
Tests pour _get_model_threshold() et run_alert_check() avec seuils par modèle.

Séparé de test_alert_thresholds.py pour éviter que supervision_reporter soit
importé avant test_config.py (qui recharge src.core.config via importlib.reload,
rendant la référence settings de supervision_reporter périmée).
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
    name: str,
    version: str = "1.0.0",
    alert_thresholds: dict | None = None,
    feature_baseline: dict | None = None,
    is_production: bool = True,
):
    """Construit un MagicMock imitant un objet ModelMetadata."""
    m = MagicMock()
    m.name = name
    m.version = version
    m.is_production = is_production
    m.alert_thresholds = alert_thresholds
    m.feature_baseline = feature_baseline
    return m


# ---------------------------------------------------------------------------
# _get_model_threshold() helper
# ---------------------------------------------------------------------------


class TestGetModelThreshold:
    def test_returns_model_value_when_set(self):
        """Retourne la valeur du modèle quand elle est définie."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold({"error_rate_max": 0.03}, "error_rate_max", default=0.10)
        assert result == pytest.approx(0.03)

    def test_returns_default_when_key_absent(self):
        """Retourne le défaut quand la clé est absente du dict."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold({"accuracy_min": 0.90}, "error_rate_max", default=0.10)
        assert result == pytest.approx(0.10)

    def test_returns_default_when_thresholds_none(self):
        """Retourne le défaut quand thresholds est None."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold(None, "error_rate_max", default=0.10)
        assert result == pytest.approx(0.10)

    def test_returns_default_when_value_is_none(self):
        """Retourne le défaut quand la clé existe mais la valeur est None."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold({"error_rate_max": None}, "error_rate_max", default=0.10)
        assert result == pytest.approx(0.10)

    def test_zero_value_is_used_not_default(self):
        """La valeur 0.0 est utilisée, pas le défaut (0.0 est falsy en Python)."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold({"error_rate_max": 0.0}, "error_rate_max", default=0.10)
        assert result == pytest.approx(0.0)

    def test_returns_default_when_thresholds_empty(self):
        """Retourne le défaut quand thresholds est un dict vide."""
        from src.tasks.supervision_reporter import _get_model_threshold

        result = _get_model_threshold({}, "error_rate_max", default=0.10)
        assert result == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# run_alert_check() avec seuils par modèle
# ---------------------------------------------------------------------------


class TestRunAlertCheckWithPerModelThresholds:
    def test_per_model_error_rate_max_triggers_alert(self):
        """
        error_rate_max=0.03 sur le modèle → alerte si error_rate=0.04,
        même si le seuil global (0.10) ne déclencherait pas d'alerte.
        """
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "strict_model",
                "error_rate": 0.04,
                "total_predictions": 100,
                "error_count": 4,
                "shadow_predictions": 0,
                "avg_latency_ms": 30,
            }
        ]
        mock_meta = _make_mock_meta("strict_model", alert_thresholds={"error_rate_max": 0.03})

        with (
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ENABLE_EMAIL_ALERTS",
                True,
            ),
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ERROR_RATE_ALERT_THRESHOLD",
                0.10,
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
                return_value=[mock_meta],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            mock_email.send_error_spike_alert = MagicMock()
            asyncio.run(run_alert_check())
            mock_email.send_error_spike_alert.assert_called_once_with("strict_model", 0.04)

    def test_per_model_error_rate_max_suppresses_alert(self):
        """
        error_rate_max=0.20 sur le modèle → pas d'alerte si error_rate=0.15,
        même si le seuil global (0.10) déclencherait une alerte.
        """
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "lenient_model",
                "error_rate": 0.15,
                "total_predictions": 100,
                "error_count": 15,
                "shadow_predictions": 0,
                "avg_latency_ms": 30,
            }
        ]
        mock_meta = _make_mock_meta("lenient_model", alert_thresholds={"error_rate_max": 0.20})

        with (
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ENABLE_EMAIL_ALERTS",
                True,
            ),
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ERROR_RATE_ALERT_THRESHOLD",
                0.10,
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
                return_value=[mock_meta],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            asyncio.run(run_alert_check())
            mock_email.send_error_spike_alert.assert_not_called()

    def test_accuracy_min_absolute_triggers_alert(self):
        """
        accuracy_min=0.90, avg_second=0.85 (< 0.90) → alerte,
        même si la chute relative (0.05) serait sous PERFORMANCE_DRIFT_ALERT_THRESHOLD (0.10).
        """
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "abs_model",
                "error_rate": 0.01,
                "total_predictions": 100,
                "error_count": 1,
                "shadow_predictions": 0,
                "avg_latency_ms": 30,
            }
        ]
        perf_data = [
            {"date": "2026-04-21", "matched_count": 10, "accuracy": 0.90},
            {"date": "2026-04-22", "matched_count": 10, "accuracy": 0.85},
        ]
        mock_meta = _make_mock_meta("abs_model", alert_thresholds={"accuracy_min": 0.90})

        with (
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ENABLE_EMAIL_ALERTS",
                True,
            ),
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "PERFORMANCE_DRIFT_ALERT_THRESHOLD",
                0.10,
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
                return_value=perf_data,
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[mock_meta],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            mock_email.send_performance_alert = MagicMock()
            mock_email.send_error_spike_alert = MagicMock()
            asyncio.run(run_alert_check())
            mock_email.send_performance_alert.assert_called_once()

    def test_accuracy_min_absolute_no_alert_when_above(self):
        """
        accuracy_min=0.80, avg_second=0.85 (≥ 0.80) → pas d'alerte.
        """
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "ok_abs_model",
                "error_rate": 0.01,
                "total_predictions": 100,
                "error_count": 1,
                "shadow_predictions": 0,
                "avg_latency_ms": 30,
            }
        ]
        perf_data = [
            {"date": "2026-04-21", "matched_count": 10, "accuracy": 0.88},
            {"date": "2026-04-22", "matched_count": 10, "accuracy": 0.85},
        ]
        mock_meta = _make_mock_meta("ok_abs_model", alert_thresholds={"accuracy_min": 0.80})

        with (
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ENABLE_EMAIL_ALERTS",
                True,
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
                return_value=perf_data,
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[mock_meta],
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            mock_email.send_performance_alert = MagicMock()
            asyncio.run(run_alert_check())
            mock_email.send_performance_alert.assert_not_called()

    def test_drift_auto_alert_false_suppresses_drift_alert(self):
        """
        drift_auto_alert=False → send_drift_alert n'est pas appelé même si drift est critique.
        """
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "no_drift_model",
                "error_rate": 0.01,
                "total_predictions": 100,
                "error_count": 1,
                "shadow_predictions": 0,
                "avg_latency_ms": 30,
            }
        ]
        mock_meta = _make_mock_meta(
            "no_drift_model",
            alert_thresholds={"drift_auto_alert": False},
            feature_baseline={"f1": {"mean": 1.0, "std": 0.1, "min": 0.0, "max": 2.0}},
        )

        critical_feat = MagicMock()
        critical_feat.drift_status = "critical"
        critical_feat.z_score = 5.0
        critical_feat.psi = 0.3

        with (
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ENABLE_EMAIL_ALERTS",
                True,
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
                return_value=[mock_meta],
            ),
            patch(
                "src.services.db_service.DBService.get_feature_production_stats",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "src.services.drift_service.compute_feature_drift",
                return_value={"f1": critical_feat},
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            asyncio.run(run_alert_check())
            mock_email.send_drift_alert.assert_not_called()

    def test_drift_auto_alert_true_sends_drift_alert(self):
        """
        drift_auto_alert=True → send_drift_alert est appelé pour un drift critique.
        """
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": "drift_model",
                "error_rate": 0.01,
                "total_predictions": 100,
                "error_count": 1,
                "shadow_predictions": 0,
                "avg_latency_ms": 30,
            }
        ]
        mock_meta = _make_mock_meta(
            "drift_model",
            alert_thresholds={"drift_auto_alert": True},
            feature_baseline={"f1": {"mean": 1.0, "std": 0.1, "min": 0.0, "max": 2.0}},
        )

        critical_feat = MagicMock()
        critical_feat.drift_status = "critical"
        critical_feat.z_score = 5.0
        critical_feat.psi = 0.3

        with (
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ENABLE_EMAIL_ALERTS",
                True,
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
                return_value=[mock_meta],
            ),
            patch(
                "src.services.db_service.DBService.get_feature_production_stats",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "src.services.drift_service.compute_feature_drift",
                return_value={"f1": critical_feat},
            ),
            patch("src.services.email_service.email_service") as mock_email,
        ):
            mock_email.send_drift_alert = MagicMock()
            asyncio.run(run_alert_check())
            mock_email.send_drift_alert.assert_called_once_with(
                model_name="drift_model",
                feature="f1",
                drift_status="critical",
                z_score=5.0,
                psi=0.3,
            )

    def test_get_all_active_models_called_once_not_per_model(self):
        """
        get_all_active_models est appelé exactement 1 fois, peu importe le nombre de modèles.
        """
        from src.tasks.supervision_reporter import run_alert_check

        mock_stats = [
            {
                "model_name": f"m{i}",
                "error_rate": 0.01,
                "total_predictions": 10,
                "error_count": 0,
                "shadow_predictions": 0,
                "avg_latency_ms": 10,
            }
            for i in range(3)
        ]
        mock_get_all = AsyncMock(return_value=[])

        with (
            patch.object(
                __import__("src.core.config", fromlist=["settings"]).settings,
                "ENABLE_EMAIL_ALERTS",
                True,
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
            patch("src.services.db_service.DBService.get_all_active_models", mock_get_all),
            patch("src.services.email_service.email_service"),
        ):
            asyncio.run(run_alert_check())
            assert mock_get_all.call_count == 1
