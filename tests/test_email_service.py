"""
Tests unitaires pour src/services/email_service.py

Couvre :
- EmailService._is_configured()
- EmailService._send_email() : STARTTLS, SSL, non configuré, erreur SMTP
- EmailService.send_weekly_report()
- EmailService.send_drift_alert()
- EmailService.send_performance_alert()
- EmailService.send_error_spike_alert()
"""

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from src.services.email_service import EmailService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_configured_service(starttls: bool = True) -> EmailService:
    """Retourne une instance EmailService avec SMTP configuré."""
    svc = EmailService()
    with (
        patch.object(type(svc).__mro__[0], "_is_configured", return_value=True),
    ):
        pass
    return svc


def _patch_settings(smtp_host: str = "smtp.example.com", alert_to: list = None):
    """Patch les settings SMTP pour les tests."""
    if alert_to is None:
        alert_to = ["admin@example.com"]
    return patch.multiple(
        "src.services.email_service.settings",
        SMTP_HOST=smtp_host,
        SMTP_PORT=587,
        SMTP_USER="user@example.com",
        SMTP_PASSWORD="secret",
        SMTP_FROM="noreply@example.com",
        SMTP_STARTTLS=True,
        ALERT_EMAIL_TO=alert_to,
        STREAMLIT_URL="http://localhost:8501",
        ERROR_RATE_ALERT_THRESHOLD=0.10,
    )


# ---------------------------------------------------------------------------
# _is_configured()
# ---------------------------------------------------------------------------


class TestEmailServiceIsConfigured:
    def test_not_configured_when_smtp_host_empty(self):
        """SMTP_HOST vide → _is_configured() retourne False."""
        svc = EmailService()
        with patch.multiple(
            "src.services.email_service.settings",
            SMTP_HOST="",
            ALERT_EMAIL_TO=["admin@example.com"],
        ):
            assert svc._is_configured() is False

    def test_not_configured_when_alert_email_empty(self):
        """ALERT_EMAIL_TO vide → _is_configured() retourne False."""
        svc = EmailService()
        with patch.multiple(
            "src.services.email_service.settings",
            SMTP_HOST="smtp.example.com",
            ALERT_EMAIL_TO=[],
        ):
            assert svc._is_configured() is False

    def test_configured_when_both_set(self):
        """SMTP_HOST et ALERT_EMAIL_TO renseignés → _is_configured() retourne True."""
        svc = EmailService()
        with patch.multiple(
            "src.services.email_service.settings",
            SMTP_HOST="smtp.example.com",
            ALERT_EMAIL_TO=["admin@example.com"],
        ):
            assert svc._is_configured() is True


# ---------------------------------------------------------------------------
# _send_email() — non configuré
# ---------------------------------------------------------------------------


class TestSendEmailUnconfigured:
    def test_send_email_returns_false_when_not_configured(self):
        """_send_email() retourne False quand SMTP non configuré (pas d'exception)."""
        svc = EmailService()
        with patch.multiple(
            "src.services.email_service.settings",
            SMTP_HOST="",
            ALERT_EMAIL_TO=[],
        ):
            result = svc._send_email(
                to=["recipient@example.com"],
                subject="Test",
                html_body="<p>Test</p>",
            )
        assert result is False


# ---------------------------------------------------------------------------
# _send_email() — STARTTLS
# ---------------------------------------------------------------------------


class TestSendEmailSTARTTLS:
    def test_send_email_starttls_success_returns_true(self):
        """_send_email() avec STARTTLS=True retourne True si SMTP réussit."""
        svc = EmailService()
        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)

        with _patch_settings(smtp_host="smtp.example.com"):
            with patch("smtplib.SMTP", return_value=mock_server):
                result = svc._send_email(
                    to=["admin@example.com"],
                    subject="Test STARTTLS",
                    html_body="<p>Hello</p>",
                )
        assert result is True

    def test_send_email_starttls_smtp_error_returns_false_no_raise(self):
        """_send_email() retourne False si smtplib.SMTP lève une exception (pas de raise)."""
        svc = EmailService()

        with _patch_settings(smtp_host="smtp.example.com"):
            with patch("smtplib.SMTP", side_effect=smtplib.SMTPException("Connection failed")):
                result = svc._send_email(
                    to=["admin@example.com"],
                    subject="Test error",
                    html_body="<p>Error test</p>",
                )
        assert result is False


# ---------------------------------------------------------------------------
# _send_email() — SSL
# ---------------------------------------------------------------------------


class TestSendEmailSSL:
    def test_send_email_ssl_success_returns_true(self):
        """_send_email() avec SMTP_STARTTLS=False (SSL) retourne True si SMTP réussit."""
        svc = EmailService()
        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)

        with patch.multiple(
            "src.services.email_service.settings",
            SMTP_HOST="smtp.example.com",
            SMTP_PORT=465,
            SMTP_USER="user@example.com",
            SMTP_PASSWORD="secret",
            SMTP_FROM="noreply@example.com",
            SMTP_STARTTLS=False,
            ALERT_EMAIL_TO=["admin@example.com"],
            STREAMLIT_URL="http://localhost:8501",
            ERROR_RATE_ALERT_THRESHOLD=0.10,
        ):
            with patch("smtplib.SMTP_SSL", return_value=mock_server):
                result = svc._send_email(
                    to=["admin@example.com"],
                    subject="Test SSL",
                    html_body="<p>SSL test</p>",
                )
        assert result is True


# ---------------------------------------------------------------------------
# Méthodes d'envoi
# ---------------------------------------------------------------------------


class TestEmailServiceAlerts:
    def _make_svc_with_mock_send(self):
        """Retourne (EmailService, mock de _send_email)."""
        svc = EmailService()
        mock_send = MagicMock(return_value=True)
        svc._send_email = mock_send
        return svc, mock_send

    def test_send_weekly_report_calls_send_email(self):
        """send_weekly_report() appelle _send_email avec un sujet cohérent."""
        svc, mock_send = self._make_svc_with_mock_send()
        overview = {
            "period": {"start": "2025-01-01T00:00:00", "end": "2025-01-08T00:00:00"},
            "global_stats": {
                "total_predictions": 100,
                "error_rate": 0.02,
                "avg_latency_ms": 50,
                "models_critical": 0,
                "models_warning": 1,
            },
            "models": [
                {
                    "model_name": "model_a",
                    "total_predictions": 100,
                    "error_rate": 0.02,
                    "avg_latency_ms": 50,
                    "feature_drift_status": "ok",
                    "performance_drift_status": "ok",
                    "health_status": "ok",
                }
            ],
        }
        with _patch_settings():
            result = svc.send_weekly_report(overview)
        assert result is True
        mock_send.assert_called_once()
        subject = mock_send.call_args.kwargs.get("subject") or mock_send.call_args.args[1]
        assert "rapport" in subject.lower() or "weekly" in subject.lower() or "2025" in subject

    def test_send_drift_alert_includes_model_name(self):
        """send_drift_alert() inclut le nom du modèle dans le sujet."""
        svc, mock_send = self._make_svc_with_mock_send()
        with _patch_settings():
            svc.send_drift_alert(
                model_name="iris_classifier",
                feature="petal_length",
                drift_status="warning",
                z_score=2.5,
                psi=0.12,
            )
        mock_send.assert_called_once()
        subject = mock_send.call_args.kwargs.get("subject") or mock_send.call_args.args[1]
        assert "iris_classifier" in subject

    def test_send_drift_alert_includes_z_score_psi(self):
        """send_drift_alert() inclut z_score et psi dans le body HTML."""
        svc, mock_send = self._make_svc_with_mock_send()
        with _patch_settings():
            svc.send_drift_alert(
                model_name="model_x",
                feature="feature_1",
                drift_status="critical",
                z_score=3.7,
                psi=0.25,
            )
        html_body = mock_send.call_args.kwargs.get("html_body") or mock_send.call_args.args[2]
        assert "3.700" in html_body
        assert "0.2500" in html_body

    def test_send_drift_alert_handles_none_z_score_psi(self):
        """send_drift_alert() fonctionne avec z_score=None et psi=None."""
        svc, mock_send = self._make_svc_with_mock_send()
        with _patch_settings():
            svc.send_drift_alert(
                model_name="model_y",
                feature="feature_2",
                drift_status="ok",
                z_score=None,
                psi=None,
            )
        html_body = mock_send.call_args.kwargs.get("html_body") or mock_send.call_args.args[2]
        assert "N/A" in html_body

    def test_send_performance_alert_includes_accuracy_values(self):
        """send_performance_alert() inclut les valeurs d'accuracy baseline et courante."""
        svc, mock_send = self._make_svc_with_mock_send()
        with _patch_settings():
            svc.send_performance_alert(
                model_name="classifier_v2",
                current_accuracy=0.78,
                baseline_accuracy=0.92,
            )
        mock_send.assert_called_once()
        html_body = mock_send.call_args.kwargs.get("html_body") or mock_send.call_args.args[2]
        assert "92.0%" in html_body or "92%" in html_body
        assert "78.0%" in html_body or "78%" in html_body

    def test_send_error_spike_alert_includes_error_rate(self):
        """send_error_spike_alert() inclut le taux d'erreur dans le body."""
        svc, mock_send = self._make_svc_with_mock_send()
        with _patch_settings():
            svc.send_error_spike_alert(model_name="prod_model", error_rate=0.35)
        mock_send.assert_called_once()
        html_body = mock_send.call_args.kwargs.get("html_body") or mock_send.call_args.args[2]
        assert "35.0%" in html_body or "35%" in html_body

    def test_all_send_methods_return_false_when_unconfigured(self):
        """Toutes les méthodes send_*() retournent False si SMTP non configuré."""
        svc = EmailService()
        with patch.multiple(
            "src.services.email_service.settings",
            SMTP_HOST="",
            ALERT_EMAIL_TO=[],
            STREAMLIT_URL="http://localhost:8501",
            ERROR_RATE_ALERT_THRESHOLD=0.10,
        ):
            overview = {
                "period": {"start": "2025-01-01T00:00:00", "end": "2025-01-08T00:00:00"},
                "global_stats": {},
                "models": [],
            }
            assert svc.send_weekly_report(overview) is False
            assert (
                svc.send_drift_alert("m", "f", "ok", z_score=1.0, psi=0.05) is False
            )
            assert svc.send_performance_alert("m", 0.8, 0.9) is False
            assert svc.send_error_spike_alert("m", 0.15) is False
