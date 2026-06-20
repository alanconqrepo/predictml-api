"""
SMTP email sending service for monitoring alerts.

Uses smtplib (Python stdlib) — no external dependency required.
If SMTP_HOST is not configured, all methods are silent no-ops.
"""

import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import structlog

from src.core.config import settings

logger = structlog.get_logger(__name__)


class EmailService:
    """Transactional email sending for ML model monitoring."""

    def _is_configured(self) -> bool:
        return bool(settings.SMTP_HOST and settings.ALERT_EMAIL_TO)

    def _send_email(self, to: list[str], subject: str, html_body: str) -> bool:
        """
        Send an HTML email via SMTP.
        Returns True if sending succeeded, False otherwise (error logged, not raised).
        """
        if not self._is_configured():
            logger.debug("SMTP not configured, email ignored", subject=subject)
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.SMTP_FROM or settings.SMTP_USER
        msg["To"] = ", ".join(to)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            if settings.SMTP_STARTTLS:
                ctx = ssl.create_default_context()
                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
                    server.ehlo()
                    server.starttls(context=ctx)
                    if settings.SMTP_USER:
                        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.sendmail(msg["From"], to, msg.as_string())
            else:
                with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
                    if settings.SMTP_USER:
                        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.sendmail(msg["From"], to, msg.as_string())

            logger.info("Email sent", subject=subject, recipients=to)
            return True

        except Exception as exc:
            logger.warning("Email send failed", subject=subject, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # HTML templates (inline, no template engine dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _base_html(title: str, body_content: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
    .card {{ background: white; border-radius: 8px; padding: 24px; max-width: 640px;
             margin: 0 auto; box-shadow: 0 2px 8px rgba(0,0,0,.1); }}
    h1 {{ color: #1a1a2e; font-size: 22px; margin-top: 0; }}
    h2 {{ color: #444; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th {{ background: #f0f0f0; text-align: left; padding: 8px 10px; font-size: 13px; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #f0f0f0; font-size: 13px; }}
    .ok    {{ color: #27ae60; font-weight: bold; }}
    .warn  {{ color: #e67e22; font-weight: bold; }}
    .crit  {{ color: #c0392b; font-weight: bold; }}
    .btn   {{ display: inline-block; margin-top: 16px; padding: 10px 22px;
              background: #2980b9; color: white; text-decoration: none;
              border-radius: 5px; font-size: 14px; }}
    .footer {{ color: #aaa; font-size: 11px; text-align: center; margin-top: 20px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>🤖 PredictML — {title}</h1>
    {body_content}
    <a class="btn" href="{settings.STREAMLIT_URL}">Open dashboard</a>
    <p class="footer">
      Generated on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M')} UTC •
      PredictML Monitoring
    </p>
  </div>
</body>
</html>"""

    def _status_badge(self, status: str) -> str:
        cls = {"ok": "ok", "warning": "warn", "critical": "crit"}.get(status, "")
        emoji = {"ok": "🟢", "warning": "🟡", "critical": "🔴"}.get(status, "⚪")
        return f'<span class="{cls}">{emoji} {status}</span>'

    # ------------------------------------------------------------------
    # Send methods
    # ------------------------------------------------------------------

    def send_weekly_report(self, overview: dict) -> bool:
        """
        Weekly report: health summary of all models.
        `overview` is the serialized response of GET /monitoring/overview.
        """
        gs = overview.get("global_stats", {})
        period = overview.get("period", {})
        models = overview.get("models", [])

        start = period.get("start", "")[:10]
        end = period.get("end", "")[:10]

        rows = ""
        for m in models:
            status = m.get("health_status", "")
            rows += f"""
            <tr>
              <td><strong>{m['model_name']}</strong></td>
              <td>{m['total_predictions']:,}</td>
              <td>{m['error_rate'] * 100:.1f} %</td>
              <td>{m.get('avg_latency_ms') or '—'}</td>
              <td>{self._status_badge(m.get('feature_drift_status',''))}</td>
              <td>{self._status_badge(m.get('performance_drift_status',''))}</td>
              <td>{self._status_badge(status)}</td>
            </tr>"""

        body = f"""
        <p>Here is the weekly monitoring report for your ML models
           for the period from <strong>{start}</strong> to <strong>{end}</strong>.</p>
        <h2>📊 Global metrics</h2>
        <table>
          <tr><th>Predictions</th><td>{gs.get('total_predictions', 0):,}</td></tr>
          <tr><th>Error rate</th>
              <td>{gs.get('error_rate', 0) * 100:.1f} %</td></tr>
          <tr><th>Average latency</th>
              <td>{gs.get('avg_latency_ms') or '—'} ms</td></tr>
          <tr><th>Models in alert</th>
              <td>🔴 {gs.get('models_critical', 0)} critical •
                  🟡 {gs.get('models_warning', 0)} warning(s)</td></tr>
        </table>
        <h2>🏥 Health by model</h2>
        <table>
          <tr>
            <th>Model</th><th>Predictions</th><th>Errors</th>
            <th>Avg latency</th><th>Feature drift</th>
            <th>Perf. drift</th><th>Status</th>
          </tr>
          {rows}
        </table>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] Weekly report — {start} to {end}",
            html_body=self._base_html("Weekly report", body),
        )

    def send_drift_alert(
        self,
        model_name: str,
        feature: str,
        drift_status: str,
        z_score: float | None = None,
        psi: float | None = None,
    ) -> bool:
        """Feature drift detected alert."""
        z_txt = f"{z_score:.3f}" if z_score is not None else "N/A"
        psi_txt = f"{psi:.4f}" if psi is not None else "N/A"
        body = f"""
        <p>Data drift has been detected on model
           <strong>{model_name}</strong> — feature <strong>{feature}</strong>.</p>
        <table>
          <tr><th>Status</th><td>{self._status_badge(drift_status)}</td></tr>
          <tr><th>Z-score</th><td>{z_txt} (warning threshold ≥ 2, critical ≥ 3)</td></tr>
          <tr><th>PSI</th><td>{psi_txt} (warning threshold ≥ 0.1, critical ≥ 0.2)</td></tr>
          <tr><th>Detected on</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>Check whether the input data distribution has changed.
           Model retraining may be necessary.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] ⚠️ Feature drift — {model_name} / {feature}",
            html_body=self._base_html("Feature drift alert", body),
        )

    def send_performance_alert(
        self,
        model_name: str,
        current_accuracy: float,
        baseline_accuracy: float,
    ) -> bool:
        """Performance degradation alert (accuracy/MAE)."""
        drop = baseline_accuracy - current_accuracy
        body = f"""
        <p>A performance degradation has been detected on model
           <strong>{model_name}</strong>.</p>
        <table>
          <tr><th>Baseline accuracy</th><td>{baseline_accuracy:.1%}</td></tr>
          <tr><th>Recent accuracy</th><td>{current_accuracy:.1%}</td></tr>
          <tr><th>Drop</th><td><strong>−{drop:.1%}</strong></td></tr>
          <tr><th>Detected on</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>Check recent observed results and consider retraining.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] 🔴 Performance drift — {model_name}",
            html_body=self._base_html("Performance drift alert", body),
        )

    def send_error_spike_alert(
        self, model_name: str, error_rate: float, threshold: float | None = None
    ) -> bool:
        """Error spike alert."""
        effective_threshold = threshold if threshold is not None else settings.ERROR_RATE_ALERT_THRESHOLD
        body = f"""
        <p>An error spike has been detected on model
           <strong>{model_name}</strong>.</p>
        <table>
          <tr><th>Current error rate</th>
              <td><strong>{error_rate:.1%}</strong></td></tr>
          <tr><th>Configured threshold</th><td>{effective_threshold:.1%}</td></tr>
          <tr><th>Detected on</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>Check the API logs and the state of the production model.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] 🔴 Error spike — {model_name} ({error_rate:.1%})",
            html_body=self._base_html("Error spike alert", body),
        )

    def send_auc_alert(
        self, model_name: str, current_auc: float, auc_min: float
    ) -> bool:
        """AUC below minimum threshold alert (binary classifiers only)."""
        gap = auc_min - current_auc
        body = f"""
        <p>The AUC of binary classifier <strong>{model_name}</strong>
           has fallen below the configured minimum threshold.</p>
        <table>
          <tr><th>Current AUC</th><td><strong>{current_auc:.4f}</strong></td></tr>
          <tr><th>Minimum required</th><td>{auc_min:.4f}</td></tr>
          <tr><th>Gap</th><td><strong>−{gap:.4f}</strong></td></tr>
          <tr><th>Detected on</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>The AUC shown is the value recorded at the last training or retrain.
           Consider retraining with more recent data or reviewing the feature pipeline.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] 🔴 AUC below threshold — {model_name} ({current_auc:.4f} < {auc_min:.4f})",
            html_body=self._base_html("AUC below minimum threshold", body),
        )

    def send_auto_promotion_alert(
        self,
        model_name: str,
        new_version: str,
        reason: str,
        accuracy: float | None = None,
        f1_score: float | None = None,
    ) -> bool:
        """Auto-promotion success notification."""
        metrics_rows = ""
        if accuracy is not None:
            metrics_rows += f"<tr><th>Accuracy</th><td><strong>{accuracy:.4f}</strong></td></tr>"
        if f1_score is not None:
            metrics_rows += f"<tr><th>F1 score</th><td><strong>{f1_score:.4f}</strong></td></tr>"
        body = f"""
        <p>The model <strong>{model_name}</strong> v<strong>{new_version}</strong>
           has been <strong>automatically promoted to production</strong>
           following a successful retrain — all quality criteria were satisfied.</p>
        <table>
          <tr><th>Model</th><td>{model_name}</td></tr>
          <tr><th>New version</th><td>{new_version}</td></tr>
          {metrics_rows}
          <tr><th>Criteria result</th><td>{reason}</td></tr>
          <tr><th>Promoted on</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>The new version is now serving production traffic.
           Open the dashboard to verify its performance metrics and configure
           alerts if needed.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] ✅ Auto-promotion — {model_name} v{new_version}",
            html_body=self._base_html("Auto-promotion — New version in production", body),
        )

    def send_auto_demotion_alert(
        self,
        model_name: str,
        version: str,
        reason: str,
        no_fallback: bool = False,
    ) -> bool:
        """Circuit breaker auto-demotion alert."""
        fallback_warning = (
            """<p style="color:#c0392b;font-weight:bold">
            ⚠️ No fallback version available — the model has NOT been demoted.
            Manual action required.</p>"""
            if no_fallback
            else ""
        )
        action_txt = (
            "removed from production" if not no_fallback else "kept in production (no fallback)"
        )
        body = f"""
        <p>The circuit breaker detected a critical degradation on model
           <strong>{model_name}</strong> v<strong>{version}</strong>
           and has <strong>{action_txt}</strong> it automatically.</p>
        {fallback_warning}
        <table>
          <tr><th>Model</th><td>{model_name}</td></tr>
          <tr><th>Version</th><td>{version}</td></tr>
          <tr><th>Reason</th><td>{reason}</td></tr>
          <tr><th>Detected on</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>Check the monitoring dashboard and promote a healthy version
           if necessary.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] 🔴 Auto-demotion — {model_name} v{version}",
            html_body=self._base_html("Circuit breaker — Auto-demotion", body),
        )


# Global instance
email_service = EmailService()
