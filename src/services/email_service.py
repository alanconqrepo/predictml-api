"""
Service d'envoi d'e-mails SMTP pour les alertes de supervision.

Utilise smtplib (stdlib Python) — aucune dépendance externe requise.
Si SMTP_HOST n'est pas configuré, toutes les méthodes sont des no-ops silencieux.
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
    """Envoi d'e-mails transactionnels pour la supervision des modèles ML."""

    def _is_configured(self) -> bool:
        return bool(settings.SMTP_HOST and settings.ALERT_EMAIL_TO)

    def _send_email(self, to: list[str], subject: str, html_body: str) -> bool:
        """
        Envoie un e-mail HTML via SMTP.
        Retourne True si l'envoi a réussi, False sinon (erreur loggée, pas propagée).
        """
        if not self._is_configured():
            logger.debug("SMTP non configuré, e-mail ignoré", subject=subject)
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

            logger.info("E-mail envoyé", subject=subject, recipients=to)
            return True

        except Exception as exc:
            logger.warning("Échec envoi e-mail", subject=subject, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Templates HTML (inline, sans dépendance moteur de template)
    # ------------------------------------------------------------------

    @staticmethod
    def _base_html(title: str, body_content: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="fr">
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
    <a class="btn" href="{settings.STREAMLIT_URL}">Ouvrir le tableau de bord</a>
    <p class="footer">
      Généré le {datetime.utcnow().strftime('%d/%m/%Y à %H:%M')} UTC •
      PredictML Supervision
    </p>
  </div>
</body>
</html>"""

    def _status_badge(self, status: str) -> str:
        cls = {"ok": "ok", "warning": "warn", "critical": "crit"}.get(status, "")
        emoji = {"ok": "🟢", "warning": "🟡", "critical": "🔴"}.get(status, "⚪")
        return f'<span class="{cls}">{emoji} {status}</span>'

    # ------------------------------------------------------------------
    # Méthodes d'envoi
    # ------------------------------------------------------------------

    def send_weekly_report(self, overview: dict) -> bool:
        """
        Rapport hebdomadaire : résumé de santé de tous les modèles.
        `overview` est la réponse sérialisée de GET /monitoring/overview.
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
        <p>Voici le rapport hebdomadaire de supervision de vos modèles ML
           pour la période du <strong>{start}</strong> au <strong>{end}</strong>.</p>
        <h2>📊 Métriques globales</h2>
        <table>
          <tr><th>Prédictions</th><td>{gs.get('total_predictions', 0):,}</td></tr>
          <tr><th>Taux d'erreur</th>
              <td>{gs.get('error_rate', 0) * 100:.1f} %</td></tr>
          <tr><th>Latence moyenne</th>
              <td>{gs.get('avg_latency_ms') or '—'} ms</td></tr>
          <tr><th>Modèles en alerte</th>
              <td>🔴 {gs.get('models_critical', 0)} critique(s) •
                  🟡 {gs.get('models_warning', 0)} avertissement(s)</td></tr>
        </table>
        <h2>🏥 Santé par modèle</h2>
        <table>
          <tr>
            <th>Modèle</th><th>Prédictions</th><th>Erreurs</th>
            <th>Latence moy.</th><th>Drift features</th>
            <th>Drift perf.</th><th>Statut</th>
          </tr>
          {rows}
        </table>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] Rapport hebdomadaire — {start} au {end}",
            html_body=self._base_html("Rapport hebdomadaire", body),
        )

    def send_drift_alert(
        self,
        model_name: str,
        feature: str,
        drift_status: str,
        z_score: float | None = None,
        psi: float | None = None,
    ) -> bool:
        """Alerte de drift de features détecté."""
        z_txt = f"{z_score:.3f}" if z_score is not None else "N/A"
        psi_txt = f"{psi:.4f}" if psi is not None else "N/A"
        body = f"""
        <p>Un drift de données a été détecté sur le modèle
           <strong>{model_name}</strong> — feature <strong>{feature}</strong>.</p>
        <table>
          <tr><th>Statut</th><td>{self._status_badge(drift_status)}</td></tr>
          <tr><th>Z-score</th><td>{z_txt} (seuil warning ≥ 2, critique ≥ 3)</td></tr>
          <tr><th>PSI</th><td>{psi_txt} (seuil warning ≥ 0.1, critique ≥ 0.2)</td></tr>
          <tr><th>Détecté le</th>
              <td>{datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC</td></tr>
        </table>
        <p>Vérifiez si les données en entrée ont changé de distribution.
           Un re-entraînement du modèle peut être nécessaire.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] ⚠️ Drift features — {model_name} / {feature}",
            html_body=self._base_html("Alerte drift features", body),
        )

    def send_performance_alert(
        self,
        model_name: str,
        current_accuracy: float,
        baseline_accuracy: float,
    ) -> bool:
        """Alerte de dégradation de performance (accuracy/MAE)."""
        drop = baseline_accuracy - current_accuracy
        body = f"""
        <p>Une dégradation de performance a été détectée sur le modèle
           <strong>{model_name}</strong>.</p>
        <table>
          <tr><th>Accuracy baseline</th><td>{baseline_accuracy:.1%}</td></tr>
          <tr><th>Accuracy récente</th><td>{current_accuracy:.1%}</td></tr>
          <tr><th>Baisse</th><td><strong>−{drop:.1%}</strong></td></tr>
          <tr><th>Détectée le</th>
              <td>{datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC</td></tr>
        </table>
        <p>Vérifiez les résultats observés récents et envisagez un re-entraînement.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] 🔴 Drift performance — {model_name}",
            html_body=self._base_html("Alerte drift performance", body),
        )

    def send_error_spike_alert(self, model_name: str, error_rate: float) -> bool:
        """Alerte de pic d'erreurs."""
        threshold = settings.ERROR_RATE_ALERT_THRESHOLD
        body = f"""
        <p>Un pic d'erreurs a été détecté sur le modèle
           <strong>{model_name}</strong>.</p>
        <table>
          <tr><th>Taux d'erreur actuel</th>
              <td><strong>{error_rate:.1%}</strong></td></tr>
          <tr><th>Seuil configuré</th><td>{threshold:.1%}</td></tr>
          <tr><th>Détecté le</th>
              <td>{datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC</td></tr>
        </table>
        <p>Vérifiez les logs de l'API et l'état du modèle en production.</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=f"[PredictML] 🔴 Pic d'erreurs — {model_name} ({error_rate:.1%})",
            html_body=self._base_html("Alerte pic d'erreurs", body),
        )


# Instance globale
email_service = EmailService()
