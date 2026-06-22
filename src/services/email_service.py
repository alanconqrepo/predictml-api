"""
SMTP email sending service for monitoring alerts.

Uses smtplib (Python stdlib) — no external dependency required.
If SMTP_HOST is not configured, all methods are silent no-ops.

Language is controlled by the DEFAULT_LANGUAGE environment variable:
  DEFAULT_LANGUAGE=EN  (default) — emails sent in English
  DEFAULT_LANGUAGE=FR            — emails sent in French
"""

import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import structlog

from src.core.config import settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Translation dictionary
# ---------------------------------------------------------------------------

_TRANSLATIONS: dict[str, dict[str, str]] = {
    # _base_html
    "btn_dashboard": {
        "EN": "Open dashboard",
        "FR": "Voir le tableau de bord",
    },
    "footer": {
        "EN": "Generated on {dt} UTC • PredictML Monitoring",
        "FR": "Généré le {dt} UTC • PredictML Supervision",
    },
    # Weekly report
    "weekly_subject": {
        "EN": "[PredictML] Weekly report — {start} to {end}",
        "FR": "[PredictML] Rapport hebdomadaire — {start} au {end}",
    },
    "weekly_intro": {
        "EN": (
            "Here is the weekly monitoring report for your ML models "
            "for the period from <strong>{start}</strong> to <strong>{end}</strong>."
        ),
        "FR": (
            "Voici le rapport hebdomadaire de supervision de vos modèles ML "
            "pour la période du <strong>{start}</strong> au <strong>{end}</strong>."
        ),
    },
    "weekly_global_title": {"EN": "📊 Global metrics", "FR": "📊 Métriques globales"},
    "weekly_model_title": {"EN": "🏥 Health by model", "FR": "🏥 Santé par modèle"},
    "col_model": {"EN": "Model", "FR": "Modèle"},
    "col_predictions": {"EN": "Predictions", "FR": "Prédictions"},
    "col_errors": {"EN": "Errors", "FR": "Erreurs"},
    "col_latency": {"EN": "Avg latency", "FR": "Latence moy."},
    "col_feat_drift": {"EN": "Feature drift", "FR": "Dérive features"},
    "col_perf_drift": {"EN": "Perf. drift", "FR": "Dérive perf."},
    "col_status": {"EN": "Status", "FR": "Statut"},
    "lbl_total_pred": {"EN": "Predictions", "FR": "Prédictions"},
    "lbl_error_rate": {"EN": "Error rate", "FR": "Taux d'erreur"},
    "lbl_avg_latency": {"EN": "Average latency", "FR": "Latence moyenne"},
    "lbl_models_alert": {"EN": "Models in alert", "FR": "Modèles en alerte"},
    "lbl_critical": {"EN": "critical", "FR": "critique"},
    "lbl_warning": {"EN": "warning(s)", "FR": "avertissement(s)"},
    # Drift alert
    "drift_subject": {
        "EN": "[PredictML] ⚠️ Feature drift — {model_name} / {feature}",
        "FR": "[PredictML] ⚠️ Dérive des données — {model_name} / {feature}",
    },
    "drift_title": {"EN": "Feature drift alert", "FR": "Alerte dérive des données"},
    "drift_intro": {
        "EN": (
            "Data drift has been detected on model "
            "<strong>{model_name}</strong> — feature <strong>{feature}</strong>."
        ),
        "FR": (
            "Une dérive des données a été détectée sur le modèle "
            "<strong>{model_name}</strong> — feature <strong>{feature}</strong>."
        ),
    },
    "drift_zscore_hint": {
        "EN": "(warning threshold ≥ 2, critical ≥ 3)",
        "FR": "(seuil avertissement ≥ 2, critique ≥ 3)",
    },
    "drift_psi_hint": {
        "EN": "(warning threshold ≥ 0.1, critical ≥ 0.2)",
        "FR": "(seuil avertissement ≥ 0.1, critique ≥ 0.2)",
    },
    "drift_advice": {
        "EN": (
            "Check whether the input data distribution has changed. "
            "Model retraining may be necessary."
        ),
        "FR": (
            "Le Z-score mesure l'écart entre la distribution actuelle et celle apprise à l'entraînement. "
            "Le PSI (Population Stability Index) quantifie la divergence globale de distribution. "
            "Vérifiez si la source des données d'entrée a changé — un réentraînement peut être nécessaire."
        ),
    },
    "lbl_status": {"EN": "Status", "FR": "Statut"},
    "lbl_zscore": {"EN": "Z-score", "FR": "Z-score"},
    "lbl_psi": {"EN": "PSI", "FR": "PSI"},
    "lbl_detected": {"EN": "Detected on", "FR": "Détecté le"},
    # Performance alert
    "perf_subject": {
        "EN": "[PredictML] 🔴 Performance drift — {model_name}",
        "FR": "[PredictML] 🔴 Dégradation de performance — {model_name}",
    },
    "perf_title": {"EN": "Performance drift alert", "FR": "Alerte dégradation de performance"},
    "perf_intro": {
        "EN": (
            "A performance degradation has been detected on model "
            "<strong>{model_name}</strong>."
        ),
        "FR": (
            "Une dégradation de performance a été détectée sur le modèle "
            "<strong>{model_name}</strong>."
        ),
    },
    "lbl_baseline_acc": {"EN": "Baseline accuracy", "FR": "Accuracy de référence"},
    "lbl_recent_acc": {"EN": "Recent accuracy", "FR": "Accuracy récente"},
    "lbl_drop": {"EN": "Drop", "FR": "Chute"},
    "perf_advice": {
        "EN": "Check recent observed results and consider retraining.",
        "FR": (
            "L'accuracy mesure la proportion de prédictions correctes sur les résultats observés récents. "
            "Vérifiez les résultats observés dans le dashboard et envisagez un réentraînement."
        ),
    },
    # Error spike alert
    "error_subject": {
        "EN": "[PredictML] 🔴 Error spike — {model_name} ({error_rate})",
        "FR": "[PredictML] 🔴 Pic d'erreurs — {model_name} ({error_rate})",
    },
    "error_title": {"EN": "Error spike alert", "FR": "Alerte pic d'erreurs"},
    "error_intro": {
        "EN": (
            "An error spike has been detected on model "
            "<strong>{model_name}</strong>."
        ),
        "FR": (
            "Un pic d'erreurs a été détecté sur le modèle "
            "<strong>{model_name}</strong>."
        ),
    },
    "lbl_current_error": {"EN": "Current error rate", "FR": "Taux d'erreur actuel"},
    "lbl_threshold": {"EN": "Configured threshold", "FR": "Seuil configuré"},
    "error_advice": {
        "EN": "Check the API logs and the state of the production model.",
        "FR": (
            "Le taux d'erreur représente la proportion de prédictions qui ont échoué (erreur technique ou timeout). "
            "Vérifiez les logs de l'API et l'état du modèle en production dans le dashboard."
        ),
    },
    # AUC alert
    "auc_subject": {
        "EN": "[PredictML] 🔴 AUC below threshold — {model_name} ({current_auc} < {auc_min})",
        "FR": "[PredictML] 🔴 AUC sous le seuil — {model_name} ({current_auc} < {auc_min})",
    },
    "auc_title": {
        "EN": "AUC below minimum threshold",
        "FR": "AUC sous le seuil minimum",
    },
    "auc_intro": {
        "EN": (
            "The AUC of binary classifier <strong>{model_name}</strong> "
            "has fallen below the configured minimum threshold."
        ),
        "FR": (
            "L'AUC du classifieur binaire <strong>{model_name}</strong> "
            "est passé en dessous du seuil minimum configuré."
        ),
    },
    "lbl_current_auc": {"EN": "Current AUC", "FR": "AUC actuel"},
    "lbl_min_auc": {"EN": "Minimum required", "FR": "Minimum requis"},
    "lbl_gap": {"EN": "Gap", "FR": "Écart"},
    "auc_advice": {
        "EN": (
            "The AUC shown is the value recorded at the last training or retrain. "
            "Consider retraining with more recent data or reviewing the feature pipeline."
        ),
        "FR": (
            "L'AUC (Aire sous la courbe ROC) mesure la capacité du modèle à distinguer les catégories. "
            "1.0 = parfait, 0.5 = aléatoire. "
            "La valeur affichée est celle enregistrée lors du dernier entraînement. "
            "Envisagez un réentraînement avec des données plus récentes ou vérifiez le pipeline de features."
        ),
    },
    # Auto-promotion alert
    "promo_subject": {
        "EN": "[PredictML] ✅ Auto-promotion — {model_name} v{new_version}",
        "FR": "[PredictML] ✅ Promotion automatique — {model_name} v{new_version}",
    },
    "promo_title": {
        "EN": "Auto-promotion — New version in production",
        "FR": "Promotion automatique — Nouvelle version en production",
    },
    "promo_intro": {
        "EN": (
            "The model <strong>{model_name}</strong> v<strong>{new_version}</strong> "
            "has been <strong>automatically promoted to production</strong> "
            "following a successful retrain — all quality criteria were satisfied."
        ),
        "FR": (
            "Le modèle <strong>{model_name}</strong> v<strong>{new_version}</strong> "
            "a été <strong>automatiquement promu en production</strong> "
            "suite à un réentraînement réussi — tous les critères de qualité ont été satisfaits."
        ),
    },
    "lbl_model": {"EN": "Model", "FR": "Modèle"},
    "lbl_new_version": {"EN": "New version", "FR": "Nouvelle version"},
    "lbl_accuracy": {"EN": "Accuracy", "FR": "Accuracy"},
    "lbl_f1": {"EN": "F1 score", "FR": "Score F1"},
    "lbl_criteria": {"EN": "Criteria result", "FR": "Résultat des critères"},
    "lbl_promoted_on": {"EN": "Promoted on", "FR": "Promu le"},
    "promo_advice": {
        "EN": (
            "The new version is now serving production traffic. "
            "Open the dashboard to verify its performance metrics and configure alerts if needed."
        ),
        "FR": (
            "La nouvelle version répond maintenant aux requêtes en production. "
            "Ouvrez le tableau de bord pour vérifier ses métriques de performance "
            "et configurer des alertes si nécessaire."
        ),
    },
    # Auto-demotion alert
    "demote_subject": {
        "EN": "[PredictML] 🔴 Auto-demotion — {model_name} v{version}",
        "FR": "[PredictML] 🔴 Rétrogradation automatique — {model_name} v{version}",
    },
    "demote_title": {
        "EN": "Circuit breaker — Auto-demotion",
        "FR": "Circuit breaker — Rétrogradation automatique",
    },
    "demote_action_removed": {
        "EN": "removed from production",
        "FR": "retiré de la production",
    },
    "demote_action_kept": {
        "EN": "kept in production (no fallback)",
        "FR": "maintenu en production (aucune version de secours disponible)",
    },
    "demote_intro": {
        "EN": (
            "The circuit breaker detected a critical degradation on model "
            "<strong>{model_name}</strong> v<strong>{version}</strong> "
            "and has <strong>{action}</strong> it automatically."
        ),
        "FR": (
            "Le circuit breaker a détecté une dégradation critique sur le modèle "
            "<strong>{model_name}</strong> v<strong>{version}</strong> "
            "et l'a <strong>{action}</strong> automatiquement."
        ),
    },
    "demote_no_fallback_warning": {
        "EN": (
            "⚠️ No fallback version available — the model has NOT been demoted. "
            "Manual action required."
        ),
        "FR": (
            "⚠️ Aucune version de secours disponible — le modèle N'A PAS été retiré de la production. "
            "Action manuelle requise."
        ),
    },
    "lbl_version": {"EN": "Version", "FR": "Version"},
    "lbl_reason": {"EN": "Reason", "FR": "Raison"},
    "demote_advice": {
        "EN": "Check the monitoring dashboard and promote a healthy version if necessary.",
        "FR": (
            "Le circuit breaker est un mécanisme de protection automatique qui retire un modèle dégradé "
            "de la production pour éviter des prédictions incorrectes. "
            "Vérifiez le tableau de bord de supervision et promouvez une version saine si nécessaire."
        ),
    },
}


class EmailService:
    """Transactional email sending for ML model monitoring."""

    def _is_configured(self) -> bool:
        return bool(settings.SMTP_HOST and settings.ALERT_EMAIL_TO)

    def _t(self, key: str, **kwargs) -> str:
        """Return translated string for the current DEFAULT_LANGUAGE."""
        lang = getattr(settings, "DEFAULT_LANGUAGE", "EN")
        entry = _TRANSLATIONS.get(key, {})
        text = entry.get(lang) or entry.get("EN", key)
        return text.format(**kwargs) if kwargs else text

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

    def _base_html(self, title: str, body_content: str) -> str:
        dt = datetime.utcnow().strftime("%Y-%m-%d at %H:%M")
        btn_label = self._t("btn_dashboard")
        footer_text = self._t("footer", dt=dt)
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
    <a class="btn" href="{settings.STREAMLIT_URL}">{btn_label}</a>
    <p class="footer">{footer_text}</p>
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

        col_m = self._t("col_model")
        col_p = self._t("col_predictions")
        col_e = self._t("col_errors")
        col_l = self._t("col_latency")
        col_fd = self._t("col_feat_drift")
        col_pd = self._t("col_perf_drift")
        col_s = self._t("col_status")

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
        <p>{self._t("weekly_intro", start=start, end=end)}</p>
        <h2>{self._t("weekly_global_title")}</h2>
        <table>
          <tr><th>{self._t("lbl_total_pred")}</th><td>{gs.get('total_predictions', 0):,}</td></tr>
          <tr><th>{self._t("lbl_error_rate")}</th>
              <td>{gs.get('error_rate', 0) * 100:.1f} %</td></tr>
          <tr><th>{self._t("lbl_avg_latency")}</th>
              <td>{gs.get('avg_latency_ms') or '—'} ms</td></tr>
          <tr><th>{self._t("lbl_models_alert")}</th>
              <td>🔴 {gs.get('models_critical', 0)} {self._t("lbl_critical")} •
                  🟡 {gs.get('models_warning', 0)} {self._t("lbl_warning")}</td></tr>
        </table>
        <h2>{self._t("weekly_model_title")}</h2>
        <table>
          <tr>
            <th>{col_m}</th><th>{col_p}</th><th>{col_e}</th>
            <th>{col_l}</th><th>{col_fd}</th>
            <th>{col_pd}</th><th>{col_s}</th>
          </tr>
          {rows}
        </table>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=self._t("weekly_subject", start=start, end=end),
            html_body=self._base_html(self._t("weekly_model_title"), body),
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
        <p>{self._t("drift_intro", model_name=model_name, feature=feature)}</p>
        <table>
          <tr><th>{self._t("lbl_status")}</th><td>{self._status_badge(drift_status)}</td></tr>
          <tr><th>{self._t("lbl_zscore")}</th>
              <td>{z_txt} {self._t("drift_zscore_hint")}</td></tr>
          <tr><th>{self._t("lbl_psi")}</th>
              <td>{psi_txt} {self._t("drift_psi_hint")}</td></tr>
          <tr><th>{self._t("lbl_detected")}</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>{self._t("drift_advice")}</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=self._t("drift_subject", model_name=model_name, feature=feature),
            html_body=self._base_html(self._t("drift_title"), body),
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
        <p>{self._t("perf_intro", model_name=model_name)}</p>
        <table>
          <tr><th>{self._t("lbl_baseline_acc")}</th><td>{baseline_accuracy:.1%}</td></tr>
          <tr><th>{self._t("lbl_recent_acc")}</th><td>{current_accuracy:.1%}</td></tr>
          <tr><th>{self._t("lbl_drop")}</th><td><strong>−{drop:.1%}</strong></td></tr>
          <tr><th>{self._t("lbl_detected")}</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>{self._t("perf_advice")}</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=self._t("perf_subject", model_name=model_name),
            html_body=self._base_html(self._t("perf_title"), body),
        )

    def send_error_spike_alert(
        self, model_name: str, error_rate: float, threshold: float | None = None
    ) -> bool:
        """Error spike alert."""
        effective_threshold = threshold if threshold is not None else settings.ERROR_RATE_ALERT_THRESHOLD
        body = f"""
        <p>{self._t("error_intro", model_name=model_name)}</p>
        <table>
          <tr><th>{self._t("lbl_current_error")}</th>
              <td><strong>{error_rate:.1%}</strong></td></tr>
          <tr><th>{self._t("lbl_threshold")}</th><td>{effective_threshold:.1%}</td></tr>
          <tr><th>{self._t("lbl_detected")}</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>{self._t("error_advice")}</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=self._t("error_subject", model_name=model_name, error_rate=f"{error_rate:.1%}"),
            html_body=self._base_html(self._t("error_title"), body),
        )

    def send_auc_alert(
        self, model_name: str, current_auc: float, auc_min: float
    ) -> bool:
        """AUC below minimum threshold alert (binary classifiers only)."""
        gap = auc_min - current_auc
        body = f"""
        <p>{self._t("auc_intro", model_name=model_name)}</p>
        <table>
          <tr><th>{self._t("lbl_current_auc")}</th><td><strong>{current_auc:.4f}</strong></td></tr>
          <tr><th>{self._t("lbl_min_auc")}</th><td>{auc_min:.4f}</td></tr>
          <tr><th>{self._t("lbl_gap")}</th><td><strong>−{gap:.4f}</strong></td></tr>
          <tr><th>{self._t("lbl_detected")}</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>{self._t("auc_advice")}</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=self._t(
                "auc_subject",
                model_name=model_name,
                current_auc=f"{current_auc:.4f}",
                auc_min=f"{auc_min:.4f}",
            ),
            html_body=self._base_html(self._t("auc_title"), body),
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
            metrics_rows += (
                f"<tr><th>{self._t('lbl_accuracy')}</th>"
                f"<td><strong>{accuracy:.4f}</strong></td></tr>"
            )
        if f1_score is not None:
            metrics_rows += (
                f"<tr><th>{self._t('lbl_f1')}</th>"
                f"<td><strong>{f1_score:.4f}</strong></td></tr>"
            )
        body = f"""
        <p>{self._t("promo_intro", model_name=model_name, new_version=new_version)}</p>
        <table>
          <tr><th>{self._t("lbl_model")}</th><td>{model_name}</td></tr>
          <tr><th>{self._t("lbl_new_version")}</th><td>{new_version}</td></tr>
          {metrics_rows}
          <tr><th>{self._t("lbl_criteria")}</th><td>{reason}</td></tr>
          <tr><th>{self._t("lbl_promoted_on")}</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>{self._t("promo_advice")}</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=self._t("promo_subject", model_name=model_name, new_version=new_version),
            html_body=self._base_html(self._t("promo_title"), body),
        )

    def send_auto_demotion_alert(
        self,
        model_name: str,
        version: str,
        reason: str,
        no_fallback: bool = False,
    ) -> bool:
        """Circuit breaker auto-demotion alert."""
        action = self._t("demote_action_kept") if no_fallback else self._t("demote_action_removed")
        fallback_warning = ""
        if no_fallback:
            warn_text = self._t("demote_no_fallback_warning")
            fallback_warning = (
                f'<p style="color:#c0392b;font-weight:bold">{warn_text}</p>'
            )
        body = f"""
        <p>{self._t("demote_intro", model_name=model_name, version=version, action=action)}</p>
        {fallback_warning}
        <table>
          <tr><th>{self._t("lbl_model")}</th><td>{model_name}</td></tr>
          <tr><th>{self._t("lbl_version")}</th><td>{version}</td></tr>
          <tr><th>{self._t("lbl_reason")}</th><td>{reason}</td></tr>
          <tr><th>{self._t("lbl_detected")}</th>
              <td>{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</td></tr>
        </table>
        <p>{self._t("demote_advice")}</p>"""

        return self._send_email(
            to=settings.ALERT_EMAIL_TO,
            subject=self._t("demote_subject", model_name=model_name, version=version),
            html_body=self._base_html(self._t("demote_title"), body),
        )


# Global instance
email_service = EmailService()
