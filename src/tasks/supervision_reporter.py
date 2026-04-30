"""
Scheduler de supervision — rapport hebdomadaire et alertes automatiques.

Utilise APScheduler (AsyncIOScheduler) intégré au cycle de vie FastAPI.
Activé via les variables d'environnement :
  - ENABLE_EMAIL_ALERTS=true   → vérification toutes les 6h (e-mails)
  - WEEKLY_REPORT_ENABLED=true → rapport le WEEKLY_REPORT_DAY à WEEKLY_REPORT_HOUR h

Le job d'alerte tourne toujours (même sans e-mail) pour déclencher les webhooks
configurés sur les modèles (drift_critical, error_rate_threshold).
"""

import asyncio
from datetime import datetime, timedelta

import structlog

from src.core.config import settings
from src.services.webhook_service import send_webhook

logger = structlog.get_logger(__name__)

_DRIFT_LEVEL: dict[str, int] = {"ok": 0, "warning": 1, "critical": 2}


def _get_model_threshold(thresholds: dict | None, key: str, default: float) -> float:
    """Retourne le seuil du modèle si défini, sinon le seuil global."""
    if thresholds and (val := thresholds.get(key)) is not None:
        return val
    return default


try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    _scheduler = AsyncIOScheduler(timezone="UTC")
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _scheduler = None  # type: ignore[assignment]
    _APSCHEDULER_AVAILABLE = False
    logger.warning("APScheduler non installé — scheduler de supervision désactivé")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


async def run_alert_check() -> None:
    """
    Vérification toutes les 6h des indicateurs de supervision sur les 24 dernières heures.
    Envoie des alertes e-mail si ENABLE_EMAIL_ALERTS=true.
    Déclenche les webhooks configurés sur les modèles indépendamment des e-mails :
      - error_rate_threshold : taux d'erreur > seuil
      - drift_critical       : drift de features critique détecté
    """
    logger.info("Vérification des alertes de supervision")

    try:
        from src.db.database import AsyncSessionLocal
        from src.services import drift_service
        from src.services.db_service import DBService
        from src.services.email_service import email_service

        end = datetime.utcnow()
        start = end - timedelta(hours=24)

        async with AsyncSessionLocal() as db:
            raw_stats = await DBService.get_global_monitoring_stats(db, start, end)

            # Charger tous les modèles actifs UNE SEULE FOIS avant la boucle
            all_metas = await DBService.get_all_active_models(db)
            meta_by_name: dict[str, list] = {}
            for m in all_metas:
                meta_by_name.setdefault(m.name, []).append(m)

            for model_stat in raw_stats:
                model_name = model_stat["model_name"]
                error_rate = model_stat["error_rate"]

                # Résoudre les métadonnées et seuils pour ce modèle
                metas = meta_by_name.get(model_name, [])
                prod_meta = next((m for m in metas if m.is_production), metas[0] if metas else None)
                thresholds = prod_meta.alert_thresholds if prod_meta else None

                # Suivi du niveau de drift max détecté pour le trigger drift-retrain
                _max_input_drift = "ok"
                _max_output_drift = "ok"

                # Alerte pic d'erreurs
                error_threshold = _get_model_threshold(
                    thresholds, "error_rate_max", settings.ERROR_RATE_ALERT_THRESHOLD
                )
                if error_rate >= error_threshold:
                    logger.warning(
                        "Pic d'erreurs détecté",
                        model=model_name,
                        error_rate=error_rate,
                    )
                    if settings.ENABLE_EMAIL_ALERTS:
                        email_service.send_error_spike_alert(model_name, error_rate)
                    if prod_meta and prod_meta.webhook_url:
                        asyncio.create_task(
                            send_webhook(
                                prod_meta.webhook_url,
                                {
                                    "model_name": model_name,
                                    "version": prod_meta.version,
                                    "timestamp": end.isoformat() + "Z",
                                    "details": {
                                        "error_rate": error_rate,
                                        "threshold": error_threshold,
                                    },
                                },
                                event_type="error_rate_threshold",
                            )
                        )

                # Alerte drift de performance
                perf_by_day = await DBService.get_accuracy_drift(db, model_name, start, end)
                if len(perf_by_day) >= 2:
                    use_mae = any(d.get("mae") is not None for d in perf_by_day)
                    if use_mae:
                        # Régression : hausse de MAE = dégradation → on inverse
                        metrics = [
                            -d["mae"]
                            for d in perf_by_day
                            if d["matched_count"] > 0 and d.get("mae") is not None
                        ]
                    else:
                        metrics = [d["accuracy"] for d in perf_by_day if d["matched_count"] > 0]
                    if len(metrics) >= 2:
                        mid = len(metrics) // 2
                        avg_first = sum(metrics[:mid]) / mid
                        avg_second = sum(metrics[mid:]) / (len(metrics) - mid)

                        # Seuil absolu (accuracy_min) si configuré, sinon seuil relatif (drop)
                        accuracy_min = thresholds.get("accuracy_min") if thresholds else None
                        if accuracy_min is not None and not use_mae:
                            should_alert = avg_second < accuracy_min
                        else:
                            drop = avg_first - avg_second
                            should_alert = drop >= settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD

                        if should_alert:
                            logger.warning(
                                "Drift de performance détecté",
                                model=model_name,
                            )
                            if settings.ENABLE_EMAIL_ALERTS:
                                email_service.send_performance_alert(
                                    model_name, avg_second, avg_first
                                )

                # Alerte drift features
                if prod_meta and prod_meta.feature_baseline:
                    drift_enabled = (
                        thresholds.get("drift_auto_alert", True) if thresholds is not None else True
                    )
                    if drift_enabled:
                        production_stats = await DBService.get_feature_production_stats(
                            db, model_name, prod_meta.version, days=1
                        )
                        features = drift_service.compute_feature_drift(
                            prod_meta.feature_baseline, production_stats, min_count=10
                        )
                        for feat_name, feat_result in features.items():
                            if _DRIFT_LEVEL.get(feat_result.drift_status, 0) > _DRIFT_LEVEL.get(_max_input_drift, 0):
                                _max_input_drift = feat_result.drift_status
                            if feat_result.drift_status == "critical":
                                logger.warning(
                                    "Drift features critique",
                                    model=model_name,
                                    feature=feat_name,
                                )
                                if settings.ENABLE_EMAIL_ALERTS:
                                    email_service.send_drift_alert(
                                        model_name=model_name,
                                        feature=feat_name,
                                        drift_status=feat_result.drift_status,
                                        z_score=feat_result.z_score,
                                        psi=feat_result.psi,
                                    )
                                if prod_meta.webhook_url:
                                    asyncio.create_task(
                                        send_webhook(
                                            prod_meta.webhook_url,
                                            {
                                                "model_name": model_name,
                                                "version": prod_meta.version,
                                                "timestamp": end.isoformat() + "Z",
                                                "details": {
                                                    "feature": feat_name,
                                                    "psi": feat_result.psi,
                                                    "z_score": feat_result.z_score,
                                                    "status": feat_result.drift_status,
                                                },
                                            },
                                            event_type="drift_critical",
                                        )
                                    )

                # Alerte drift de sortie (label shift)
                if prod_meta:
                    drift_enabled = (
                        thresholds.get("drift_auto_alert", True) if thresholds is not None else True
                    )
                    if drift_enabled:
                        output_report = await drift_service.compute_output_drift(
                            model_name=model_name,
                            period_days=1,
                            db=db,
                            model_version=prod_meta.version,
                            min_predictions=10,
                        )
                        _max_output_drift = output_report.status
                        if output_report.status == "critical":
                            logger.warning(
                                "Drift de sortie critique (label shift)",
                                model=model_name,
                                psi=output_report.psi,
                            )
                            if prod_meta.webhook_url:
                                asyncio.create_task(
                                    send_webhook(
                                        prod_meta.webhook_url,
                                        {
                                            "model_name": model_name,
                                            "version": prod_meta.version,
                                            "timestamp": end.isoformat() + "Z",
                                            "details": {
                                                "psi": output_report.psi,
                                                "status": output_report.status,
                                                "predictions_analyzed": output_report.predictions_analyzed,
                                            },
                                        },
                                        event_type="output_drift_critical",
                                    )
                                )

                # Circuit breaker — auto-demotion
                if prod_meta and prod_meta.promotion_policy:
                    policy = prod_meta.promotion_policy
                    if policy.get("auto_demote"):
                        from src.services.auto_promotion_service import evaluate_auto_demotion

                        demoted, reason = await evaluate_auto_demotion(db, model_name, policy)
                        if demoted:
                            logger.warning(
                                "Modèle auto-démis par le circuit breaker",
                                model=model_name,
                                version=prod_meta.version,
                                reason=reason,
                            )

                # Retrain déclenché par drift
                if prod_meta:
                    sched = prod_meta.retrain_schedule
                    if sched and sched.get("trigger_on_drift") and prod_meta.train_script_object_key:
                        threshold_level = _DRIFT_LEVEL.get(sched["trigger_on_drift"], 999)
                        detected_level = max(
                            _DRIFT_LEVEL.get(_max_input_drift, 0),
                            _DRIFT_LEVEL.get(_max_output_drift, 0),
                        )
                        if detected_level >= threshold_level:
                            cooldown_hours = int(sched.get("drift_retrain_cooldown_hours", 24))
                            last_run_str = sched.get("last_run_at")
                            cooldown_ok = last_run_str is None or (
                                datetime.utcnow()
                                >= datetime.fromisoformat(last_run_str) + timedelta(hours=cooldown_hours)
                            )
                            if cooldown_ok:
                                from src.tasks.retrain_scheduler import _run_retrain_job

                                asyncio.create_task(_run_retrain_job(model_name, prod_meta.version))
                                logger.info(
                                    "Retrain déclenché par drift",
                                    model=model_name,
                                    version=prod_meta.version,
                                )

    except Exception as exc:
        logger.error("Erreur lors de la vérification des alertes", error=str(exc))


async def run_weekly_report() -> None:
    """
    Rapport hebdomadaire : envoie un e-mail récapitulatif de la semaine écoulée.
    """
    if not settings.WEEKLY_REPORT_ENABLED:
        return

    logger.info("Génération du rapport hebdomadaire")

    try:
        from src.db.database import AsyncSessionLocal
        from src.services.db_service import DBService
        from src.services.email_service import email_service

        end = datetime.utcnow()
        start = end - timedelta(days=7)

        async with AsyncSessionLocal() as db:
            raw_stats = await DBService.get_global_monitoring_stats(db, start, end)
            all_metas = await DBService.get_all_active_models(db)
            meta_by_name = {}
            for m in all_metas:
                meta_by_name.setdefault(m.name, []).append(m)

        # Construire un dict compatible avec email_service.send_weekly_report()
        models_summary = []
        total_pred = 0
        total_errors = 0
        for raw in raw_stats:
            total_pred += raw["total_predictions"]
            total_errors += raw["error_count"]
            models_summary.append(
                {
                    "model_name": raw["model_name"],
                    "total_predictions": raw["total_predictions"],
                    "error_rate": raw["error_rate"],
                    "avg_latency_ms": raw["avg_latency_ms"],
                    "feature_drift_status": "no_data",
                    "performance_drift_status": "no_data",
                    "health_status": (
                        "critical"
                        if raw["error_rate"] >= settings.ERROR_RATE_ALERT_THRESHOLD
                        else "ok"
                    ),
                }
            )

        overview = {
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "global_stats": {
                "total_predictions": total_pred,
                "total_shadow": sum(r["shadow_predictions"] for r in raw_stats),
                "total_errors": total_errors,
                "error_rate": round(total_errors / total_pred, 4) if total_pred > 0 else 0.0,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "active_models": len(raw_stats),
                "models_critical": sum(
                    1 for m in models_summary if m["health_status"] == "critical"
                ),
                "models_warning": 0,
                "models_ok": sum(1 for m in models_summary if m["health_status"] == "ok"),
            },
            "models": models_summary,
        }

        email_service.send_weekly_report(overview)

    except Exception as exc:
        logger.error("Erreur lors de la génération du rapport hebdomadaire", error=str(exc))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def start_scheduler() -> None:
    """Démarre le scheduler avec les jobs configurés."""
    if not _APSCHEDULER_AVAILABLE or _scheduler is None:
        logger.warning("APScheduler indisponible, scheduler non démarré")
        return

    _scheduler.add_job(
        run_alert_check,
        "interval",
        hours=6,
        id="alert_check",
        replace_existing=True,
    )
    logger.info("Job de vérification d'alertes configuré (toutes les 6h)")

    if settings.WEEKLY_REPORT_ENABLED:
        _scheduler.add_job(
            run_weekly_report,
            "cron",
            day_of_week=settings.WEEKLY_REPORT_DAY,
            hour=settings.WEEKLY_REPORT_HOUR,
            minute=0,
            id="weekly_report",
            replace_existing=True,
        )
        logger.info(
            "Job de rapport hebdomadaire configuré",
            day=settings.WEEKLY_REPORT_DAY,
            hour=settings.WEEKLY_REPORT_HOUR,
        )

    _scheduler.start()


def stop_scheduler() -> None:
    """Arrête proprement le scheduler."""
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler de supervision arrêté")
