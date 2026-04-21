"""
Scheduler de supervision — rapport hebdomadaire et alertes automatiques.

Utilise APScheduler (AsyncIOScheduler) intégré au cycle de vie FastAPI.
Activé via les variables d'environnement :
  - ENABLE_EMAIL_ALERTS=true   → vérification toutes les 6h
  - WEEKLY_REPORT_ENABLED=true → rapport le WEEKLY_REPORT_DAY à WEEKLY_REPORT_HOUR h

Aucun scheduler n'est démarré si les deux variables sont false.
"""

from datetime import datetime, timedelta

import structlog

from src.core.config import settings

logger = structlog.get_logger(__name__)

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
    Envoie des alertes e-mail si :
      - taux d'erreur d'un modèle > ERROR_RATE_ALERT_THRESHOLD
      - drift de performance détecté sur un modèle
      - drift de features critique détecté sur un modèle
    """
    if not settings.ENABLE_EMAIL_ALERTS:
        return

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

            for model_stat in raw_stats:
                model_name = model_stat["model_name"]
                error_rate = model_stat["error_rate"]

                # Alerte pic d'erreurs
                if error_rate >= settings.ERROR_RATE_ALERT_THRESHOLD:
                    logger.warning(
                        "Pic d'erreurs détecté",
                        model=model_name,
                        error_rate=error_rate,
                    )
                    email_service.send_error_spike_alert(model_name, error_rate)

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
                        drop = avg_first - avg_second
                        if drop >= settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD:
                            logger.warning(
                                "Drift de performance détecté",
                                model=model_name,
                                drop=drop,
                            )
                            email_service.send_performance_alert(model_name, avg_second, avg_first)

                # Alerte drift features
                all_metas = await DBService.get_all_active_models(db)
                metas = [m for m in all_metas if m.name == model_name]
                if metas:
                    prod_meta = next((m for m in metas if m.is_production), metas[0])
                    if prod_meta.feature_baseline:
                        production_stats = await DBService.get_feature_production_stats(
                            db, model_name, prod_meta.version, days=1
                        )
                        features = drift_service.compute_feature_drift(
                            prod_meta.feature_baseline, production_stats, min_count=10
                        )
                        for feat_name, feat_result in features.items():
                            if feat_result.drift_status == "critical":
                                logger.warning(
                                    "Drift features critique",
                                    model=model_name,
                                    feature=feat_name,
                                )
                                email_service.send_drift_alert(
                                    model_name=model_name,
                                    feature=feat_name,
                                    drift_status=feat_result.drift_status,
                                    z_score=feat_result.z_score,
                                    psi=feat_result.psi,
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

    if settings.ENABLE_EMAIL_ALERTS:
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
