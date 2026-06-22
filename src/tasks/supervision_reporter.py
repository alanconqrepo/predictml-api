"""
Supervision reporter — weekly report and automatic alerts.

Scheduled by the ARQ worker (``arq_worker.py``) in the ``retrain-worker``
container — runs every 6 h via ``alert_check_task`` (0 h, 6 h, 12 h, 18 h UTC).
Enabled via environment variables:
  - ENABLE_EMAIL_ALERTS=true   → send alert emails
  - WEEKLY_REPORT_ENABLED=true → weekly report on WEEKLY_REPORT_DAY at WEEKLY_REPORT_HOUR h

The alert job always runs (even without email) to trigger webhooks
configured on models (drift_critical, error_rate_threshold, auto_demote…).
Each check result is logged in alert_check_logs for audit and Streamlit display.
Emails are suppressed when no new production predictions have arrived since the
last check for the model (to avoid duplicate alerts on stale data).
"""

import asyncio
from datetime import datetime, timedelta

import structlog

from src.core.config import settings
from src.services.webhook_service import send_webhook

logger = structlog.get_logger(__name__)

_DRIFT_LEVEL: dict[str, int] = {"ok": 0, "warning": 1, "critical": 2}


def _get_model_threshold(thresholds: dict | None, key: str, default: float) -> float:
    """Return the model-specific threshold if set, otherwise the global threshold."""
    if thresholds and (val := thresholds.get(key)) is not None:
        return val
    return default


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


async def run_alert_check() -> None:
    """
    Check supervision indicators every 6 h over the last 24 hours.
    Sends email alerts if ENABLE_EMAIL_ALERTS=true AND new predictions exist since last check.
    Triggers webhooks configured on models regardless of email setting:
      - error_rate_threshold : error rate > threshold
      - drift_critical       : critical feature drift detected
    Logs every check result in alert_check_logs.
    """
    from src.core.ml_metrics import drift_detected_total

    logger.info("Checking supervision alerts")
    structlog.contextvars.bind_contextvars(event_type="supervision")

    try:
        from src.db.database import AsyncSessionLocal
        from src.services import drift_service
        from src.services.db_service import DBService
        from src.services.email_service import email_service

        end = datetime.utcnow()
        start = end - timedelta(hours=24)

        async with AsyncSessionLocal() as db:
            raw_stats = await DBService.get_global_monitoring_stats(db, start, end)

            # Load all active models ONCE before the loop
            all_metas = await DBService.get_all_active_models(db)
            meta_by_name: dict[str, list] = {}
            for m in all_metas:
                meta_by_name.setdefault(m.name, []).append(m)

            for model_stat in raw_stats:
                model_name = model_stat["model_name"]
                error_rate = model_stat["error_rate"]

                # Resolve metadata and thresholds for this model
                metas = meta_by_name.get(model_name, [])
                prod_meta = next((m for m in metas if m.is_production), metas[0] if metas else None)
                thresholds = prod_meta.alert_thresholds if prod_meta else None

                # --- Guard: check for new predictions since last check ---
                last_check_at = await DBService.get_last_alert_check_at(db, model_name)
                if last_check_at is not None:
                    new_pred_count = await DBService.count_predictions_since(
                        db, model_name, last_check_at
                    )
                    email_blocked = new_pred_count == 0
                else:
                    # First check ever for this model — use total predictions in window
                    new_pred_count = model_stat.get("total_predictions", 0)
                    email_blocked = False

                if email_blocked:
                    logger.info(
                        "No new predictions since last check — email suppressed",
                        model=model_name,
                        last_check_at=last_check_at.isoformat() if last_check_at else None,
                    )

                # Track the max detected drift level for the drift-retrain trigger
                _max_input_drift = "ok"
                _max_output_drift = "ok"

                # ── Error spike alert ────────────────────────────────────────
                error_threshold = _get_model_threshold(
                    thresholds, "error_rate_max", settings.ERROR_RATE_ALERT_THRESHOLD
                )
                error_triggered = error_rate >= error_threshold
                error_alert_sent = False
                error_webhook_sent = False

                if error_triggered:
                    severity = "critical" if error_rate >= error_threshold * 2 else "warning"
                    drift_detected_total.labels(
                        model_name=model_name, drift_type="error_rate", severity=severity
                    ).inc()
                    logger.warning(
                        "Error spike detected",
                        model=model_name,
                        error_rate=error_rate,
                    )
                    if not email_blocked and settings.ENABLE_EMAIL_ALERTS:
                        error_alert_sent = bool(
                            email_service.send_error_spike_alert(model_name, error_rate)
                        )
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
                        error_webhook_sent = True

                _result = (
                    "alert_triggered"
                    if error_triggered and not email_blocked
                    else "skipped_no_predictions"
                    if error_triggered and email_blocked
                    else "ok"
                )
                await DBService.create_alert_check_log(
                    db,
                    check_type="error_spike",
                    model_name=model_name,
                    model_version=prod_meta.version if prod_meta else None,
                    result=_result,
                    alert_sent=error_alert_sent,
                    webhook_sent=error_webhook_sent,
                    new_predictions_count=new_pred_count,
                    details={"error_rate": round(error_rate, 4), "threshold": error_threshold},
                )

                # ── AUC minimum alert (binary classifiers only) ──────────────
                auc_min = thresholds.get("auc_min") if thresholds else None
                auc_triggered = (
                    auc_min is not None
                    and prod_meta
                    and prod_meta.model_task == "classification_binary"
                    and prod_meta.auc is not None
                    and prod_meta.auc < auc_min
                )
                auc_alert_sent = False
                auc_webhook_sent = False

                if auc_triggered:
                    logger.warning(
                        "AUC below minimum threshold",
                        model=model_name,
                        auc=prod_meta.auc,
                        auc_min=auc_min,
                    )
                    if not email_blocked and settings.ENABLE_EMAIL_ALERTS:
                        auc_alert_sent = bool(
                            email_service.send_auc_alert(model_name, prod_meta.auc, auc_min)
                        )
                    if prod_meta and prod_meta.webhook_url:
                        asyncio.create_task(
                            send_webhook(
                                prod_meta.webhook_url,
                                {
                                    "model_name": model_name,
                                    "version": prod_meta.version,
                                    "timestamp": end.isoformat() + "Z",
                                    "details": {
                                        "auc": prod_meta.auc,
                                        "auc_min": auc_min,
                                    },
                                },
                                event_type="auc_below_threshold",
                            )
                        )
                        auc_webhook_sent = True

                if auc_min is not None and prod_meta:
                    _auc_result = (
                        "alert_triggered"
                        if auc_triggered and not email_blocked
                        else "skipped_no_predictions"
                        if auc_triggered and email_blocked
                        else "ok"
                    )
                    await DBService.create_alert_check_log(
                        db,
                        check_type="auc",
                        model_name=model_name,
                        model_version=prod_meta.version if prod_meta else None,
                        result=_auc_result,
                        alert_sent=auc_alert_sent,
                        webhook_sent=auc_webhook_sent,
                        new_predictions_count=new_pred_count,
                        details={
                            "auc": prod_meta.auc if prod_meta else None,
                            "auc_min": auc_min,
                        },
                    )

                # ── Performance drift alert ──────────────────────────────────
                perf_alert_sent = False
                perf_by_day = await DBService.get_accuracy_drift(db, model_name, start, end)
                perf_triggered = False
                perf_details: dict = {}

                if len(perf_by_day) >= 2:
                    use_mae = any(d.get("mae") is not None for d in perf_by_day)
                    if use_mae:
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

                        accuracy_min = thresholds.get("accuracy_min") if thresholds else None
                        if accuracy_min is not None and not use_mae:
                            perf_triggered = avg_second < accuracy_min
                            perf_details = {
                                "accuracy_recent": round(avg_second, 4),
                                "accuracy_min": accuracy_min,
                            }
                        else:
                            drop = avg_first - avg_second
                            perf_triggered = drop >= settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD
                            perf_details = {
                                "accuracy_first_half": round(avg_first, 4),
                                "accuracy_second_half": round(avg_second, 4),
                                "drop": round(drop, 4),
                                "threshold": settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD,
                            }

                        if perf_triggered:
                            drift_detected_total.labels(
                                model_name=model_name,
                                drift_type="performance",
                                severity="warning",
                            ).inc()
                            logger.warning(
                                "Performance drift detected",
                                model=model_name,
                            )
                            if not email_blocked and settings.ENABLE_EMAIL_ALERTS:
                                perf_alert_sent = bool(
                                    email_service.send_performance_alert(
                                        model_name, avg_second, avg_first
                                    )
                                )

                        _perf_result = (
                            "alert_triggered"
                            if perf_triggered and not email_blocked
                            else "skipped_no_predictions"
                            if perf_triggered and email_blocked
                            else "ok"
                        )
                        await DBService.create_alert_check_log(
                            db,
                            check_type="performance_drift",
                            model_name=model_name,
                            model_version=prod_meta.version if prod_meta else None,
                            result=_perf_result,
                            alert_sent=perf_alert_sent,
                            webhook_sent=False,
                            new_predictions_count=new_pred_count,
                            details=perf_details,
                        )

                # ── Feature drift alert ──────────────────────────────────────
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
                            if _DRIFT_LEVEL.get(feat_result.drift_status, 0) > _DRIFT_LEVEL.get(
                                _max_input_drift, 0
                            ):
                                _max_input_drift = feat_result.drift_status
                            if feat_result.drift_status in ("warning", "critical"):
                                drift_detected_total.labels(
                                    model_name=model_name,
                                    drift_type="feature",
                                    severity=feat_result.drift_status,
                                ).inc()

                            feat_alert_sent = False
                            feat_webhook_sent = False
                            if feat_result.drift_status == "critical":
                                logger.warning(
                                    "Critical feature drift",
                                    model=model_name,
                                    feature=feat_name,
                                )
                                if not email_blocked and settings.ENABLE_EMAIL_ALERTS:
                                    feat_alert_sent = bool(
                                        email_service.send_drift_alert(
                                            model_name=model_name,
                                            feature=feat_name,
                                            drift_status=feat_result.drift_status,
                                            z_score=feat_result.z_score,
                                            psi=feat_result.psi,
                                        )
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
                                    feat_webhook_sent = True

                            if feat_result.drift_status in ("warning", "critical"):
                                _feat_result = (
                                    "alert_triggered"
                                    if feat_result.drift_status == "critical" and not email_blocked
                                    else "skipped_no_predictions"
                                    if feat_result.drift_status == "critical" and email_blocked
                                    else "ok"
                                )
                                await DBService.create_alert_check_log(
                                    db,
                                    check_type="feature_drift",
                                    model_name=model_name,
                                    model_version=prod_meta.version,
                                    result=_feat_result,
                                    alert_sent=feat_alert_sent,
                                    webhook_sent=feat_webhook_sent,
                                    new_predictions_count=new_pred_count,
                                    details={
                                        "feature": feat_name,
                                        "drift_status": feat_result.drift_status,
                                        "z_score": feat_result.z_score,
                                        "psi": feat_result.psi,
                                    },
                                )

                # ── Output drift alert (label shift) ─────────────────────────
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
                        if output_report.status in ("warning", "critical"):
                            drift_detected_total.labels(
                                model_name=model_name,
                                drift_type="output",
                                severity=output_report.status,
                            ).inc()

                        output_webhook_sent = False
                        output_alert_sent = False
                        if output_report.status == "critical":
                            logger.warning(
                                "Critical output drift (label shift)",
                                model=model_name,
                                psi=output_report.psi,
                            )
                            if not email_blocked and settings.ENABLE_EMAIL_ALERTS:
                                # No dedicated email method for output drift currently
                                pass
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
                                output_webhook_sent = True

                        if output_report.status in ("warning", "critical"):
                            _out_result = (
                                "alert_triggered"
                                if output_report.status == "critical" and not email_blocked
                                else "skipped_no_predictions"
                                if output_report.status == "critical" and email_blocked
                                else "ok"
                            )
                            await DBService.create_alert_check_log(
                                db,
                                check_type="output_drift",
                                model_name=model_name,
                                model_version=prod_meta.version,
                                result=_out_result,
                                alert_sent=output_alert_sent,
                                webhook_sent=output_webhook_sent,
                                new_predictions_count=new_pred_count,
                                details={
                                    "psi": output_report.psi,
                                    "drift_status": output_report.status,
                                    "predictions_analyzed": output_report.predictions_analyzed,
                                },
                            )

                # ── Circuit breaker — auto-demotion ──────────────────────────
                if prod_meta and prod_meta.promotion_policy:
                    policy = prod_meta.promotion_policy
                    if policy.get("auto_demote"):
                        from src.services.auto_promotion_service import evaluate_auto_demotion

                        demoted, reason = await evaluate_auto_demotion(db, model_name, policy)
                        if demoted:
                            logger.warning(
                                "Model auto-demoted by circuit breaker",
                                model=model_name,
                                version=prod_meta.version,
                                reason=reason,
                            )

                # ── Drift-triggered retrain ───────────────────────────────────
                if prod_meta:
                    sched = prod_meta.retrain_schedule
                    if (
                        sched
                        and sched.get("trigger_on_drift")
                        and prod_meta.train_script_object_key
                    ):
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
                                >= datetime.fromisoformat(last_run_str)
                                + timedelta(hours=cooldown_hours)
                            )
                            if cooldown_ok:
                                from src.core.arq_pool import get_arq_pool

                                pool = await get_arq_pool()
                                await pool.enqueue_job(
                                    "scheduled_retrain_task",
                                    model_name=model_name,
                                    source_version=prod_meta.version,
                                )
                                logger.info(
                                    "Retrain triggered by drift",
                                    model=model_name,
                                    version=prod_meta.version,
                                )

    except Exception as exc:
        logger.error("Error during alert check", error=str(exc))
    finally:
        structlog.contextvars.clear_contextvars()


async def run_weekly_report() -> None:
    """
    Weekly report: sends a summary email for the past week.
    """
    if not settings.WEEKLY_REPORT_ENABLED:
        return

    logger.info("Generating weekly report")

    try:
        from src.db.database import AsyncSessionLocal
        from src.services.db_service import DBService
        from src.services.email_service import email_service

        end = datetime.utcnow()
        start = end - timedelta(days=7)

        async with AsyncSessionLocal() as db:
            raw_stats = await DBService.get_global_monitoring_stats(db, start, end)
            all_metas = await DBService.get_all_active_models(db)
            meta_by_name: dict[str, list] = {}
            for m in all_metas:
                meta_by_name.setdefault(m.name, []).append(m)

            # Build a dict compatible with email_service.send_weekly_report()
            models_summary = []
            total_pred = 0
            total_errors = 0
            for raw in raw_stats:
                model_name = raw["model_name"]
                total_pred += raw["total_predictions"]
                total_errors += raw["error_count"]

                # Compute performance drift status from 7-day accuracy trend
                perf_status = "no_data"
                metas = meta_by_name.get(model_name, [])
                prod_meta = next((m for m in metas if m.is_production), metas[0] if metas else None)
                thresholds = prod_meta.alert_thresholds if prod_meta else None

                perf_by_day = await DBService.get_accuracy_drift(db, model_name, start, end)
                if len(perf_by_day) >= 2:
                    use_mae = any(d.get("mae") is not None for d in perf_by_day)
                    if use_mae:
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
                        accuracy_min = thresholds.get("accuracy_min") if thresholds else None
                        if accuracy_min is not None and not use_mae:
                            if avg_second < accuracy_min:
                                perf_status = "critical"
                            else:
                                perf_status = "ok"
                        else:
                            drop = avg_first - avg_second
                            if drop >= settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD:
                                perf_status = "critical"
                            elif drop >= settings.PERFORMANCE_DRIFT_ALERT_THRESHOLD / 2:
                                perf_status = "warning"
                            else:
                                perf_status = "ok"

                error_rate = raw["error_rate"]
                error_threshold = _get_model_threshold(
                    thresholds, "error_rate_max", settings.ERROR_RATE_ALERT_THRESHOLD
                )
                if error_rate >= error_threshold:
                    health_status = "critical"
                elif perf_status in ("warning", "critical"):
                    health_status = "warning"
                else:
                    health_status = "ok"

                models_summary.append(
                    {
                        "model_name": model_name,
                        "total_predictions": raw["total_predictions"],
                        "error_rate": error_rate,
                        "avg_latency_ms": raw["avg_latency_ms"],
                        "feature_drift_status": "no_data",
                        "performance_drift_status": perf_status,
                        "health_status": health_status,
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
                "models_warning": sum(1 for m in models_summary if m["health_status"] == "warning"),
                "models_ok": sum(1 for m in models_summary if m["health_status"] == "ok"),
            },
            "models": models_summary,
        }

        email_service.send_weekly_report(overview)

    except Exception as exc:
        logger.error("Error during weekly report generation", error=str(exc))


# ---------------------------------------------------------------------------
# Scheduler lifecycle (APScheduler — optional dependency)
# ---------------------------------------------------------------------------

try:
    from apscheduler.schedulers.background import BackgroundScheduler

    _APSCHEDULER_AVAILABLE = True
    _scheduler: BackgroundScheduler = BackgroundScheduler()
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    _scheduler = None  # type: ignore[assignment]


def start_scheduler() -> None:
    """Start the APScheduler background scheduler and register periodic jobs."""
    if not _APSCHEDULER_AVAILABLE or _scheduler is None:
        logger.warning("APScheduler not available — supervision scheduler not started")
        return

    if settings.ENABLE_EMAIL_ALERTS or True:
        _scheduler.add_job(
            run_alert_check,
            "interval",
            hours=6,
            id="alert_check",
            replace_existing=True,
        )

    if settings.WEEKLY_REPORT_ENABLED:
        _scheduler.add_job(
            run_weekly_report,
            "cron",
            day_of_week=getattr(settings, "WEEKLY_REPORT_DAY", "mon"),
            hour=getattr(settings, "WEEKLY_REPORT_HOUR", 8),
            id="weekly_report",
            replace_existing=True,
        )

    if not _scheduler.running:
        _scheduler.start()


def stop_scheduler() -> None:
    """Stop the APScheduler background scheduler if it is running."""
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
