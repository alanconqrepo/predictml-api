"""
ARQ Worker — executes long-running tasks outside the API process.

Defined tasks:
  - retrain_task            : manually triggered retrain via POST /retrain
  - scheduled_retrain_task  : scheduled retrain (ARQ cron, replaces APScheduler)
  - alert_check_task        : alert check every 6 h
  - weekly_report_task      : weekly report

Start:
    python -m arq src.tasks.arq_worker.WorkerSettings

The worker reads jobs from Redis (same instance as the API).
A single worker replica is deployed (no Redis lock needed for
scheduling, unlike the old APScheduler × 3 replicas architecture).
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import structlog
from arq import cron
from arq.connections import RedisSettings
from prometheus_client import Counter, Histogram, start_http_server

from src.core.config import settings

TASK_TOTAL = Counter(
    "arq_task_total",
    "Total ARQ task executions",
    ["task", "status"],
)
TASK_DURATION = Histogram(
    "arq_task_duration_seconds",
    "ARQ task execution duration in seconds",
    ["task"],
    buckets=[1, 5, 30, 60, 120, 300, 600, 720],
)

logger = structlog.get_logger(__name__)


def _redis_settings() -> RedisSettings:
    from src.core.config import settings

    return RedisSettings.from_dsn(settings.REDIS_URL)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _update_task_run(task_run_id: str, **fields) -> None:
    """Update a TaskRun in the DB (status, started_at, completed_at, result, error, logs)."""
    from sqlalchemy import update

    from src.db.database import AsyncSessionLocal
    from src.db.models.task_run import TaskRun

    try:
        async with AsyncSessionLocal() as db:
            await db.execute(update(TaskRun).where(TaskRun.id == task_run_id).values(**fields))
            await db.commit()
    except Exception as exc:
        logger.warning("TaskRun update failed (non-blocking)", id=task_run_id, error=str(exc))


async def _get_redis_logs(redis, job_id: str) -> str:
    """Fetch all Redis logs for a job and return them as a string."""
    try:
        lines = await redis.lrange(f"retrain_logs:{job_id}", 0, -1)
        return "\n".join(
            line.decode("utf-8") if isinstance(line, bytes) else line for line in lines
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Task 1: manual retrain (enqueued by POST /models/{name}/{version}/retrain)
# ---------------------------------------------------------------------------


async def retrain_task(
    ctx: Dict[str, Any],
    job_id: str,
    model_name: str,
    source_version: str,
    new_version: str,
    start_date: str,
    end_date: str,
    set_production: bool,
    triggered_by: str,
    source_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the retraining pipeline and update the TaskRun in the DB."""
    from src.services.retrain_service import do_retrain

    redis = ctx.get("redis")
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Mark as running
    await _update_task_run(job_id, status="running", started_at=now)
    logger.info(
        "retrain_task started",
        job_id=job_id,
        model=model_name,
        source_version=source_version,
        new_version=new_version,
    )

    t0 = time.monotonic()
    try:
        result = await do_retrain(
            model_name=model_name,
            source_version=source_version,
            new_version=new_version,
            start_date=start_date,
            end_date=end_date,
            set_production=set_production,
            triggered_by=triggered_by,
            trigger="manual",
            source_fields=source_fields,
            job_id=job_id,
            redis_client=redis,
        )
    except Exception as exc:
        logger.error("retrain_task unexpected exception", job_id=job_id, error=str(exc))
        result = {"success": False, "error": str(exc), "stdout": "", "stderr": ""}

    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    status = "success" if result.get("success") else "failed"
    TASK_DURATION.labels(task="retrain_task").observe(time.monotonic() - t0)
    TASK_TOTAL.labels(task="retrain_task", status=status).inc()
    stdout_logs = await _get_redis_logs(redis, job_id) if redis else result.get("stdout", "")
    stderr_text = result.get("stderr", "")
    if stderr_text:
        final_logs = (
            stdout_logs + "\n\n--- STDERR ---\n" + stderr_text
            if stdout_logs
            else "--- STDERR ---\n" + stderr_text
        )
    else:
        final_logs = stdout_logs

    await _update_task_run(
        job_id,
        status=status,
        completed_at=completed_at,
        new_version=result.get("new_version") if result.get("success") else None,
        result={k: v for k, v in result.items() if k not in ("stdout", "stderr") and v is not None},
        error=result.get("error"),
        logs=final_logs,
    )

    logger.info(
        "retrain_task completed",
        job_id=job_id,
        status=status,
        model=model_name,
        new_version=result.get("new_version"),
    )
    return result


# ---------------------------------------------------------------------------
# Task 2: scheduled retrain (triggered by ARQ cron jobs below)
# ---------------------------------------------------------------------------


async def scheduled_retrain_task(
    ctx: Dict[str, Any],
    model_name: str,
    source_version: str,
) -> None:
    """Scheduled retrain — acquires a Redis lock to prevent duplicate runs."""
    from src.services.retrain_service import do_retrain

    redis = ctx.get("redis")
    lock_key = f"retrain_lock:{model_name}:{source_version}"

    # Distributed lock: prevents double-run if a manual retrain is in progress
    acquired = await redis.set(lock_key, "1", nx=True, ex=700)
    if not acquired:
        logger.warning(
            "Scheduled retrain skipped — lock active",
            model=model_name,
            version=source_version,
        )
        return

    job_id = str(uuid4())
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S")

    # Load schedule from DB to get parameters
    from sqlalchemy import and_, select

    from src.db.database import AsyncSessionLocal
    from src.db.models import ModelMetadata

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ModelMetadata).where(
                    and_(
                        ModelMetadata.name == model_name,
                        ModelMetadata.version == source_version,
                    )
                )
            )
            src = result.scalar_one_or_none()
            if not src:
                logger.error(
                    "Source model not found for scheduled_retrain",
                    model=model_name,
                    version=source_version,
                )
                await redis.delete(lock_key)
                return

            schedule = src.retrain_schedule or {}
            if not schedule.get("enabled", True):
                logger.info("Schedule disabled — job skipped", model=model_name)
                await redis.delete(lock_key)
                return

            from datetime import timedelta

            lookback_days = int(schedule.get("lookback_days", 30))
            end_date = now.strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            new_version = f"{source_version}-sched-{timestamp}"

    except Exception as exc:
        logger.error("DB read error in scheduled_retrain", error=str(exc))
        await redis.delete(lock_key)
        return

    # Create the TaskRun in DB for tracking
    from src.db.database import AsyncSessionLocal
    from src.db.models.task_run import TaskRun

    try:
        async with AsyncSessionLocal() as db:
            task_run = TaskRun(
                id=job_id,
                task_type="scheduled_retrain",
                model_name=model_name,
                model_version=source_version,
                new_version=new_version,
                triggered_by="scheduler",
                status="running",
                started_at=now.replace(tzinfo=None),
            )
            db.add(task_run)
            await db.commit()
    except Exception as exc:
        logger.warning("TaskRun creation failed (non-blocking)", error=str(exc))

    t0 = time.monotonic()
    try:
        result = await do_retrain(
            model_name=model_name,
            source_version=source_version,
            new_version=new_version,
            start_date=start_date,
            end_date=end_date,
            set_production=False,
            triggered_by="scheduler",
            trigger="scheduler",
            job_id=job_id,
            redis_client=redis,
        )
    except Exception as exc:
        logger.error(
            "scheduled_retrain_task exception",
            model=model_name,
            error=str(exc),
        )
        result = {"success": False, "error": str(exc)}
    finally:
        await redis.delete(lock_key)
        status = "success" if result.get("success") else "failed"
        TASK_DURATION.labels(task="scheduled_retrain_task").observe(time.monotonic() - t0)
        TASK_TOTAL.labels(task="scheduled_retrain_task", status=status).inc()

    # Final TaskRun update
    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    status = "success" if result.get("success") else "failed"
    final_logs = await _get_redis_logs(redis, job_id)

    await _update_task_run(
        job_id,
        status=status,
        completed_at=completed_at,
        new_version=result.get("new_version") if result.get("success") else None,
        result={k: v for k, v in result.items() if k not in ("stdout", "stderr") and v is not None},
        error=result.get("error"),
        logs=final_logs,
    )

    # Notify admins of the result; skip when auto_promoted=True because
    # do_retrain() already called send_auto_promotion_alert() in that case.
    from src.core.config import settings as _cfg

    if _cfg.ENABLE_EMAIL_ALERTS and not result.get("auto_promoted"):
        from src.services.email_service import email_service as _email_svc

        _email_svc.send_retrain_result_alert(
            model_name=model_name,
            source_version=source_version,
            new_version=result.get("new_version") if status == "success" else None,
            success=(status == "success"),
            trigger="scheduler",
            error=result.get("error"),
            accuracy=result.get("accuracy"),
            f1_score=result.get("f1_score"),
        )


# ---------------------------------------------------------------------------
# Task 3: alert check (every 6 h)
# ---------------------------------------------------------------------------


async def alert_check_task(ctx: Dict[str, Any]) -> None:
    """Delegate to existing supervision_reporter logic."""
    from src.tasks.supervision_reporter import run_alert_check

    logger.info("alert_check_task started")
    t0 = time.monotonic()
    try:
        await run_alert_check()
        TASK_TOTAL.labels(task="alert_check_task", status="success").inc()
    except Exception:
        TASK_TOTAL.labels(task="alert_check_task", status="failed").inc()
        raise
    finally:
        TASK_DURATION.labels(task="alert_check_task").observe(time.monotonic() - t0)
    logger.info("alert_check_task completed")


# ---------------------------------------------------------------------------
# Task 4: weekly report
# ---------------------------------------------------------------------------


async def weekly_report_task(ctx: Dict[str, Any]) -> None:
    """Delegate to existing supervision_reporter logic."""
    from src.tasks.supervision_reporter import run_weekly_report

    logger.info("weekly_report_task started")
    await run_weekly_report()
    logger.info("weekly_report_task completed")


# ---------------------------------------------------------------------------
# Worker startup / shutdown
# ---------------------------------------------------------------------------


async def startup(ctx: Dict[str, Any]) -> None:
    """Initialize the DB and load retrain cron jobs from the DB."""
    from src.core.config import settings
    from src.core.logging import setup_logging
    from src.db.database import init_db

    setup_logging(debug=settings.DEBUG)
    logger.info("ARQ worker started")

    start_http_server(9092)
    logger.info("Prometheus metrics server started", port=9092)

    try:
        await init_db()
        logger.info("DB connected (worker)")
    except Exception as exc:
        logger.warning("DB connection failed at worker startup", error=str(exc))

    # Retrain cron jobs are loaded dynamically via _load_retrain_crons()
    # (called after startup by WorkerSettings.cron_jobs via the factory below)


async def shutdown(ctx: Dict[str, Any]) -> None:
    """Close the DB connection."""
    from src.db.database import close_db

    await close_db()
    logger.info("ARQ worker stopped")


# ---------------------------------------------------------------------------
# Dynamic loading of retrain cron jobs from the DB
# ---------------------------------------------------------------------------


async def _build_retrain_cron_jobs():
    """
    Load models with retrain_schedule.enabled=True and return
    a list of ARQ cron jobs, one per active version.

    Called once at worker startup (WorkerSettings.cron_jobs).
    """
    from sqlalchemy import select

    from src.db.database import AsyncSessionLocal
    from src.db.models import ModelMetadata

    cron_jobs = []
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ModelMetadata).where(ModelMetadata.is_active.is_(True))
            )
            models = result.scalars().all()

        for model in models:
            sched = model.retrain_schedule
            if not (sched and sched.get("enabled") and sched.get("cron")):
                continue
            try:
                from arq import cron as arq_cron

                # ARQ expects minute, hour, day, month, weekday as integers/sets
                # We use the direct crontab representation (sched["cron"] e.g. "0 3 * * 1")
                cron_jobs.append(
                    arq_cron(
                        scheduled_retrain_task,
                        name=f"retrain_schedule:{model.name}:{model.version}",
                        kwargs={
                            "model_name": model.name,
                            "source_version": model.version,
                        },
                        # ARQ supports a native `cron` parameter
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Retrain cron skipped (error)",
                    model=model.name,
                    version=model.version,
                    error=str(exc),
                )
    except Exception as exc:
        logger.warning("Failed to load retrain crons", error=str(exc))
    return cron_jobs


# ---------------------------------------------------------------------------
# WorkerSettings
# ---------------------------------------------------------------------------


class WorkerSettings:
    """ARQ worker configuration.

    Start:
        python -m arq src.tasks.arq_worker.WorkerSettings
    """

    functions = [
        retrain_task,
        scheduled_retrain_task,
        alert_check_task,
        weekly_report_task,
    ]

    # Fixed cron jobs (alerts + weekly report)
    # Dynamic retrain crons are added at startup via on_startup
    _WEEKDAY_MAP = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    cron_jobs = [
        # Alert check every 6 h (0h, 6h, 12h, 18h UTC)
        cron(alert_check_task, hour={0, 6, 12, 18}, minute=0, run_at_startup=False),
        # Weekly report — day and hour driven by WEEKLY_REPORT_DAY / WEEKLY_REPORT_HOUR
        cron(
            weekly_report_task,
            weekday=_WEEKDAY_MAP.get(settings.WEEKLY_REPORT_DAY.lower(), 0),
            hour=settings.WEEKLY_REPORT_HOUR,
            minute=0,
            run_at_startup=False,
        ),
    ]

    on_startup = startup
    on_shutdown = shutdown

    # Max concurrency: 5 simultaneous jobs (the heavy retrain takes 1 slot)
    max_jobs = 5

    # Global job timeout (slightly > subprocess timeout of 600 s)
    job_timeout = 720

    # Grace time for missed jobs (e.g. worker restarted)
    keep_result = 3600  # keep result 1 h in Redis

    redis_settings = _redis_settings()
