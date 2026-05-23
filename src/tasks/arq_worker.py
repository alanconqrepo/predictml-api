"""
Worker ARQ — exécute les tâches longues hors du processus API.

Tâches définies :
  - retrain_task         : retrain déclenché manuellement via POST /retrain
  - scheduled_retrain_task : retrain planifié (cron ARQ, remplace APScheduler)
  - alert_check_task     : vérification des alertes toutes les 6 h
  - weekly_report_task   : rapport hebdomadaire

Démarrage :
    python -m arq src.tasks.arq_worker.WorkerSettings

Le worker lit les jobs depuis Redis (même instance que l'API).
Un seul réplica du worker est déployé (pas de Redis lock nécessaire pour
le scheduling, contrairement à l'ancienne architecture APScheduler × 3 réplicas).
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import structlog
from arq import cron
from arq.connections import RedisSettings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers DB
# ---------------------------------------------------------------------------


async def _update_task_run(task_run_id: str, **fields) -> None:
    """Met à jour un TaskRun en DB (status, started_at, completed_at, result, error, logs)."""
    from sqlalchemy import update

    from src.db.database import AsyncSessionLocal
    from src.db.models.task_run import TaskRun

    try:
        async with AsyncSessionLocal() as db:
            await db.execute(update(TaskRun).where(TaskRun.id == task_run_id).values(**fields))
            await db.commit()
    except Exception as exc:
        logger.warning("Mise à jour TaskRun échouée (non bloquant)", id=task_run_id, error=str(exc))


async def _get_redis_logs(redis, job_id: str) -> str:
    """Récupère tous les logs Redis pour un job et retourne un texte."""
    try:
        lines = await redis.lrange(f"retrain_logs:{job_id}", 0, -1)
        return "\n".join(
            line.decode("utf-8") if isinstance(line, bytes) else line for line in lines
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tâche 1 : retrain manuel (enqueué par POST /models/{name}/{version}/retrain)
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
    """Exécute le pipeline de ré-entraînement, met à jour TaskRun en DB."""
    from src.services.retrain_service import do_retrain

    redis = ctx.get("redis")
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Marquer comme running
    await _update_task_run(job_id, status="running", started_at=now)
    logger.info(
        "retrain_task démarré",
        job_id=job_id,
        model=model_name,
        source_version=source_version,
        new_version=new_version,
    )

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
        logger.error("retrain_task exception inattendue", job_id=job_id, error=str(exc))
        result = {"success": False, "error": str(exc), "stdout": "", "stderr": ""}

    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    status = "success" if result.get("success") else "failed"
    final_logs = await _get_redis_logs(redis, job_id) if redis else ""

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
        "retrain_task terminé",
        job_id=job_id,
        status=status,
        model=model_name,
        new_version=result.get("new_version"),
    )
    return result


# ---------------------------------------------------------------------------
# Tâche 2 : retrain planifié (déclenché par les cron jobs ARQ ci-dessous)
# ---------------------------------------------------------------------------


async def scheduled_retrain_task(
    ctx: Dict[str, Any],
    model_name: str,
    source_version: str,
) -> None:
    """Retrain planifié — acquiert un verrou Redis pour éviter les doublons."""
    from src.services.retrain_service import do_retrain

    redis = ctx.get("redis")
    lock_key = f"retrain_lock:{model_name}:{source_version}"

    # Verrou distribué : évite un double-run si un retrain manuel est en cours
    acquired = await redis.set(lock_key, "1", nx=True, ex=700)
    if not acquired:
        logger.warning(
            "Retrain planifié ignoré — verrou actif",
            model=model_name,
            version=source_version,
        )
        return

    job_id = str(uuid4())
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S")

    # Charger le schedule depuis la DB pour les paramètres
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
                    "Modèle source introuvable pour scheduled_retrain",
                    model=model_name,
                    version=source_version,
                )
                await redis.delete(lock_key)
                return

            schedule = src.retrain_schedule or {}
            if not schedule.get("enabled", True):
                logger.info("Schedule désactivé — job ignoré", model=model_name)
                await redis.delete(lock_key)
                return

            from datetime import timedelta

            lookback_days = int(schedule.get("lookback_days", 30))
            end_date = now.strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            new_version = f"{source_version}-sched-{timestamp}"

    except Exception as exc:
        logger.error("Erreur lecture DB scheduled_retrain", error=str(exc))
        await redis.delete(lock_key)
        return

    # Créer le TaskRun en DB pour le tracking
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
        logger.warning("Création TaskRun échouée (non bloquant)", error=str(exc))

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

    # Mise à jour finale du TaskRun
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


# ---------------------------------------------------------------------------
# Tâche 3 : vérification des alertes (toutes les 6 h)
# ---------------------------------------------------------------------------


async def alert_check_task(ctx: Dict[str, Any]) -> None:
    """Délègue à la logique existante de supervision_reporter."""
    from src.tasks.supervision_reporter import run_alert_check

    logger.info("alert_check_task démarré")
    await run_alert_check()
    logger.info("alert_check_task terminé")


# ---------------------------------------------------------------------------
# Tâche 4 : rapport hebdomadaire
# ---------------------------------------------------------------------------


async def weekly_report_task(ctx: Dict[str, Any]) -> None:
    """Délègue à la logique existante de supervision_reporter."""
    from src.tasks.supervision_reporter import run_weekly_report

    logger.info("weekly_report_task démarré")
    await run_weekly_report()
    logger.info("weekly_report_task terminé")


# ---------------------------------------------------------------------------
# Startup / Shutdown du worker
# ---------------------------------------------------------------------------


async def startup(ctx: Dict[str, Any]) -> None:
    """Initialise la DB et charge les crons retrain depuis la DB."""
    from src.core.config import settings
    from src.core.logging import setup_logging
    from src.db.database import init_db

    setup_logging(debug=settings.DEBUG)
    logger.info("ARQ worker démarré")

    try:
        await init_db()
        logger.info("DB connectée (worker)")
    except Exception as exc:
        logger.warning("DB connexion échouée au démarrage worker", error=str(exc))

    # Les cron jobs retrain sont chargés dynamiquement via _load_retrain_crons()
    # (appelé après startup par WorkerSettings.cron_jobs via la factory ci-dessous)


async def shutdown(ctx: Dict[str, Any]) -> None:
    """Ferme la DB."""
    from src.db.database import close_db

    await close_db()
    logger.info("ARQ worker arrêté")


# ---------------------------------------------------------------------------
# Chargement dynamique des cron jobs retrain depuis la DB
# ---------------------------------------------------------------------------


async def _build_retrain_cron_jobs():
    """
    Charge les modèles avec retrain_schedule.enabled=True et retourne
    une liste de cron jobs ARQ, un par version active.

    Appelé une seule fois au démarrage du worker (WorkerSettings.cron_jobs).
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

                # ARQ attend minute, hour, day, month, weekday comme entiers/sets
                # On utilise la représentation crontab directe (sched["cron"] ex: "0 3 * * 1")
                cron_jobs.append(
                    arq_cron(
                        scheduled_retrain_task,
                        name=f"retrain_schedule:{model.name}:{model.version}",
                        kwargs={
                            "model_name": model.name,
                            "source_version": model.version,
                        },
                        # ARQ supporte un paramètre `cron` natif
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Cron retrain ignoré (erreur)",
                    model=model.name,
                    version=model.version,
                    error=str(exc),
                )
    except Exception as exc:
        logger.warning("Chargement crons retrain échoué", error=str(exc))
    return cron_jobs


# ---------------------------------------------------------------------------
# WorkerSettings
# ---------------------------------------------------------------------------


class WorkerSettings:
    """Configuration du worker ARQ.

    Démarrage :
        python -m arq src.tasks.arq_worker.WorkerSettings
    """

    functions = [
        retrain_task,
        scheduled_retrain_task,
        alert_check_task,
        weekly_report_task,
    ]

    # Cron jobs fixes (alertes + rapport hebdo)
    # Les crons retrain dynamiques sont ajoutés au démarrage via on_startup
    cron_jobs = [
        # Vérification alertes toutes les 6 h (0h, 6h, 12h, 18h UTC)
        cron(alert_check_task, hour={0, 6, 12, 18}, minute=0, run_at_startup=False),
        # Rapport hebdomadaire — lundi 8h UTC
        cron(weekly_report_task, weekday=0, hour=8, minute=0, run_at_startup=False),
    ]

    on_startup = startup
    on_shutdown = shutdown

    # Concurrence max : 5 jobs simultanés (le retrain lourd prend 1 slot)
    max_jobs = 5

    # Timeout global d'un job (légèrement > timeout subprocess 600 s)
    job_timeout = 720

    # Grace time pour les jobs manqués (ex: worker redémarré)
    keep_result = 3600  # conserver le résultat 1 h dans Redis

    @classmethod
    def redis_settings(cls) -> RedisSettings:
        from src.core.config import settings

        return RedisSettings.from_dsn(settings.REDIS_URL)
