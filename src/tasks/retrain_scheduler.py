"""
Scheduler de ré-entraînement automatique basé sur des expressions cron.

Un job APScheduler est créé par version de modèle ayant un ``retrain_schedule``
actif en base. Au démarrage de l'application, tous les schedules actifs sont
chargés depuis la DB.

Chaque job :
1. Acquiert un verrou Redis (SET NX EX 700) pour éviter les exécutions concurrentes
   en environnement multi-réplicas.
2. Télécharge le script ``train.py`` depuis MinIO.
3. Exécute le script dans un sous-processus (timeout 600 s).
4. Upload le ``.pkl`` produit et crée une nouvelle version en base.
5. Évalue l'auto-promotion si ``schedule.auto_promote=True``.
6. Met à jour ``last_run_at`` et ``next_run_at`` sur la version source.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = structlog.get_logger(__name__)

_retrain_scheduler = AsyncIOScheduler(timezone="UTC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_next_run_at(cron: str) -> Optional[datetime]:
    """Retourne le prochain déclenchement (naive UTC) pour une expression cron."""
    try:
        trigger = CronTrigger.from_crontab(cron, timezone="UTC")
        now = datetime.now(timezone.utc)
        next_fire = trigger.get_next_fire_time(None, now)
        if next_fire is None:
            return None
        return next_fire.replace(tzinfo=None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gestion des jobs
# ---------------------------------------------------------------------------


def add_retrain_job(name: str, version: str, schedule: dict) -> None:
    """Ajoute ou remplace un job cron dans le scheduler."""
    if not schedule.get("enabled", True):
        return
    cron = schedule.get("cron")
    if not cron:
        return
    job_id = f"retrain_schedule:{name}:{version}"
    trigger = CronTrigger.from_crontab(cron, timezone="UTC")
    _retrain_scheduler.add_job(
        _run_retrain_job,
        trigger=trigger,
        id=job_id,
        args=[name, version],
        replace_existing=True,
        misfire_grace_time=300,
    )
    logger.info("Job de ré-entraînement planifié", model=name, version=version, cron=cron)


def remove_retrain_job(name: str, version: str) -> None:
    """Retire un job du scheduler — silencieux si absent."""
    job_id = f"retrain_schedule:{name}:{version}"
    try:
        _retrain_scheduler.remove_job(job_id)
        logger.info("Job de ré-entraînement supprimé", model=name, version=version)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Job principal
# ---------------------------------------------------------------------------


async def _run_retrain_job(name: str, version: str) -> None:
    """Point d'entrée APScheduler. Acquiert le verrou Redis puis délègue."""
    from src.services.model_service import model_service

    redis_client = await model_service._get_redis()
    lock_key = f"retrain_lock:{name}:{version}"
    acquired = await redis_client.set(lock_key, "1", nx=True, ex=700)
    if not acquired:
        logger.warning(
            "Job de ré-entraînement ignoré — verrou Redis actif",
            model=name,
            version=version,
        )
        return

    try:
        await _do_retrain(name, version)
    except Exception as exc:
        logger.error(
            "Erreur inattendue dans le job de ré-entraînement",
            model=name,
            version=version,
            error=str(exc),
        )
    finally:
        await redis_client.delete(lock_key)


async def _do_retrain(name: str, version: str) -> None:
    """Logique core : télécharge train.py, exécute subprocess, upload, crée ModelMetadata."""
    # Imports lazy pour éviter les effets de bord à l'import et faciliter le mock en tests
    from sqlalchemy import and_, select

    from src.db.database import AsyncSessionLocal
    from src.db.models import HistoryActionType, ModelMetadata
    from src.services.auto_promotion_service import evaluate_auto_promotion
    from src.services.db_service import DBService
    from src.services.minio_service import minio_service

    logger.info("Démarrage du ré-entraînement planifié", model=name, version=version)

    async with AsyncSessionLocal() as db:
        # 1. Charger le modèle source
        result = await db.execute(
            select(ModelMetadata).where(
                and_(ModelMetadata.name == name, ModelMetadata.version == version)
            )
        )
        source_model = result.scalar_one_or_none()
        if not source_model:
            logger.error("Modèle source introuvable", model=name, version=version)
            return

        schedule = source_model.retrain_schedule or {}
        if not schedule.get("enabled", True):
            logger.info("Schedule désactivé, job ignoré", model=name, version=version)
            return

        if not source_model.train_script_object_key:
            logger.error(
                "Pas de train_script_object_key — retrain impossible",
                model=name,
                version=version,
            )
            return

        # 2. Calculer la plage de dates
        lookback_days = int(schedule.get("lookback_days", 30))
        now_utc = datetime.now(timezone.utc)
        end_date = now_utc.strftime("%Y-%m-%d")
        start_date = (now_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # 3. Télécharger train.py depuis MinIO
        try:
            script_bytes = minio_service.download_file_bytes(source_model.train_script_object_key)
        except Exception as exc:
            logger.error(
                "Téléchargement train.py échoué",
                model=name,
                version=version,
                error=str(exc),
            )
            return

        # 4. Exécuter le subprocess (timeout 600 s — identique à l'endpoint manuel)
        timestamp = now_utc.strftime("%Y%m%d%H%M%S")
        new_version = f"{version}-sched-{timestamp}"
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        stdout_text = ""
        stderr_text = ""
        new_model_bytes: Optional[bytes] = None

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "train.py")
            output_model_path = os.path.join(tmpdir, "output_model.pkl")

            with open(script_path, "wb") as f:
                f.write(script_bytes)

            env = {
                **os.environ,
                "TRAIN_START_DATE": start_date,
                "TRAIN_END_DATE": end_date,
                "OUTPUT_MODEL_PATH": output_model_path,
                "MLFLOW_TRACKING_URI": mlflow_uri,
                "MODEL_NAME": name,
            }

            try:
                proc = await asyncio.create_subprocess_exec(
                    "python",
                    script_path,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmpdir,
                )
                try:
                    raw_stdout, raw_stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=600.0
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    logger.error(
                        "Timeout ré-entraînement planifié (600 s)",
                        model=name,
                        version=version,
                    )
                    return

                stdout_text = raw_stdout.decode("utf-8", errors="replace")
                stderr_text = raw_stderr.decode("utf-8", errors="replace")

                if proc.returncode != 0:
                    logger.error(
                        "Script de ré-entraînement échoué",
                        model=name,
                        returncode=proc.returncode,
                        stderr=stderr_text[:500],
                    )
                    return

                if not os.path.exists(output_model_path):
                    logger.error(
                        "Fichier modèle absent après l'exécution du script",
                        model=name,
                        version=version,
                    )
                    return

                with open(output_model_path, "rb") as f:
                    new_model_bytes = f.read()

            except Exception as exc:
                logger.error(
                    "Erreur lors de l'exécution du subprocess",
                    model=name,
                    version=version,
                    error=str(exc),
                )
                return

        # 5. Extraire les métriques depuis la dernière ligne JSON de stdout
        new_accuracy = source_model.accuracy
        new_f1 = source_model.f1_score
        parsed_metrics: dict = {}
        for line in reversed(stdout_text.strip().splitlines()):
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed_metrics = json.loads(stripped)
                    new_accuracy = parsed_metrics.get("accuracy", new_accuracy)
                    new_f1 = parsed_metrics.get("f1_score", new_f1)
                except json.JSONDecodeError:
                    pass
                break

        training_stats = {
            "train_start_date": start_date,
            "train_end_date": end_date,
            "trained_at": now_utc.isoformat(),
            "n_rows": parsed_metrics.get("n_rows"),
            "feature_stats": parsed_metrics.get("feature_stats"),
            "label_distribution": parsed_metrics.get("label_distribution"),
        }

        # 6. Uploader le nouveau modèle et le script dans MinIO
        new_object_key = f"{name}/v{new_version}.pkl"
        upload_info = minio_service.upload_model_bytes(new_model_bytes, new_object_key)
        new_train_key = f"{name}/v{new_version}_train.py"
        minio_service.upload_file_bytes(script_bytes, new_train_key, content_type="text/x-python")

        # 7. Créer la nouvelle entrée ModelMetadata
        now_naive = now_utc.replace(tzinfo=None)
        new_metadata = ModelMetadata(
            name=name,
            version=new_version,
            minio_bucket=upload_info.get("bucket"),
            minio_object_key=new_object_key,
            file_size_bytes=upload_info.get("size"),
            train_script_object_key=new_train_key,
            description=source_model.description,
            algorithm=source_model.algorithm,
            mlflow_run_id=None,
            accuracy=new_accuracy,
            f1_score=new_f1,
            features_count=source_model.features_count,
            classes=source_model.classes,
            training_params=source_model.training_params,
            training_dataset=(
                f"{source_model.training_dataset or name} [{start_date} → {end_date}]"
            ),
            feature_baseline=source_model.feature_baseline,
            confidence_threshold=source_model.confidence_threshold,
            tags=source_model.tags,
            webhook_url=source_model.webhook_url,
            promotion_policy=source_model.promotion_policy,
            trained_by="scheduler",
            training_date=now_naive,
            user_id_creator=None,
            is_active=True,
            is_production=False,
            parent_version=version,
            training_stats=training_stats,
        )
        db.add(new_metadata)
        await db.flush()

        await DBService.log_model_history(
            db, new_metadata, HistoryActionType.CREATED, None, "scheduler"
        )

        # 8. Auto-promotion si activée et policy définie
        _auto_promoted = False
        _auto_promote_reason: Optional[str] = None
        if schedule.get("auto_promote") and source_model.promotion_policy:
            should_promote, reason = await evaluate_auto_promotion(
                db, name, source_model.promotion_policy
            )
            _auto_promoted = should_promote
            _auto_promote_reason = reason
            if should_promote:
                other_result = await db.execute(
                    select(ModelMetadata).where(
                        and_(
                            ModelMetadata.name == name,
                            ModelMetadata.version != new_version,
                            ModelMetadata.is_production.is_(True),
                        )
                    )
                )
                for other in other_result.scalars().all():
                    other.is_production = False
                new_metadata.is_production = True
                await db.flush()
            logger.info(
                "Auto-promotion évaluée",
                model=name,
                new_version=new_version,
                promoted=should_promote,
                reason=reason,
            )

        # Persist auto_promoted outcome in training_stats for retrain-history
        if schedule.get("auto_promote") and source_model.promotion_policy:
            new_metadata.training_stats = {
                **(new_metadata.training_stats or {}),
                "auto_promoted": _auto_promoted,
                "auto_promote_reason": _auto_promote_reason,
            }

        # 9. Mettre à jour last_run_at et next_run_at sur le modèle source
        cron_expr = schedule.get("cron", "")
        next_run = _compute_next_run_at(cron_expr) if cron_expr else None
        source_model.retrain_schedule = {
            **schedule,
            "last_run_at": now_naive.isoformat(),
            "next_run_at": next_run.isoformat() if next_run else None,
        }

        await db.commit()

        _wh = new_metadata.webhook_url
        if _wh:
            _ts = datetime.utcnow().isoformat() + "Z"
            from src.services.webhook_service import send_webhook

            asyncio.create_task(
                send_webhook(
                    _wh,
                    {
                        "model_name": name,
                        "version": new_version,
                        "timestamp": _ts,
                        "details": {
                            "source_version": version,
                            "accuracy": new_metadata.accuracy,
                            "f1_score": new_metadata.f1_score,
                        },
                    },
                    event_type="retrain_completed",
                )
            )
            if _auto_promoted:
                asyncio.create_task(
                    send_webhook(
                        _wh,
                        {
                            "model_name": name,
                            "version": new_version,
                            "timestamp": _ts,
                            "details": {"reason": _auto_promote_reason},
                        },
                        event_type="model_promoted",
                    )
                )

        logger.info(
            "Ré-entraînement planifié terminé",
            model=name,
            source_version=version,
            new_version=new_version,
            stdout_lines=len(stdout_text.splitlines()),
        )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def start_retrain_scheduler() -> None:
    """Charge les schedules actifs depuis la DB et démarre le scheduler.

    Cette fonction est ``async`` (contrairement à ``start_scheduler`` du
    supervision_reporter) car elle effectue une requête DB.
    """
    from sqlalchemy import select

    from src.db.database import AsyncSessionLocal
    from src.db.models import ModelMetadata

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ModelMetadata).where(ModelMetadata.is_active.is_(True))
            )
            models = result.scalars().all()

        loaded = 0
        for model in models:
            sched = model.retrain_schedule
            if sched and sched.get("enabled") and sched.get("cron"):
                add_retrain_job(model.name, model.version, sched)
                loaded += 1

        _retrain_scheduler.start()
        logger.info("Scheduler de ré-entraînement démarré", jobs_loaded=loaded)

    except Exception as exc:
        logger.warning("Impossible de démarrer le scheduler de ré-entraînement", error=str(exc))


def stop_retrain_scheduler() -> None:
    """Arrête proprement le scheduler."""
    if _retrain_scheduler.running:
        _retrain_scheduler.shutdown(wait=False)
        logger.info("Scheduler de ré-entraînement arrêté")
