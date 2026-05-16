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
import csv
import json
import os
import resource
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


_SAFE_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV",
}


def _set_subprocess_limits() -> None:
    """Appelé dans le processus enfant après fork — réduit la surface d'attaque.

    RLIMIT_AS : on ne limite pas la mémoire virtuelle — mlflow + numpy + sklearn + boto3
    nécessitent plus de 2 GB d'espace d'adressage virtuel (shared libs + compilations regex
    de email._header_value_parser en Python 3.13). Un plafond trop bas provoque MemoryError
    à l'import, bien avant que le script d'entraînement lui-même s'exécute.
    La protection mémoire est assurée par les cgroups Docker du container.

    RLIMIT_NOFILE : 1024 minimum requis — importlib_metadata ouvre les métadonnées de
    tous les paquets installés à l'import de mlflow (200+ fd simultanément).
    """
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _fd_limit = min(1024, _hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_fd_limit, _fd_limit))


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

    from src.core.config import settings
    from src.core.ml_metrics import retrain_total
    from src.db.database import AsyncSessionLocal
    from src.db.models import HistoryActionType, ModelMetadata
    from src.services.auto_promotion_service import evaluate_auto_promotion
    from src.services.db_service import DBService
    from src.services.minio_service import minio_service
    from src.services.mlflow_service import mlflow_service
    from src.services.model_service import compute_model_hmac, model_service

    logger.info("Démarrage du ré-entraînement planifié", model=name, version=version)
    structlog.contextvars.bind_contextvars(event_type="retrain", model_name=name)

    # --- Session 1 : lecture pré-subprocess ---
    # Fermer la connexion DB avant le subprocess (600 s) pour libérer le pool.
    async with AsyncSessionLocal() as db:
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

        # Extraire tous les champs nécessaires en mémoire avant de fermer la session
        source_fields = {
            "train_script_object_key": source_model.train_script_object_key,
            "description": source_model.description,
            "algorithm": source_model.algorithm,
            "features_count": source_model.features_count,
            "classes": source_model.classes,
            "training_params": source_model.training_params,
            "training_dataset": source_model.training_dataset,
            "feature_baseline": source_model.feature_baseline,
            "confidence_threshold": source_model.confidence_threshold,
            "tags": source_model.tags,
            "webhook_url": source_model.webhook_url,
            "promotion_policy": source_model.promotion_policy,
            "accuracy": source_model.accuracy,
            "f1_score": source_model.f1_score,
        }

        # 2. Calculer la plage de dates (ici pour pouvoir exporter avant de fermer la session)
        lookback_days = int(schedule.get("lookback_days", 30))
        now_utc = datetime.now(timezone.utc)
        end_date = now_utc.strftime("%Y-%m-%d")
        start_date = (now_utc - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Exporter les données de production avant de libérer la connexion
        retrain_rows = await DBService.export_retrain_data(db, name, start_date, end_date)
    # ← connexion DB libérée ici, avant le subprocess de 600 s

    # 3. Télécharger train.py depuis MinIO
    try:
        script_bytes = await minio_service.async_download_file_bytes(
            source_fields["train_script_object_key"]
        )
    except Exception as exc:
        logger.error(
            "Téléchargement train.py échoué",
            model=name,
            version=version,
            error=str(exc),
        )
        retrain_total.labels(model_name=name, status="failure").inc()
        return

    # 4. Exécuter le subprocess (timeout 600 s — identique à l'endpoint manuel)
    timestamp = now_utc.strftime("%Y%m%d%H%M%S")
    new_version = f"{version}-sched-{timestamp}"
    mlflow_uri = settings.MLFLOW_TRACKING_URI
    stdout_text = ""
    stderr_text = ""
    new_model_bytes: Optional[bytes] = None

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "train.py")
        output_model_path = os.path.join(tmpdir, "output_model.joblib")

        with open(script_path, "wb") as f:
            f.write(script_bytes)

        # Écrire le CSV de données de production si disponible
        train_data_path = None
        if retrain_rows:
            train_data_path = os.path.join(tmpdir, "train_data.csv")
            with open(train_data_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "id_obs",
                    "input_features",
                    "prediction_result",
                    "observed_result",
                    "timestamp",
                    "model_version",
                    "response_time_ms",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in retrain_rows:
                    writer.writerow(
                        {
                            "id_obs": row["id_obs"],
                            "input_features": json.dumps(row["input_features"]),
                            "prediction_result": json.dumps(row["prediction_result"]),
                            "observed_result": json.dumps(row["observed_result"])
                            if row["observed_result"] is not None
                            else "",
                            "timestamp": row["timestamp"],
                            "model_version": row["model_version"],
                            "response_time_ms": row["response_time_ms"],
                        }
                    )
            logger.info(
                "Données de production exportées pour le retrain planifié",
                model=name,
                rows=len(retrain_rows),
            )

        env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
        env.update(
            {
                "TRAIN_START_DATE": start_date,
                "TRAIN_END_DATE": end_date,
                "OUTPUT_MODEL_PATH": output_model_path,
                "MLFLOW_TRACKING_URI": mlflow_uri,
                "MODEL_NAME": name,
            }
        )
        if train_data_path:
            env["TRAIN_DATA_PATH"] = train_data_path

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                script_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                preexec_fn=_set_subprocess_limits,
            )
            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(proc.communicate(), timeout=600.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.error(
                    "Timeout ré-entraînement planifié (600 s)",
                    model=name,
                    version=version,
                )
                retrain_total.labels(model_name=name, status="failure").inc()
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
                retrain_total.labels(model_name=name, status="failure").inc()
                return

            if not os.path.exists(output_model_path):
                logger.error(
                    "Fichier modèle absent après l'exécution du script",
                    model=name,
                    version=version,
                )
                retrain_total.labels(model_name=name, status="failure").inc()
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
            retrain_total.labels(model_name=name, status="failure").inc()
            return

    # 5. Extraire les métriques depuis la dernière ligne JSON de stdout
    new_accuracy = source_fields["accuracy"]
    new_f1 = source_fields["f1_score"]
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

    # 5b. Snapshot des versions de librairies (généré ici, uploadé MinIO en step 6)
    # Priorité : "dependencies" dans le stdout JSON (auto-reporté par le script),
    # fallback sur AST + importlib.metadata si absent.
    from src.services.env_snapshot_service import (
        dependencies_to_requirements_txt as _deps_to_req,
        generate_requirements_txt as _gen_req,
    )

    _deps_from_stdout = parsed_metrics.get("dependencies")
    if _deps_from_stdout and isinstance(_deps_from_stdout, dict):
        _req_txt = _deps_to_req(_deps_from_stdout)
    else:
        _req_txt = _gen_req(script_bytes.decode("utf-8", errors="replace"))

    # 5c. Créer le run MLflow (dégradation gracieuse si MLflow indisponible)
    _mlflow_run_id: Optional[str] = None
    try:
        _mlflow_run_id = mlflow_service.log_retrain_run(
            model_name=name,
            new_version=new_version,
            source_version=version,
            trigger="scheduler",
            trained_by="scheduler",
            train_start_date=start_date,
            train_end_date=end_date,
            accuracy=new_accuracy,
            f1_score=new_f1,
            n_rows=parsed_metrics.get("n_rows"),
            feature_stats=parsed_metrics.get("feature_stats"),
            label_distribution=parsed_metrics.get("label_distribution"),
            algorithm=source_fields["algorithm"],
            training_params=source_fields["training_params"],
            auto_promoted=False,
            auto_promote_reason=None,
            model_bytes=new_model_bytes,
            lookback_days=lookback_days,
            requirements_txt=_req_txt,
        )
    except Exception as _exc:
        logger.warning("MLflow logging échoué (scheduler)", error=str(_exc))

    # 6. Uploader le nouveau modèle, le script et le requirements.txt dans MinIO
    new_object_key = f"{name}/v{new_version}.joblib"
    new_model_hmac_signature = compute_model_hmac(new_model_bytes)
    upload_info = await minio_service.async_upload_model_bytes(new_model_bytes, new_object_key)
    new_train_key = f"{name}/v{new_version}_train.py"
    await minio_service.async_upload_file_bytes(
        script_bytes, new_train_key, content_type="text/x-python"
    )
    _req_object_key: Optional[str] = f"{name}/v{new_version}_requirements.txt"
    try:
        await minio_service.async_upload_file_bytes(
            _req_txt.encode("utf-8"), _req_object_key, content_type="text/plain"
        )
    except Exception as _exc:
        logger.warning("Upload requirements.txt échoué (scheduler) — non bloquant", error=str(_exc))
        _req_object_key = None

    # --- Session 2 : écriture post-subprocess ---
    now_naive = now_utc.replace(tzinfo=None)
    async with AsyncSessionLocal() as db:
        # 7. Créer la nouvelle entrée ModelMetadata
        new_metadata = ModelMetadata(
            name=name,
            version=new_version,
            minio_bucket=upload_info.get("bucket"),
            minio_object_key=new_object_key,
            file_size_bytes=upload_info.get("size"),
            model_hmac_signature=new_model_hmac_signature,
            train_script_object_key=new_train_key,
            requirements_object_key=_req_object_key,
            description=source_fields["description"],
            algorithm=source_fields["algorithm"],
            mlflow_run_id=_mlflow_run_id,
            accuracy=new_accuracy,
            f1_score=new_f1,
            features_count=source_fields["features_count"],
            classes=source_fields["classes"],
            training_params=source_fields["training_params"],
            training_dataset=(
                f"{source_fields['training_dataset'] or name} [{start_date} → {end_date}]"
            ),
            feature_baseline=source_fields["feature_baseline"],
            confidence_threshold=source_fields["confidence_threshold"],
            tags=source_fields["tags"],
            webhook_url=source_fields["webhook_url"],
            promotion_policy=source_fields["promotion_policy"],
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
        if schedule.get("auto_promote") and source_fields["promotion_policy"]:
            should_promote, reason = await evaluate_auto_promotion(
                db, name, source_fields["promotion_policy"]
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
        if schedule.get("auto_promote") and source_fields["promotion_policy"]:
            new_metadata.training_stats = {
                **(new_metadata.training_stats or {}),
                "auto_promoted": _auto_promoted,
                "auto_promote_reason": _auto_promote_reason,
            }

        # Mettre à jour les tags MLflow avec le résultat final de promotion
        if _mlflow_run_id:
            try:
                mlflow_service.update_run_tags(
                    _mlflow_run_id,
                    {
                        "auto_promoted": str(_auto_promoted),
                        "auto_promote_reason": _auto_promote_reason or "",
                        "is_production": str(new_metadata.is_production),
                    },
                )
            except Exception as _exc:
                logger.warning("MLflow tag update échoué (scheduler)", error=str(_exc))

        # 9. Mettre à jour last_run_at et next_run_at sur le modèle source
        cron_expr = schedule.get("cron", "")
        next_run = _compute_next_run_at(cron_expr) if cron_expr else None
        source_update_result = await db.execute(
            select(ModelMetadata).where(
                and_(ModelMetadata.name == name, ModelMetadata.version == version)
            )
        )
        source_model_for_update = source_update_result.scalar_one_or_none()
        if source_model_for_update:
            source_model_for_update.retrain_schedule = {
                **schedule,
                "last_run_at": now_naive.isoformat(),
                "next_run_at": next_run.isoformat() if next_run else None,
            }

        await db.commit()

        # 10. Invalider le cache Redis du modèle (forcer le rechargement de la nouvelle version)
        await model_service.invalidate_model_cache(name)

        # 11. Pré-chauffer le cache pour la nouvelle version si promue en production
        if _auto_promoted or new_metadata.is_production:
            try:
                async with AsyncSessionLocal() as warmup_db:
                    await model_service.load_model(warmup_db, name, new_version)
                logger.info(
                    "Cache pré-chauffé pour la nouvelle version",
                    model=name,
                    new_version=new_version,
                )
            except Exception as _exc:
                logger.warning("Pré-chauffe cache échouée (non bloquant)", error=str(_exc))

        _wh = source_fields["webhook_url"]
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

        retrain_total.labels(model_name=name, status="success").inc()
        logger.info(
            "Ré-entraînement planifié terminé",
            model=name,
            source_version=version,
            new_version=new_version,
            stdout_lines=len(stdout_text.splitlines()),
        )

    structlog.contextvars.clear_contextvars()


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
