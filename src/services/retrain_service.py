"""
Retraining service shared between:
  - src/tasks/arq_worker.py  (retrain_task, scheduled_retrain_task)
  - src/api/models.py        (pre-enqueue validation only)

Key features:
  - Subprocess train.py execution with 600 s timeout
  - Line-by-line stdout streaming to Redis (key retrain_logs:{job_id})
  - Creation of the new ModelMetadata version in DB
  - Auto-promotion according to the configured PromotionPolicy
  - TaskRun updates (status, result, logs, timestamps)
"""

import asyncio
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from src.core.platform_limits import SUBPROCESS_PREEXEC_KWARGS

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Subprocess constants
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

_SUBPROCESS_TIMEOUT = 600.0  # seconds


# ---------------------------------------------------------------------------
# Subprocess streaming
# ---------------------------------------------------------------------------


async def _stream_stdout(
    proc: asyncio.subprocess.Process,
    job_id: Optional[str],
    redis_client,
) -> List[str]:
    """Read stdout line-by-line and push each line to Redis if job_id is provided.

    Returns:
        List of decoded lines (without trailing \\n).
    """
    lines: List[str] = []
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        decoded = raw.decode("utf-8", errors="replace").rstrip("\n\r")
        lines.append(decoded)
        if redis_client and job_id:
            key = f"retrain_logs:{job_id}"
            await redis_client.rpush(key, decoded)
            await redis_client.expire(key, 86400)  # TTL 24 h
    return lines


# ---------------------------------------------------------------------------
# Retrain core
# ---------------------------------------------------------------------------


async def do_retrain(  # noqa: C901 — complexity inherent to the pipeline
    *,
    # Identité
    model_name: str,
    source_version: str,
    new_version: str,
    # Paramètres d'entraînement
    start_date: str,  # "YYYY-MM-DD"
    end_date: str,
    set_production: bool = False,
    triggered_by: str = "unknown",
    trigger: str = "manual",  # "manual" | "scheduler" | "drift"
    # Champs source (extraits en mémoire avant d'appeler cette fonction)
    source_fields: Dict[str, Any] = None,
    # Suivi du job
    job_id: Optional[str] = None,
    redis_client=None,
    # Injection (fournie par le contexte ARQ ou les tests)
    db_session=None,  # AsyncSession — si None, une nouvelle session est créée
) -> Dict[str, Any]:
    """Execute the full retraining pipeline.

    Returns a dict with:
        success, stdout, stderr, new_version, accuracy, f1_score,
        auto_promoted, auto_promote_reason, training_stats, error
    """
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

    structlog.contextvars.bind_contextvars(
        event_type="retrain",
        model_name=model_name,
        source_version=source_version,
        job_id=job_id,
    )

    def _log(job_id, redis_client, line: str):
        """Push a log line to Redis synchronously (async wrap)."""
        # Used for API-level messages (outside the subprocess)
        pass  # API messages are in structlog, subprocess logs are in Redis

    # ------------------------------------------------------------------ #
    # 1. Load source fields if not provided
    # ------------------------------------------------------------------ #
    if source_fields is None:
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
                return {
                    "success": False,
                    "error": f"Source model '{model_name}' v{source_version} not found.",
                    "stdout": "",
                    "stderr": "",
                }
            source_fields = {
                "train_script_object_key": src.train_script_object_key,
                "description": src.description,
                "algorithm": src.algorithm,
                "features_count": src.features_count,
                "classes": src.classes,
                "model_task": src.model_task,
                "training_params": src.training_params,
                "hyperparameters": src.hyperparameters,
                "training_dataset": src.training_dataset,
                "feature_baseline": src.feature_baseline,
                "confidence_threshold": src.confidence_threshold,
                "tags": src.tags,
                "webhook_url": src.webhook_url,
                "promotion_policy": src.promotion_policy,
                "retrain_schedule": src.retrain_schedule,
                "accuracy": src.accuracy,
                "auc": src.auc,
                "f1_score": src.f1_score,
            }

    if not source_fields.get("train_script_object_key"):
        return {
            "success": False,
            "error": f"Model '{model_name}' v{source_version} has no train.py script.",
            "stdout": "",
            "stderr": "",
        }

    # ------------------------------------------------------------------ #
    # 2. Export production data (before the 600 s subprocess)
    # ------------------------------------------------------------------ #
    async with AsyncSessionLocal() as db:
        retrain_rows = await DBService.export_retrain_data(db, model_name, start_date, end_date)

    # ------------------------------------------------------------------ #
    # 3. Download train.py from MinIO
    # ------------------------------------------------------------------ #
    try:
        script_bytes = await minio_service.async_download_file_bytes(
            source_fields["train_script_object_key"]
        )
    except Exception as exc:
        logger.error(
            "train.py download failed",
            model=model_name,
            error=str(exc),
        )
        retrain_total.labels(model_name=model_name, status="failure").inc()
        return {
            "success": False,
            "error": f"train.py download failed: {exc}",
            "stdout": "",
            "stderr": "",
        }

    # ------------------------------------------------------------------ #
    # 4. Run the subprocess with stdout streaming → Redis
    # ------------------------------------------------------------------ #
    logger.info(
        "Starting train.py subprocess",
        model=model_name,
        new_version=new_version,
        timeout=_SUBPROCESS_TIMEOUT,
    )

    stdout_lines: List[str] = []
    stderr_text = ""
    new_model_bytes: Optional[bytes] = None

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "train.py")
        output_model_path = os.path.join(tmpdir, "output_model.joblib")

        with open(script_path, "wb") as f:
            f.write(script_bytes)

        # Write the production data CSV if available
        train_data_path: Optional[str] = None
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
                            "observed_result": (
                                json.dumps(row["observed_result"])
                                if row["observed_result"] is not None
                                else ""
                            ),
                            "timestamp": row["timestamp"],
                            "model_version": row["model_version"],
                            "response_time_ms": row["response_time_ms"],
                        }
                    )
            logger.info(
                "Production data exported",
                model=model_name,
                rows=len(retrain_rows),
            )

        # Minimal environment (no SECRET_KEY, DATABASE_URL, etc.)
        _minio_scheme = "https" if settings.MINIO_SECURE else "http"
        env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
        env.update(
            {
                "TRAIN_START_DATE": start_date,
                "TRAIN_END_DATE": end_date,
                "OUTPUT_MODEL_PATH": output_model_path,
                "MLFLOW_TRACKING_URI": settings.MLFLOW_TRACKING_URI,
                "MODEL_NAME": model_name,
                "MINIO_ENDPOINT": settings.MINIO_ENDPOINT,
                "MINIO_ACCESS_KEY": settings.MINIO_ACCESS_KEY,
                "MINIO_SECRET_KEY": settings.MINIO_SECRET_KEY,
                "MINIO_BUCKET": settings.MINIO_BUCKET,
                "MINIO_SECURE": str(settings.MINIO_SECURE).lower(),
                "MLFLOW_S3_ENDPOINT_URL": (
                    settings.MLFLOW_S3_ENDPOINT_URL
                    or f"{_minio_scheme}://{settings.MINIO_ENDPOINT}"
                ),
                "AWS_ACCESS_KEY_ID": settings.MINIO_ACCESS_KEY,
                "AWS_SECRET_ACCESS_KEY": settings.MINIO_SECRET_KEY,
            }
        )
        if train_data_path:
            env["TRAIN_DATA_PATH"] = train_data_path

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                **SUBPROCESS_PREEXEC_KWARGS,
            )

            try:
                stdout_lines, stderr_bytes = await asyncio.wait_for(
                    asyncio.gather(
                        _stream_stdout(proc, job_id, redis_client),
                        proc.stderr.read(),
                    ),
                    timeout=_SUBPROCESS_TIMEOUT,
                )
                await proc.wait()
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error(
                    "Retrain subprocess timeout",
                    model=model_name,
                    timeout=_SUBPROCESS_TIMEOUT,
                )
                retrain_total.labels(model_name=model_name, status="failure").inc()
                return {
                    "success": False,
                    "error": f"Timeout: script exceeded {int(_SUBPROCESS_TIMEOUT)} s.",
                    "stdout": "",
                    "stderr": "",
                }

            stdout_text = "\n".join(stdout_lines)

            if proc.returncode != 0:
                logger.error(
                    "Retrain script failed",
                    model=model_name,
                    returncode=proc.returncode,
                    stderr=stderr_text[:500],
                )
                retrain_total.labels(model_name=model_name, status="failure").inc()
                return {
                    "success": False,
                    "error": f"Script exited with code {proc.returncode}.",
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                }

            if not os.path.exists(output_model_path):
                logger.error(
                    "Script did not produce a .joblib",
                    model=model_name,
                    new_version=new_version,
                )
                retrain_total.labels(model_name=model_name, status="failure").inc()
                return {
                    "success": False,
                    "error": "Script did not produce a file at OUTPUT_MODEL_PATH.",
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                }

            with open(output_model_path, "rb") as f:
                new_model_bytes = f.read()

        except Exception as exc:
            logger.error(
                "Unexpected subprocess error",
                model=model_name,
                error=str(exc),
            )
            retrain_total.labels(model_name=model_name, status="failure").inc()
            return {
                "success": False,
                "error": f"Unexpected execution error: {exc}",
                "stdout": "\n".join(stdout_lines),
                "stderr": stderr_text,
            }

    stdout_text = "\n".join(stdout_lines)

    # ------------------------------------------------------------------ #
    # 5. Extract JSON metrics from the last stdout line
    # ------------------------------------------------------------------ #
    new_accuracy = source_fields.get("accuracy")
    new_auc = source_fields.get("auc")
    new_f1 = source_fields.get("f1_score")
    parsed_metrics: dict = {}
    for line in reversed(stdout_lines):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed_metrics = json.loads(stripped)
                new_accuracy = parsed_metrics.get("accuracy", new_accuracy)
                _parsed_auc = parsed_metrics.get("auc")
                if _parsed_auc is None:
                    _parsed_auc = parsed_metrics.get("roc_auc")
                if _parsed_auc is not None:
                    new_auc = _parsed_auc
                new_f1 = parsed_metrics.get("f1_score", new_f1)
            except json.JSONDecodeError:
                pass
            break

    training_stats = {
        "train_start_date": start_date,
        "train_end_date": end_date,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": parsed_metrics.get("n_rows"),
        "feature_stats": parsed_metrics.get("feature_stats"),
        "label_distribution": parsed_metrics.get("label_distribution"),
        "regression_bins": parsed_metrics.get("regression_bins"),
    }

    # ------------------------------------------------------------------ #
    # 5b. Requirements snapshot
    # ------------------------------------------------------------------ #
    from src.services.env_snapshot_service import (
        dependencies_to_requirements_txt as _deps_to_req,
    )
    from src.services.env_snapshot_service import (
        generate_requirements_txt as _gen_req,
    )

    _deps_from_stdout = parsed_metrics.get("dependencies")
    if _deps_from_stdout and isinstance(_deps_from_stdout, dict):
        _req_txt = _deps_to_req(_deps_from_stdout)
    else:
        _req_txt = _gen_req(script_bytes.decode("utf-8", errors="replace"))

    # ------------------------------------------------------------------ #
    # 5c. Extract feature importances from the newly produced model
    # (must happen before MLflow so the run is logged with complete data)
    # ------------------------------------------------------------------ #
    from src.api.models import _extract_feature_importances

    new_feature_importances: Optional[dict] = None
    try:
        new_feature_importances = _extract_feature_importances(new_model_bytes)
        if new_feature_importances:
            logger.info(
                "Feature importances extracted after retrain",
                model=model_name,
                version=new_version,
                n_features=len(new_feature_importances),
            )
    except Exception as _exc:
        logger.warning("Feature importances extraction failed (non-blocking)", error=str(_exc))

    # ------------------------------------------------------------------ #
    # 5d. MLflow run (graceful degradation)
    # The MinIO path is computed here (deterministic) to be passed to MLflow
    # before the actual upload, without duplicating the binary in the mlflow bucket.
    # ------------------------------------------------------------------ #
    now_utc = datetime.now(timezone.utc)
    new_object_key = f"{model_name}/{new_version}/model.joblib"
    _mlflow_run_id: Optional[str] = None
    try:
        _mlflow_run_id = mlflow_service.log_retrain_run(
            model_name=model_name,
            new_version=new_version,
            source_version=source_version,
            trigger=trigger,
            trained_by=triggered_by,
            train_start_date=start_date,
            train_end_date=end_date,
            accuracy=new_accuracy,
            auc=new_auc,
            f1_score=new_f1,
            n_rows=parsed_metrics.get("n_rows"),
            feature_stats=parsed_metrics.get("feature_stats"),
            label_distribution=parsed_metrics.get("label_distribution"),
            algorithm=source_fields.get("algorithm"),
            training_params=source_fields.get("training_params"),
            hyperparameters=(
                parsed_metrics.get("hyperparameters") or source_fields.get("hyperparameters")
            ),
            feature_importances=new_feature_importances,
            auto_promoted=False,
            auto_promote_reason=None,
            minio_object_key=new_object_key,
            minio_bucket="models",
        )
    except Exception as _exc:
        logger.warning("MLflow logging failed (non-blocking)", error=str(_exc))

    # ------------------------------------------------------------------ #
    # 6. MinIO upload: .joblib + train.py + requirements.txt
    # ------------------------------------------------------------------ #
    # new_object_key already computed in section 5c
    new_model_hmac_signature = compute_model_hmac(new_model_bytes)
    try:
        upload_info = await minio_service.async_upload_model_bytes(new_model_bytes, new_object_key)
    except Exception as exc:
        logger.error("MinIO .joblib upload failed", error=str(exc))
        retrain_total.labels(model_name=model_name, status="failure").inc()
        return {
            "success": False,
            "error": f"MinIO .joblib upload failed: {exc}",
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

    new_train_key = f"{model_name}/{new_version}/train.py"
    try:
        await minio_service.async_upload_file_bytes(
            script_bytes, new_train_key, content_type="text/x-python"
        )
    except Exception as exc:
        logger.error("MinIO train.py upload failed", error=str(exc))
        retrain_total.labels(model_name=model_name, status="failure").inc()
        return {
            "success": False,
            "error": f"MinIO train.py upload failed: {exc}",
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

    _req_object_key: Optional[str] = f"{model_name}/{new_version}/requirements.txt"
    try:
        await minio_service.async_upload_file_bytes(
            _req_txt.encode("utf-8"), _req_object_key, content_type="text/plain"
        )
    except Exception as _exc:
        logger.warning("requirements.txt upload failed (non-blocking)", error=str(_exc))
        _req_object_key = None

    # ------------------------------------------------------------------ #
    # 7. DB write: new version + history + auto-promotion
    # ------------------------------------------------------------------ #
    now_naive = now_utc.replace(tzinfo=None)
    auto_promoted = False
    auto_promote_reason: Optional[str] = None

    async with AsyncSessionLocal() as write_db:
        new_metadata = ModelMetadata(
            name=model_name,
            version=new_version,
            minio_bucket=upload_info.get("bucket"),
            minio_object_key=new_object_key,
            file_size_bytes=upload_info.get("size"),
            model_hmac_signature=new_model_hmac_signature,
            train_script_object_key=new_train_key,
            requirements_object_key=_req_object_key,
            description=source_fields.get("description"),
            algorithm=source_fields.get("algorithm"),
            mlflow_run_id=_mlflow_run_id,
            accuracy=new_accuracy,
            auc=new_auc,
            f1_score=new_f1,
            features_count=source_fields.get("features_count"),
            classes=source_fields.get("classes"),
            model_task=source_fields.get("model_task"),
            training_params=source_fields.get("training_params"),
            hyperparameters=(
                parsed_metrics.get("hyperparameters") or source_fields.get("hyperparameters")
            ),
            training_dataset=(
                parsed_metrics.get("training_dataset")
                or f"{source_fields.get('training_dataset') or model_name}"
                f" [{start_date} → {end_date}]"
            ),
            feature_baseline=source_fields.get("feature_baseline"),
            feature_importances=new_feature_importances,
            confidence_threshold=source_fields.get("confidence_threshold"),
            tags=source_fields.get("tags"),
            webhook_url=source_fields.get("webhook_url"),
            promotion_policy=source_fields.get("promotion_policy"),
            trained_by=triggered_by,
            training_date=now_naive,
            user_id_creator=None,  # scheduler — no user
            is_active=True,
            is_production=False,
            parent_version=source_version,
            training_stats=training_stats,
        )
        write_db.add(new_metadata)
        await write_db.flush()

        await DBService.log_model_history(
            write_db, new_metadata, HistoryActionType.CREATED, None, triggered_by
        )

        # Manual promotion or auto-promotion according to policy
        promotion_policy = source_fields.get("promotion_policy") or {}
        if set_production:
            # Explicit manual promotion
            other_result = await write_db.execute(
                select(ModelMetadata).where(
                    and_(
                        ModelMetadata.name == model_name,
                        ModelMetadata.version != new_version,
                        ModelMetadata.is_production.is_(True),
                    )
                )
            )
            for other in other_result.scalars().all():
                other.is_production = False
            new_metadata.is_production = True
            await write_db.flush()
        elif promotion_policy.get("auto_promote"):
            should_promote, reason = await evaluate_auto_promotion(
                write_db, model_name, promotion_policy, version=new_version
            )
            auto_promote_reason = reason
            if should_promote:
                other_result = await write_db.execute(
                    select(ModelMetadata).where(
                        and_(
                            ModelMetadata.name == model_name,
                            ModelMetadata.version != new_version,
                            ModelMetadata.is_production.is_(True),
                        )
                    )
                )
                for other in other_result.scalars().all():
                    other.is_production = False
                new_metadata.is_production = True
                auto_promoted = True
                await write_db.flush()
            logger.info(
                "Auto-promotion evaluated",
                model=model_name,
                new_version=new_version,
                promoted=should_promote,
                reason=reason,
            )
            if should_promote and settings.ENABLE_EMAIL_ALERTS:
                from src.services.email_service import email_service as _email_svc

                _email_svc.send_auto_promotion_alert(
                    model_name,
                    new_version,
                    reason,
                    accuracy=new_metadata.accuracy,
                    f1_score=new_metadata.f1_score,
                )

        # Persist auto_promoted in training_stats
        if promotion_policy.get("auto_promote") and not set_production:
            new_metadata.training_stats = {
                **(new_metadata.training_stats or {}),
                "auto_promoted": auto_promoted,
                "auto_promote_reason": auto_promote_reason,
            }

        # Update last_run_at / next_run_at for scheduled retrains
        if trigger == "scheduler":
            from croniter import croniter as _croniter

            sched = source_fields.get("retrain_schedule") or {}
            cron_expr = sched.get("cron", "")
            next_run = None
            if cron_expr and _croniter.is_valid(cron_expr):
                try:
                    it = _croniter(cron_expr, now_utc)
                    next_run = it.get_next(datetime).replace(tzinfo=None)
                except Exception:
                    pass

            src_result = await write_db.execute(
                select(ModelMetadata).where(
                    and_(
                        ModelMetadata.name == model_name,
                        ModelMetadata.version == source_version,
                    )
                )
            )
            src_for_update = src_result.scalar_one_or_none()
            if src_for_update:
                src_for_update.retrain_schedule = {
                    **sched,
                    "last_run_at": now_naive.isoformat(),
                    "next_run_at": next_run.isoformat() if next_run else None,
                }

        # Final MLflow tags
        if _mlflow_run_id:
            try:
                mlflow_service.update_run_tags(
                    _mlflow_run_id,
                    {
                        "auto_promoted": str(auto_promoted),
                        "auto_promote_reason": auto_promote_reason or "",
                        "is_production": str(new_metadata.is_production),
                    },
                )
            except Exception as _exc:
                logger.warning("MLflow tag update failed", error=str(_exc))

        await write_db.commit()
        await write_db.refresh(new_metadata)

    # ------------------------------------------------------------------ #
    # 8. Redis cache: invalidation + warm-up if in production
    # ------------------------------------------------------------------ #
    try:
        await model_service.invalidate_model_cache(model_name)
    except Exception as _exc:
        logger.warning("Cache invalidation failed (non-blocking)", error=str(_exc))

    if new_metadata.is_production:
        try:
            async with AsyncSessionLocal() as warmup_db:
                await model_service.load_model(warmup_db, model_name, new_version)
            logger.info("Cache warmed up", model=model_name, new_version=new_version)
        except Exception as _exc:
            logger.warning("Cache warm-up failed (non-blocking)", error=str(_exc))

    # ------------------------------------------------------------------ #
    # 9. Webhooks
    # ------------------------------------------------------------------ #

    _wh = source_fields.get("webhook_url")
    if _wh:
        import asyncio as _asyncio

        from src.services.webhook_service import send_webhook

        _ts = now_utc.isoformat()
        _asyncio.create_task(
            send_webhook(
                _wh,
                {
                    "model_name": model_name,
                    "version": new_version,
                    "timestamp": _ts,
                    "details": {
                        "source_version": source_version,
                        "accuracy": new_metadata.accuracy,
                        "f1_score": new_metadata.f1_score,
                    },
                },
                event_type="retrain_completed",
            )
        )
        if auto_promoted:
            _asyncio.create_task(
                send_webhook(
                    _wh,
                    {
                        "model_name": model_name,
                        "version": new_version,
                        "timestamp": _ts,
                        "details": {"reason": auto_promote_reason},
                    },
                    event_type="model_promoted",
                )
            )

    retrain_total.labels(model_name=model_name, status="success").inc()
    logger.info(
        "Retraining completed successfully",
        model=model_name,
        source_version=source_version,
        new_version=new_version,
    )
    structlog.contextvars.clear_contextvars()

    return {
        "success": True,
        "new_version": new_version,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "error": None,
        "accuracy": new_accuracy,
        "auc": new_auc,
        "f1_score": new_f1,
        "auto_promoted": (
            None
            if set_production  # manual promotion → auto-promotion not evaluated
            else (auto_promoted if promotion_policy.get("auto_promote") else None)
        ),
        "auto_promote_reason": auto_promote_reason,
        "training_stats": training_stats,
        "is_production": new_metadata.is_production,
    }
