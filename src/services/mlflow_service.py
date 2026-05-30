"""
MLflow integration service.

All public methods wrap MLflow calls in a try/except and return None instead of
raising an exception, so as to never block a retraining if the MLflow server is
unavailable (graceful degradation).

Storage principle:
  - MinIO (bucket "models") is the operational source of truth for binaries.
  - MLflow stores only metadata (metrics, params, lineage) and a link
    to the MinIO path — never a copy of the binary.
"""

import os
from datetime import datetime, timezone
from typing import Optional

import mlflow
import structlog
from mlflow.tracking import MlflowClient

from src.core.config import settings

logger = structlog.get_logger(__name__)


class MLflowService:
    def _configure(self) -> bool:
        """Configure the MLflow URI and MinIO S3 environment variables."""
        if not settings.MLFLOW_ENABLE:
            return False
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            if settings.MLFLOW_S3_ENDPOINT_URL:
                os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.MLFLOW_S3_ENDPOINT_URL)
                os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.MINIO_ACCESS_KEY)
                os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.MINIO_SECRET_KEY)
            return True
        except Exception as exc:
            logger.warning("MLflow configuration failed", error=str(exc))
            return False

    def _experiment_name(self, model_name: str) -> str:
        return f"{settings.MLFLOW_EXPERIMENT_PREFIX}/{model_name}"

    def log_retrain_run(
        self,
        *,
        model_name: str,
        new_version: str,
        source_version: str,
        trigger: str,
        trained_by: str,
        train_start_date: str,
        train_end_date: str,
        accuracy: Optional[float],
        auc: Optional[float] = None,
        f1_score: Optional[float] = None,
        n_rows: Optional[int] = None,
        feature_stats: Optional[dict],
        label_distribution: Optional[dict],
        algorithm: Optional[str],
        training_params: Optional[dict],
        hyperparameters: Optional[dict] = None,
        feature_importances: Optional[dict] = None,
        auto_promoted: bool = False,
        auto_promote_reason: Optional[str] = None,
        minio_object_key: Optional[str] = None,
        minio_bucket: str = "models",
        lookback_days: Optional[int] = None,
    ) -> Optional[str]:
        """
        Create an MLflow run for a retraining.

        MinIO is the source of truth for the binary; MLflow stores only
        metrics, parameters and lineage (+ a link to the MinIO path).

        Returns the run_id if successful, None if MLflow is unavailable.
        """
        if not self._configure():
            return None
        try:
            mlflow.set_experiment(self._experiment_name(model_name))

            # Enriched run name: model + version + date + trigger
            _date = datetime.now(timezone.utc).strftime("%Y%m%d")
            run_name = f"{model_name}_v{new_version}_{_date}_{trigger}"

            with mlflow.start_run(run_name=run_name) as run:
                # Params — lineage and configuration information
                params: dict = {
                    "model_name": model_name,
                    "new_version": new_version,
                    "source_version": source_version,
                    "trigger": trigger,
                    "trained_by": trained_by,
                    "train_start_date": train_start_date,
                    "train_end_date": train_end_date,
                }
                if algorithm:
                    params["algorithm"] = algorithm
                if lookback_days is not None:
                    params["lookback_days"] = str(lookback_days)
                # MinIO link — source of truth for the binary
                if minio_object_key:
                    params["minio_bucket"] = minio_bucket
                    params["minio_object_key"] = minio_object_key
                if training_params:
                    for k, v in training_params.items():
                        params[f"param_{k}"] = str(v)
                if hyperparameters:
                    for k, v in hyperparameters.items():
                        params[f"hparam_{k}"] = str(v)
                mlflow.log_params(params)

                # Scalar metrics
                if accuracy is not None:
                    mlflow.log_metric("accuracy", accuracy)
                if auc is not None:
                    mlflow.log_metric("auc", auc)
                if f1_score is not None:
                    mlflow.log_metric("f1_score", f1_score)
                if n_rows is not None:
                    mlflow.log_metric("n_rows_train", float(n_rows))

                # Feature stats — one metric per feature × statistic
                if feature_stats:
                    for feat_name, stats_dict in feature_stats.items():
                        if not isinstance(stats_dict, dict):
                            continue
                        safe = feat_name.replace(" ", "_")[:40]
                        for stat_key in ("mean", "std", "min", "max", "null_rate"):
                            val = stats_dict.get(stat_key)
                            if val is not None:
                                try:
                                    mlflow.log_metric(f"feat_{safe}_{stat_key}", float(val))
                                except Exception:
                                    pass

                # Label distribution — ratio per class
                if label_distribution:
                    total = sum(float(v) for v in label_distribution.values() if v is not None)
                    for label, count in label_distribution.items():
                        if count is None:
                            continue
                        ratio = float(count) / total if total > 0 else 0.0
                        safe_label = str(label).replace(" ", "_")[:30]
                        try:
                            mlflow.log_metric(f"label_{safe_label}_ratio", ratio)
                        except Exception:
                            pass

                # Feature importances — one metric per feature
                if feature_importances:
                    for feat_name, importance in feature_importances.items():
                        safe = feat_name.replace(" ", "_")[:50]
                        try:
                            mlflow.log_metric(f"fi_{safe}", float(importance))
                        except Exception:
                            pass

                # Tags — for filtering in MLflow UI
                mlflow.set_tags(
                    {
                        "run_type": "training",
                        "trigger": trigger,
                        "auto_promoted": str(auto_promoted),
                        "auto_promote_reason": auto_promote_reason or "",
                    }
                )

                run_id = run.info.run_id

            logger.info(
                "MLflow run created",
                model=model_name,
                version=new_version,
                run_id=run_id,
                run_name=run_name,
                trigger=trigger,
            )
            return run_id

        except Exception as exc:
            logger.warning(
                "MLflow log_retrain_run failed — graceful degradation",
                model=model_name,
                version=new_version,
                error=str(exc),
            )
            return None

    def update_run_tags(self, run_id: str, tags: dict) -> bool:
        """Update tags on an existing run (e.g. auto_promoted after promotion)."""
        if not self._configure():
            return False
        try:
            client = MlflowClient()
            for key, value in tags.items():
                client.set_tag(run_id, key, str(value))
            return True
        except Exception as exc:
            logger.warning("MLflow update_run_tags failed", run_id=run_id, error=str(exc))
            return False

    def log_production_snapshot(
        self,
        *,
        model_name: str,
        version: str,
        metrics: dict,
    ) -> Optional[str]:
        """
        Log a production metrics snapshot.

        Uses the same experiment as training (predictml/{model_name})
        with the tag run_type=monitoring to distinguish the two run types.
        """
        if not self._configure():
            return None
        try:
            # Same experiment as training — distinguished by the run_type tag
            mlflow.set_experiment(self._experiment_name(model_name))
            _date = datetime.now(timezone.utc).strftime("%Y%m%d")
            run_name = f"{model_name}_v{version}_monitoring_{_date}"

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tags(
                    {
                        "run_type": "monitoring",
                        "model_name": model_name,
                        "version": version,
                    }
                )
                for key, value in metrics.items():
                    if value is not None:
                        try:
                            mlflow.log_metric(key, float(value))
                        except Exception:
                            pass
                return run.info.run_id

        except Exception as exc:
            logger.warning(
                "MLflow log_production_snapshot failed",
                model=model_name,
                version=version,
                error=str(exc),
            )
            return None

    def delete_run(self, run_id: str) -> bool:
        """Delete an MLflow run. Returns False if MLflow is unavailable."""
        if not self._configure():
            return False
        try:
            MlflowClient().delete_run(run_id)
            logger.info("MLflow run deleted", run_id=run_id)
            return True
        except Exception as exc:
            logger.warning("MLflow delete_run failed", run_id=run_id, error=str(exc))
            return False


mlflow_service = MLflowService()
