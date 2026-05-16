"""
Service d'intégration MLflow.

Toutes les méthodes publiques encapsulent les appels MLflow dans un try/except et
retournent None au lieu de lever une exception, de façon à ne jamais bloquer un
ré-entraînement si le serveur MLflow est indisponible (dégradation gracieuse).
"""

import os
import pickle
import tempfile
from typing import Optional

import mlflow
import mlflow.sklearn
import structlog
from mlflow.tracking import MlflowClient

from src.core.config import settings

logger = structlog.get_logger(__name__)


class MLflowService:
    def _configure(self) -> bool:
        """Configure l'URI MLflow et les variables d'env S3 MinIO."""
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
            logger.warning("MLflow configuration échouée", error=str(exc))
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
        f1_score: Optional[float],
        n_rows: Optional[int],
        feature_stats: Optional[dict],
        label_distribution: Optional[dict],
        algorithm: Optional[str],
        training_params: Optional[dict],
        auto_promoted: bool,
        auto_promote_reason: Optional[str],
        model_bytes: Optional[bytes] = None,
        lookback_days: Optional[int] = None,
        requirements_txt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Crée un run MLflow complet pour un ré-entraînement.

        Retourne le run_id si succès, None si MLflow est indisponible.
        Le modèle sklearn est loggué comme artifact si model_bytes est fourni.
        La version est enregistrée dans le Model Registry si MLFLOW_REGISTER_MODELS=true.
        """
        if not self._configure():
            return None
        try:
            mlflow.set_experiment(self._experiment_name(model_name))

            with mlflow.start_run(run_name=f"{model_name}_v{new_version}") as run:
                # Params
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
                if training_params:
                    for k, v in training_params.items():
                        params[f"param_{k}"] = str(v)
                mlflow.log_params(params)

                # Metrics scalaires
                if accuracy is not None:
                    mlflow.log_metric("accuracy", accuracy)
                if f1_score is not None:
                    mlflow.log_metric("f1_score", f1_score)
                if n_rows is not None:
                    mlflow.log_metric("n_rows_train", float(n_rows))

                # Feature stats — une metric par feature × statistique
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

                # Label distribution — ratio par classe
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

                # Tags
                mlflow.set_tags(
                    {
                        "model_name": model_name,
                        "new_version": new_version,
                        "source_version": source_version,
                        "trigger": trigger,
                        "auto_promoted": str(auto_promoted),
                        "auto_promote_reason": auto_promote_reason or "",
                    }
                )

                # Artifact : log du modèle sklearn
                if model_bytes:
                    try:
                        model_obj = pickle.loads(model_bytes)  # noqa: S301
                        mlflow.sklearn.log_model(model_obj, artifact_path="model")
                    except Exception as exc:
                        logger.warning("MLflow log_model échoué — artifact ignoré", error=str(exc))

                # Artifact : requirements.txt (snapshot des versions de librairies)
                if requirements_txt:
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix="_requirements.txt", delete=False, encoding="utf-8"
                        ) as tmp:
                            tmp.write(requirements_txt)
                            _tmp_path = tmp.name
                        try:
                            mlflow.log_artifact(_tmp_path, artifact_path="environment")
                        finally:
                            os.unlink(_tmp_path)
                    except Exception as exc:
                        logger.warning("MLflow log requirements.txt échoué — artifact ignoré", error=str(exc))

                run_id = run.info.run_id

            # Enregistrement dans le Model Registry
            if settings.MLFLOW_REGISTER_MODELS:
                try:
                    mlflow.register_model(f"runs:/{run_id}/model", model_name)
                except Exception as exc:
                    logger.warning("MLflow register_model échoué — run_id conservé", error=str(exc))

            logger.info(
                "Run MLflow créé",
                model=model_name,
                version=new_version,
                run_id=run_id,
                trigger=trigger,
            )
            return run_id

        except Exception as exc:
            logger.warning(
                "MLflow log_retrain_run échoué — dégradation gracieuse",
                model=model_name,
                version=new_version,
                error=str(exc),
            )
            return None

    def update_run_tags(self, run_id: str, tags: dict) -> bool:
        """Met à jour les tags d'un run existant (ex: auto_promoted après promotion)."""
        if not self._configure():
            return False
        try:
            client = MlflowClient()
            for key, value in tags.items():
                client.set_tag(run_id, key, str(value))
            return True
        except Exception as exc:
            logger.warning("MLflow update_run_tags échoué", run_id=run_id, error=str(exc))
            return False

    def log_production_snapshot(
        self,
        *,
        model_name: str,
        version: str,
        metrics: dict,
    ) -> Optional[str]:
        """Logue un snapshot de métriques de production dans une expérience dédiée."""
        if not self._configure():
            return None
        try:
            experiment_name = f"{settings.MLFLOW_EXPERIMENT_PREFIX}/{model_name}_monitoring"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{model_name}_v{version}_monitoring") as run:
                mlflow.set_tags(
                    {"model_name": model_name, "version": version, "type": "monitoring"}
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
                "MLflow log_production_snapshot échoué",
                model=model_name,
                version=version,
                error=str(exc),
            )
            return None

    def delete_run(self, run_id: str) -> bool:
        """Supprime un run MLflow. Retourne False si MLflow est indisponible."""
        if not self._configure():
            return False
        try:
            MlflowClient().delete_run(run_id)
            logger.info("Run MLflow supprimé", run_id=run_id)
            return True
        except Exception as exc:
            logger.warning("MLflow delete_run échoué", run_id=run_id, error=str(exc))
            return False


mlflow_service = MLflowService()
