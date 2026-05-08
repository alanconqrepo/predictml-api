"""
Tests pour les branches non couvertes de src/services/mlflow_service.py.

Couvre :
- _configure() avec MLFLOW_S3_ENDPOINT_URL défini → variables AWS env vars configurées
- log_retrain_run quand MLFLOW_ENABLE=False → retourne None
- update_run_tags quand MLFLOW_ENABLE=False → retourne False
- log_production_snapshot quand MLFLOW_ENABLE=False → retourne None
- log_production_snapshot quand exception interne → retourne None (dégradation gracieuse)
"""

import os
from unittest.mock import MagicMock, patch

import pytest


def _make_service():
    """Instancie un MLflowService frais (non patchable globalement)."""
    from src.services.mlflow_service import MLflowService

    return MLflowService()


def _default_retrain_kwargs():
    return dict(
        model_name="test_model",
        new_version="1.0.0",
        source_version="0.9.0",
        trigger="manual",
        trained_by="admin",
        train_start_date="2025-01-01",
        train_end_date="2025-12-31",
        accuracy=0.95,
        f1_score=0.94,
        n_rows=None,
        feature_stats=None,
        label_distribution=None,
        algorithm=None,
        training_params=None,
        auto_promoted=False,
        auto_promote_reason=None,
    )


class TestMLflowDisabled:
    """MLflow désactivé → toutes les méthodes retournent None/False."""

    def test_log_retrain_run_returns_none_when_disabled(self):
        """MLFLOW_ENABLE=False → log_retrain_run retourne None."""
        svc = _make_service()
        with patch("src.services.mlflow_service.settings") as mock_s:
            mock_s.MLFLOW_ENABLE = False
            result = svc.log_retrain_run(**_default_retrain_kwargs())
        assert result is None

    def test_update_run_tags_returns_false_when_disabled(self):
        """MLFLOW_ENABLE=False → update_run_tags retourne False."""
        svc = _make_service()
        with patch("src.services.mlflow_service.settings") as mock_s:
            mock_s.MLFLOW_ENABLE = False
            result = svc.update_run_tags("run-id-abc", {"key": "val"})
        assert result is False

    def test_log_production_snapshot_returns_none_when_disabled(self):
        """MLFLOW_ENABLE=False → log_production_snapshot retourne None."""
        svc = _make_service()
        with patch("src.services.mlflow_service.settings") as mock_s:
            mock_s.MLFLOW_ENABLE = False
            result = svc.log_production_snapshot(
                model_name="m", version="1.0.0", metrics={"accuracy": 0.9}
            )
        assert result is None


class TestMLflowS3Config:
    """_configure() avec MLFLOW_S3_ENDPOINT_URL → env vars AWS configurées."""

    def test_configure_sets_s3_env_vars_when_endpoint_defined(self):
        """MLFLOW_S3_ENDPOINT_URL non vide → AWS_ACCESS_KEY_ID défini dans l'env."""
        svc = _make_service()

        with patch("src.services.mlflow_service.settings") as mock_s:
            mock_s.MLFLOW_ENABLE = True
            mock_s.MLFLOW_TRACKING_URI = "http://localhost:5000"
            mock_s.MLFLOW_S3_ENDPOINT_URL = "http://minio:9000"
            mock_s.MINIO_ACCESS_KEY = "test_key"
            mock_s.MINIO_SECRET_KEY = "test_secret"

            with patch("src.services.mlflow_service.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri = MagicMock()

                saved = {}
                for k in ("MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
                    saved[k] = os.environ.pop(k, None)

                try:
                    result = svc._configure()
                    assert result is True
                    # os.environ.setdefault doit avoir été appelé — la valeur doit être présente
                    assert os.environ.get("AWS_ACCESS_KEY_ID") == "test_key"
                finally:
                    for k, v in saved.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v

    def test_configure_returns_false_on_exception(self):
        """Exception dans _configure() → retourne False."""
        svc = _make_service()
        with patch("src.services.mlflow_service.settings") as mock_s:
            mock_s.MLFLOW_ENABLE = True
            mock_s.MLFLOW_TRACKING_URI = "http://localhost:5000"
            mock_s.MLFLOW_S3_ENDPOINT_URL = ""

            with patch("src.services.mlflow_service.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri.side_effect = Exception("connexion refusée")
                result = svc._configure()

        assert result is False


class TestMLflowProductionSnapshotException:
    """log_production_snapshot : exception interne → dégradation gracieuse."""

    def test_exception_in_set_experiment_returns_none(self):
        """mlflow.set_experiment lève une exception → retourne None sans crash."""
        svc = _make_service()

        with patch("src.services.mlflow_service.settings") as mock_s:
            mock_s.MLFLOW_ENABLE = True
            mock_s.MLFLOW_TRACKING_URI = "http://localhost:5000"
            mock_s.MLFLOW_S3_ENDPOINT_URL = ""
            mock_s.MLFLOW_EXPERIMENT_PREFIX = "predictml"

            with patch("src.services.mlflow_service.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri = MagicMock()
                mock_mlflow.set_experiment.side_effect = Exception("MLflow unavailable")

                result = svc.log_production_snapshot(
                    model_name="iris", version="1.0.0", metrics={"accuracy": 0.92}
                )

        assert result is None

    def test_exception_in_log_retrain_returns_none(self):
        """Exception interne dans log_retrain_run → retourne None."""
        svc = _make_service()

        with patch("src.services.mlflow_service.settings") as mock_s:
            mock_s.MLFLOW_ENABLE = True
            mock_s.MLFLOW_TRACKING_URI = "http://localhost:5000"
            mock_s.MLFLOW_S3_ENDPOINT_URL = ""
            mock_s.MLFLOW_EXPERIMENT_PREFIX = "predictml"
            mock_s.MLFLOW_REGISTER_MODELS = False

            with patch("src.services.mlflow_service.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri = MagicMock()
                mock_mlflow.set_experiment.side_effect = Exception("MLflow down")

                result = svc.log_retrain_run(**_default_retrain_kwargs())

        assert result is None
