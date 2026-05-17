"""
Tests unitaires pour MLflowService.
Tous les appels au SDK mlflow sont mockés — aucun serveur MLflow requis.
"""

import io
import joblib
from unittest.mock import MagicMock, patch

import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.services.mlflow_service import MLflowService

_MODULE = "src.services.mlflow_service"


def _make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _make_service() -> MLflowService:
    return MLflowService()


def _mock_settings(enabled: bool = True, register: bool = False):
    m = MagicMock()
    m.MLFLOW_ENABLE = enabled
    m.MLFLOW_TRACKING_URI = "http://mock-mlflow:5000"
    m.MLFLOW_EXPERIMENT_PREFIX = "predictml"
    m.MLFLOW_S3_ENDPOINT_URL = ""
    m.MLFLOW_REGISTER_MODELS = register
    m.MINIO_ACCESS_KEY = "minioadmin"
    m.MINIO_SECRET_KEY = "minioadmin"
    return m


def _build_mock_mlflow(run_id: str = "test-run-id-xyz"):
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_run)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value = ctx
    return mock_mlflow, mock_run


def _run_log(svc, mock_mlflow, mock_settings=None, **overrides):
    """Helper : appelle log_retrain_run avec des valeurs par défaut patchables."""
    if mock_settings is None:
        mock_settings = _mock_settings()
    defaults = dict(
        model_name="iris",
        new_version="2.0",
        source_version="1.0",
        trigger="manual",
        trained_by="admin",
        train_start_date="2025-01-01",
        train_end_date="2025-12-31",
        accuracy=0.95,
        f1_score=0.93,
        n_rows=120,
        feature_stats=None,
        label_distribution=None,
        algorithm="RandomForest",
        training_params=None,
        auto_promoted=False,
        auto_promote_reason=None,
    )
    defaults.update(overrides)
    with patch(f"{_MODULE}.settings", mock_settings):
        with patch(f"{_MODULE}.mlflow", mock_mlflow):
            with patch(f"{_MODULE}.MlflowClient", MagicMock()):
                return svc.log_retrain_run(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# Disabled MLflow
# ──────────────────────────────────────────────────────────────────────────────


class TestMLflowDisabled:
    def test_log_retrain_run_returns_none_when_disabled(self):
        svc = _make_service()
        with patch(f"{_MODULE}.settings", _mock_settings(enabled=False)):
            result = svc.log_retrain_run(
                model_name="iris",
                new_version="2.0",
                source_version="1.0",
                trigger="manual",
                trained_by="admin",
                train_start_date="2025-01-01",
                train_end_date="2025-12-31",
                accuracy=0.95,
                f1_score=0.93,
                n_rows=100,
                feature_stats=None,
                label_distribution=None,
                algorithm=None,
                training_params=None,
                auto_promoted=False,
                auto_promote_reason=None,
            )
        assert result is None

    def test_delete_run_returns_false_when_disabled(self):
        svc = _make_service()
        with patch(f"{_MODULE}.settings", _mock_settings(enabled=False)):
            assert svc.delete_run("some-run-id") is False

    def test_log_production_snapshot_returns_none_when_disabled(self):
        svc = _make_service()
        with patch(f"{_MODULE}.settings", _mock_settings(enabled=False)):
            result = svc.log_production_snapshot(
                model_name="iris", version="1.0", metrics={"error_rate": 0.02}
            )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Configuration failure
# ──────────────────────────────────────────────────────────────────────────────


class TestMLflowConfigError:
    def test_returns_none_on_configure_exception(self):
        svc = _make_service()
        mock_mlflow = MagicMock()
        mock_mlflow.set_tracking_uri.side_effect = Exception("Connection refused")
        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow", mock_mlflow):
                result = svc.log_retrain_run(
                    model_name="iris",
                    new_version="2.0",
                    source_version="1.0",
                    trigger="manual",
                    trained_by="admin",
                    train_start_date="2025-01-01",
                    train_end_date="2025-12-31",
                    accuracy=0.95,
                    f1_score=0.93,
                    n_rows=None,
                    feature_stats=None,
                    label_distribution=None,
                    algorithm=None,
                    training_params=None,
                    auto_promoted=False,
                    auto_promote_reason=None,
                )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Successful run logging
# ──────────────────────────────────────────────────────────────────────────────


class TestLogRetrainRun:
    def test_returns_run_id_on_success(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow("abc-123")
        result = _run_log(svc, mock_mlflow)
        assert result == "abc-123"

    def test_sets_experiment_with_prefix(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow)
        mock_mlflow.set_experiment.assert_called_once_with("predictml/iris")

    def test_logs_scalar_metrics(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow)
        logged = {c.args[0]: c.args[1] for c in mock_mlflow.log_metric.call_args_list}
        assert logged["accuracy"] == 0.95
        assert logged["f1_score"] == 0.93
        assert logged["n_rows_train"] == 120.0

    def test_omits_none_metrics(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow, accuracy=None, f1_score=None, n_rows=None)
        metric_names = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
        assert "accuracy" not in metric_names
        assert "f1_score" not in metric_names
        assert "n_rows_train" not in metric_names

    def test_logs_feature_stats_as_individual_metrics(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        feature_stats = {
            "sepal_length": {"mean": 5.8, "std": 0.83, "min": 4.3, "max": 7.9, "null_rate": 0.0},
        }
        _run_log(svc, mock_mlflow, feature_stats=feature_stats, n_rows=None)
        metric_names = {c.args[0] for c in mock_mlflow.log_metric.call_args_list}
        assert "feat_sepal_length_mean" in metric_names
        assert "feat_sepal_length_std" in metric_names
        assert "feat_sepal_length_null_rate" in metric_names

    def test_logs_label_distribution_as_ratios(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        label_distribution = {"setosa": 50, "versicolor": 50}
        _run_log(svc, mock_mlflow, label_distribution=label_distribution, n_rows=None)
        logged = {c.args[0]: c.args[1] for c in mock_mlflow.log_metric.call_args_list}
        assert "label_setosa_ratio" in logged
        assert "label_versicolor_ratio" in logged
        assert logged["label_setosa_ratio"] == pytest.approx(0.5)
        assert logged["label_versicolor_ratio"] == pytest.approx(0.5)

    def test_logs_params_with_algorithm_and_training_params(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow, training_params={"n_estimators": 100})
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["model_name"] == "iris"
        assert params["new_version"] == "2.0"
        assert params["trigger"] == "manual"
        assert params["algorithm"] == "RandomForest"
        assert params["param_n_estimators"] == "100"

    def test_logs_lookback_days_as_param_for_scheduler(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow, lookback_days=30)
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["lookback_days"] == "30"

    def test_sets_tags(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow, auto_promoted=True, auto_promote_reason="accuracy ok")
        tags = mock_mlflow.set_tags.call_args[0][0]
        assert tags["model_name"] == "iris"
        assert tags["auto_promoted"] == "True"
        assert tags["auto_promote_reason"] == "accuracy ok"

    def test_logs_model_bytes_as_artifact(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow, model_bytes=_make_pkl_bytes())
        mock_mlflow.sklearn.log_model.assert_called_once()
        positional = mock_mlflow.sklearn.log_model.call_args[0]
        kw = mock_mlflow.sklearn.log_model.call_args[1]
        artifact_path = positional[1] if len(positional) > 1 else kw.get("artifact_path")
        assert artifact_path == "model"

    def test_skips_artifact_on_invalid_bytes(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow, model_bytes=b"not-a-valid-joblib")
        mock_mlflow.sklearn.log_model.assert_not_called()

    def test_skips_artifact_when_model_bytes_is_none(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        _run_log(svc, mock_mlflow, model_bytes=None)
        mock_mlflow.sklearn.log_model.assert_not_called()

    def test_calls_register_model_when_enabled(self):
        svc = _make_service()
        mock_mlflow, mock_run = _build_mock_mlflow("reg-run-id")
        mock_settings = _mock_settings(register=True)
        _run_log(svc, mock_mlflow, mock_settings=mock_settings)
        mock_mlflow.register_model.assert_called_once_with("runs:/reg-run-id/model", "iris")

    def test_skips_register_model_when_disabled(self):
        svc = _make_service()
        mock_mlflow, _ = _build_mock_mlflow()
        mock_settings = _mock_settings(register=False)
        _run_log(svc, mock_mlflow, mock_settings=mock_settings)
        mock_mlflow.register_model.assert_not_called()

    def test_returns_none_on_mlflow_exception(self):
        svc = _make_service()
        mock_mlflow = MagicMock()
        mock_mlflow.set_tracking_uri.return_value = None  # configure succeeds
        mock_mlflow.set_experiment.side_effect = Exception("MLflow down")
        result = _run_log(svc, mock_mlflow)
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# update_run_tags
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateRunTags:
    def test_calls_set_tag_for_each_entry(self):
        svc = _make_service()
        mock_client_instance = MagicMock()
        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri.return_value = None
                with patch(f"{_MODULE}.MlflowClient", return_value=mock_client_instance):
                    result = svc.update_run_tags("run-123", {"auto_promoted": "True", "k": "v"})
        assert result is True
        assert mock_client_instance.set_tag.call_count == 2

    def test_returns_false_on_error(self):
        svc = _make_service()
        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri.return_value = None
                with patch(f"{_MODULE}.MlflowClient", side_effect=Exception("err")):
                    result = svc.update_run_tags("run-123", {"k": "v"})
        assert result is False


# ──────────────────────────────────────────────────────────────────────────────
# delete_run
# ──────────────────────────────────────────────────────────────────────────────


class TestDeleteRun:
    def test_calls_delete_on_mlflow_client(self):
        svc = _make_service()
        mock_client_instance = MagicMock()
        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri.return_value = None
                with patch(f"{_MODULE}.MlflowClient", return_value=mock_client_instance):
                    result = svc.delete_run("run-abc")
        assert result is True
        mock_client_instance.delete_run.assert_called_once_with("run-abc")

    def test_returns_false_on_error(self):
        svc = _make_service()
        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow") as mock_mlflow:
                mock_mlflow.set_tracking_uri.return_value = None
                with patch(f"{_MODULE}.MlflowClient", side_effect=Exception("down")):
                    result = svc.delete_run("run-abc")
        assert result is False


# ──────────────────────────────────────────────────────────────────────────────
# log_production_snapshot
# ──────────────────────────────────────────────────────────────────────────────


class TestLogProductionSnapshot:
    def test_returns_run_id_on_success(self):
        svc = _make_service()
        mock_run = MagicMock()
        mock_run.info.run_id = "monitoring-run-id"
        mock_mlflow = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_run)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = ctx

        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow", mock_mlflow):
                result = svc.log_production_snapshot(
                    model_name="iris",
                    version="1.0",
                    metrics={"error_rate": 0.02, "p95_latency_ms": 45.0},
                )
        assert result == "monitoring-run-id"
        metric_names = {c.args[0] for c in mock_mlflow.log_metric.call_args_list}
        assert "error_rate" in metric_names
        assert "p95_latency_ms" in metric_names

    def test_uses_monitoring_experiment_name(self):
        svc = _make_service()
        mock_run = MagicMock()
        mock_run.info.run_id = "x"
        mock_mlflow = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_run)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = ctx

        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow", mock_mlflow):
                svc.log_production_snapshot(
                    model_name="iris", version="1.0", metrics={"error_rate": 0.01}
                )
        mock_mlflow.set_experiment.assert_called_once_with("predictml/iris_monitoring")

    def test_returns_none_on_exception(self):
        svc = _make_service()
        mock_mlflow = MagicMock()
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.set_experiment.side_effect = Exception("boom")
        with patch(f"{_MODULE}.settings", _mock_settings()):
            with patch(f"{_MODULE}.mlflow", mock_mlflow):
                result = svc.log_production_snapshot(
                    model_name="iris", version="1.0", metrics={"error_rate": 0.01}
                )
        assert result is None
