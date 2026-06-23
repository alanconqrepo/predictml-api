"""
Tests for the scheduled retraining feature (cron_schedule).

Covers:
- PATCH /models/{name}/{version}/schedule
  - Auth / permissions
  - Non-existent model → 404
  - Invalid cron expression → 422
  - enabled=True without cron → 422
  - Success: retrain_schedule stored with non-null next_run_at
  - next_run_at is in the future
  - Disable (enabled=False): schedule stored with enabled=False
- Drift-triggered retrain
  - Retrain enqueued to ARQ on critical drift + expired cooldown
  - No retrain when drift below threshold
  - No retrain when cooldown not expired
  - No retrain when no train script
"""

import asyncio
import io
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import joblib
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-sched-admin-cc33"
USER_TOKEN = "test-token-sched-user-dd44"
MODEL_PREFIX = "sched_model"

VALID_TRAIN_SCRIPT = """\
import os
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    joblib.dump(model, f)

print(json.dumps({"accuracy": 0.97, "f1_score": 0.96}))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _test_session_cm():
    """Replaces AsyncSessionLocal with the test SQLite session."""
    async with _TestSessionLocal() as session:
        yield session


def _make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)  # noqa: N806
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _create_model(name: str, version: str = "1.0.0", with_train_script: bool = False) -> dict:
    files: dict = {
        "file": ("model.joblib", io.BytesIO(_make_pkl_bytes()), "application/octet-stream"),
    }
    if with_train_script:
        files["train_file"] = (
            "train.py",
            io.BytesIO(VALID_TRAIN_SCRIPT.encode()),
            "text/x-python",
        )
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files=files,
        data={"name": name, "version": version, "accuracy": "0.90", "f1_score": "0.89"},
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# User setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="sched_admin",
                email="sched_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="sched_user",
                email="sched_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Tests for endpoint PATCH /models/{name}/{version}/schedule
# ---------------------------------------------------------------------------


class TestScheduleEndpointAuth:
    def test_no_auth_returns_401(self):
        r = client.patch("/models/unknown/1.0.0/schedule", json={"cron": "0 3 * * 1"})
        assert r.status_code in (401, 403)

    def test_non_admin_returns_403(self):
        name = f"{MODEL_PREFIX}_auth"
        _create_model(name)
        r = client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"cron": "0 3 * * 1"},
        )
        assert r.status_code == 403

    def test_unknown_model_returns_404(self):
        r = client.patch(
            "/models/does_not_exist/9.9.9/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "0 3 * * 1"},
        )
        assert r.status_code == 404


class TestScheduleEndpointValidation:
    def test_invalid_cron_returns_422(self):
        name = f"{MODEL_PREFIX}_val_cron"
        _create_model(name)
        r = client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "not a cron expression"},
        )
        assert r.status_code == 422
        assert "cron" in r.text.lower() or "invalide" in r.text.lower()

    def test_enabled_true_without_cron_returns_422(self):
        name = f"{MODEL_PREFIX}_val_nocron"
        _create_model(name)
        r = client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": None, "enabled": True},
        )
        assert r.status_code == 422


class TestScheduleEndpointSuccess:
    def test_schedule_set_success(self):
        name = f"{MODEL_PREFIX}_success"
        _create_model(name)
        r = client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "0 3 * * 1", "lookback_days": 30, "enabled": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == name
        assert data["version"] == "1.0.0"
        sched = data["retrain_schedule"]
        assert sched is not None
        assert sched["cron"] == "0 3 * * 1"
        assert sched["enabled"] is True
        assert sched["next_run_at"] is not None

    def test_next_run_at_is_in_future(self):
        name = f"{MODEL_PREFIX}_nextrun"
        _create_model(name)
        r = client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "0 3 * * 1", "lookback_days": 7, "enabled": True},
        )
        assert r.status_code == 200
        next_run_str = r.json()["retrain_schedule"]["next_run_at"]
        next_run = datetime.fromisoformat(next_run_str)
        assert next_run > datetime.utcnow()

    def test_disable_schedule(self):
        name = f"{MODEL_PREFIX}_disable"
        _create_model(name)
        # First enable
        client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "0 3 * * 1", "enabled": True},
        )
        # Then disable
        r = client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "0 3 * * 1", "enabled": False},
        )
        assert r.status_code == 200
        sched = r.json()["retrain_schedule"]
        assert sched["enabled"] is False


# ---------------------------------------------------------------------------
# Drift-triggered retrain tests
# ---------------------------------------------------------------------------


class TestDriftTriggeredRetrain:
    """
    Verifies that run_alert_check() enqueues a scheduled_retrain_task via ARQ
    when drift reaches or exceeds the threshold configured in retrain_schedule.
    """

    @staticmethod
    def _make_prod_meta(
        name: str,
        trigger_on_drift: str | None,
        cooldown_hours: int = 24,
        last_run_at: str | None = None,
        has_script: bool = True,
    ):
        m = MagicMock()
        m.name = name
        m.version = "1.0.0"
        m.is_production = True
        m.feature_baseline = {"sepal_length": {"mean": 5.8, "std": 0.83}}
        m.alert_thresholds = None
        m.webhook_url = None
        m.promotion_policy = None
        m.train_script_object_key = f"{name}/v1.0.0_train.py" if has_script else None
        m.retrain_schedule = {
            "cron": "0 3 * * 1",
            "enabled": True,
            "lookback_days": 30,
            "auto_promote": False,
            "trigger_on_drift": trigger_on_drift,
            "drift_retrain_cooldown_hours": cooldown_hours,
            "last_run_at": last_run_at,
        }
        return m

    @staticmethod
    def _feat_result(status: str):
        r = MagicMock()
        r.drift_status = status
        r.z_score = 3.5
        r.psi = 0.25
        return r

    @staticmethod
    def _output_report(status: str):
        r = MagicMock()
        r.status = status
        r.psi = 0.05
        r.predictions_analyzed = 50
        return r

    def _run_check_with_mocks(self, prod_meta, feat_status: str, out_status: str):
        """Runs run_alert_check() with all dependencies mocked."""
        from src.tasks.supervision_reporter import run_alert_check

        model_stat = {
            "model_name": prod_meta.name,
            "error_rate": 0.0,
            "total_predictions": 100,
            "error_count": 0,
            "avg_latency_ms": 50.0,
            "shadow_predictions": 0,
        }

        mock_pool = AsyncMock()
        mock_pool.enqueue_job = AsyncMock()
        mock_get_arq_pool = AsyncMock(return_value=mock_pool)

        with (
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new_callable=AsyncMock,
                return_value=[model_stat],
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new_callable=AsyncMock,
                return_value=[prod_meta],
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.services.db_service.DBService.get_feature_production_stats",
                new_callable=AsyncMock,
                return_value={"sepal_length": {"mean": 7.5, "std": 0.5, "count": 50}},
            ),
            patch(
                "src.services.drift_service.compute_feature_drift",
                return_value={"sepal_length": self._feat_result(feat_status)},
            ),
            patch(
                "src.services.drift_service.compute_output_drift",
                new_callable=AsyncMock,
                return_value=self._output_report(out_status),
            ),
            patch("src.core.arq_pool.get_arq_pool", mock_get_arq_pool),
        ):
            asyncio.run(run_alert_check())

        return mock_pool.enqueue_job

    def test_trigger_fires_on_critical_input_drift(self):
        """Critical feature drift + threshold=critical + expired cooldown → retrain enqueued."""
        meta = self._make_prod_meta("drift_fire_input", trigger_on_drift="critical")
        mock_enqueue = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_enqueue.assert_called_once_with(
            "scheduled_retrain_task", model_name=meta.name, source_version="1.0.0"
        )

    def test_trigger_fires_on_critical_output_drift(self):
        """Critical output drift + threshold=critical + expired cooldown → retrain enqueued."""
        meta = self._make_prod_meta("drift_fire_output", trigger_on_drift="critical")
        mock_enqueue = self._run_check_with_mocks(meta, feat_status="ok", out_status="critical")
        mock_enqueue.assert_called_once_with(
            "scheduled_retrain_task", model_name=meta.name, source_version="1.0.0"
        )

    def test_no_trigger_if_warning_with_critical_threshold(self):
        """Warning feature drift + threshold=critical → no retrain."""
        meta = self._make_prod_meta("drift_no_fire_warn", trigger_on_drift="critical")
        mock_enqueue = self._run_check_with_mocks(meta, feat_status="warning", out_status="ok")
        mock_enqueue.assert_not_called()

    def test_trigger_fires_when_threshold_is_warning(self):
        """Warning feature drift + threshold=warning → retrain enqueued."""
        meta = self._make_prod_meta("drift_fire_warn_thresh", trigger_on_drift="warning")
        mock_enqueue = self._run_check_with_mocks(meta, feat_status="warning", out_status="ok")
        mock_enqueue.assert_called_once_with(
            "scheduled_retrain_task", model_name=meta.name, source_version="1.0.0"
        )

    def test_cooldown_blocks_second_fire(self):
        """Recent last_run_at (cooldown not expired) → no retrain triggered."""
        from datetime import timezone

        recent = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        meta = self._make_prod_meta(
            "drift_cooldown",
            trigger_on_drift="critical",
            cooldown_hours=24,
            last_run_at=recent,
        )
        mock_enqueue = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_enqueue.assert_not_called()

    def test_no_trigger_without_train_script(self):
        """No train_script_object_key → retrain not triggered even on critical drift."""
        meta = self._make_prod_meta(
            "drift_no_script", trigger_on_drift="critical", has_script=False
        )
        mock_enqueue = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_enqueue.assert_not_called()

    def test_no_trigger_when_trigger_on_drift_is_null(self):
        """trigger_on_drift=None → no retrain even on critical drift."""
        meta = self._make_prod_meta("drift_null_trigger", trigger_on_drift=None)
        meta.retrain_schedule = None  # no schedule configured
        mock_enqueue = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for scheduled retrain email notifications
# ---------------------------------------------------------------------------


class TestRetrainResultAlert:
    """Unit tests for email_service.send_retrain_result_alert().

    These verify that the correct subject, title, and parameters are used
    for success and failure notifications without requiring SMTP.
    """

    def _send_with_mock(self, **kwargs):
        """Call send_retrain_result_alert() with _send_email mocked out.
        Returns the mock so callers can inspect call args.
        """
        from src.services.email_service import email_service

        with patch.object(email_service, "_send_email", return_value=True) as mock_send:
            with patch.object(email_service, "_is_configured", return_value=True):
                email_service.send_retrain_result_alert(**kwargs)
        return mock_send

    def test_success_email_subject_and_content(self):
        """Successful retrain: subject contains model name and new version."""
        mock_send = self._send_with_mock(
            model_name="iris",
            source_version="1.0.0",
            new_version="1.0.0-sched-20260623",
            success=True,
            trigger="scheduler",
            accuracy=0.95,
            f1_score=0.93,
        )
        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert "iris" in kwargs["subject"]
        assert "1.0.0-sched-20260623" in kwargs["subject"]
        assert "iris" in kwargs["html_body"]
        assert "0.9500" in kwargs["html_body"]  # accuracy formatted
        assert "0.9300" in kwargs["html_body"]  # f1_score formatted

    def test_failure_email_subject_and_error(self):
        """Failed retrain: subject mentions failure, error is in body."""
        mock_send = self._send_with_mock(
            model_name="wine",
            source_version="2.0.0",
            new_version=None,
            success=False,
            trigger="scheduler",
            error="subprocess timed out after 600s",
        )
        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert "wine" in kwargs["subject"]
        assert "subprocess timed out" in kwargs["html_body"]

    def test_success_without_metrics_does_not_crash(self):
        """send_retrain_result_alert(success=True) with no accuracy/f1 must not raise."""
        mock_send = self._send_with_mock(
            model_name="model_x",
            source_version="1.0.0",
            new_version="1.0.0-sched-x",
            success=True,
            trigger="scheduler",
        )
        mock_send.assert_called_once()

    def test_no_email_when_smtp_not_configured(self):
        """Returns False (silent no-op) when SMTP is not configured."""
        from src.services.email_service import email_service

        with patch.object(email_service, "_is_configured", return_value=False):
            result = email_service.send_retrain_result_alert(
                model_name="iris",
                source_version="1.0.0",
                new_version="1.0.0-x",
                success=True,
                trigger="scheduler",
            )
        assert result is False
