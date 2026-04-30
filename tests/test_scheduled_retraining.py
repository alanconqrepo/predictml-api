"""
Tests pour la fonctionnalité de ré-entraînement planifié (cron_schedule).

Couvre :
- PATCH /models/{name}/{version}/schedule
  - Auth / permissions
  - Modèle inexistant → 404
  - Expression cron invalide → 422
  - enabled=True sans cron → 422
  - Succès : retrain_schedule stocké avec next_run_at non-null
  - next_run_at est dans le futur
  - Désactivation (enabled=False) : schedule stocké avec enabled=False
- _run_retrain_job() / _do_retrain() (tests unitaires du job)
  - Skip si schedule.enabled=False
  - Skip si train_script_object_key absent
  - Skip si verrou Redis déjà actif
  - Succès : nouvelle version créée en base
  - last_run_at mis à jour après succès
  - Subprocess en échec : pas de nouvelle version
- Lifecycle scheduler
  - start_retrain_scheduler() charge les jobs actifs
  - stop_retrain_scheduler() appelle shutdown(wait=False)
"""

import asyncio
import io
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sqlalchemy import func, select

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _minio_mock, _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-sched-admin-cc33"
USER_TOKEN = "test-token-sched-user-dd44"
MODEL_PREFIX = "sched_model"

VALID_TRAIN_SCRIPT = """\
import os
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(json.dumps({"accuracy": 0.97, "f1_score": 0.96}))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _test_session_cm():
    """Remplace AsyncSessionLocal par la session SQLite de test."""
    async with _TestSessionLocal() as session:
        yield session


def _make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)  # noqa: N806
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


def _create_model(name: str, version: str = "1.0.0", with_train_script: bool = False) -> dict:
    files: dict = {
        "file": ("model.pkl", io.BytesIO(_make_pkl_bytes()), "application/octet-stream"),
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
# Setup utilisateurs
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

# Configurer le mock MinIO pour le scheduler
_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


# ---------------------------------------------------------------------------
# Mock subprocess pour les tests de job
# ---------------------------------------------------------------------------


async def _mock_exec_success(*args, **kwargs):
    """Subprocess mock : écrit un modèle factice et retourne succès (code 0)."""
    env = kwargs.get("env", {})
    output_path = env.get("OUTPUT_MODEL_PATH", "")
    if output_path:
        X, y = load_iris(return_X_y=True)  # noqa: N806
        model = LogisticRegression(max_iter=200).fit(X, y)
        with open(output_path, "wb") as f:
            pickle.dump(model, f)

    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(
        return_value=(b'Train done\n{"accuracy": 0.95, "f1_score": 0.93}\n', b"")
    )
    proc.kill = MagicMock()
    return proc


async def _mock_exec_failure(*args, **kwargs):
    """Subprocess mock : returncode != 0."""
    proc = MagicMock()
    proc.returncode = 1
    proc.communicate = AsyncMock(return_value=(b"", b"Error: training failed\n"))
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Tests endpoint PATCH /models/{name}/{version}/schedule
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
        # D'abord activer
        client.patch(
            f"/models/{name}/1.0.0/schedule",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"cron": "0 3 * * 1", "enabled": True},
        )
        # Puis désactiver
        with patch("src.tasks.retrain_scheduler.remove_retrain_job") as mock_remove:
            r = client.patch(
                f"/models/{name}/1.0.0/schedule",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"cron": "0 3 * * 1", "enabled": False},
            )
        assert r.status_code == 200
        sched = r.json()["retrain_schedule"]
        assert sched["enabled"] is False
        mock_remove.assert_called_once_with(name, "1.0.0")


# ---------------------------------------------------------------------------
# Tests unitaires du job _run_retrain_job / _do_retrain
# ---------------------------------------------------------------------------


class TestRunRetrainJob:
    def test_skips_when_disabled(self):
        """schedule.enabled=False → _do_retrain s'arrête sans créer de nouveau modèle."""
        from src.db.models import ModelMetadata

        name = f"{MODEL_PREFIX}_job_disabled"
        _create_model(name, with_train_script=True)

        # Stocker un schedule désactivé directement en base
        async def _set_disabled():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name, ModelMetadata.version == "1.0.0"
                    )
                )
                m = result.scalar_one()
                m.retrain_schedule = {
                    "cron": "0 3 * * 1",
                    "lookback_days": 30,
                    "auto_promote": False,
                    "enabled": False,
                }
                await db.commit()

        asyncio.run(_set_disabled())

        async def _count():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(func.count(ModelMetadata.id)).where(ModelMetadata.name == name)
                )
                return result.scalar()

        count_before = asyncio.run(_count())

        from src.tasks.retrain_scheduler import _run_retrain_job

        with patch("src.db.database.AsyncSessionLocal", new=_test_session_cm):
            asyncio.run(_run_retrain_job(name, "1.0.0"))

        count_after = asyncio.run(_count())
        assert count_after == count_before

    def test_skips_when_no_train_script(self):
        """train_script_object_key=None → job s'arrête sans créer de nouvelle version."""
        from src.db.models import ModelMetadata

        name = f"{MODEL_PREFIX}_job_noscript"
        _create_model(name, with_train_script=False)  # sans script

        async def _set_schedule():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name, ModelMetadata.version == "1.0.0"
                    )
                )
                m = result.scalar_one()
                m.retrain_schedule = {
                    "cron": "0 3 * * 1",
                    "lookback_days": 30,
                    "auto_promote": False,
                    "enabled": True,
                }
                await db.commit()

        asyncio.run(_set_schedule())

        async def _count():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(func.count(ModelMetadata.id)).where(ModelMetadata.name == name)
                )
                return result.scalar()

        count_before = asyncio.run(_count())

        from src.tasks.retrain_scheduler import _run_retrain_job

        with patch("src.db.database.AsyncSessionLocal", new=_test_session_cm):
            asyncio.run(_run_retrain_job(name, "1.0.0"))

        count_after = asyncio.run(_count())
        assert count_after == count_before

    def test_skips_when_redis_lock_held(self):
        """Verrou Redis déjà actif → _do_retrain n'est pas appelé."""
        from src.services.model_service import model_service
        from src.tasks.retrain_scheduler import _run_retrain_job

        name = f"{MODEL_PREFIX}_job_lock"
        _create_model(name, with_train_script=True)

        # Pré-setter le verrou sur le FakeRedis de test
        lock_key = f"retrain_lock:{name}:1.0.0"

        async def _run():
            await model_service._redis.set(lock_key, "1", nx=True, ex=700)
            try:
                with patch("src.db.database.AsyncSessionLocal", new=_test_session_cm):
                    with patch(
                        "src.tasks.retrain_scheduler._do_retrain", new_callable=AsyncMock
                    ) as mock_do:
                        await _run_retrain_job(name, "1.0.0")
                        mock_do.assert_not_called()
            finally:
                await model_service._redis.delete(lock_key)

        asyncio.run(_run())

    def test_success_creates_new_version(self):
        """Subprocess réussit → une nouvelle version est créée en base."""
        from src.db.models import ModelMetadata

        name = f"{MODEL_PREFIX}_job_ok"
        _create_model(name, with_train_script=True)

        async def _set_schedule():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name, ModelMetadata.version == "1.0.0"
                    )
                )
                m = result.scalar_one()
                m.retrain_schedule = {
                    "cron": "0 3 * * 1",
                    "lookback_days": 30,
                    "auto_promote": False,
                    "enabled": True,
                }
                await db.commit()

        asyncio.run(_set_schedule())

        async def _count():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(func.count(ModelMetadata.id)).where(ModelMetadata.name == name)
                )
                return result.scalar()

        count_before = asyncio.run(_count())

        from src.tasks.retrain_scheduler import _run_retrain_job

        with (
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch("src.services.minio_service.minio_service", _minio_mock),
            patch("asyncio.create_subprocess_exec", side_effect=_mock_exec_success),
        ):
            asyncio.run(_run_retrain_job(name, "1.0.0"))

        count_after = asyncio.run(_count())
        assert count_after == count_before + 1

        async def _get_new():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name,
                        ModelMetadata.version != "1.0.0",
                    )
                )
                return result.scalars().all()

        new_models = asyncio.run(_get_new())
        assert len(new_models) == 1
        assert "sched" in new_models[0].version
        assert new_models[0].trained_by == "scheduler"

    def test_updates_last_run_at(self):
        """Après succès, retrain_schedule.last_run_at est mis à jour sur le modèle source."""
        from src.db.models import ModelMetadata

        name = f"{MODEL_PREFIX}_last_run"
        _create_model(name, with_train_script=True)

        async def _set_schedule():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name, ModelMetadata.version == "1.0.0"
                    )
                )
                m = result.scalar_one()
                m.retrain_schedule = {
                    "cron": "0 3 * * 1",
                    "lookback_days": 30,
                    "auto_promote": False,
                    "enabled": True,
                    "last_run_at": None,
                }
                await db.commit()

        asyncio.run(_set_schedule())

        from src.tasks.retrain_scheduler import _run_retrain_job

        with (
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch("src.services.minio_service.minio_service", _minio_mock),
            patch("asyncio.create_subprocess_exec", side_effect=_mock_exec_success),
        ):
            asyncio.run(_run_retrain_job(name, "1.0.0"))

        async def _get_schedule():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name, ModelMetadata.version == "1.0.0"
                    )
                )
                m = result.scalar_one()
                return m.retrain_schedule

        sched = asyncio.run(_get_schedule())
        assert sched is not None
        assert sched["last_run_at"] is not None

    def test_subprocess_failure_does_not_create_version(self):
        """Script retourne returncode != 0 → aucune nouvelle version."""
        from src.db.models import ModelMetadata

        name = f"{MODEL_PREFIX}_job_fail"
        _create_model(name, with_train_script=True)

        async def _set_schedule():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name, ModelMetadata.version == "1.0.0"
                    )
                )
                m = result.scalar_one()
                m.retrain_schedule = {
                    "cron": "0 3 * * 1",
                    "lookback_days": 30,
                    "auto_promote": False,
                    "enabled": True,
                }
                await db.commit()

        asyncio.run(_set_schedule())

        async def _count():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(func.count(ModelMetadata.id)).where(ModelMetadata.name == name)
                )
                return result.scalar()

        count_before = asyncio.run(_count())

        from src.tasks.retrain_scheduler import _run_retrain_job

        with (
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch("src.services.minio_service.minio_service", _minio_mock),
            patch("asyncio.create_subprocess_exec", side_effect=_mock_exec_failure),
        ):
            asyncio.run(_run_retrain_job(name, "1.0.0"))

        count_after = asyncio.run(_count())
        assert count_after == count_before


# ---------------------------------------------------------------------------
# Tests lifecycle scheduler
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    def test_start_scheduler_loads_active_jobs(self):
        """start_retrain_scheduler() ajoute un job pour chaque modèle avec schedule actif."""
        from src.db.models import ModelMetadata
        from src.tasks.retrain_scheduler import start_retrain_scheduler

        name = f"{MODEL_PREFIX}_lifecycle"
        _create_model(name, with_train_script=True)

        async def _set_schedule():
            async with _TestSessionLocal() as db:
                result = await db.execute(
                    select(ModelMetadata).where(
                        ModelMetadata.name == name, ModelMetadata.version == "1.0.0"
                    )
                )
                m = result.scalar_one()
                m.retrain_schedule = {
                    "cron": "0 3 * * 1",
                    "lookback_days": 30,
                    "auto_promote": False,
                    "enabled": True,
                }
                await db.commit()

        asyncio.run(_set_schedule())

        mock_sched = MagicMock()
        mock_sched.running = False

        with (
            patch("src.db.database.AsyncSessionLocal", new=_test_session_cm),
            patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_sched),
        ):
            asyncio.run(start_retrain_scheduler())

        mock_sched.add_job.assert_called()
        job_ids = [call.kwargs.get("id") for call in mock_sched.add_job.call_args_list]
        assert f"retrain_schedule:{name}:1.0.0" in job_ids
        mock_sched.start.assert_called_once()

    def test_stop_scheduler_calls_shutdown_when_running(self):
        """stop_retrain_scheduler() avec running=True → shutdown(wait=False) appelé."""
        from src.tasks.retrain_scheduler import stop_retrain_scheduler

        mock_sched = MagicMock()
        mock_sched.running = True

        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_sched):
            stop_retrain_scheduler()

        mock_sched.shutdown.assert_called_once_with(wait=False)

    def test_stop_scheduler_noop_when_not_running(self):
        """stop_retrain_scheduler() avec running=False → shutdown n'est pas appelé."""
        from src.tasks.retrain_scheduler import stop_retrain_scheduler

        mock_sched = MagicMock()
        mock_sched.running = False

        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_sched):
            stop_retrain_scheduler()

        mock_sched.shutdown.assert_not_called()


# ---------------------------------------------------------------------------
# Tests drift-triggered retrain
# ---------------------------------------------------------------------------


class TestDriftTriggeredRetrain:
    """
    Vérifie que run_alert_check() déclenche _run_retrain_job via asyncio.create_task
    quand le drift atteint ou dépasse le seuil configuré dans retrain_schedule.
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
        """Exécute run_alert_check() avec toutes les dépendances mockées."""
        from src.tasks.supervision_reporter import run_alert_check

        model_stat = {
            "model_name": prod_meta.name,
            "error_rate": 0.0,
            "total_predictions": 100,
            "error_count": 0,
            "avg_latency_ms": 50.0,
            "shadow_predictions": 0,
        }

        mock_retrain_job = AsyncMock()

        def _close_coro(coro, **_):
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

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
            patch("src.tasks.retrain_scheduler._run_retrain_job", mock_retrain_job),
            patch("asyncio.create_task", side_effect=_close_coro),
        ):
            asyncio.run(run_alert_check())

        return mock_retrain_job

    def test_trigger_fires_on_critical_input_drift(self):
        """Drift feature critique + threshold=critical + cooldown expiré → retrain déclenché."""
        meta = self._make_prod_meta("drift_fire_input", trigger_on_drift="critical")
        mock_job = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_job.assert_called_once_with(meta.name, "1.0.0")

    def test_trigger_fires_on_critical_output_drift(self):
        """Drift de sortie critique + threshold=critical + cooldown expiré → retrain déclenché."""
        meta = self._make_prod_meta("drift_fire_output", trigger_on_drift="critical")
        mock_job = self._run_check_with_mocks(meta, feat_status="ok", out_status="critical")
        mock_job.assert_called_once_with(meta.name, "1.0.0")

    def test_no_trigger_if_warning_with_critical_threshold(self):
        """Drift feature warning + threshold=critical → pas de retrain."""
        meta = self._make_prod_meta("drift_no_fire_warn", trigger_on_drift="critical")
        mock_job = self._run_check_with_mocks(meta, feat_status="warning", out_status="ok")
        mock_job.assert_not_called()

    def test_trigger_fires_when_threshold_is_warning(self):
        """Drift feature warning + threshold=warning → retrain déclenché."""
        meta = self._make_prod_meta("drift_fire_warn_thresh", trigger_on_drift="warning")
        mock_job = self._run_check_with_mocks(meta, feat_status="warning", out_status="ok")
        mock_job.assert_called_once_with(meta.name, "1.0.0")

    def test_cooldown_blocks_second_fire(self):
        """last_run_at récent (cooldown non expiré) → pas de retrain déclenché."""
        from datetime import timezone

        recent = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        meta = self._make_prod_meta(
            "drift_cooldown",
            trigger_on_drift="critical",
            cooldown_hours=24,
            last_run_at=recent,
        )
        mock_job = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_job.assert_not_called()

    def test_no_trigger_without_train_script(self):
        """Pas de train_script_object_key → retrain non déclenché même si drift critique."""
        meta = self._make_prod_meta(
            "drift_no_script", trigger_on_drift="critical", has_script=False
        )
        mock_job = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_job.assert_not_called()

    def test_no_trigger_when_trigger_on_drift_is_null(self):
        """trigger_on_drift=None → aucun retrain même si drift critique."""
        meta = self._make_prod_meta("drift_null_trigger", trigger_on_drift=None)
        meta.retrain_schedule = None  # pas de schedule configuré
        mock_job = self._run_check_with_mocks(meta, feat_status="critical", out_status="ok")
        mock_job.assert_not_called()
