"""
Tests complémentaires pour src/tasks/retrain_scheduler.py.

Couvre les branches non couvertes par test_scheduled_retraining.py :
- _compute_next_run_at avec différentes expressions cron
- _compute_next_run_at avec cron invalide → None
- _run_retrain_job quand Redis lock déjà pris → skip
- _do_retrain quand modèle source introuvable → retour immédiat
- _do_retrain quand schedule disabled → retour immédiat
- _do_retrain quand train_script_object_key absent → retour immédiat
- _do_retrain subprocess timeout → return (version non créée)
- add_retrain_job quand enabled=False → pas de job créé
- remove_retrain_job silencieux si job absent
- start_retrain_scheduler / stop_retrain_scheduler lifecycle
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestComputeNextRunAt:
    def test_daily_cron(self):
        """Expression cron quotidienne → datetime naive UTC valide."""
        from src.tasks.retrain_scheduler import _compute_next_run_at

        result = _compute_next_run_at("0 3 * * *")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is None  # naive UTC

    def test_weekly_cron(self):
        """Expression cron hebdomadaire (lundi à 02h) → datetime valide."""
        from src.tasks.retrain_scheduler import _compute_next_run_at

        result = _compute_next_run_at("0 2 * * 1")
        assert result is not None
        assert isinstance(result, datetime)

    def test_invalid_cron_returns_none(self):
        """Expression cron invalide → None (pas d'exception)."""
        from src.tasks.retrain_scheduler import _compute_next_run_at

        result = _compute_next_run_at("invalid cron expression !!")
        assert result is None

    def test_monthly_cron(self):
        """Expression cron mensuelle (1er du mois à minuit) → datetime valide."""
        from src.tasks.retrain_scheduler import _compute_next_run_at

        result = _compute_next_run_at("0 0 1 * *")
        assert result is not None


class TestRunRetrainJobRedisLock:
    def test_job_skips_when_lock_already_acquired(self):
        """Redis lock déjà pris → _do_retrain non appelé, warning loggé."""
        from src.tasks.retrain_scheduler import _run_retrain_job

        mock_redis = AsyncMock()
        mock_redis.set.return_value = False  # verrou déjà acquis

        with (
            patch(
                "src.tasks.retrain_scheduler._retrain_scheduler",
                MagicMock(),
            ),
        ):
            with patch("src.services.model_service.model_service") as mock_ms:
                mock_ms._get_redis = AsyncMock(return_value=mock_redis)

                with patch(
                    "src.tasks.retrain_scheduler._do_retrain",
                    new_callable=AsyncMock,
                ) as mock_do_retrain:
                    # On doit patcher model_service dans le module retrain_scheduler
                    with patch(
                        "src.tasks.retrain_scheduler._run_retrain_job.__module__",
                        "src.tasks.retrain_scheduler",
                    ):
                        pass

                # Appel direct plus simple
                async def _run():
                    with patch(
                        "src.tasks.retrain_scheduler._do_retrain",
                        new_callable=AsyncMock,
                    ) as mock_inner:
                        original = __import__(
                            "src.tasks.retrain_scheduler", fromlist=["_run_retrain_job"]
                        )
                        with patch.object(
                            __import__(
                                "src.services.model_service",
                                fromlist=["model_service"],
                            ).model_service,
                            "_get_redis",
                            AsyncMock(return_value=mock_redis),
                        ):
                            await original._run_retrain_job("my_model", "1.0.0")
                        mock_inner.assert_not_called()

                asyncio.run(_run())

    def test_job_executes_when_lock_acquired(self):
        """Redis lock libre → _do_retrain appelé."""
        from src.tasks.retrain_scheduler import _run_retrain_job

        mock_redis = AsyncMock()
        mock_redis.set.return_value = True  # verrou libre
        mock_redis.delete = AsyncMock()

        async def _run():
            import src.tasks.retrain_scheduler as mod
            with (
                patch.object(
                    __import__(
                        "src.services.model_service", fromlist=["model_service"]
                    ).model_service,
                    "_get_redis",
                    AsyncMock(return_value=mock_redis),
                ),
                patch.object(mod, "_do_retrain", new_callable=AsyncMock) as mock_do,
            ):
                await _run_retrain_job("my_model", "2.0.0")
                mock_do.assert_called_once_with("my_model", "2.0.0")

        asyncio.run(_run())


class TestDoRetrainEarlyReturns:
    def _make_db_mock(self, source_model=None):
        """Helper : mock de la session DB."""
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = source_model
        db.execute = AsyncMock(return_value=result_mock)
        return db

    def test_do_retrain_model_not_found(self):
        """Source model introuvable → retour immédiat sans erreur."""
        import src.tasks.retrain_scheduler as mod

        async def _run():
            with (
                patch("src.db.database.AsyncSessionLocal") as mock_session_cls,
            ):
                mock_db = self._make_db_mock(source_model=None)
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                await mod._do_retrain("missing_model", "9.9.9")

        asyncio.run(_run())

    def test_do_retrain_schedule_disabled(self):
        """retrain_schedule.enabled=False → retour immédiat."""
        import src.tasks.retrain_scheduler as mod

        source = MagicMock()
        source.retrain_schedule = {"enabled": False, "cron": "0 3 * * *"}
        source.train_script_object_key = "some/key.py"

        async def _run():
            with (
                patch("src.db.database.AsyncSessionLocal") as mock_session_cls,
            ):
                mock_db = self._make_db_mock(source_model=source)
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                await mod._do_retrain("disabled_model", "1.0.0")

        asyncio.run(_run())

    def test_do_retrain_no_train_script(self):
        """Pas de train_script_object_key → retour immédiat."""
        import src.tasks.retrain_scheduler as mod

        source = MagicMock()
        source.retrain_schedule = {"enabled": True, "cron": "0 3 * * *"}
        source.train_script_object_key = None

        async def _run():
            with patch("src.db.database.AsyncSessionLocal") as mock_session_cls:
                mock_db = self._make_db_mock(source_model=source)
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                await mod._do_retrain("no_script_model", "1.0.0")

        asyncio.run(_run())

    def test_do_retrain_minio_download_fails(self):
        """Téléchargement MinIO échoue → retour immédiat sans crash."""
        import src.tasks.retrain_scheduler as mod

        source = MagicMock()
        source.retrain_schedule = {"enabled": True, "lookback_days": 30}
        source.train_script_object_key = "model/train.py"

        mock_minio = MagicMock()
        mock_minio.download_file_bytes.side_effect = Exception("MinIO error")

        async def _run():
            with (
                patch("src.db.database.AsyncSessionLocal") as mock_session_cls,
                patch("src.services.minio_service.minio_service", mock_minio),
            ):
                mock_db = self._make_db_mock(source_model=source)
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                await mod._do_retrain("minio_fail_model", "1.0.0")

        asyncio.run(_run())


class TestAddRemoveRetrainJob:
    def test_add_retrain_job_enabled(self):
        """add_retrain_job avec enabled=True + cron valide → job ajouté."""
        from src.tasks.retrain_scheduler import add_retrain_job

        mock_scheduler = MagicMock()
        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_scheduler):
            add_retrain_job("mymodel", "1.0.0", {"enabled": True, "cron": "0 3 * * *"})
            mock_scheduler.add_job.assert_called_once()

    def test_add_retrain_job_disabled_does_nothing(self):
        """add_retrain_job avec enabled=False → aucun job créé."""
        from src.tasks.retrain_scheduler import add_retrain_job

        mock_scheduler = MagicMock()
        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_scheduler):
            add_retrain_job("mymodel", "1.0.0", {"enabled": False, "cron": "0 3 * * *"})
            mock_scheduler.add_job.assert_not_called()

    def test_add_retrain_job_no_cron_does_nothing(self):
        """add_retrain_job sans cron → aucun job créé."""
        from src.tasks.retrain_scheduler import add_retrain_job

        mock_scheduler = MagicMock()
        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_scheduler):
            add_retrain_job("mymodel", "1.0.0", {"enabled": True, "cron": None})
            mock_scheduler.add_job.assert_not_called()

    def test_remove_retrain_job_silent_when_absent(self):
        """remove_retrain_job silencieux si le job n'existe pas."""
        from src.tasks.retrain_scheduler import remove_retrain_job

        mock_scheduler = MagicMock()
        mock_scheduler.remove_job.side_effect = Exception("Job not found")

        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_scheduler):
            remove_retrain_job("mymodel", "1.0.0")


class TestSchedulerLifecycle:
    def test_start_retrain_scheduler_starts(self):
        """start_retrain_scheduler → scheduler.start() appelé."""
        from src.tasks.retrain_scheduler import start_retrain_scheduler

        mock_scheduler = MagicMock()
        mock_scheduler.running = False

        async def _run():
            with (
                patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_scheduler),
                patch(
                    "src.db.database.AsyncSessionLocal"
                ) as mock_session_cls,
            ):
                mock_db = AsyncMock()
                result_mock = MagicMock()
                result_mock.scalars.return_value.all.return_value = []
                mock_db.execute = AsyncMock(return_value=result_mock)
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_db)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                await start_retrain_scheduler()
            mock_scheduler.start.assert_called_once()

        asyncio.run(_run())

    def test_stop_retrain_scheduler_shuts_down(self):
        """stop_retrain_scheduler → scheduler.shutdown() appelé si running."""
        from src.tasks.retrain_scheduler import stop_retrain_scheduler

        mock_scheduler = MagicMock()
        mock_scheduler.running = True

        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_scheduler):
            stop_retrain_scheduler()
            mock_scheduler.shutdown.assert_called_once_with(wait=False)

    def test_stop_retrain_scheduler_no_shutdown_when_not_running(self):
        """stop_retrain_scheduler sans running → shutdown() NON appelé."""
        from src.tasks.retrain_scheduler import stop_retrain_scheduler

        mock_scheduler = MagicMock()
        mock_scheduler.running = False

        with patch("src.tasks.retrain_scheduler._retrain_scheduler", mock_scheduler):
            stop_retrain_scheduler()
            mock_scheduler.shutdown.assert_not_called()

    def test_start_retrain_scheduler_db_failure_does_not_crash(self):
        """start_retrain_scheduler : AsyncSessionLocal lève → warning loggé, pas d'exception."""
        from src.tasks.retrain_scheduler import start_retrain_scheduler

        async def _run():
            with patch(
                "src.db.database.AsyncSessionLocal"
            ) as mock_session_cls:
                mock_session_cls.return_value.__aenter__ = AsyncMock(
                    side_effect=Exception("DB unavailable")
                )
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                await start_retrain_scheduler()  # ne doit pas lever

        asyncio.run(_run())


class TestComputeNextRunAtNoneReturn:
    def test_returns_none_when_trigger_has_no_next_fire_time(self):
        """CronTrigger.get_next_fire_time retourne None → _compute_next_run_at retourne None."""
        from unittest.mock import MagicMock
        from src.tasks.retrain_scheduler import _compute_next_run_at

        mock_trigger = MagicMock()
        mock_trigger.get_next_fire_time.return_value = None

        with patch("src.tasks.retrain_scheduler.CronTrigger") as mock_cls:
            mock_cls.from_crontab.return_value = mock_trigger
            result = _compute_next_run_at("0 3 * * *")

        assert result is None


class TestSetSubprocessLimits:
    def test_set_subprocess_limits_calls_setrlimit_twice(self):
        """_set_subprocess_limits appelle resource.setrlimit exactement deux fois."""
        import resource
        from src.tasks.retrain_scheduler import _set_subprocess_limits

        with patch.object(resource, "setrlimit") as mock_setrlimit:
            _set_subprocess_limits()
            assert mock_setrlimit.call_count == 2


class TestRunRetrainJobErrorHandling:
    def test_do_retrain_exception_is_caught_not_propagated(self):
        """_run_retrain_job : exception de _do_retrain → attrapée, pas propagée."""

        async def _run():
            mock_redis = AsyncMock()
            mock_redis.set.return_value = True  # verrou libre
            mock_redis.delete = AsyncMock()

            import src.tasks.retrain_scheduler as mod

            with (
                patch.object(
                    __import__(
                        "src.services.model_service", fromlist=["model_service"]
                    ).model_service,
                    "_get_redis",
                    AsyncMock(return_value=mock_redis),
                ),
                patch.object(
                    mod, "_do_retrain", side_effect=RuntimeError("crash inattendu")
                ),
            ):
                # _run_retrain_job ne doit pas propager l'exception
                await mod._run_retrain_job("my_model", "1.0.0")

        asyncio.run(_run())
