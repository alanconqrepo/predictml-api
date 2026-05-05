"""
Tests pour le cycle de vie (lifespan) de src/main.py.

Couvre les branches try/except du startup et du shutdown :
- Échec Alembic → warning, app démarre quand même
- Échec init_db → warning, app démarre quand même
- Échec start_scheduler (supervision) → warning, app démarre quand même
- Échec start_retrain_scheduler → warning, app démarre quand même
- Shutdown propre (fermeture DB + Redis)
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLifespanStartupFailures:
    """Les exceptions dans le startup ne doivent jamais crasher l'app."""

    def test_migration_failure_does_not_crash(self):
        """Échec Alembic → warning loggé, lifespan continue."""
        from src.main import lifespan
        from src.main import app

        async def _run():
            with (
                patch("src.main.run_migrations", side_effect=Exception("alembic error")),
                patch("src.main.init_db", new_callable=AsyncMock),
                patch("src.main.start_retrain_scheduler", new_callable=AsyncMock),
                patch("src.tasks.supervision_reporter.start_scheduler"),
                patch("src.main.close_db", new_callable=AsyncMock),
                patch("src.main.model_service.close", new_callable=AsyncMock),
            ):
                async with lifespan(app):
                    pass  # startup réussi malgré l'exception

        asyncio.run(_run())

    def test_init_db_failure_does_not_crash(self):
        """Échec init_db → warning loggé, lifespan continue."""
        from src.main import lifespan
        from src.main import app

        async def _run():
            with (
                patch("src.main.run_migrations", new_callable=AsyncMock),
                patch("src.main.init_db", side_effect=Exception("db init error")),
                patch("src.main.start_retrain_scheduler", new_callable=AsyncMock),
                patch("src.tasks.supervision_reporter.start_scheduler"),
                patch("src.main.close_db", new_callable=AsyncMock),
                patch("src.main.model_service.close", new_callable=AsyncMock),
            ):
                async with lifespan(app):
                    pass

        asyncio.run(_run())

    def test_supervision_scheduler_failure_does_not_crash(self):
        """Échec start_scheduler supervision → warning loggé, lifespan continue."""
        from src.main import lifespan
        from src.main import app

        async def _run():
            with (
                patch("src.main.run_migrations", new_callable=AsyncMock),
                patch("src.main.init_db", new_callable=AsyncMock),
                patch("src.main.start_retrain_scheduler", new_callable=AsyncMock),
                patch(
                    "src.tasks.supervision_reporter.start_scheduler",
                    side_effect=Exception("scheduler error"),
                ),
                patch("src.main.close_db", new_callable=AsyncMock),
                patch("src.main.model_service.close", new_callable=AsyncMock),
            ):
                async with lifespan(app):
                    pass

        asyncio.run(_run())

    def test_retrain_scheduler_failure_does_not_crash(self):
        """Échec start_retrain_scheduler → warning loggé, lifespan continue."""
        from src.main import lifespan
        from src.main import app

        async def _run():
            with (
                patch("src.main.run_migrations", new_callable=AsyncMock),
                patch("src.main.init_db", new_callable=AsyncMock),
                patch(
                    "src.main.start_retrain_scheduler",
                    side_effect=Exception("retrain scheduler error"),
                ),
                patch("src.tasks.supervision_reporter.start_scheduler"),
                patch("src.main.close_db", new_callable=AsyncMock),
                patch("src.main.model_service.close", new_callable=AsyncMock),
            ):
                async with lifespan(app):
                    pass

        asyncio.run(_run())


class TestLifespanShutdown:
    """Le shutdown doit fermer proprement les ressources."""

    def test_shutdown_closes_db(self):
        """Shutdown → close_db appelé."""
        from src.main import lifespan
        from src.main import app

        mock_close_db = AsyncMock()

        async def _run():
            with (
                patch("src.main.run_migrations", new_callable=AsyncMock),
                patch("src.main.init_db", new_callable=AsyncMock),
                patch("src.main.start_retrain_scheduler", new_callable=AsyncMock),
                patch("src.tasks.supervision_reporter.start_scheduler"),
                patch("src.main.close_db", mock_close_db),
                patch("src.main.model_service.close", new_callable=AsyncMock),
            ):
                async with lifespan(app):
                    pass

        asyncio.run(_run())
        mock_close_db.assert_called_once()

    def test_shutdown_closes_redis(self):
        """Shutdown → model_service.close appelé."""
        from src.main import lifespan
        from src.main import app

        mock_model_close = AsyncMock()

        async def _run():
            with (
                patch("src.main.run_migrations", new_callable=AsyncMock),
                patch("src.main.init_db", new_callable=AsyncMock),
                patch("src.main.start_retrain_scheduler", new_callable=AsyncMock),
                patch("src.tasks.supervision_reporter.start_scheduler"),
                patch("src.main.close_db", new_callable=AsyncMock),
                patch("src.main.model_service.close", mock_model_close),
            ):
                async with lifespan(app):
                    pass

        asyncio.run(_run())
        mock_model_close.assert_called_once()
