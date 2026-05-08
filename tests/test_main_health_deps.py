"""
Tests pour les branches non couvertes de src/main.py.

Couvre :
- GET /health quand exception interne → réponse dégradée (status=degraded)
- _check_db exception → DependencyDetail(status="error")
- _check_redis exception → DependencyDetail(status="error")
- _check_minio exception → DependencyDetail(status="error")
- _check_mlflow exception → DependencyDetail(status="error")
- lifespan shutdown : close_db() lève → warning loggé, fermeture continue
- lifespan shutdown : model_service.close() lève → warning loggé, fermeture continue
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

_ADMIN_TOKEN = "test-token-health-deps-001"


async def _setup():
    from src.services.db_service import DBService
    from tests.conftest import _TestSessionLocal

    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, _ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="health_deps_admin",
                email="health_deps_admin@test.com",
                api_token=_ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())

_ADMIN_HDR = {"Authorization": f"Bearer {_ADMIN_TOKEN}"}


# ---------------------------------------------------------------------------
# GET /health — chemin dégradé
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_degraded_on_exception():
    """Exception dans get_available_models → GET /health retourne status=degraded."""
    with patch(
        "src.main.model_service.get_available_models",
        side_effect=Exception("DB connection lost"),
    ):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["database"] == "error"


# ---------------------------------------------------------------------------
# _check_db — chemin d'erreur
# ---------------------------------------------------------------------------


def test_check_db_returns_error_on_exception():
    """_check_db : db.execute lève une exception → DependencyDetail status=error."""
    from src.main import _check_db

    mock_db = AsyncMock()
    mock_db.execute.side_effect = Exception("connexion perdue")

    result = asyncio.run(_check_db(mock_db))

    assert result.status == "error"
    assert result.latency_ms is None
    assert result.detail is not None


# ---------------------------------------------------------------------------
# _check_redis — chemin d'erreur
# ---------------------------------------------------------------------------


def test_check_redis_returns_error_on_exception():
    """_check_redis : redis.ping lève une exception → DependencyDetail status=error."""
    from src.main import _check_redis

    mock_redis = AsyncMock()
    mock_redis.ping.side_effect = Exception("redis down")

    with patch(
        "src.main.model_service._get_redis",
        AsyncMock(return_value=mock_redis),
    ):
        result = asyncio.run(_check_redis())

    assert result.status == "error"
    assert result.latency_ms is None


# ---------------------------------------------------------------------------
# _check_minio — chemin d'erreur
# ---------------------------------------------------------------------------


def test_check_minio_returns_error_on_exception():
    """_check_minio : bucket_exists lève une exception → DependencyDetail status=error."""
    from src.main import _check_minio

    with patch("src.main.minio_service") as mock_minio:
        mock_minio.client.bucket_exists.side_effect = Exception("minio down")

        result = asyncio.run(_check_minio())

    assert result.status == "error"
    assert result.latency_ms is None


# ---------------------------------------------------------------------------
# _check_mlflow — chemin d'erreur
# ---------------------------------------------------------------------------


def test_check_mlflow_returns_error_on_exception():
    """_check_mlflow : httpx lève une exception → DependencyDetail status=error."""
    from src.main import _check_mlflow

    with patch("src.main.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("mlflow unreachable")
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        result = asyncio.run(_check_mlflow())

    assert result.status == "error"
    assert result.latency_ms is None


# ---------------------------------------------------------------------------
# Lifespan shutdown — close_db / model_service.close failures
# ---------------------------------------------------------------------------


def test_shutdown_close_db_failure_does_not_crash():
    """close_db() lève → warning loggé, shutdown continue sans exception."""
    from src.main import lifespan

    async def _run():
        with (
            patch("src.main.run_migrations", new_callable=AsyncMock),
            patch("src.main.init_db", new_callable=AsyncMock),
            patch("src.tasks.supervision_reporter.start_scheduler"),
            patch(
                "src.tasks.retrain_scheduler.start_retrain_scheduler",
                new_callable=AsyncMock,
            ),
            patch("src.main.close_db", side_effect=Exception("DB fermeture échouée")),
            patch("src.main.model_service.close", new_callable=AsyncMock),
        ):
            async with lifespan(app):
                pass

    asyncio.run(_run())


def test_shutdown_model_service_close_failure_does_not_crash():
    """model_service.close() lève → warning loggé, shutdown continue."""
    from src.main import lifespan

    async def _run():
        with (
            patch("src.main.run_migrations", new_callable=AsyncMock),
            patch("src.main.init_db", new_callable=AsyncMock),
            patch("src.tasks.supervision_reporter.start_scheduler"),
            patch(
                "src.tasks.retrain_scheduler.start_retrain_scheduler",
                new_callable=AsyncMock,
            ),
            patch("src.main.close_db", new_callable=AsyncMock),
            patch(
                "src.main.model_service.close",
                side_effect=Exception("Redis fermeture échouée"),
            ),
        ):
            async with lifespan(app):
                pass

    asyncio.run(_run())
