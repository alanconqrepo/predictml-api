"""
Singleton ARQ Redis pool — shared between the API and /jobs endpoints.

Usage:
    import src.core.arq_pool as arq_pool_module
    pool = await arq_pool_module.get_arq_pool()
    await pool.enqueue_job("retrain_task", ...)

Going through the module (rather than a direct import) allows patching
``src.core.arq_pool.get_arq_pool`` in tests to inject a mock.
"""

import structlog

logger = structlog.get_logger(__name__)

_arq_pool = None


async def get_arq_pool():
    """Return the ARQ pool (creates the connection if necessary)."""
    global _arq_pool
    if _arq_pool is None:
        try:
            import arq
            from arq.connections import RedisSettings

            from src.core.config import settings

            # Build RedisSettings from the existing Redis URL
            redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
            _arq_pool = await arq.create_pool(redis_settings)
            logger.info("ARQ Redis pool created")
        except Exception as exc:
            logger.warning("ARQ pool unavailable — enqueue disabled", error=str(exc))
            raise
    return _arq_pool


async def close_arq_pool() -> None:
    """Gracefully close the ARQ pool."""
    global _arq_pool
    if _arq_pool is not None:
        try:
            await _arq_pool.aclose()
            logger.info("ARQ pool closed")
        except Exception as exc:
            logger.warning("Error closing ARQ pool", error=str(exc))
        finally:
            _arq_pool = None
