"""
Singleton ARQ Redis pool — partagé entre l'API et les endpoints /jobs.

Usage :
    import src.core.arq_pool as arq_pool_module
    pool = await arq_pool_module.get_arq_pool()
    await pool.enqueue_job("retrain_task", ...)

Le fait de passer par le module (plutôt qu'un import direct) permet de
patcher ``src.core.arq_pool.get_arq_pool`` en tests pour injecter un mock.
"""

import structlog

logger = structlog.get_logger(__name__)

_arq_pool = None


async def get_arq_pool():
    """Retourne le pool ARQ (crée la connexion si nécessaire)."""
    global _arq_pool
    if _arq_pool is None:
        try:
            import arq
            from arq.connections import RedisSettings

            from src.core.config import settings

            # Construire les RedisSettings depuis l'URL Redis existante
            redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
            _arq_pool = await arq.create_pool(redis_settings)
            logger.info("ARQ pool Redis créé")
        except Exception as exc:
            logger.warning("ARQ pool indisponible — enqueue désactivé", error=str(exc))
            raise
    return _arq_pool


async def close_arq_pool() -> None:
    """Ferme proprement le pool ARQ."""
    global _arq_pool
    if _arq_pool is not None:
        try:
            await _arq_pool.aclose()
            logger.info("ARQ pool fermé")
        except Exception as exc:
            logger.warning("Erreur fermeture ARQ pool", error=str(exc))
        finally:
            _arq_pool = None
