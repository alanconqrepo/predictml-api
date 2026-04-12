"""
Service d'envoi de webhooks sortants post-prédiction
"""

import httpx
import structlog

logger = structlog.get_logger(__name__)


async def send_webhook(url: str, payload: dict) -> None:
    """Envoie un POST async vers l'URL de callback.

    Les erreurs (réseau, timeout, HTTP 4xx/5xx) sont loggées sans bloquer la réponse API.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            logger.info("webhook_sent", url=url, status_code=response.status_code)
    except Exception as e:
        logger.warning("webhook_failed", url=url, error=str(e))
