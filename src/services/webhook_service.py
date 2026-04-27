"""
Service d'envoi de webhooks sortants post-prédiction
"""

import asyncio

import httpx
import structlog

logger = structlog.get_logger(__name__)


async def send_webhook(
    url: str,
    payload: dict,
    event_type: str | None = None,
) -> None:
    """Envoie un POST async vers l'URL de callback.

    Si event_type est fourni, l'inclut sous la clé "event" dans le payload.
    Retry max 3 fois avec backoff exponentiel (1 s, 2 s, 4 s). Timeout 5 s.
    Les erreurs sont loggées sans bloquer l'appelant.
    """
    if event_type is not None:
        payload = {"event": event_type, **payload}

    last_error: Exception | None = None
    for attempt in range(4):  # 1 tentative initiale + 3 retries
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=payload)
                logger.info(
                    "webhook_sent",
                    url=url,
                    status_code=response.status_code,
                    webhook_event=event_type,
                )
                return
        except Exception as e:
            last_error = e
            if attempt < 3:
                await asyncio.sleep(2**attempt)  # 1 s, 2 s, 4 s

    logger.warning("webhook_failed", url=url, error=str(last_error), webhook_event=event_type)
