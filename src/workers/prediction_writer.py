"""
Worker autonome — consomme Redis Stream "predictions:new" et insère en batch dans PostgreSQL.

Lancement : python -m src.workers.prediction_writer

Fonctionnement :
- XREADGROUP avec consumer group "prediction-writers" (at-least-once delivery)
- Batch INSERT via SQLAlchemy add_all + commit
- XACK uniquement après succès ; en cas d'erreur DB les messages restent pending
- XAUTOCLAIM périodique pour récupérer les messages bloqués
- Dead-Letter Queue (predictions:dlq) après MAX_RETRIES livraisons échouées
- Arrêt propre sur SIGTERM / SIGINT
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

# Assure que le package src/ est trouvable quand on lance le script depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.config import settings
from src.db.models.prediction import Prediction
from src.services.model_service import model_service

logger = structlog.get_logger(__name__)

STREAM = settings.PREDICTION_STREAM_NAME
DLQ = settings.PREDICTION_STREAM_DLQ
GROUP = "prediction-writers"
CONSUMER = f"worker-{os.getpid()}"
BATCH_SIZE = settings.PREDICTION_STREAM_BATCH_SIZE
FLUSH_MS = settings.PREDICTION_STREAM_FLUSH_MS
MAX_RETRIES = settings.PREDICTION_STREAM_MAX_RETRIES

# Délai avant de reconsidérer un message pending comme "orphelin" (ms)
PENDING_CLAIM_MIN_IDLE_MS = 30_000
# Intervalle de vérification des messages pending orphelins (secondes)
PENDING_CHECK_INTERVAL = 60


def _to_optional_str(value: str) -> Optional[str]:
    return value if value else None


def _decode_fields(fields: dict) -> dict:
    """Normalise les clés/valeurs bytes → str (redis.asyncio sans decode_responses)."""
    result = {}
    for k, v in fields.items():
        key = k.decode() if isinstance(k, bytes) else k
        val = v.decode() if isinstance(v, bytes) else v
        result[key] = val
    return result


def _deserialize(fields: dict) -> Prediction:
    """Convertit les champs bruts du stream en objet ORM Prediction."""
    fields = _decode_fields(fields)
    raw_proba = fields.get("probabilities", "")
    raw_conf = fields.get("max_confidence", "")

    return Prediction(
        user_id=int(fields["user_id"]),
        model_name=fields["model_name"],
        model_version=_to_optional_str(fields.get("model_version", "")),
        id_obs=_to_optional_str(fields.get("id_obs", "")),
        input_features=json.loads(fields["input_features"]),
        prediction_result=json.loads(fields["prediction_result"]),
        probabilities=json.loads(raw_proba) if raw_proba else None,
        response_time_ms=float(fields["response_time_ms"]),
        client_ip=_to_optional_str(fields.get("client_ip", "")),
        user_agent=_to_optional_str(fields.get("user_agent", "")),
        status=fields.get("status", "success"),
        error_message=_to_optional_str(fields.get("error_message", "")),
        is_shadow=fields.get("is_shadow", "false") == "true",
        max_confidence=float(raw_conf) if raw_conf else None,
    )


async def _move_to_dlq(redis, msg_id: str, fields: dict, reason: str) -> None:
    """Déplace un message dans la DLQ et l'acquitte du stream principal."""
    try:
        dlq_payload = _decode_fields(fields)
        dlq_payload["_dlq_reason"] = reason
        dlq_payload["_original_id"] = msg_id
        dlq_payload["_failed_at"] = datetime.utcnow().isoformat()
        await redis.xadd(DLQ, dlq_payload)
        await redis.xack(STREAM, GROUP, msg_id)
        logger.error(
            "Message déplacé en DLQ",
            msg_id=msg_id,
            reason=reason,
        )
    except Exception as exc:
        logger.error("Impossible de déplacer le message en DLQ", msg_id=msg_id, error=str(exc))


async def _flush_batch(
    session_factory: async_sessionmaker,
    redis,
    messages: list[tuple[str, dict]],
) -> None:
    """Insère un batch de messages en DB et acquitte les succès."""
    orm_objects: list[Prediction] = []
    valid_ids: list[str] = []
    dlq_items: list[tuple[str, dict, str]] = []

    for msg_id, fields in messages:
        try:
            obj = _deserialize(fields)
            orm_objects.append(obj)
            valid_ids.append(msg_id)
        except Exception as exc:
            dlq_items.append((msg_id, fields, f"deserialize_error: {exc}"))

    # Traiter les erreurs de désérialisation → DLQ immédiat
    for msg_id, fields, reason in dlq_items:
        await _move_to_dlq(redis, msg_id, fields, reason)

    if not orm_objects:
        return

    async with session_factory() as db:
        try:
            db.add_all(orm_objects)
            await db.commit()
            await redis.xack(STREAM, GROUP, *valid_ids)
            logger.info("Batch inséré", count=len(orm_objects))
        except Exception as exc:
            await db.rollback()
            # Ne pas XACK : les messages restent pending pour retry via XAUTOCLAIM
            logger.error(
                "Erreur DB lors du batch INSERT — messages en attente de retry",
                count=len(orm_objects),
                error=str(exc),
            )


async def _reclaim_pending(session_factory: async_sessionmaker, redis) -> None:
    """Réclame les messages pending depuis trop longtemps et les retraite ou les envoie en DLQ."""
    try:
        # XAUTOCLAIM : récupère les messages idle depuis PENDING_CLAIM_MIN_IDLE_MS
        result = await redis.xautoclaim(
            STREAM,
            GROUP,
            CONSUMER,
            min_idle_time=PENDING_CLAIM_MIN_IDLE_MS,
            start_id="0-0",
            count=BATCH_SIZE,
        )
        # result = (next_start_id, [(msg_id, fields), ...], [deleted_ids])
        claimed_messages = result[1] if result and len(result) > 1 else []

        if not claimed_messages:
            return

        # Vérifier le nombre de livraisons via XPENDING pour chaque message
        to_retry: list[tuple[str, dict]] = []
        for msg_id, fields in claimed_messages:
            pending_info = await redis.xpending_range(STREAM, GROUP, msg_id, msg_id, 1)
            delivery_count = pending_info[0]["times_delivered"] if pending_info else 1

            if delivery_count > MAX_RETRIES:
                await _move_to_dlq(
                    redis,
                    msg_id,
                    fields,
                    f"max_retries_exceeded: {delivery_count} livraisons",
                )
            else:
                to_retry.append((msg_id, fields))

        if to_retry:
            logger.info("Retry de messages pending", count=len(to_retry))
            await _flush_batch(session_factory, redis, to_retry)

    except Exception as exc:
        logger.warning("Erreur lors du reclaim des messages pending", error=str(exc))


async def _ensure_consumer_group(redis) -> None:
    """Crée le consumer group si inexistant (MKSTREAM crée le stream si nécessaire)."""
    try:
        await redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        logger.info("Consumer group créé", stream=STREAM, group=GROUP)
    except Exception as exc:
        # BUSYGROUP = groupe déjà existant → normal au redémarrage
        if "BUSYGROUP" in str(exc):
            logger.debug("Consumer group déjà existant", stream=STREAM, group=GROUP)
        else:
            raise


async def run() -> None:
    """Boucle principale du worker."""
    engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        echo=False,
    )
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    redis = await model_service._get_redis()
    await _ensure_consumer_group(redis)

    running = True
    last_pending_check = asyncio.get_event_loop().time()

    def _stop(sig, frame):
        nonlocal running
        logger.info("Signal d'arrêt reçu", signal=sig)
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logger.info(
        "Worker démarré",
        stream=STREAM,
        group=GROUP,
        consumer=CONSUMER,
        batch_size=BATCH_SIZE,
        flush_ms=FLUSH_MS,
    )

    while running:
        try:
            results = await redis.xreadgroup(
                GROUP,
                CONSUMER,
                {STREAM: ">"},
                count=BATCH_SIZE,
                block=FLUSH_MS,
            )

            if results:
                for _stream_name, messages in results:
                    await _flush_batch(session_factory, redis, messages)

            # Vérification périodique des messages pending orphelins
            now = asyncio.get_event_loop().time()
            if now - last_pending_check >= PENDING_CHECK_INTERVAL:
                await _reclaim_pending(session_factory, redis)
                last_pending_check = now

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Erreur inattendue dans la boucle principale", error=str(exc))
            await asyncio.sleep(1)

    logger.info("Worker arrêté proprement")
    await engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
