"""
Standalone worker — consumes the "predictions:new" Redis Stream and batch-inserts into PostgreSQL.

Launch: python -m src.workers.prediction_writer

How it works:
- XREADGROUP with consumer group "prediction-writers" (at-least-once delivery)
- Batch INSERT via SQLAlchemy add_all + commit
- XACK only on success; on DB error, messages remain pending for retry
- Periodic XAUTOCLAIM to reclaim stuck messages
- Dead-Letter Queue (predictions:dlq) after MAX_RETRIES failed deliveries
- Graceful shutdown on SIGTERM / SIGINT
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis
import structlog
from prometheus_client import Counter, Histogram, start_http_server
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

MESSAGES_PROCESSED = Counter(
    "prediction_writer_messages_processed_total",
    "Messages successfully inserted into PostgreSQL",
)
MESSAGES_FAILED = Counter(
    "prediction_writer_messages_failed_total",
    "Messages that failed DB insertion (left pending for retry)",
)
MESSAGES_DLQ = Counter(
    "prediction_writer_dlq_messages_total",
    "Messages moved to the dead-letter queue",
    ["reason"],
)
BATCH_INSERT_DURATION = Histogram(
    "prediction_writer_batch_insert_seconds",
    "Batch INSERT duration in PostgreSQL",
    buckets=[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Ensure the src/ package is importable when running the script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.config import settings  # noqa: E402
from src.db.models.prediction import Prediction  # noqa: E402

logger = structlog.get_logger(__name__)

STREAM = settings.PREDICTION_STREAM_NAME
DLQ = settings.PREDICTION_STREAM_DLQ
GROUP = "prediction-writers"
CONSUMER = f"worker-{os.getpid()}"
BATCH_SIZE = settings.PREDICTION_STREAM_BATCH_SIZE
FLUSH_MS = settings.PREDICTION_STREAM_FLUSH_MS
MAX_RETRIES = settings.PREDICTION_STREAM_MAX_RETRIES

# Idle time before a pending message is considered "orphaned" (ms)
PENDING_CLAIM_MIN_IDLE_MS = 30_000
# Interval for checking orphaned pending messages (seconds)
PENDING_CHECK_INTERVAL = 60


def _to_optional_str(value: str) -> Optional[str]:
    return value if value else None


def _decode_fields(fields: dict) -> dict:
    """Normalise bytes → str keys/values (redis.asyncio without decode_responses)."""
    result = {}
    for k, v in fields.items():
        key = k.decode() if isinstance(k, bytes) else k
        val = v.decode() if isinstance(v, bytes) else v
        result[key] = val
    return result


def _deserialize(fields: dict) -> Prediction:
    """Convert raw stream fields into a Prediction ORM object."""
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
    """Move a message to the DLQ and acknowledge it from the main stream."""
    try:
        dlq_payload = _decode_fields(fields)
        dlq_payload["_dlq_reason"] = reason
        dlq_payload["_original_id"] = msg_id
        dlq_payload["_failed_at"] = datetime.utcnow().isoformat()
        await redis.xadd(DLQ, dlq_payload)
        await redis.xack(STREAM, GROUP, msg_id)
        MESSAGES_DLQ.labels(reason=reason.split(":")[0]).inc()
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
    """Insert a batch of messages into the DB and acknowledge successful ones."""
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

    # Handle deserialization errors → immediate DLQ
    for msg_id, fields, reason in dlq_items:
        await _move_to_dlq(redis, msg_id, fields, reason)

    if not orm_objects:
        return

    async with session_factory() as db:
        try:
            with BATCH_INSERT_DURATION.time():
                db.add_all(orm_objects)
                await db.commit()
            await redis.xack(STREAM, GROUP, *valid_ids)
            MESSAGES_PROCESSED.inc(len(orm_objects))
            logger.info("Batch inséré", count=len(orm_objects))
        except Exception as exc:
            await db.rollback()
            MESSAGES_FAILED.inc(len(orm_objects))
            # Do not XACK: messages remain pending for retry via XAUTOCLAIM
            logger.error(
                "DB error during batch INSERT — messages pending for retry",
                count=len(orm_objects),
                error=str(exc),
            )


async def _reclaim_pending(session_factory: async_sessionmaker, redis) -> None:
    """Reclaim long-pending messages and retry them or send them to the DLQ."""
    try:
        # XAUTOCLAIM: fetch messages idle for at least PENDING_CLAIM_MIN_IDLE_MS
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

        # Check delivery count via XPENDING for each message
        to_retry: list[tuple[str, dict]] = []
        for msg_id, fields in claimed_messages:
            pending_info = await redis.xpending_range(STREAM, GROUP, msg_id, msg_id, 1)
            delivery_count = pending_info[0]["times_delivered"] if pending_info else 1

            if delivery_count > MAX_RETRIES:
                await _move_to_dlq(
                    redis,
                    msg_id,
                    fields,
                    f"max_retries_exceeded: {delivery_count} deliveries",
                )
            else:
                to_retry.append((msg_id, fields))

        if to_retry:
            logger.info("Retry de messages pending", count=len(to_retry))
            await _flush_batch(session_factory, redis, to_retry)

    except Exception as exc:
        logger.warning("Erreur lors du reclaim des messages pending", error=str(exc))


async def _ensure_consumer_group(redis) -> None:
    """Create the consumer group if it does not exist (MKSTREAM creates the stream if needed)."""
    try:
        await redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        logger.info("Consumer group created", stream=STREAM, group=GROUP)
    except Exception as exc:
        # BUSYGROUP = group already exists → normal on restart
        if "BUSYGROUP" in str(exc):
            logger.debug("Consumer group already exists", stream=STREAM, group=GROUP)
        else:
            raise


async def _build_redis() -> aioredis.Redis:
    """
    Dedicated direct Redis connection for the prediction-writer.

    Uses REDIS_URL (Docker DNS hostname → redis-master) instead of Sentinel.
    This avoids stale IPs cached by Sentinel after restarts.
    socket_timeout must be > FLUSH_MS (in seconds).
    """
    # ×5 margin over the XREADGROUP block time to avoid race conditions
    data_socket_timeout = max(2.0, (FLUSH_MS / 1000) * 5)
    return aioredis.Redis.from_url(
        settings.REDIS_URL,
        decode_responses=False,
        socket_timeout=data_socket_timeout,
        socket_connect_timeout=2.0,
    )


async def run() -> None:
    """Main worker loop."""
    engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        echo=False,
    )
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    redis = await _build_redis()
    await _ensure_consumer_group(redis)

    running = True
    last_pending_check = asyncio.get_event_loop().time()

    def _stop(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received", signal=sig)
        running = False

    if sys.platform != "win32":  # SIGTERM is not available on Windows
        signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logger.info(
        "Worker started",
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

            # Periodic check for orphaned pending messages
            now = asyncio.get_event_loop().time()
            if now - last_pending_check >= PENDING_CHECK_INTERVAL:
                await _reclaim_pending(session_factory, redis)
                last_pending_check = now

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Erreur inattendue dans la boucle principale", error=str(exc))
            await asyncio.sleep(1)

    logger.info("Worker shut down cleanly")
    await engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    start_http_server(9091)
    asyncio.run(run())


if __name__ == "__main__":
    main()
