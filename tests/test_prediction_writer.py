"""
Tests unitaires du worker prediction_writer.

- DB : SQLite en mémoire (FKs non enforced par défaut → pas besoin de créer un User)
- Redis : FakeRedis en mémoire (supporte les Streams)
- Pas de serveur externe requis
"""

import asyncio
import json
import os
import tempfile

import fakeredis.aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

# Variables d'env requises par src.core.config avant tout import de src/
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-do-not-use-in-production")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-minio-access-key")
os.environ.setdefault("MINIO_SECRET_KEY", "test-minio-secret-key-safe-value")

from src.db.database import Base
from src.db.models.prediction import Prediction
from src.workers.prediction_writer import (
    GROUP,
    STREAM,
    _deserialize,
    _ensure_consumer_group,
    _flush_batch,
    _move_to_dlq,
)

# ---------------------------------------------------------------------------
# Infrastructure de test
# ---------------------------------------------------------------------------

# Fichier SQLite temporaire : NullPool + :memory: perdrait les tables entre sessions
_test_db_file = os.path.join(tempfile.gettempdir(), f"predictml_worker_test_{os.getpid()}.db")
_engine = create_async_engine(
    f"sqlite+aiosqlite:///{_test_db_file}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
    echo=False,
)
_SessionFactory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def _create_tables():
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


asyncio.run(_create_tables())

# Payload représentatif d'une prédiction réussie dans le stream
_SAMPLE_FIELDS = {
    "user_id": "42",
    "model_name": "iris",
    "model_version": "1.0.0",
    "id_obs": "obs-test-001",
    "input_features": json.dumps({"f1": 1.5, "f2": 2.3, "f3": 0.7}),
    "prediction_result": json.dumps("setosa"),
    "probabilities": json.dumps([0.9, 0.06, 0.04]),
    "response_time_ms": "14.2",
    "client_ip": "192.168.1.10",
    "user_agent": "pytest/8.0",
    "status": "success",
    "error_message": "",
    "is_shadow": "false",
    "max_confidence": "0.9",
}


def _make_redis():
    """Retourne un FakeRedis frais avec consumer group initialisé."""

    async def _init():
        redis = fakeredis.aioredis.FakeRedis()
        await _ensure_consumer_group(redis)
        return redis

    return asyncio.run(_init())


# ---------------------------------------------------------------------------
# Tests _deserialize
# ---------------------------------------------------------------------------


def test_deserialize_basic_fields():
    obj = _deserialize(_SAMPLE_FIELDS)
    assert obj.user_id == 42
    assert obj.model_name == "iris"
    assert obj.model_version == "1.0.0"
    assert obj.id_obs == "obs-test-001"
    assert obj.input_features == {"f1": 1.5, "f2": 2.3, "f3": 0.7}
    assert obj.prediction_result == "setosa"
    assert obj.probabilities == [0.9, 0.06, 0.04]
    assert obj.response_time_ms == 14.2
    assert obj.client_ip == "192.168.1.10"
    assert obj.user_agent == "pytest/8.0"
    assert obj.status == "success"
    assert obj.is_shadow is False
    assert obj.max_confidence == 0.9


def test_deserialize_optional_fields_empty_string():
    fields = dict(_SAMPLE_FIELDS)
    fields["model_version"] = ""
    fields["id_obs"] = ""
    fields["client_ip"] = ""
    fields["user_agent"] = ""
    fields["error_message"] = ""
    fields["probabilities"] = ""
    fields["max_confidence"] = ""

    obj = _deserialize(fields)
    assert obj.model_version is None
    assert obj.id_obs is None
    assert obj.client_ip is None
    assert obj.user_agent is None
    assert obj.error_message is None
    assert obj.probabilities is None
    assert obj.max_confidence is None


def test_deserialize_shadow_flag():
    fields = dict(_SAMPLE_FIELDS)
    fields["is_shadow"] = "true"
    obj = _deserialize(fields)
    assert obj.is_shadow is True


def test_deserialize_error_status():
    fields = dict(_SAMPLE_FIELDS)
    fields["status"] = "error"
    fields["error_message"] = "Model not found"
    fields["prediction_result"] = json.dumps(None)
    obj = _deserialize(fields)
    assert obj.status == "error"
    assert obj.error_message == "Model not found"
    assert obj.prediction_result is None


# ---------------------------------------------------------------------------
# Tests _flush_batch — succès
# ---------------------------------------------------------------------------


def test_flush_batch_inserts_to_db():
    """Un batch de messages est correctement inséré en DB."""
    redis = _make_redis()

    async def _run():
        # Publier un message dans le stream
        msg_id = await redis.xadd(STREAM, _SAMPLE_FIELDS)

        # Lire le message comme le worker le ferait
        results = await redis.xreadgroup(GROUP, "test-consumer", {STREAM: ">"}, count=10)
        messages = results[0][1] if results else []

        await _flush_batch(_SessionFactory, redis, messages)

        # Vérifier l'insertion en DB
        async with _SessionFactory() as db:
            rows = (await db.execute(select(Prediction))).scalars().all()
        return rows

    rows = asyncio.run(_run())
    assert len(rows) >= 1
    last = rows[-1]
    assert last.model_name == "iris"
    assert last.prediction_result == "setosa"
    assert last.user_id == 42


def test_flush_batch_acks_on_success():
    """Après un INSERT réussi, les messages sont XACK'd (plus de pending)."""
    redis = _make_redis()

    async def _run():
        await redis.xadd(STREAM, _SAMPLE_FIELDS)
        results = await redis.xreadgroup(GROUP, "test-consumer", {STREAM: ">"}, count=10)
        messages = results[0][1]

        await _flush_batch(_SessionFactory, redis, messages)

        # Après XACK, XPENDING renvoie 0 messages en attente
        pending = await redis.xpending(STREAM, GROUP)
        return pending

    pending = asyncio.run(_run())
    assert pending["pending"] == 0


def test_flush_batch_multiple_messages():
    """Un batch de plusieurs messages est inséré en une seule transaction."""
    redis = _make_redis()

    async def _run():
        for i in range(5):
            fields = dict(_SAMPLE_FIELDS)
            fields["id_obs"] = f"bulk-obs-{i}"
            await redis.xadd(STREAM, fields)

        results = await redis.xreadgroup(GROUP, "test-consumer", {STREAM: ">"}, count=10)
        messages = results[0][1]
        before_count = len(messages)

        await _flush_batch(_SessionFactory, redis, messages)

        async with _SessionFactory() as db:
            rows = (
                await db.execute(
                    select(Prediction).where(Prediction.id_obs.like("bulk-obs-%"))
                )
            ).scalars().all()
        return before_count, rows

    batch_size, rows = asyncio.run(_run())
    assert batch_size == 5
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# Tests _flush_batch — erreur désérialisation → DLQ
# ---------------------------------------------------------------------------


def test_flush_batch_bad_json_goes_to_dlq():
    """Un message avec JSON malformé est déplacé en DLQ sans bloquer le batch."""
    from src.workers.prediction_writer import DLQ

    redis = _make_redis()

    async def _run():
        bad_fields = dict(_SAMPLE_FIELDS)
        bad_fields["input_features"] = "not-valid-json{"
        bad_fields["id_obs"] = "bad-json-obs"
        await redis.xadd(STREAM, bad_fields)

        results = await redis.xreadgroup(GROUP, "test-consumer", {STREAM: ">"}, count=10)
        messages = results[0][1]

        await _flush_batch(_SessionFactory, redis, messages)

        dlq_len = await redis.xlen(DLQ)
        pending = await redis.xpending(STREAM, GROUP)
        return dlq_len, pending["pending"]

    dlq_len, pending = asyncio.run(_run())
    assert dlq_len >= 1
    assert pending == 0  # message acquitté après déplacement en DLQ


# ---------------------------------------------------------------------------
# Tests _move_to_dlq
# ---------------------------------------------------------------------------


def test_move_to_dlq_adds_message_and_acks():
    """_move_to_dlq publie dans la DLQ et acquitte du stream principal."""
    from src.workers.prediction_writer import DLQ

    redis = _make_redis()

    async def _run():
        msg_id = await redis.xadd(STREAM, _SAMPLE_FIELDS)
        # Consommer pour mettre en pending
        await redis.xreadgroup(GROUP, "test-consumer", {STREAM: ">"}, count=1)

        await _move_to_dlq(redis, msg_id, _SAMPLE_FIELDS, "test_reason")

        dlq_msgs = await redis.xrange(DLQ)
        pending = await redis.xpending(STREAM, GROUP)
        return dlq_msgs, pending["pending"]

    dlq_msgs, pending = asyncio.run(_run())
    assert len(dlq_msgs) >= 1
    # FakeRedis retourne des bytes — le worker decode via _decode_fields → str dans le DLQ
    _, raw_fields = dlq_msgs[-1]
    fields = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in raw_fields.items()}
    assert fields.get("_dlq_reason") == "test_reason"
    assert pending == 0


# ---------------------------------------------------------------------------
# Tests _ensure_consumer_group — idempotence
# ---------------------------------------------------------------------------


def test_ensure_consumer_group_idempotent():
    """Appeler _ensure_consumer_group deux fois ne lève pas d'exception."""
    redis = _make_redis()

    async def _run():
        # Déjà créé par _make_redis() ; un second appel doit être silencieux
        await _ensure_consumer_group(redis)

    asyncio.run(_run())  # Ne doit pas lever


# ---------------------------------------------------------------------------
# Tests _publish_prediction_to_stream (helper API-side)
# ---------------------------------------------------------------------------


def test_publish_to_stream_success(monkeypatch):
    """_publish_prediction_to_stream retourne True et ajoute un message dans le stream."""
    from src.workers.prediction_writer import STREAM as W_STREAM
    from src.api.predict import _publish_prediction_to_stream
    from src.services import model_service as ms_module

    redis = _make_redis()

    async def _fake_get_redis():
        return redis

    monkeypatch.setattr(ms_module.model_service, "_get_redis", _fake_get_redis)

    payload = {
        "user_id": 1,
        "model_name": "iris",
        "model_version": "1.0.0",
        "id_obs": "pub-test",
        "input_features": {"f1": 1.0},
        "prediction_result": "setosa",
        "probabilities": [0.9, 0.1],
        "response_time_ms": 10.0,
        "client_ip": "127.0.0.1",
        "user_agent": None,
        "status": "success",
        "error_message": None,
        "is_shadow": False,
        "max_confidence": 0.9,
    }

    async def _run():
        result = await _publish_prediction_to_stream(payload)
        length = await redis.xlen(W_STREAM)
        return result, length

    ok, length = asyncio.run(_run())
    assert ok is True
    assert length >= 1


def test_publish_to_stream_fallback_on_redis_error(monkeypatch):
    """Si Redis est indisponible, _publish_prediction_to_stream retourne False (fallback sync)."""
    from src.api.predict import _publish_prediction_to_stream
    from src.services import model_service as ms_module

    async def _failing_get_redis():
        raise ConnectionError("Redis indisponible")

    monkeypatch.setattr(ms_module.model_service, "_get_redis", _failing_get_redis)

    payload = {
        "user_id": 1,
        "model_name": "iris",
        "model_version": "1.0.0",
        "id_obs": None,
        "input_features": {"f1": 1.0},
        "prediction_result": 0,
        "probabilities": None,
        "response_time_ms": 5.0,
        "client_ip": None,
        "user_agent": None,
        "status": "success",
        "error_message": None,
        "is_shadow": False,
        "max_confidence": None,
    }

    result = asyncio.run(_publish_prediction_to_stream(payload))
    assert result is False
