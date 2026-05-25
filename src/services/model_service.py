"""
ML model management service (v3 - distributed Redis cache)
"""

import asyncio
import hashlib
import hmac as _hmac_mod
import io
import random
from typing import Any, Dict, List, Optional, Tuple

import joblib
import redis.asyncio as aioredis
import structlog
from fastapi import HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models.model_metadata import DeploymentMode, ModelMetadata
from src.services.db_service import DBService
from src.services.minio_service import minio_service

logger = structlog.get_logger(__name__)

_CACHE_PREFIX = "model:"
_HMAC_HEX_LEN = 64  # SHA-256 hex digest = 64 chars

# Intra-process locks per cache key — avoids concurrent MinIO downloads
# within the same worker on a cache miss (thundering herd).
_LOAD_LOCKS: Dict[str, asyncio.Lock] = {}


def _build_sentinel_redis() -> aioredis.Redis:
    """Return a Redis client connected to the Sentinel-elected master."""
    from urllib.parse import urlparse

    from redis.asyncio.sentinel import Sentinel

    hosts = [
        (part.strip().rsplit(":", 1)[0], int(part.strip().rsplit(":", 1)[1]))
        for part in settings.REDIS_SENTINEL_HOSTS.split(",")
        if part.strip()
    ]
    redis_password = urlparse(settings.REDIS_URL).password or None
    sentinel = Sentinel(hosts, password=redis_password, socket_timeout=0.5)
    return sentinel.master_for("mymaster", decode_responses=False)


def compute_model_hmac(data: bytes) -> str:
    """Return HMAC-SHA256 hex digest of data, keyed with SECRET_KEY."""
    return _hmac_mod.new(settings.SECRET_KEY.encode(), data, hashlib.sha256).hexdigest()


def _sign_for_cache(payload: bytes) -> bytes:
    """Prepend 64-char HMAC hex to payload for tamper-evident Redis storage."""
    return compute_model_hmac(payload).encode() + payload


def _verify_cache_signature(signed: bytes) -> bytes:
    """
    Verify the HMAC prefix and return the payload. Raises ValueError on mismatch.

    Compares bytes (not str) to avoid TypeError on non-ASCII input.
    """
    sig_bytes = signed[:_HMAC_HEX_LEN]
    payload = signed[_HMAC_HEX_LEN:]
    expected_bytes = compute_model_hmac(payload).encode()
    if not _hmac_mod.compare_digest(sig_bytes, expected_bytes):
        raise ValueError("Cache HMAC signature mismatch")
    return payload


def _verify_minio_hmac(raw_bytes: bytes, metadata: Any, model_name: str) -> None:
    """
    Verify raw pkl bytes against the HMAC-SHA256 stored in DB metadata.

    Raises HTTPException 403 if the model has no stored signature (legacy, un-signed).
    Raises HTTPException 500 if the signature does not match (tampering detected).
    """
    if not metadata.model_hmac_signature:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Model '{model_name}' v{metadata.version} has no HMAC signature. "
                "Run init_data/resign_models.py to re-sign existing models "
                "before they can be loaded."
            ),
        )
    expected = compute_model_hmac(raw_bytes)
    if not _hmac_mod.compare_digest(expected, metadata.model_hmac_signature):
        logger.error(
            "Invalid HMAC signature — loading refused",
            model=model_name,
            version=str(metadata.version),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Model '{model_name}' v{metadata.version} signature invalid — "
                "loading refused (possible model file tampering)."
            ),
        )


def _extract_model_from_payload(payload: bytes) -> Any:
    """
    Deserialize the cache payload into a model object.

    Handles two formats for backward compatibility:
    - New format: raw pickle of the model object directly.
    - Old format (legacy): pickle of {"model": obj, "metadata": ...}; model is extracted.

    The HMAC was already verified by the caller before this is invoked.
    """
    loaded = joblib.load(io.BytesIO(payload))  # caller verified HMAC
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]
    return loaded


class ModelService:
    """Service for loading and managing ML models from MinIO/MLflow, with Redis cache."""

    def __init__(self):
        # Redis connection lazily initialized on first access
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        """Return the Redis client (direct or via Sentinel), creating it if necessary."""
        if self._redis is None:
            if settings.REDIS_SENTINEL_HOSTS:
                self._redis = _build_sentinel_redis()
            else:
                self._redis = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=False)
        return self._redis

    async def get_available_models(
        self,
        db: AsyncSession,
        is_production: Optional[bool] = None,
        algorithm: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        deployment_mode: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the list of available models from the database.

        Args:
            db: Database session
            is_production: filter on is_production
            algorithm: exact filter on algorithm
            min_accuracy: filter accuracy >= value
            deployment_mode: filter on deployment mode
            search: text search on name and description (ILIKE)

        Returns:
            List of active models with their metadata
        """
        models = await DBService.get_all_active_models(
            db,
            is_production=is_production,
            algorithm=algorithm,
            min_accuracy=min_accuracy,
            deployment_mode=deployment_mode,
            search=search,
        )
        last_seen_map = await DBService.get_models_last_seen(db)
        return [
            {
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "is_production": m.is_production,
                "accuracy": m.accuracy,
                "f1_score": m.f1_score,
                "algorithm": m.algorithm,
                "mlflow_run_id": m.mlflow_run_id,
                "minio_bucket": m.minio_bucket,
                "minio_object_key": m.minio_object_key,
                "file_size_bytes": m.file_size_bytes,
                "features_count": m.features_count,
                "classes": m.classes,
                "tags": m.tags,
                "webhook_url": m.webhook_url,
                "is_active": m.is_active,
                "status": m.status,
                "traffic_weight": m.traffic_weight,
                "deployment_mode": m.deployment_mode,
                "train_script_object_key": m.train_script_object_key,
                "requirements_object_key": m.requirements_object_key,
                "feature_baseline": m.feature_baseline,
                "confidence_threshold": m.confidence_threshold,
                "training_dataset": m.training_dataset,
                "trained_by": m.trained_by,
                "training_metrics": m.training_metrics,
                "hyperparameters": m.hyperparameters,
                "precision": m.precision,
                "recall": m.recall,
                "model_task": m.model_task,
                "alert_thresholds": m.alert_thresholds,
                "promotion_policy": m.promotion_policy,
                "created_at": m.created_at,
                "user_id_creator": m.user_id_creator,
                "creator_username": m.creator.username if m.creator else None,
                "last_seen": last_seen_map.get(m.name),
            }
            for m in models
        ]

    async def load_model(
        self,
        db: AsyncSession,
        model_name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a model from the Redis cache or from MinIO/MLflow on a cache miss.

        Security:
        - Redis cache: verifies the HMAC-SHA256 envelope (64 ASCII bytes) before joblib.load().
          Protects against Redis cache poisoning (attacker does not know SECRET_KEY).
        - MinIO: verifies the HMAC signature stored in DB before joblib.load().
          Protects against tampering of the .joblib file in MinIO.

        Redis format: HMAC_HEX(64 ASCII bytes) + payload_bytes

        Args:
            db: Database session
            model_name: Model name
            version: Specific version (optional, takes most recent otherwise)

        Returns:
            Dict containing the model and its metadata

        Raises:
            HTTPException 403: MinIO model without HMAC signature (unsigned legacy model)
            HTTPException 500: Invalid HMAC signature (tampering detected)
            HTTPException 404: Model not found
        """
        # 1. Retrieve metadata from DB (source of truth for the signature)
        metadata = await DBService.get_model_metadata(db, model_name, version)

        if not metadata:
            available = await self.get_available_models(db)
            available_names = [m["name"] for m in available]
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' (version '{version or 'latest'}') not found. "
                f"Available models: {available_names}",
            )

        # 2. Check Redis cache (quick first check without lock)
        cache_key = f"{_CACHE_PREFIX}{model_name}:{metadata.version}"
        redis = await self._get_redis()

        cached_signed = await redis.get(cache_key)
        if cached_signed:
            try:
                payload = _verify_cache_signature(cached_signed)
            except ValueError:
                # Tampered or old-format (unsigned) cache entry — reject and reload from source
                logger.warning(
                    "Invalid Redis cache HMAC — entry invalidated",
                    cache_key=cache_key,
                    model=model_name,
                    version=str(metadata.version),
                )
                await redis.delete(cache_key)
            else:
                logger.info(
                    "Model loaded from Redis cache",
                    model_name=model_name,
                    version=str(metadata.version),
                )
                # HMAC envelope verified; extract model (handles both old and new formats)
                model = _extract_model_from_payload(payload)
                return {"model": model, "metadata": metadata}

        # 3. Cache miss — acquire intra-process lock to avoid thundering herd.
        # Multiple concurrent coroutines in the same worker will only trigger one
        # MinIO download thanks to double-checked locking.
        if cache_key not in _LOAD_LOCKS:
            _LOAD_LOCKS[cache_key] = asyncio.Lock()
        lock = _LOAD_LOCKS[cache_key]

        async with lock:
            # Double-check after lock acquisition (another coroutine may have already loaded)
            cached_signed = await redis.get(cache_key)
            if cached_signed:
                try:
                    payload = _verify_cache_signature(cached_signed)
                except ValueError:
                    await redis.delete(cache_key)
                else:
                    model = _extract_model_from_payload(payload)
                    return {"model": model, "metadata": metadata}

            # 4. Load the model from MinIO (operational source of truth).
            #    mlflow_run_id remains in DB as a traceability link (MLflow UI) but
            #    is no longer used for loading — ensures operation even if MLflow is
            #    disabled, unreachable, or absent at the time of initial upload.
            try:
                logger.info(
                    "Downloading model from MinIO",
                    model_name=model_name,
                    version=str(metadata.version),
                    minio_object_key=metadata.minio_object_key,
                )
                # Download raw bytes — do NOT call joblib.load() until HMAC is verified
                raw_bytes = await minio_service.async_download_file_bytes(metadata.minio_object_key)
                _verify_minio_hmac(raw_bytes, metadata, model_name)
                model = joblib.load(io.BytesIO(raw_bytes))  # HMAC verified above
                cache_payload = raw_bytes

                # 5. Store in Redis cache with HMAC envelope (configurable TTL)
                signed = _sign_for_cache(cache_payload)
                await redis.setex(cache_key, settings.REDIS_CACHE_TTL, signed)

                logger.info(
                    "Model loaded and cached in Redis",
                    model_name=model_name,
                    version=str(metadata.version),
                    ttl=settings.REDIS_CACHE_TTL,
                )
                return {"model": model, "metadata": metadata}

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error loading model '{model_name}': {str(e)}",
                )

    async def get_cached_models(self) -> List[str]:
        """
        Return the list of models currently in the Redis cache.

        Returns:
            List of keys in the format ``name:version``
        """
        redis = await self._get_redis()
        keys = await redis.keys(f"{_CACHE_PREFIX}*")
        return [k.decode().removeprefix(_CACHE_PREFIX) for k in keys]

    async def invalidate_model_cache(self, model_name: str, version: Optional[str] = None) -> None:
        """
        Invalidate Redis entries for a given model.

        Args:
            model_name: Model name.
            version: If provided, only invalidates this version.
                     If None, invalidates all versions of the model.
        """
        redis = await self._get_redis()
        if version:
            key = f"{_CACHE_PREFIX}{model_name}:{version}"
            await redis.delete(key)
            logger.info("Model cache invalidated", model=model_name, version=version)
        else:
            keys = await redis.keys(f"{_CACHE_PREFIX}{model_name}:*")
            if keys:
                await redis.delete(*keys)
            logger.info("Model cache invalidated (all versions)", model=model_name)

    async def clear_cache(self, cache_key: Optional[str] = None):
        """
        Invalidate the Redis cache.

        Args:
            cache_key: If provided (``name:version``), invalidates only that entry.
                       If None, invalidates all ``model:*`` entries.
        """
        redis = await self._get_redis()
        if cache_key:
            await redis.delete(f"{_CACHE_PREFIX}{cache_key}")
            logger.info("Redis cache invalidated", cache_key=cache_key)
        else:
            keys = await redis.keys(f"{_CACHE_PREFIX}*")
            if keys:
                await redis.delete(*keys)
            logger.info("Full Redis cache invalidated")

    async def select_routing_versions(
        self,
        db: AsyncSession,
        model_name: str,
    ) -> Tuple[Optional[ModelMetadata], List[ModelMetadata]]:
        """
        Determine which version(s) to use for a call without an explicit version.

        Selection rules:
        1. Versions with deployment_mode="ab_test" and traffic_weight > 0 participate
           in weighted routing (random.choices) → primary version.
           Unselected A/B candidates run automatically as shadow,
           allowing comparison of all versions on the same inputs.
        2. Versions with deployment_mode="shadow" are added to the shadow list.
        3. If no A/B version: fallback to get_model_metadata() (is_production=True
           or most recent).

        Returns:
            (primary_metadata, shadow_list)
            shadow_list may contain: unselected A/B candidates + dedicated shadow version.
        """
        result = await db.execute(
            select(ModelMetadata).where(
                and_(
                    ModelMetadata.name == model_name,
                    ModelMetadata.is_active.is_(True),
                    ModelMetadata.status != "deprecated",
                )
            )
        )
        all_versions = result.scalars().all()

        shadow_candidates = [m for m in all_versions if m.deployment_mode == DeploymentMode.SHADOW]
        ab_candidates = [
            m
            for m in all_versions
            if m.deployment_mode == DeploymentMode.AB_TEST
            and m.traffic_weight is not None
            and m.traffic_weight > 0.0
        ]

        if ab_candidates:
            weights = [m.traffic_weight for m in ab_candidates]
            primary_metadata = random.choices(ab_candidates, weights=weights, k=1)[0]
            # Les autres candidats A/B non sélectionnés tournent en shadow
            ab_shadows = [m for m in ab_candidates if m.version != primary_metadata.version]
            logger.info(
                "A/B routing",
                model_name=model_name,
                selected_version=primary_metadata.version,
                candidates=[m.version for m in ab_candidates],
                ab_shadows=[m.version for m in ab_shadows],
            )
        else:
            # Legacy fallback: is_production=True or most recent
            primary_metadata = await DBService.get_model_metadata(db, model_name, version=None)
            ab_shadows = []

        # Dedicated shadow (deployment_mode="shadow") + unselected A/B candidates
        shadow_list = ab_shadows + shadow_candidates

        return primary_metadata, shadow_list

    async def close(self):
        """Cleanly close the Redis connection (to be called at app shutdown)."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis connection closed")


# Global service instance
model_service = ModelService()
