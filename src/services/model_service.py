"""
Service de gestion des modèles ML (v3 - cache Redis distribué)
"""

import asyncio
import hashlib
import hmac as _hmac_mod
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

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

# Verrous intra-process par clé cache — évite les téléchargements MinIO simultanés
# dans le même worker lors d'un cache miss (thundering herd).
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
    if not metadata.pkl_hmac_signature:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Le modèle '{model_name}' v{metadata.version} n'a pas de signature HMAC. "
                "Exécutez init_data/resign_models.py pour re-signer les modèles existants "
                "avant de pouvoir les charger."
            ),
        )
    expected = compute_model_hmac(raw_bytes)
    if not _hmac_mod.compare_digest(expected, metadata.pkl_hmac_signature):
        logger.error(
            "Signature HMAC invalide — chargement refusé",
            model=model_name,
            version=str(metadata.version),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Signature du modèle '{model_name}' v{metadata.version} invalide — "
                "chargement refusé (possible falsification du fichier .pkl)."
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
    loaded = pickle.loads(payload)  # noqa: S301 - caller verified HMAC
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]
    return loaded


class ModelService:
    """Service pour charger et gérer les modèles ML depuis MinIO/MLflow, avec cache Redis."""

    def __init__(self):
        # Connexion Redis initialisée paresseusement au premier accès
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        """Retourne le client Redis (direct ou via Sentinel), en le créant si nécessaire."""
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
        Retourne la liste des modèles disponibles depuis la base de données

        Args:
            db: Session de base de données
            is_production: filtre sur is_production
            algorithm: filtre exact sur l'algorithme
            min_accuracy: filtre accuracy >= valeur
            deployment_mode: filtre sur le mode de déploiement
            search: recherche textuelle sur name et description (ILIKE)

        Returns:
            Liste des modèles actifs avec leurs métadonnées
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
                "feature_baseline": m.feature_baseline,
                "confidence_threshold": m.confidence_threshold,
                "training_dataset": m.training_dataset,
                "trained_by": m.trained_by,
                "training_metrics": m.training_metrics,
                "hyperparameters": m.hyperparameters,
                "precision": m.precision,
                "recall": m.recall,
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
        Charge un modèle depuis le cache Redis ou depuis MinIO/MLflow en cas de cache miss.

        Sécurité :
        - Cache Redis : vérifie l'enveloppe HMAC-SHA256 (64 octets ASCII) avant pickle.loads().
          Protège contre l'empoisonnement du cache Redis (l'attaquant ne connaît pas SECRET_KEY).
        - MinIO : vérifie la signature HMAC stockée en DB avant pickle.loads().
          Protège contre la falsification du fichier .pkl dans MinIO.

        Format Redis : HMAC_HEX(64 octets ASCII) + payload_bytes

        Args:
            db: Session de base de données
            model_name: Nom du modèle
            version: Version spécifique (optionnel, prend la plus récente sinon)

        Returns:
            Dict contenant le modèle et ses métadonnées

        Raises:
            HTTPException 403: Modèle MinIO sans signature HMAC (modèle legacy non signé)
            HTTPException 500: Signature HMAC invalide (falsification détectée)
            HTTPException 404: Modèle introuvable
        """
        # 1. Récupérer les métadonnées depuis la DB (source de vérité pour la signature)
        metadata = await DBService.get_model_metadata(db, model_name, version)

        if not metadata:
            available = await self.get_available_models(db)
            available_names = [m["name"] for m in available]
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modèle '{model_name}' (version '{version or 'latest'}') non trouvé. "
                f"Modèles disponibles: {available_names}",
            )

        # 2. Vérifier le cache Redis (première vérification rapide sans lock)
        cache_key = f"{_CACHE_PREFIX}{model_name}:{metadata.version}"
        redis = await self._get_redis()

        cached_signed = await redis.get(cache_key)
        if cached_signed:
            try:
                payload = _verify_cache_signature(cached_signed)
            except ValueError:
                # Tampered or old-format (unsigned) cache entry — reject and reload from source
                logger.warning(
                    "Cache Redis HMAC invalide — entrée invalidée",
                    cache_key=cache_key,
                    model=model_name,
                    version=str(metadata.version),
                )
                await redis.delete(cache_key)
            else:
                logger.info(
                    "Modèle chargé depuis le cache Redis",
                    model_name=model_name,
                    version=str(metadata.version),
                )
                # HMAC envelope verified; extract model (handles both old and new formats)
                model = _extract_model_from_payload(payload)
                return {"model": model, "metadata": metadata}

        # 3. Cache miss — acquérir le verrou intra-process pour éviter le thundering herd.
        # Plusieurs coroutines concurrentes dans le même worker ne déclencheront qu'un seul
        # téléchargement MinIO grâce au double-checked locking.
        if cache_key not in _LOAD_LOCKS:
            _LOAD_LOCKS[cache_key] = asyncio.Lock()
        lock = _LOAD_LOCKS[cache_key]

        async with lock:
            # Double-check après acquisition du lock (un autre coroutine a peut-être déjà chargé)
            cached_signed = await redis.get(cache_key)
            if cached_signed:
                try:
                    payload = _verify_cache_signature(cached_signed)
                except ValueError:
                    await redis.delete(cache_key)
                else:
                    model = _extract_model_from_payload(payload)
                    return {"model": model, "metadata": metadata}

            # 4. Charger le modèle depuis la source
            try:
                if metadata.mlflow_run_id:
                    logger.info(
                        "Chargement du modèle depuis MLflow",
                        model_name=model_name,
                        version=str(metadata.version),
                        mlflow_run_id=metadata.mlflow_run_id,
                    )
                    import mlflow.sklearn

                    model = mlflow.sklearn.load_model(f"runs:/{metadata.mlflow_run_id}/model")
                    cache_payload = pickle.dumps(model)  # noqa: S301
                else:
                    logger.info(
                        "Téléchargement du modèle depuis MinIO",
                        model_name=model_name,
                        version=str(metadata.version),
                    )
                    # Download raw pkl bytes — do NOT call pickle.loads() until HMAC is verified
                    raw_bytes = await minio_service.async_download_file_bytes(
                        metadata.minio_object_key
                    )
                    _verify_minio_hmac(raw_bytes, metadata, model_name)
                    model = pickle.loads(raw_bytes)  # noqa: S301 - HMAC verified above
                    cache_payload = raw_bytes

                # 5. Stocker en cache Redis avec enveloppe HMAC (TTL configurable)
                signed = _sign_for_cache(cache_payload)
                await redis.setex(cache_key, settings.REDIS_CACHE_TTL, signed)

                logger.info(
                    "Modèle chargé et mis en cache Redis",
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
                    detail=f"Erreur lors du chargement du modèle '{model_name}': {str(e)}",
                )

    async def get_cached_models(self) -> List[str]:
        """
        Retourne la liste des modèles actuellement en cache Redis.

        Returns:
            Liste de clés au format ``name:version``
        """
        redis = await self._get_redis()
        keys = await redis.keys(f"{_CACHE_PREFIX}*")
        return [k.decode().removeprefix(_CACHE_PREFIX) for k in keys]

    async def invalidate_model_cache(self, model_name: str, version: Optional[str] = None) -> None:
        """
        Invalide les entrées Redis pour un modèle donné.

        Args:
            model_name: Nom du modèle.
            version: Si fournie, n'invalide que cette version.
                     Si None, invalide toutes les versions du modèle.
        """
        redis = await self._get_redis()
        if version:
            key = f"{_CACHE_PREFIX}{model_name}:{version}"
            await redis.delete(key)
            logger.info("Cache modèle invalidé", model=model_name, version=version)
        else:
            keys = await redis.keys(f"{_CACHE_PREFIX}{model_name}:*")
            if keys:
                await redis.delete(*keys)
            logger.info("Cache modèle invalidé (toutes versions)", model=model_name)

    async def clear_cache(self, cache_key: Optional[str] = None):
        """
        Invalide le cache Redis.

        Args:
            cache_key: Si fourni (``name:version``), invalide uniquement cette entrée.
                       Si None, invalide toutes les entrées ``model:*``.
        """
        redis = await self._get_redis()
        if cache_key:
            await redis.delete(f"{_CACHE_PREFIX}{cache_key}")
            logger.info("Cache Redis invalidé", cache_key=cache_key)
        else:
            keys = await redis.keys(f"{_CACHE_PREFIX}*")
            if keys:
                await redis.delete(*keys)
            logger.info("Cache Redis complet invalidé")

    async def select_routing_versions(
        self,
        db: AsyncSession,
        model_name: str,
    ) -> Tuple[Optional[ModelMetadata], Optional[ModelMetadata]]:
        """
        Détermine quelle(s) version(s) utiliser pour un appel sans version explicite.

        Règles de sélection :
        1. Les versions avec deployment_mode="shadow" deviennent le candidat shadow.
        2. Les versions avec deployment_mode="ab_test" et traffic_weight > 0 participent
           au routage pondéré (random.choices) → version primaire.
        3. Si aucune version A/B : fallback sur get_model_metadata() (comportement legacy :
           is_production=True ou la plus récente).
        4. Le shadow n'est activé que s'il existe exactement une version shadow.

        Retourne:
            (primary_metadata, shadow_metadata)
            shadow_metadata est None si aucune version shadow unique n'existe.
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

        # Un seul shadow autorisé (ambiguïté sinon → désactivé)
        shadow_metadata = shadow_candidates[0] if len(shadow_candidates) == 1 else None

        if ab_candidates:
            weights = [m.traffic_weight for m in ab_candidates]
            primary_metadata = random.choices(ab_candidates, weights=weights, k=1)[0]
            logger.info(
                "Routage A/B",
                model_name=model_name,
                selected_version=primary_metadata.version,
                candidates=[m.version for m in ab_candidates],
            )
        else:
            # Fallback legacy : is_production=True ou plus récent
            primary_metadata = await DBService.get_model_metadata(db, model_name, version=None)

        return primary_metadata, shadow_metadata

    async def close(self):
        """Ferme proprement la connexion Redis (à appeler au shutdown de l'app)."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Connexion Redis fermée")


# Instance globale du service
model_service = ModelService()
