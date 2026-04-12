"""
Service de gestion des modèles ML (v3 - cache Redis distribué)
"""

import pickle
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
import structlog
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.services.db_service import DBService
from src.services.minio_service import minio_service

logger = structlog.get_logger(__name__)

_CACHE_PREFIX = "model:"


class ModelService:
    """Service pour charger et gérer les modèles ML depuis MinIO/MLflow, avec cache Redis."""

    def __init__(self):
        # Connexion Redis initialisée paresseusement au premier accès
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        """Retourne le client Redis, en le créant si nécessaire."""
        if self._redis is None:
            self._redis = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=False)
        return self._redis

    async def get_available_models(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Retourne la liste des modèles disponibles depuis la base de données

        Args:
            db: Session de base de données

        Returns:
            Liste des modèles actifs avec leurs métadonnées
        """
        models = await DBService.get_all_active_models(db)
        last_seen_map = await DBService.get_models_last_seen(db)
        return [
            {
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "is_production": m.is_production,
                "accuracy": m.accuracy,
                "features_count": m.features_count,
                "classes": m.classes,
                "tags": m.tags,
                "webhook_url": m.webhook_url,
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

        Clé Redis : ``model:{name}:{version}`` — TTL configurable via REDIS_CACHE_TTL.

        Args:
            db: Session de base de données
            model_name: Nom du modèle
            version: Version spécifique (optionnel, prend la plus récente sinon)

        Returns:
            Dict contenant le modèle et ses métadonnées

        Raises:
            HTTPException: Si le modèle n'existe pas ou ne peut pas être chargé
        """
        # 1. Récupérer les métadonnées depuis la DB
        metadata = await DBService.get_model_metadata(db, model_name, version)

        if not metadata:
            available = await self.get_available_models(db)
            available_names = [m["name"] for m in available]
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modèle '{model_name}' (version '{version or 'latest'}') non trouvé. "
                f"Modèles disponibles: {available_names}",
            )

        # 2. Vérifier le cache Redis
        cache_key = f"{_CACHE_PREFIX}{model_name}:{metadata.version}"
        redis = await self._get_redis()

        cached_bytes = await redis.get(cache_key)
        if cached_bytes:
            logger.info(
                "Modèle chargé depuis le cache Redis",
                model_name=model_name,
                version=str(metadata.version),
            )
            return pickle.loads(cached_bytes)  # noqa: S301

        # 3. Charger le modèle depuis la source
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
            else:
                logger.info(
                    "Téléchargement du modèle depuis MinIO",
                    model_name=model_name,
                    version=str(metadata.version),
                )
                model = minio_service.download_model(metadata.minio_object_key)

            # 4. Stocker en cache Redis avec TTL
            cached_data = {"model": model, "metadata": metadata}
            await redis.setex(cache_key, settings.REDIS_CACHE_TTL, pickle.dumps(cached_data))

            logger.info(
                "Modèle chargé et mis en cache Redis",
                model_name=model_name,
                version=str(metadata.version),
                ttl=settings.REDIS_CACHE_TTL,
            )
            return cached_data

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

    async def close(self):
        """Ferme proprement la connexion Redis (à appeler au shutdown de l'app)."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Connexion Redis fermée")


# Instance globale du service
model_service = ModelService()
