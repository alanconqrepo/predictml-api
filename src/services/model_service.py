"""
Service de gestion des modèles ML (v2 - avec MinIO + DB)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.db_service import DBService
from src.services.minio_service import minio_service

logger = logging.getLogger(__name__)


class ModelService:
    """Service pour charger et gérer les modèles ML depuis MinIO"""

    def __init__(self):
        # Cache: {"name:version": {"model": model, "metadata": ModelMetadata}}
        self.models_cache: Dict[str, Dict[str, Any]] = {}

    async def get_available_models(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Retourne la liste des modèles disponibles depuis la base de données

        Args:
            db: Session de base de données

        Returns:
            Liste des modèles actifs avec leurs métadonnées
        """
        models = await DBService.get_all_active_models(db)
        return [
            {
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "is_production": m.is_production,
                "accuracy": m.accuracy,
                "features_count": m.features_count,
                "classes": m.classes,
                "user_id_creator": m.user_id_creator,
                "creator_username": m.creator.username if m.creator else None,
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
        Charge un modèle depuis MinIO (via cache ou download)

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

        # 2. Vérifier le cache
        cache_key = f"{model_name}:{metadata.version}"
        if cache_key in self.models_cache:
            logger.info("Modèle '%s' v%s chargé depuis le cache", model_name, metadata.version)
            return self.models_cache[cache_key]

        # 3. Charger le modèle
        try:
            if metadata.mlflow_run_id:
                logger.info(
                    "Chargement du modèle '%s' v%s depuis MLflow (run=%s)...",
                    model_name,
                    metadata.version,
                    metadata.mlflow_run_id,
                )
                import mlflow.sklearn

                model = mlflow.sklearn.load_model(f"runs:/{metadata.mlflow_run_id}/model")
            else:
                logger.info(
                    "Téléchargement du modèle '%s' v%s depuis MinIO...",
                    model_name,
                    metadata.version,
                )
                model = minio_service.download_model(metadata.minio_object_key)

            # 4. Mettre en cache
            cached_data = {"model": model, "metadata": metadata}
            self.models_cache[cache_key] = cached_data

            logger.info("Modèle '%s' v%s chargé et mis en cache", model_name, metadata.version)
            return cached_data

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur lors du chargement du modèle '{model_name}': {str(e)}",
            )

    def get_cached_models(self) -> List[str]:
        """
        Retourne la liste des modèles actuellement en cache

        Returns:
            Liste des object keys MinIO en cache
        """
        return list(self.models_cache.keys())

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Vide le cache des modèles

        Args:
            cache_key: Si fourni ("name:version"), vide uniquement ce modèle du cache
        """
        if cache_key:
            self.models_cache.pop(cache_key, None)
            logger.info("Cache vidé pour: %s", cache_key)
        else:
            self.models_cache.clear()
            logger.info("Cache complet vidé")


# Instance globale du service
model_service = ModelService()
