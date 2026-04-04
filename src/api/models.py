"""
Endpoints pour la gestion des modèles
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_db
from src.services.model_service import model_service

router = APIRouter(tags=["models"])


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(db: AsyncSession = Depends(get_db)):
    """
    Liste tous les modèles disponibles depuis la base de données

    Returns:
        Liste des modèles actifs avec leurs métadonnées
    """
    models = await model_service.get_available_models(db)
    return models


@router.get("/models/cached")
async def list_cached_models():
    """
    Liste les modèles actuellement en cache mémoire

    Returns:
        Liste des object keys MinIO en cache
    """
    cached = model_service.get_cached_models()
    return {
        "cached_models": cached,
        "count": len(cached)
    }
