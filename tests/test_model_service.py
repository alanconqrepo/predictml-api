"""
Tests pour le service de gestion des modèles
"""
import asyncio
from unittest.mock import AsyncMock, patch
from src.services.model_service import ModelService


def test_get_available_models():
    """Test de récupération des modèles disponibles - retourne une liste"""
    service = ModelService()
    db_mock = AsyncMock()

    with patch(
        "src.services.db_service.DBService.get_all_active_models",
        new_callable=AsyncMock,
        return_value=[],
    ):
        models = asyncio.run(service.get_available_models(db_mock))

    assert isinstance(models, list)


def test_get_cached_models():
    """Test de récupération des modèles en cache"""
    service = ModelService()
    cached = service.get_cached_models()

    assert isinstance(cached, list)


def test_clear_cache():
    """Test du vidage du cache"""
    service = ModelService()
    service.clear_cache()

    assert len(service.get_cached_models()) == 0
