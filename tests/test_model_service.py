"""
Tests pour le service de gestion des modèles
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.model_service import ModelService


def _fake_metadata(mlflow_run_id=None, minio_object_key="model/v1.0.0.pkl"):
    """Retourne un faux objet ModelMetadata."""
    return SimpleNamespace(
        name="test_model",
        version="1.0.0",
        mlflow_run_id=mlflow_run_id,
        minio_object_key=minio_object_key,
    )


def test_get_available_models():
    """Test de récupération des modèles disponibles - retourne une liste"""
    service = ModelService()
    db_mock = AsyncMock()

    with patch(
        "src.services.db_service.DBService.get_all_active_models",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.services.db_service.DBService.get_models_last_seen",
        new_callable=AsyncMock,
        return_value={},
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


# ---------------------------------------------------------------------------
# load_model — chemin MinIO
# ---------------------------------------------------------------------------

def test_load_model_via_minio():
    """load_model sans mlflow_run_id → charge depuis MinIO"""
    service = ModelService()
    fake_model = MagicMock()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl")

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.download_model.return_value = fake_model
        result = asyncio.run(service.load_model(AsyncMock(), "test_model"))

    assert result["model"] is fake_model
    assert result["metadata"] is metadata
    minio_mock.download_model.assert_called_once_with("iris/v1.0.0.pkl")


def test_load_model_via_mlflow():
    """load_model avec mlflow_run_id → charge depuis MLflow, MinIO non appelé"""
    service = ModelService()
    fake_model = MagicMock()
    run_id = "abc123def456abc123def456abc123de"
    metadata = _fake_metadata(mlflow_run_id=run_id, minio_object_key=None)

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("mlflow.sklearn.load_model", return_value=fake_model) as mlflow_mock, \
       patch("src.services.model_service.minio_service") as minio_mock:
        result = asyncio.run(service.load_model(AsyncMock(), "test_model"))

    assert result["model"] is fake_model
    mlflow_mock.assert_called_once_with(f"runs:/{run_id}/model")
    minio_mock.download_model.assert_not_called()


def test_load_model_cache_hit():
    """load_model appelle le stockage une seule fois : le second appel vient du cache"""
    service = ModelService()
    fake_model = MagicMock()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl")

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.download_model.return_value = fake_model
        asyncio.run(service.load_model(AsyncMock(), "test_model"))
        asyncio.run(service.load_model(AsyncMock(), "test_model"))

    # MinIO appelé une seule fois malgré deux load_model
    assert minio_mock.download_model.call_count == 1


def test_load_model_forwards_explicit_version():
    """load_model avec version explicite → get_model_metadata appelé avec cette version"""
    service = ModelService()
    fake_model = MagicMock()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v2.0.0.pkl")
    metadata.version = "2.0.0"

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ) as mock_get, patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.download_model.return_value = fake_model
        asyncio.run(service.load_model(AsyncMock(), "test_model", "2.0.0"))

    mock_get.assert_called_once()
    _, args, kwargs = mock_get.mock_calls[0]
    # Le troisième argument positionnel (ou kwarg 'version') doit être "2.0.0"
    assert args[2] == "2.0.0" or kwargs.get("version") == "2.0.0"


def test_load_model_no_version_passes_none():
    """load_model sans version → get_model_metadata appelé avec version=None (production en priorité)"""
    service = ModelService()
    fake_model = MagicMock()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl")

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ) as mock_get, patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.download_model.return_value = fake_model
        asyncio.run(service.load_model(AsyncMock(), "test_model"))

    mock_get.assert_called_once()
    _, args, kwargs = mock_get.mock_calls[0]
    assert args[2] is None or kwargs.get("version") is None


def test_load_model_not_found():
    """load_model avec un modèle inexistant → HTTPException 404"""
    from fastapi import HTTPException

    service = ModelService()

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "src.services.db_service.DBService.get_all_active_models",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.services.db_service.DBService.get_models_last_seen",
        new_callable=AsyncMock,
        return_value={},
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(service.load_model(AsyncMock(), "inexistant"))

    assert exc_info.value.status_code == 404
