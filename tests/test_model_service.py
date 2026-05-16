"""
Tests pour le service de gestion des modèles ML
"""
import asyncio
import io
import joblib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest

from src.services.model_service import ModelService, compute_model_hmac, _sign_for_cache


def _make_service() -> ModelService:
    """Crée un ModelService avec un FakeRedis en mémoire (pas de serveur Redis requis)."""
    service = ModelService()
    service._redis = fakeredis.aioredis.FakeRedis()
    return service


def _fake_metadata(mlflow_run_id=None, minio_object_key="model/v1.0.0.pkl", model_bytes=None):
    """
    Retourne un faux objet ModelMetadata avec une signature HMAC valide.

    model_bytes : bytes pkl du modèle ; si None, une valeur par défaut est utilisée.
    """
    if model_bytes is None:
        model_bytes = b"fake_pkl_bytes"
    sig = compute_model_hmac(model_bytes) if minio_object_key else None
    return SimpleNamespace(
        name="test_model",
        version="1.0.0",
        mlflow_run_id=mlflow_run_id,
        minio_object_key=minio_object_key,
        model_hmac_signature=sig,
    )


def test_get_available_models():
    """Test de récupération des modèles disponibles - retourne une liste"""
    service = _make_service()
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
    """Test de récupération des modèles en cache — Redis vide au départ"""
    service = _make_service()
    cached = asyncio.run(service.get_cached_models())

    assert isinstance(cached, list)
    assert cached == []


def test_clear_cache():
    """Test du vidage du cache — Redis vide après clear"""
    service = _make_service()
    asyncio.run(service.clear_cache())

    assert asyncio.run(service.get_cached_models()) == []


# ---------------------------------------------------------------------------
# load_model — chemin MinIO
# ---------------------------------------------------------------------------

def test_load_model_via_minio():
    """load_model sans mlflow_run_id → charge depuis MinIO via download_file_bytes"""
    service = _make_service()
    fake_model = SimpleNamespace(marker="fake_minio_model")
    _jbuf = io.BytesIO()
    joblib.dump(fake_model, _jbuf)
    fake_pkl = _jbuf.getvalue()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl",
                               model_bytes=fake_pkl)

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock(return_value=fake_pkl)
        result = asyncio.run(service.load_model(AsyncMock(), "test_model"))

    assert result["model"].marker == "fake_minio_model"
    assert result["metadata"].name == "test_model"
    assert result["metadata"].minio_object_key == "iris/v1.0.0.pkl"
    minio_mock.async_download_file_bytes.assert_called_once_with("iris/v1.0.0.pkl")


def test_load_model_via_mlflow():
    """load_model avec mlflow_run_id → charge depuis MLflow, MinIO non appelé"""
    service = _make_service()
    fake_model = SimpleNamespace(marker="fake_mlflow_model")
    run_id = "abc123def456abc123def456abc123de"
    metadata = _fake_metadata(mlflow_run_id=run_id, minio_object_key=None, model_bytes=None)

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("mlflow.sklearn.load_model", return_value=fake_model) as mlflow_mock, \
       patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock()
        result = asyncio.run(service.load_model(AsyncMock(), "test_model"))

    assert result["model"].marker == "fake_mlflow_model"
    mlflow_mock.assert_called_once_with(f"runs:/{run_id}/model")
    minio_mock.async_download_file_bytes.assert_not_called()


def test_load_model_cache_hit():
    """load_model : le second appel est servi depuis le cache Redis (MinIO non rappelé)"""
    service = _make_service()
    fake_model = SimpleNamespace(marker="cache_hit_model")
    _jbuf = io.BytesIO()
    joblib.dump(fake_model, _jbuf)
    fake_pkl = _jbuf.getvalue()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl",
                               model_bytes=fake_pkl)

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock(return_value=fake_pkl)
        asyncio.run(service.load_model(AsyncMock(), "test_model"))
        asyncio.run(service.load_model(AsyncMock(), "test_model"))

    # MinIO appelé une seule fois malgré deux load_model
    assert minio_mock.async_download_file_bytes.call_count == 1


def test_load_model_forwards_explicit_version():
    """load_model avec version explicite → get_model_metadata appelé avec cette version"""
    service = _make_service()
    fake_model = SimpleNamespace(marker="fake_model")
    _jbuf = io.BytesIO()
    joblib.dump(fake_model, _jbuf)
    fake_pkl = _jbuf.getvalue()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v2.0.0.pkl",
                               model_bytes=fake_pkl)
    metadata.version = "2.0.0"

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ) as mock_get, patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock(return_value=fake_pkl)
        asyncio.run(service.load_model(AsyncMock(), "test_model", "2.0.0"))

    mock_get.assert_called_once()
    _, args, kwargs = mock_get.mock_calls[0]
    assert args[2] == "2.0.0" or kwargs.get("version") == "2.0.0"


def test_load_model_no_version_passes_none():
    """load_model sans version → get_model_metadata appelé avec version=None"""
    service = _make_service()
    fake_model = SimpleNamespace(marker="fake_model")
    _jbuf = io.BytesIO()
    joblib.dump(fake_model, _jbuf)
    fake_pkl = _jbuf.getvalue()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl",
                               model_bytes=fake_pkl)

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ) as mock_get, patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock(return_value=fake_pkl)
        asyncio.run(service.load_model(AsyncMock(), "test_model"))

    mock_get.assert_called_once()
    _, args, kwargs = mock_get.mock_calls[0]
    assert args[2] is None or kwargs.get("version") is None


def test_load_model_not_found():
    """load_model avec un modèle inexistant → HTTPException 404"""
    from fastapi import HTTPException

    service = _make_service()

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


# ---------------------------------------------------------------------------
# Sécurité HMAC
# ---------------------------------------------------------------------------

def test_load_model_missing_signature_raises_403():
    """load_model sans model_hmac_signature → HTTPException 403 (modèle legacy non signé)"""
    from fastapi import HTTPException

    service = _make_service()
    _jbuf = io.BytesIO()
    joblib.dump(SimpleNamespace(marker="x"), _jbuf)
    fake_pkl = _jbuf.getvalue()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl",
                               model_bytes=fake_pkl)
    metadata.model_hmac_signature = None  # simule un modèle sans signature

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock(return_value=fake_pkl)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(service.load_model(AsyncMock(), "test_model"))

    assert exc_info.value.status_code == 403
    assert "resign_models" in exc_info.value.detail


def test_load_model_tampered_pkl_raises_500():
    """load_model avec pkl falsifié → HTTPException 500 (signature invalide)"""
    from fastapi import HTTPException

    service = _make_service()
    _jbuf = io.BytesIO()
    joblib.dump(SimpleNamespace(marker="real"), _jbuf)
    real_pkl = _jbuf.getvalue()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl",
                               model_bytes=real_pkl)
    tampered_pkl = b"this_is_not_the_signed_file"

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock(return_value=tampered_pkl)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(service.load_model(AsyncMock(), "test_model"))

    assert exc_info.value.status_code == 500
    assert "signature" in exc_info.value.detail.lower()


def test_load_model_tampered_redis_cache_invalidated():
    """
    Si le cache Redis contient des bytes avec un HMAC invalide,
    load_model invalide l'entrée et charge depuis MinIO.
    """
    service = _make_service()
    fake_model = SimpleNamespace(marker="fresh_from_minio")
    _jbuf = io.BytesIO()
    joblib.dump(fake_model, _jbuf)
    fake_pkl = _jbuf.getvalue()
    metadata = _fake_metadata(mlflow_run_id=None, minio_object_key="iris/v1.0.0.pkl",
                               model_bytes=fake_pkl)

    # Injecter des bytes corrompus dans le cache (HMAC invalide)
    cache_key = f"model:test_model:1.0.0"
    corrupted = b"A" * 64 + b"malicious_payload"
    asyncio.run(service._redis.setex(cache_key, 3600, corrupted))

    with patch(
        "src.services.db_service.DBService.get_model_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    ), patch("src.services.model_service.minio_service") as minio_mock:
        minio_mock.async_download_file_bytes = AsyncMock(return_value=fake_pkl)
        result = asyncio.run(service.load_model(AsyncMock(), "test_model"))

    # L'entrée corrompue doit être ignorée ; MinIO doit avoir été appelé
    assert result["model"].marker == "fresh_from_minio"
    minio_mock.async_download_file_bytes.assert_called_once_with("iris/v1.0.0.pkl")
    # Le cache doit maintenant contenir l'entrée valide
    new_cached = asyncio.run(service._redis.get(cache_key))
    assert new_cached is not None and len(new_cached) > 64
