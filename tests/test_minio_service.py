"""
Tests unitaires — MinIOService (src/services/minio_service.py).

Stratégie :
  Patcher src.services.minio_service.Minio (PAS le mock global de conftest)
  pour tester la logique interne du service en isolation.
  S3Error importé réellement pour valider les chemins d'exception.

Couvre :
- _ensure_bucket_exists() : création, idempotence, S3Error
- upload_model() / upload_model_bytes() / upload_file_bytes()
- download_model() / download_file_bytes()
- list_models() / delete_model() / get_object_info()
"""

import io
import pickle
from unittest.mock import MagicMock, patch

import pytest
from minio.error import S3Error


def _make_s3error():
    """Crée un S3Error minimal pour les tests."""
    return S3Error(
        code="NoSuchBucket",
        message="bucket does not exist",
        resource="/test-bucket",
        request_id="test-req",
        host_id="test-host",
        response=MagicMock(status=404, headers={}, data=b""),
    )


def _make_service():
    """Instancie MinIOService avec un client Minio mocké."""
    with patch("src.services.minio_service.Minio") as MockMinio:
        mock_client = MockMinio.return_value
        mock_client.bucket_exists.return_value = True  # défaut : bucket existe
        from src.services.minio_service import MinIOService

        svc = MinIOService()
        return svc, mock_client


class TestEnsureBucketExists:
    """Tests pour _ensure_bucket_exists()."""

    def test_skips_make_bucket_when_exists(self):
        """Si le bucket existe déjà, make_bucket n'est PAS appelé."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            svc._ensure_bucket_exists()
            mock_client.make_bucket.assert_not_called()
            assert svc._bucket_ready is True

    def test_creates_bucket_when_not_exists(self):
        """Si le bucket n'existe pas, make_bucket est appelé et _bucket_ready=True."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = False
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            svc._ensure_bucket_exists()
            mock_client.make_bucket.assert_called_once()
            assert svc._bucket_ready is True

    def test_idempotent_when_already_ready(self):
        """Si _bucket_ready=True, bucket_exists n'est PAS rappelé."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            svc._bucket_ready = True
            svc._ensure_bucket_exists()
            mock_client.bucket_exists.assert_not_called()

    def test_s3error_does_not_set_bucket_ready(self):
        """S3Error dans _ensure_bucket_exists → _bucket_ready reste False."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            svc._ensure_bucket_exists()  # ne doit pas lever d'exception
            assert svc._bucket_ready is False


class TestUploadModel:
    """Tests pour upload_model()."""

    def test_upload_model_returns_correct_dict(self):
        """upload_model() sérialise et upload, retourne le dict attendu."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_result = MagicMock()
            mock_result.etag = "abc123"
            mock_result.version_id = None
            mock_client.put_object.return_value = mock_result
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            dummy_model = {"weights": [1, 2, 3]}  # picklable
            result = svc.upload_model(dummy_model, "test/v1.0.0.pkl")
            assert result["bucket"] == svc.bucket
            assert result["object_name"] == "test/v1.0.0.pkl"
            assert result["etag"] == "abc123"
            assert result["size"] > 0
            mock_client.put_object.assert_called_once()

    def test_upload_model_propagates_s3error(self):
        """upload_model() relance S3Error si put_object échoue."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.put_object.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            with pytest.raises(S3Error):
                svc.upload_model({"x": 1}, "fail/v1.pkl")


class TestUploadModelBytes:
    """Tests pour upload_model_bytes()."""

    def test_upload_model_bytes_calls_put_object_with_correct_length(self):
        """upload_model_bytes() appelle put_object avec la bonne taille."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_result = MagicMock()
            mock_result.etag = "etag-bytes"
            mock_client.put_object.return_value = mock_result
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            model_bytes = pickle.dumps({"model": "data"})
            result = svc.upload_model_bytes(model_bytes, "test/v2.0.0.pkl")
            call_kwargs = mock_client.put_object.call_args
            assert call_kwargs[1].get("length") == len(model_bytes) or (
                len(call_kwargs[0]) > 3 and call_kwargs[0][3] == len(model_bytes)
            )
            assert result["size"] == len(model_bytes)
            assert result["etag"] == "etag-bytes"

    def test_upload_model_bytes_propagates_s3error(self):
        """upload_model_bytes() relance S3Error."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.put_object.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            with pytest.raises(S3Error):
                svc.upload_model_bytes(b"data", "fail.pkl")


class TestUploadFileBytes:
    """Tests pour upload_file_bytes()."""

    def test_upload_file_bytes_transmits_content_type(self):
        """upload_file_bytes() transmet le content_type à put_object."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.put_object.return_value = MagicMock()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            content = b"import os\nprint('hello')"
            svc.upload_file_bytes(content, "scripts/train.py", content_type="text/x-python")
            call_kwargs = mock_client.put_object.call_args[1]
            assert call_kwargs.get("content_type") == "text/x-python"

    def test_upload_file_bytes_returns_dict_with_size(self):
        """upload_file_bytes() retourne dict avec bucket, object_name, size."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.put_object.return_value = MagicMock()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            content = b"script content"
            result = svc.upload_file_bytes(content, "scripts/train.py")
            assert result["size"] == len(content)
            assert result["object_name"] == "scripts/train.py"

    def test_upload_file_bytes_propagates_s3error(self):
        """upload_file_bytes() relance S3Error."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.put_object.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            with pytest.raises(S3Error):
                svc.upload_file_bytes(b"data", "fail.py")


class TestDownloadModel:
    """Tests pour download_model()."""

    def test_download_model_deserializes_pickle(self):
        """download_model() lit et désérialise le pickle correctement."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            original_obj = {"weights": [0.5, 1.0], "bias": 0.1}
            pickled_bytes = pickle.dumps(original_obj)
            mock_response = MagicMock()
            mock_response.read.return_value = pickled_bytes
            mock_client.get_object.return_value = mock_response
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.download_model("test/v1.0.0.pkl")
            assert result == original_obj

    def test_download_model_calls_close_and_release(self):
        """download_model() appelle close() et release_conn() sur la réponse."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_response = MagicMock()
            mock_response.read.return_value = pickle.dumps({"x": 1})
            mock_client.get_object.return_value = mock_response
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            svc.download_model("test/v1.pkl")
            mock_response.close.assert_called_once()
            mock_response.release_conn.assert_called_once()

    def test_download_model_propagates_s3error(self):
        """download_model() relance S3Error si get_object échoue."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.get_object.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            with pytest.raises(S3Error):
                svc.download_model("fail/v1.pkl")


class TestDownloadFileBytes:
    """Tests pour download_file_bytes()."""

    def test_download_file_bytes_returns_raw_bytes(self):
        """download_file_bytes() retourne les bytes bruts sans désérialisation."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            raw_bytes = b"import os\nprint(os.environ['TRAIN_START_DATE'])"
            mock_response = MagicMock()
            mock_response.read.return_value = raw_bytes
            mock_client.get_object.return_value = mock_response
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.download_file_bytes("scripts/train.py")
            assert result == raw_bytes

    def test_download_file_bytes_propagates_s3error(self):
        """download_file_bytes() relance S3Error."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.get_object.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            with pytest.raises(S3Error):
                svc.download_file_bytes("fail.py")


class TestListModels:
    """Tests pour list_models()."""

    def test_list_models_returns_object_names(self):
        """list_models() retourne la liste des noms d'objets."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            obj1 = MagicMock()
            obj1.object_name = "model_a/v1.pkl"
            obj2 = MagicMock()
            obj2.object_name = "model_b/v2.pkl"
            mock_client.list_objects.return_value = [obj1, obj2]
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.list_models()
            assert "model_a/v1.pkl" in result
            assert "model_b/v2.pkl" in result

    def test_list_models_with_prefix(self):
        """list_models(prefix=...) passe le préfixe à list_objects."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.list_objects.return_value = []
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            svc.list_models(prefix="my_model/")
            call_kwargs = mock_client.list_objects.call_args
            assert call_kwargs[1].get("prefix") == "my_model/"

    def test_list_models_returns_empty_on_s3error(self):
        """list_models() retourne [] si S3Error."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.bucket_exists.return_value = True
            mock_client.list_objects.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.list_models()
            assert result == []


class TestDeleteModel:
    """Tests pour delete_model()."""

    def test_delete_model_returns_true_on_success(self):
        """delete_model() retourne True quand remove_object réussit."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.delete_model("test/v1.pkl")
            assert result is True
            mock_client.remove_object.assert_called_once()

    def test_delete_model_returns_false_on_s3error(self):
        """delete_model() retourne False si S3Error."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.remove_object.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.delete_model("test/v1.pkl")
            assert result is False


class TestGetObjectInfo:
    """Tests pour get_object_info()."""

    def test_get_object_info_returns_stat_dict(self):
        """get_object_info() retourne un dict avec les clés attendues."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_stat = MagicMock()
            mock_stat.size = 1024
            mock_stat.etag = "etag-stat"
            mock_stat.content_type = "application/octet-stream"
            mock_stat.last_modified = "2025-01-01T00:00:00Z"
            mock_stat.metadata = {}
            mock_client.stat_object.return_value = mock_stat
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.get_object_info("test/v1.pkl")
            assert result is not None
            assert result["size"] == 1024
            assert result["etag"] == "etag-stat"
            assert "content_type" in result
            assert "last_modified" in result

    def test_get_object_info_returns_none_on_s3error(self):
        """get_object_info() retourne None si S3Error."""
        with patch("src.services.minio_service.Minio") as MockMinio:
            mock_client = MockMinio.return_value
            mock_client.stat_object.side_effect = _make_s3error()
            from src.services.minio_service import MinIOService

            svc = MinIOService()
            result = svc.get_object_info("nonexistent.pkl")
            assert result is None
