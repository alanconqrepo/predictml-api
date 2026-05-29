"""
MinIO storage management service
"""

import asyncio
import io
from typing import Any, List, Optional

import joblib
import structlog
from minio import Minio
from minio.error import S3Error

from src.core.config import settings

logger = structlog.get_logger(__name__)


class MinIOService:
    """Service for managing models in MinIO"""

    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.bucket = settings.MINIO_BUCKET
        self._bucket_ready = False

    def _ensure_bucket_exists(self):
        """Create the bucket if it does not exist (called on first access)"""
        if self._bucket_ready:
            return
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("MinIO bucket created", bucket=self.bucket)
            else:
                logger.info("MinIO bucket already exists", bucket=self.bucket)
            self._bucket_ready = True
        except S3Error as e:
            logger.error("MinIO error while creating bucket", error=str(e))

    async def ensure_bucket_ready(self) -> None:
        """Eagerly create the bucket at startup (called from lifespan)."""
        await asyncio.to_thread(self._ensure_bucket_exists)

    def upload_model(self, model: Any, object_name: str, metadata: Optional[dict] = None) -> dict:
        """
        Upload a model to MinIO

        Args:
            model: The scikit-learn model
            object_name: Object name in MinIO (e.g. "iris_model/v1.0.0.joblib")
            metadata: Optional metadata

        Returns:
            dict with upload information
        """
        self._ensure_bucket_exists()
        try:
            model_stream = io.BytesIO()
            joblib.dump(model, model_stream)
            model_bytes = model_stream.getvalue()
            model_stream = io.BytesIO(model_bytes)
            file_size = len(model_bytes)

            result = self.client.put_object(
                self.bucket,
                object_name,
                model_stream,
                length=file_size,
                metadata=metadata,
            )

            logger.info("Model uploaded", object_name=object_name, size_bytes=file_size)

            return {
                "bucket": self.bucket,
                "object_name": object_name,
                "size": file_size,
                "etag": result.etag,
                "version_id": result.version_id,
            }

        except S3Error as e:
            logger.error("Upload error", error=str(e))
            raise

    def upload_model_bytes(self, model_bytes: bytes, object_name: str) -> dict:
        """
        Upload raw pkl bytes to MinIO (without re-serialization)

        Args:
            model_bytes: Already serialized pkl content
            object_name: Object name in MinIO

        Returns:
            dict with upload information
        """
        self._ensure_bucket_exists()
        try:
            model_stream = io.BytesIO(model_bytes)
            result = self.client.put_object(
                self.bucket,
                object_name,
                model_stream,
                length=len(model_bytes),
            )
            logger.info("Model uploaded", object_name=object_name, size_bytes=len(model_bytes))
            return {
                "bucket": self.bucket,
                "object_name": object_name,
                "size": len(model_bytes),
                "etag": result.etag,
            }
        except S3Error as e:
            logger.error("Upload error", error=str(e))
            raise

    def download_model(self, object_name: str) -> Any:
        """
        Download and deserialize a model from MinIO

        Args:
            object_name: Object name in MinIO

        Returns:
            The deserialized scikit-learn model
        """
        self._ensure_bucket_exists()
        try:
            response = self.client.get_object(self.bucket, object_name)
            model_bytes = response.read()
            response.close()
            response.release_conn()

            model = joblib.load(io.BytesIO(model_bytes))

            logger.info("Model downloaded", object_name=object_name)
            return model

        except S3Error as e:
            logger.error("Download error", error=str(e))
            raise

    def list_models(self, prefix: str = "") -> List[str]:
        """
        List all available models

        Args:
            prefix: Prefix to filter objects

        Returns:
            List of object names
        """
        self._ensure_bucket_exists()
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error("Listing error", error=str(e))
            return []

    def delete_model(self, object_name: str) -> bool:
        """
        Delete a model from MinIO

        Args:
            object_name: Object name to delete

        Returns:
            True if successful
        """
        try:
            self.client.remove_object(self.bucket, object_name)
            logger.info("Model deleted", object_name=object_name)
            return True
        except S3Error as e:
            logger.error("Deletion error", error=str(e))
            return False

    def upload_file_bytes(
        self, content: bytes, object_name: str, content_type: str = "text/plain"
    ) -> dict:
        """
        Upload raw bytes (e.g. Python script, JSON, CSV) to MinIO.

        Args:
            content: Raw content to upload
            object_name: Object name in MinIO
            content_type: MIME type of the object

        Returns:
            dict with upload information
        """
        self._ensure_bucket_exists()
        try:
            self.client.put_object(
                self.bucket,
                object_name,
                io.BytesIO(content),
                length=len(content),
                content_type=content_type,
            )
            logger.info("File uploaded", object_name=object_name, size_bytes=len(content))
            return {"bucket": self.bucket, "object_name": object_name, "size": len(content)}
        except S3Error as e:
            logger.error("File upload error", error=str(e))
            raise

    def download_file_bytes(self, object_name: str) -> bytes:
        """
        Download and return the raw content of a MinIO object (without deserialization).

        Args:
            object_name: Object name in MinIO

        Returns:
            Raw content in bytes
        """
        self._ensure_bucket_exists()
        try:
            response = self.client.get_object(self.bucket, object_name)
            content = response.read()
            response.close()
            response.release_conn()
            logger.info("File downloaded", object_name=object_name, size_bytes=len(content))
            return content
        except S3Error as e:
            logger.error("File download error", error=str(e))
            raise

    async def async_download_file_bytes(self, object_name: str) -> bytes:
        """Async version of download_file_bytes — does not block the event loop during network I/O."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.download_file_bytes, object_name)

    async def async_upload_model_bytes(self, model_bytes: bytes, object_name: str) -> dict:
        """Async version of upload_model_bytes — does not block the event loop during network I/O."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.upload_model_bytes, model_bytes, object_name)

    async def async_upload_file_bytes(
        self, content: bytes, object_name: str, content_type: str = "text/plain"
    ) -> dict:
        """Async version of upload_file_bytes — does not block the event loop during network I/O."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.upload_file_bytes, content, object_name, content_type
        )

    def get_object_info(self, object_name: str) -> Optional[dict]:
        """
        Retrieve information about an object

        Args:
            object_name: Object name

        Returns:
            Dict with object information
        """
        try:
            stat = self.client.stat_object(self.bucket, object_name)
            return {
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "metadata": stat.metadata,
            }
        except S3Error as e:
            logger.error("Error retrieving object info", error=str(e))
            return None


# Global instance
minio_service = MinIOService()
