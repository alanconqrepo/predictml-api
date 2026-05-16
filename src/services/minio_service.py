"""
Service de gestion du stockage MinIO
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
    """Service pour gérer les modèles dans MinIO"""

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
        """Crée le bucket s'il n'existe pas (appelé au premier accès)"""
        if self._bucket_ready:
            return
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Bucket MinIO créé", bucket=self.bucket)
            else:
                logger.info("Bucket MinIO existant", bucket=self.bucket)
            self._bucket_ready = True
        except S3Error as e:
            logger.error("Erreur MinIO lors de la création du bucket", error=str(e))

    def upload_model(self, model: Any, object_name: str, metadata: Optional[dict] = None) -> dict:
        """
        Upload un modèle vers MinIO

        Args:
            model: Le modèle scikit-learn
            object_name: Nom de l'objet dans MinIO (ex: "iris_model/v1.0.0.joblib")
            metadata: Métadonnées optionnelles

        Returns:
            dict avec les informations de l'upload
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

            logger.info("Modèle uploadé", object_name=object_name, size_bytes=file_size)

            return {
                "bucket": self.bucket,
                "object_name": object_name,
                "size": file_size,
                "etag": result.etag,
                "version_id": result.version_id,
            }

        except S3Error as e:
            logger.error("Erreur lors de l'upload", error=str(e))
            raise

    def upload_model_bytes(self, model_bytes: bytes, object_name: str) -> dict:
        """
        Upload des bytes pkl bruts vers MinIO (sans re-sérialisation)

        Args:
            model_bytes: Contenu pkl déjà sérialisé
            object_name: Nom de l'objet dans MinIO

        Returns:
            dict avec les informations de l'upload
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
            logger.info("Modèle uploadé", object_name=object_name, size_bytes=len(model_bytes))
            return {
                "bucket": self.bucket,
                "object_name": object_name,
                "size": len(model_bytes),
                "etag": result.etag,
            }
        except S3Error as e:
            logger.error("Erreur lors de l'upload", error=str(e))
            raise

    def download_model(self, object_name: str) -> Any:
        """
        Télécharge et désérialise un modèle depuis MinIO

        Args:
            object_name: Nom de l'objet dans MinIO

        Returns:
            Le modèle scikit-learn désérialisé
        """
        self._ensure_bucket_exists()
        try:
            response = self.client.get_object(self.bucket, object_name)
            model_bytes = response.read()
            response.close()
            response.release_conn()

            model = joblib.load(io.BytesIO(model_bytes))

            logger.info("Modèle téléchargé", object_name=object_name)
            return model

        except S3Error as e:
            logger.error("Erreur lors du téléchargement", error=str(e))
            raise

    def list_models(self, prefix: str = "") -> List[str]:
        """
        Liste tous les modèles disponibles

        Args:
            prefix: Préfixe pour filtrer les objets

        Returns:
            Liste des noms d'objets
        """
        self._ensure_bucket_exists()
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error("Erreur lors du listing", error=str(e))
            return []

    def delete_model(self, object_name: str) -> bool:
        """
        Supprime un modèle de MinIO

        Args:
            object_name: Nom de l'objet à supprimer

        Returns:
            True si succès
        """
        try:
            self.client.remove_object(self.bucket, object_name)
            logger.info("Modèle supprimé", object_name=object_name)
            return True
        except S3Error as e:
            logger.error("Erreur lors de la suppression", error=str(e))
            return False

    def upload_file_bytes(
        self, content: bytes, object_name: str, content_type: str = "text/plain"
    ) -> dict:
        """
        Upload des bytes bruts (ex: script Python, JSON, CSV) dans MinIO.

        Args:
            content: Contenu brut à uploader
            object_name: Nom de l'objet dans MinIO
            content_type: Type MIME de l'objet

        Returns:
            dict avec les informations de l'upload
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
            logger.info("Fichier uploadé", object_name=object_name, size_bytes=len(content))
            return {"bucket": self.bucket, "object_name": object_name, "size": len(content)}
        except S3Error as e:
            logger.error("Erreur lors de l'upload du fichier", error=str(e))
            raise

    def download_file_bytes(self, object_name: str) -> bytes:
        """
        Télécharge et retourne le contenu brut d'un objet MinIO (sans désérialisation).

        Args:
            object_name: Nom de l'objet dans MinIO

        Returns:
            Contenu brut en bytes
        """
        self._ensure_bucket_exists()
        try:
            response = self.client.get_object(self.bucket, object_name)
            content = response.read()
            response.close()
            response.release_conn()
            logger.info("Fichier téléchargé", object_name=object_name, size_bytes=len(content))
            return content
        except S3Error as e:
            logger.error("Erreur lors du téléchargement du fichier", error=str(e))
            raise

    async def async_download_file_bytes(self, object_name: str) -> bytes:
        """Version async de download_file_bytes — n'occupe pas l'event loop pendant l'I/O réseau."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.download_file_bytes, object_name)

    async def async_upload_model_bytes(self, model_bytes: bytes, object_name: str) -> dict:
        """Version async de upload_model_bytes — n'occupe pas l'event loop pendant l'I/O réseau."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.upload_model_bytes, model_bytes, object_name)

    async def async_upload_file_bytes(
        self, content: bytes, object_name: str, content_type: str = "text/plain"
    ) -> dict:
        """Version async de upload_file_bytes — n'occupe pas l'event loop pendant l'I/O réseau."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.upload_file_bytes, content, object_name, content_type
        )

    def get_object_info(self, object_name: str) -> Optional[dict]:
        """
        Récupère les informations d'un objet

        Args:
            object_name: Nom de l'objet

        Returns:
            Dict avec les informations de l'objet
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
            logger.error("Erreur lors de la récupération des infos", error=str(e))
            return None


# Instance globale
minio_service = MinIOService()
