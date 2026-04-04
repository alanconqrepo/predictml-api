"""
Service de gestion du stockage MinIO
"""
import io
import pickle
from typing import Optional, List
from minio import Minio
from minio.error import S3Error

from src.core.config import settings


class MinIOService:
    """Service pour gérer les modèles dans MinIO"""

    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
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
                print(f"✅ Bucket MinIO '{self.bucket}' créé")
            else:
                print(f"📦 Bucket MinIO '{self.bucket}' existe déjà")
            self._bucket_ready = True
        except S3Error as e:
            print(f"❌ Erreur MinIO lors de la création du bucket: {e}")

    def upload_model(
        self,
        model: any,
        object_name: str,
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Upload un modèle vers MinIO

        Args:
            model: Le modèle scikit-learn
            object_name: Nom de l'objet dans MinIO (ex: "iris_model/v1.0.0.pkl")
            metadata: Métadonnées optionnelles

        Returns:
            dict avec les informations de l'upload
        """
        self._ensure_bucket_exists()
        try:
            # Sérialiser le modèle en bytes
            model_bytes = pickle.dumps(model)
            model_stream = io.BytesIO(model_bytes)
            file_size = len(model_bytes)

            # Upload vers MinIO
            result = self.client.put_object(
                self.bucket,
                object_name,
                model_stream,
                length=file_size,
                metadata=metadata
            )

            print(f"✅ Modèle uploadé: {object_name} ({file_size} bytes)")

            return {
                "bucket": self.bucket,
                "object_name": object_name,
                "size": file_size,
                "etag": result.etag,
                "version_id": result.version_id
            }

        except S3Error as e:
            print(f"❌ Erreur lors de l'upload: {e}")
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
            print(f"✅ Modèle uploadé: {object_name} ({len(model_bytes)} bytes)")
            return {
                "bucket": self.bucket,
                "object_name": object_name,
                "size": len(model_bytes),
                "etag": result.etag,
            }
        except S3Error as e:
            print(f"❌ Erreur lors de l'upload: {e}")
            raise

    def download_model(self, object_name: str) -> any:
        """
        Télécharge et désérialise un modèle depuis MinIO

        Args:
            object_name: Nom de l'objet dans MinIO

        Returns:
            Le modèle scikit-learn désérialisé
        """
        self._ensure_bucket_exists()
        try:
            # Télécharger l'objet
            response = self.client.get_object(self.bucket, object_name)
            model_bytes = response.read()
            response.close()
            response.release_conn()

            # Désérialiser le modèle
            model = pickle.loads(model_bytes)

            print(f"✅ Modèle téléchargé: {object_name}")
            return model

        except S3Error as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
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
            print(f"❌ Erreur lors du listing: {e}")
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
            print(f"✅ Modèle supprimé: {object_name}")
            return True
        except S3Error as e:
            print(f"❌ Erreur lors de la suppression: {e}")
            return False

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
                "metadata": stat.metadata
            }
        except S3Error as e:
            print(f"❌ Erreur lors de la récupération des infos: {e}")
            return None


# Instance globale
minio_service = MinIOService()
