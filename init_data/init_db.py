"""
Script d'initialisation de la base de données et MinIO
"""
import asyncio
import secrets
from pathlib import Path
import pickle

from src.db.database import init_db, AsyncSessionLocal
from src.db.models import User, ModelMetadata
from src.services.minio_service import minio_service
from src.core.config import settings


async def create_default_user():
    """Crée un utilisateur par défaut"""
    async with AsyncSessionLocal() as db:
        # Générer un token unique
        admin_token = secrets.token_urlsafe(32)

        user = User(
            username="admin",
            email="admin@sklearn-api.local",
            api_token=admin_token,
            role="admin",
            rate_limit_per_day=10000,
            is_active=True
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        print(f"\nUtilisateur admin cree!")
        print(f"   Username: {user.username}")
        print(f"   Email: {user.email}")
        print(f"   API Token: {admin_token}")
        print(f"   \nSAUVEGARDEZ CE TOKEN - Il ne sera plus affiche!\n")

        return user


async def upload_local_models_to_minio():
    """Upload les modèles locaux vers MinIO"""
    models_dir = Path("Models")

    if not models_dir.exists():
        print("Dossier Models/ introuvable")
        return

    pkl_files = list(models_dir.glob("*.pkl"))

    if not pkl_files:
        print("Aucun modele .pkl trouve dans Models/")
        return

    async with AsyncSessionLocal() as db:
        for pkl_file in pkl_files:
            try:
                print(f"\nUpload de {pkl_file.name}...")

                # Charger le modèle
                with open(pkl_file, "rb") as f:
                    model = pickle.load(f)

                # Nom et version
                model_name = pkl_file.stem
                version = "1.0.0"

                # Upload vers MinIO
                object_key = f"{model_name}/{version}.pkl"
                upload_result = minio_service.upload_model(
                    model=model,
                    object_name=object_key,
                    metadata={"original_filename": pkl_file.name}
                )

                # Enregistrer les métadonnées en DB
                metadata = ModelMetadata(
                    name=model_name,
                    version=version,
                    minio_bucket=settings.MINIO_BUCKET,
                    minio_object_key=object_key,
                    file_size_bytes=upload_result["size"],
                    description=f"Modele {model_name} - Version initiale",
                    is_active=True,
                    is_production=True
                )

                db.add(metadata)
                await db.commit()

                print(f"   {model_name} v{version} uploade et enregistre")

            except Exception as e:
                print(f"   Erreur pour {pkl_file.name}: {e}")
                continue


async def main():
    """Fonction principale d'initialisation"""
    print("=" * 60)
    print("Initialisation de la Base de Donnees et MinIO")
    print("=" * 60)

    # 1. Créer les tables
    print("\n1. Creation des tables PostgreSQL...")
    await init_db()
    print("   Tables creees")

    # 2. Créer l'utilisateur admin
    print("\n2. Creation de l'utilisateur admin...")
    await create_default_user()

    # 3. Upload des modèles vers MinIO
    print("\n3. Upload des modeles vers MinIO...")
    await upload_local_models_to_minio()

    print("\n" + "=" * 60)
    print("Initialisation terminee!")
    print("=" * 60)
    print("\nProchaines etapes:")
    print("   1. Sauvegardez le token API admin ci-dessus")
    print("   2. Testez avec: curl -H 'Authorization: Bearer <TOKEN>' http://localhost:8000/models")
    print()


if __name__ == "__main__":
    asyncio.run(main())
