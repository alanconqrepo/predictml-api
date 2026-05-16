"""
Script d'initialisation de la base de données et MinIO.
Idempotent : peut être relancé sans créer de doublons.
"""
import asyncio
import os
import secrets
from pathlib import Path

import joblib

from sqlalchemy import select

from src.db.database import AsyncSessionLocal
from src.db.models import User, ModelMetadata
from src.services.minio_service import minio_service
from src.core.config import settings


async def create_default_user():
    """
    Crée l'utilisateur admin s'il n'existe pas encore.
    Le token est lu depuis ADMIN_TOKEN (env) ; sinon généré aléatoirement.
    Idempotent : ne fait rien si 'admin' existe déjà.
    """
    async with AsyncSessionLocal() as db:
        existing = await db.execute(select(User).where(User.username == "admin"))
        if existing.scalar_one_or_none():
            print("   Utilisateur admin deja present — ignore.")
            return

        admin_token = os.environ.get("ADMIN_TOKEN") or secrets.token_urlsafe(32)

        user = User(
            username="admin",
            email=os.environ.get("ADMIN_EMAIL", "admin@predictml.local"),
            api_token=admin_token,
            role="admin",
            rate_limit_per_day=10000,
            is_active=True
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        print(f"\n   Utilisateur admin cree!")
        print(f"   Username : {user.username}")
        if not os.environ.get("ADMIN_TOKEN"):
            # Token généré aléatoirement — à conserver de façon sécurisée
            print(f"   API Token: {admin_token}")
            print(f"   SAUVEGARDEZ CE TOKEN - Il ne sera plus affiche!")
        else:
            print(f"   API Token: défini via ADMIN_TOKEN (variable d'environnement)")
        print()

        return user


async def upload_local_models_to_minio():
    """Upload les modèles locaux vers MinIO"""
    models_dir = Path("Models")

    if not models_dir.exists():
        print("Dossier Models/ introuvable")
        return

    joblib_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))

    if not joblib_files:
        print("Aucun fichier modèle (.joblib ou .pkl) trouvé dans Models/")
        return

    async with AsyncSessionLocal() as db:
        for model_file in joblib_files:
            try:
                print(f"\nUpload de {model_file.name}...")

                # Charger le modèle
                model = joblib.load(model_file)

                # Nom et version
                model_name = model_file.stem
                version = "1.0.0"

                # Upload vers MinIO
                object_key = f"{model_name}/{version}.joblib"
                upload_result = minio_service.upload_model(
                    model=model,
                    object_name=object_key,
                    metadata={"original_filename": model_file.name}
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
                print(f"   Erreur pour {model_file.name}: {e}")
                continue


async def main():
    """Fonction principale d'initialisation"""
    print("=" * 60)
    print("Initialisation de la Base de Donnees et MinIO")
    print("=" * 60)

    # 1. Appliquer les migrations Alembic (crée ou met à jour le schéma)
    print("\n1. Application des migrations Alembic...")
    import os
    import subprocess
    import sys

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=_root,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Alembic upgrade failed (code {result.returncode})")
    print("   Migrations appliquees")

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
