"""
Database and MinIO initialization script.
Idempotent: can be re-run without creating duplicates.
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
    Creates the admin user if it does not already exist.
    The token is read from ADMIN_TOKEN (env); otherwise generated randomly.
    Idempotent: does nothing if 'admin' already exists.
    """
    async with AsyncSessionLocal() as db:
        existing = await db.execute(select(User).where(User.username == "admin"))
        if existing.scalar_one_or_none():
            print("   Admin user already present — skipped.")
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

        print(f"\n   Admin user created!")
        print(f"   Username : {user.username}")
        if not os.environ.get("ADMIN_TOKEN"):
            # Randomly generated token — store it securely
            print(f"   API Token: {admin_token}")
            print(f"   SAVE THIS TOKEN - It will not be displayed again!")
        else:
            print(f"   API Token: set via ADMIN_TOKEN (environment variable)")
        print()

        return user


async def upload_local_models_to_minio():
    """Upload local models to MinIO"""
    models_dir = Path("Models")

    if not models_dir.exists():
        print("Models/ directory not found")
        return

    joblib_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))

    if not joblib_files:
        print("No model files (.joblib or .pkl) found in Models/")
        return

    async with AsyncSessionLocal() as db:
        for model_file in joblib_files:
            try:
                print(f"\nUploading {model_file.name}...")

                # Load the model
                model = joblib.load(model_file)

                # Name and version
                model_name = model_file.stem
                version = "1.0.0"

                # Upload to MinIO
                object_key = f"{model_name}/{version}.joblib"
                upload_result = minio_service.upload_model(
                    model=model,
                    object_name=object_key,
                    metadata={"original_filename": model_file.name}
                )

                # Save metadata to DB
                metadata = ModelMetadata(
                    name=model_name,
                    version=version,
                    minio_bucket=settings.MINIO_BUCKET,
                    minio_object_key=object_key,
                    file_size_bytes=upload_result["size"],
                    description=f"Model {model_name} - Initial version",
                    is_active=True,
                    is_production=True
                )

                db.add(metadata)
                await db.commit()

                print(f"   {model_name} v{version} uploaded and registered")

            except Exception as e:
                print(f"   Error for {model_file.name}: {e}")
                continue


async def main():
    """Main initialization function"""
    print("=" * 60)
    print("Database and MinIO Initialization")
    print("=" * 60)

    # 1. Apply Alembic migrations (creates or updates the schema)
    print("\n1. Applying Alembic migrations...")
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
    print("   Migrations applied")

    # 2. Create the admin user
    print("\n2. Creating the admin user...")
    await create_default_user()

    # 3. Upload models to MinIO
    print("\n3. Uploading models to MinIO...")
    await upload_local_models_to_minio()

    print("\n" + "=" * 60)
    print("Initialization complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("   1. Save the admin API token shown above")
    print("   2. Test with: curl -H 'Authorization: Bearer <TOKEN>' http://localhost:8000/models")
    print()


if __name__ == "__main__":
    asyncio.run(main())
