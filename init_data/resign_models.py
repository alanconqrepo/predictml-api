#!/usr/bin/env python3
"""
One-shot script: re-signs all existing MinIO models that have no model_hmac_signature.

Context
-------
Since the introduction of HMAC-SHA256 verification on .joblib files,
loading a model without a signature is rejected with an HTTPException 403.
This script downloads each .joblib file from MinIO, computes its HMAC-SHA256
(using the application SECRET_KEY) and stores the signature in the
`model_hmac_signature` column of the `model_metadata` table.

Prerequisites
-------------
- Docker Compose running (PostgreSQL + MinIO accessible)
- Environment variables configured (DATABASE_URL, MINIO_*, SECRET_KEY)
  or development default values

Usage
-----
    docker exec predictml-api python init_data/resign_models.py

    # Show all models (including already-signed ones):
    docker exec predictml-api python init_data/resign_models.py --all

    # Dry run without writing:
    docker exec predictml-api python init_data/resign_models.py --dry-run
"""

import argparse
import asyncio
import sys

# Ensure the project root is on the path when run directly
sys.path.insert(0, "/app")

from sqlalchemy import select, update

from src.core.config import settings
from src.db.database import AsyncSessionLocal
from src.db.models.model_metadata import ModelMetadata
from src.services.minio_service import minio_service
from src.services.model_service import compute_model_hmac


async def resign_models(dry_run: bool = False, force_all: bool = False) -> None:
    """Re-signs MinIO models that are missing a model_hmac_signature in the database."""
    async with AsyncSessionLocal() as db:
        stmt = select(ModelMetadata).where(ModelMetadata.is_active.is_(True))
        if not force_all:
            stmt = stmt.where(ModelMetadata.model_hmac_signature.is_(None))
        result = await db.execute(stmt)
        models = result.scalars().all()

    if not models:
        print("No models to re-sign.")
        return

    print(f"{'[DRY-RUN] ' if dry_run else ''}Models to re-sign: {len(models)}")
    print()

    signed = 0
    skipped = 0
    errors = 0

    for m in models:
        label = f"{m.name} v{m.version}"

        if not m.minio_object_key:
            print(f"  SKIP  {label} — MLflow model (no MinIO model file)")
            skipped += 1
            continue

        try:
            raw_bytes = minio_service.download_file_bytes(m.minio_object_key)
        except Exception as exc:
            print(f"  ERROR {label} — MinIO download failed: {exc}")
            errors += 1
            continue

        signature = compute_model_hmac(raw_bytes)

        if dry_run:
            print(f"  DRY   {label} — computed signature: {signature[:16]}…")
            signed += 1
            continue

        async with AsyncSessionLocal() as db:
            await db.execute(
                update(ModelMetadata)
                .where(ModelMetadata.id == m.id)
                .values(model_hmac_signature=signature)
            )
            await db.commit()

        print(f"  OK    {label} — signature saved: {signature[:16]}…")
        signed += 1

    print()
    print(
        f"Summary: {signed} {'simulated' if dry_run else 're-signed'}, "
        f"{skipped} skipped, {errors} error(s)."
    )
    if errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-sign joblib models that have no HMAC signature.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signatures without writing to the database.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="force_all",
        help="Re-sign models that already have a signature (force).",
    )
    args = parser.parse_args()
    asyncio.run(resign_models(dry_run=args.dry_run, force_all=args.force_all))


if __name__ == "__main__":
    main()
