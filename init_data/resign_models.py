#!/usr/bin/env python3
"""
Script one-shot : re-signe tous les modèles MinIO existants sans model_hmac_signature.

Contexte
--------
Depuis l'introduction de la vérification HMAC-SHA256 sur les fichiers .pkl,
tout chargement d'un modèle sans signature est refusé avec une HTTPException 403.
Ce script télécharge chaque fichier .pkl depuis MinIO, calcule son HMAC-SHA256
(avec la SECRET_KEY de l'application) et enregistre la signature dans la colonne
`model_hmac_signature` de la table `model_metadata`.

Prérequis
---------
- Docker Compose actif (PostgreSQL + MinIO accessibles)
- Variables d'environnement configurées (DATABASE_URL, MINIO_*, SECRET_KEY)
  ou valeurs par défaut de développement

Usage
-----
    docker exec predictml-api python init_data/resign_models.py

    # Voir tous les modèles (y compris ceux déjà signés) :
    docker exec predictml-api python init_data/resign_models.py --all

    # Simulation sans écriture :
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
    """Re-signe les modèles MinIO manquant un model_hmac_signature en base."""
    async with AsyncSessionLocal() as db:
        stmt = select(ModelMetadata).where(ModelMetadata.is_active.is_(True))
        if not force_all:
            stmt = stmt.where(ModelMetadata.model_hmac_signature.is_(None))
        result = await db.execute(stmt)
        models = result.scalars().all()

    if not models:
        print("Aucun modèle à re-signer.")
        return

    print(f"{'[DRY-RUN] ' if dry_run else ''}Modèles à re-signer : {len(models)}")
    print()

    signed = 0
    skipped = 0
    errors = 0

    for m in models:
        label = f"{m.name} v{m.version}"

        if not m.minio_object_key:
            print(f"  SKIP  {label} — modèle MLflow (pas de fichier modèle MinIO)")
            skipped += 1
            continue

        try:
            raw_bytes = minio_service.download_file_bytes(m.minio_object_key)
        except Exception as exc:
            print(f"  ERROR {label} — téléchargement MinIO échoué : {exc}")
            errors += 1
            continue

        signature = compute_model_hmac(raw_bytes)

        if dry_run:
            print(f"  DRY   {label} — signature calculée : {signature[:16]}…")
            signed += 1
            continue

        async with AsyncSessionLocal() as db:
            await db.execute(
                update(ModelMetadata)
                .where(ModelMetadata.id == m.id)
                .values(model_hmac_signature=signature)
            )
            await db.commit()

        print(f"  OK    {label} — signature enregistrée : {signature[:16]}…")
        signed += 1

    print()
    print(
        f"Résumé : {signed} {'simulé(s)' if dry_run else 're-signé(s)'}, "
        f"{skipped} ignoré(s), {errors} erreur(s)."
    )
    if errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-signe les modèles pkl sans signature HMAC.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcule les signatures sans écrire en base.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="force_all",
        help="Re-signe aussi les modèles déjà signés (force).",
    )
    args = parser.parse_args()
    asyncio.run(resign_models(dry_run=args.dry_run, force_all=args.force_all))


if __name__ == "__main__":
    main()
