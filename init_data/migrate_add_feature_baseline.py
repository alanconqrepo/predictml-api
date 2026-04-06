"""
Migration : ajout de la colonne feature_baseline dans model_metadata.
Idempotent — utilise ADD COLUMN IF NOT EXISTS.

Exécution :
    docker exec predictml-api python init_data/migrate_add_feature_baseline.py
"""

import asyncio

from sqlalchemy import text

from src.db.database import AsyncSessionLocal


async def main() -> None:
    print("Migration : ajout de feature_baseline dans model_metadata...")
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                "ALTER TABLE model_metadata "
                "ADD COLUMN IF NOT EXISTS feature_baseline JSON;"
            )
        )
        await db.commit()
    print("Migration terminee.")


if __name__ == "__main__":
    asyncio.run(main())
