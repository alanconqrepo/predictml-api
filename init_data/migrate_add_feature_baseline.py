"""
Migration: add the feature_baseline column to model_metadata.
Idempotent — uses ADD COLUMN IF NOT EXISTS.

Execution:
    docker exec predictml-api python init_data/migrate_add_feature_baseline.py
"""

import asyncio

from sqlalchemy import text

from src.db.database import AsyncSessionLocal


async def main() -> None:
    print("Migration: adding feature_baseline to model_metadata...")
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(
                "ALTER TABLE model_metadata "
                "ADD COLUMN IF NOT EXISTS feature_baseline JSON;"
            )
        )
        await db.commit()
    print("Migration complete.")


if __name__ == "__main__":
    asyncio.run(main())
