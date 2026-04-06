"""
Alembic environment configuration — async SQLAlchemy support (asyncpg)
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Ajouter la racine du projet au sys.path pour importer src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import settings  # noqa: E402
from src.db.database import Base  # noqa: E402
from src.db import models  # noqa: E402, F401 — enregistre tous les modèles dans Base.metadata

# Objet de configuration Alembic
config = context.config

# Surcharger l'URL depuis les settings de l'application (DRY — pas de duplication)
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Configurer le logging depuis alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Métadonnées cible pour l'autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Migrations en mode offline (génération SQL sans connexion DB)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Exécute les migrations sur une connexion synchrone fournie."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Migrations en mode online avec engine async (asyncpg)."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # obligatoire pour éviter les conflits de pool lors des migrations
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
