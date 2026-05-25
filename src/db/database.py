"""
Database configuration
"""

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src.core.config import settings

_ENGINE_KWARGS = dict(
    echo=True if settings.DEBUG else False,
    future=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    # Recycle connections every 10 min (prevents stale connections)
    pool_recycle=600,
    # Validate connection before use (detects silent disconnections)
    pool_pre_ping=True,
    # Max wait time to obtain a connection from the SQLAlchemy pool.
    # Must be > PgBouncer QUERY_WAIT_TIMEOUT (10 s) so PgBouncer errors
    # surface cleanly before SQLAlchemy gives up on its side.
    pool_timeout=15,
    # LIFO: prefer reusing recent connections → better cache on Postgres side
    pool_use_lifo=True,
    # PgBouncer transaction mode does not support server-side prepared statements.
    # prepared_statement_cache_size=0 disables the LRU cache but asyncpg 0.30+ still
    # creates *named* prepared statements (e.g. __asyncpg_stmt_39__).  When PgBouncer
    # reassigns a backend connection to a different asyncpg connection, both may try to
    # create the same name → "prepared statement already exists".
    # prepared_statement_name_func forces a globally-unique name per statement so
    # collisions are impossible regardless of connection reuse.
    connect_args={
        "prepared_statement_cache_size": 0,
        "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex}__",
    },
)

# Write engine — primary (all mutations)
engine = create_async_engine(settings.DATABASE_URL, **_ENGINE_KWARGS)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Read engine — replica when DATABASE_READ_REPLICA_URL is set, otherwise reuses write engine
if settings.DATABASE_READ_REPLICA_URL:
    read_engine = create_async_engine(settings.DATABASE_READ_REPLICA_URL, **_ENGINE_KWARGS)
    ReadAsyncSessionLocal = async_sessionmaker(
        read_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    _separate_read_engine = True
else:
    read_engine = engine
    ReadAsyncSessionLocal = AsyncSessionLocal
    _separate_read_engine = False

# ORM declarative base
Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency for write/transactional endpoints (primary)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_read_db() -> AsyncSession:
    """Dependency for analytics/read-only endpoints. Routes to read replica when configured."""
    async with ReadAsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Verify the database connection.

    Schema creation and migrations are managed by Alembic (alembic upgrade head).
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    if _separate_read_engine:
        async with read_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))


async def close_db():
    """Close database connections"""
    await engine.dispose()
    if _separate_read_engine:
        await read_engine.dispose()
