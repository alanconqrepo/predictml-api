#!/bin/sh
# Entrypoint for the PostgreSQL streaming-replication standby.
# On first start (empty volume), initialises the data directory from the primary
# via pg_basebackup -R, which writes standby.signal and primary_conninfo so the
# replica enters hot-standby mode automatically.
set -e

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PRIMARY_HOST="${POSTGRES_MASTER_HOST:-postgres}"
PRIMARY_PORT="${POSTGRES_MASTER_PORT:-5432}"

if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "[replica] Data directory empty — running pg_basebackup from ${PRIMARY_HOST}:${PRIMARY_PORT}..."
    mkdir -p "$PGDATA"
    chown postgres:postgres "$PGDATA"
    chmod 700 "$PGDATA"

    su-exec postgres sh -c "
        PGPASSWORD='${POSTGRES_PASSWORD:-postgres}' pg_basebackup \
            -h '${PRIMARY_HOST}' \
            -p '${PRIMARY_PORT}' \
            -U '${PGUSER:-postgres}' \
            -D '${PGDATA}' \
            -R \
            -X stream \
            -P
    "
    echo "[replica] Base backup complete — standby.signal written."
fi

exec su-exec postgres postgres "$@"
