#!/bin/sh
# docker/patroni/post-init.sh
#
# Script exécuté UNE SEULE FOIS par Patroni après l'initialisation du cluster.
# Équivalent de init_data/create_mlflow_db.sql en N1 (monté dans initdb.d).
#
# Patroni appelle ce script depuis le nœud primary juste après le bootstrap.
# Il ne sera plus appelé lors des redémarrages ultérieurs.

set -e

PGUSER="${POSTGRES_USER:-postgres}"

echo "[post-init] Création de la base de données MLflow..."

psql -v ON_ERROR_STOP=1 --username "$PGUSER" <<-EOSQL
    SELECT 'CREATE DATABASE mlflow'
    WHERE NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'mlflow'
    )\gexec
EOSQL

echo "[post-init] Base de données MLflow prête."
