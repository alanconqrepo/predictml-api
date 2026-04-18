#!/bin/sh
set -e

echo "=================================================="
echo " predictml-api — initialisation au démarrage"
echo "=================================================="

echo "[1/2] Initialisation DB + admin user..."
python init_data/init_db.py

mkdir -p "${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc}"

echo "[2/2] Démarrage de l'API..."
exec uvicorn src.main:app --host 0.0.0.0 --port 8000
