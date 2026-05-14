#!/bin/bash
set -e

mkdir -p "${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc}"

exec uvicorn src.main:app --host 0.0.0.0 --port "${API_PORT:-8000}" --workers 2
