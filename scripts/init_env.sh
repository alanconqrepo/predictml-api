#!/usr/bin/env bash
# New terminal > git bash > bash scripts/init_env.sh
# init_env.sh — Génère un .env avec des secrets forts à partir de .env.example
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
EXAMPLE_FILE="$ROOT_DIR/.env.example"

# ── Vérifications préalables ───────────────────────────────────────────────────
if [ ! -f "$EXAMPLE_FILE" ]; then
  echo "❌  .env.example introuvable dans $ROOT_DIR" >&2
  exit 1
fi

if [ -f "$ENV_FILE" ]; then
  echo "⚠️  Un fichier .env existe déjà."
  read -r -p "   Écraser ? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Annulé."; exit 0; }
fi

# ── Génération des secrets ─────────────────────────────────────────────────────
gen() { python -c "import secrets; print(secrets.token_urlsafe($1))"; }

SECRET_KEY=$(gen 32)
ADMIN_TOKEN=$(gen 24)
POSTGRES_PASSWORD=$(gen 24)
MINIO_ROOT_PASSWORD=$(gen 32)
REDIS_PASSWORD=$(gen 24)
GRAFANA_ADMIN_PASSWORD=$(gen 24)
MLFLOW_ADMIN_PASSWORD=$(gen 24)
MLFLOW_FLASK_SERVER_SECRET_KEY=$(gen 32)
METRICS_TOKEN=$(gen 32)

# Utilisateur MinIO fixe (pas un secret)
MINIO_USER="minioadmin"

# ── Construction du .env ───────────────────────────────────────────────────────
# On part de .env.example et on substitue chaque valeur placeholder
cp "$EXAMPLE_FILE" "$ENV_FILE"

replace() {
  local key="$1" val="$2"
  if sed --version &>/dev/null 2>&1; then
    sed -i "s|^#\?[[:space:]]*${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    sed -i '' "s|^#\?[[:space:]]*${key}=.*|${key}=${val}|" "$ENV_FILE"
  fi
}

replace SECRET_KEY          "$SECRET_KEY"
replace ADMIN_TOKEN         "$ADMIN_TOKEN"
replace POSTGRES_PASSWORD   "$POSTGRES_PASSWORD"
replace MINIO_ROOT_USER     "$MINIO_USER"
replace MINIO_ROOT_PASSWORD "$MINIO_ROOT_PASSWORD"
replace MINIO_ACCESS_KEY    "$MINIO_USER"
replace MINIO_SECRET_KEY    "$MINIO_ROOT_PASSWORD"
replace REDIS_PASSWORD      "$REDIS_PASSWORD"
replace GRAFANA_ADMIN_USER  "admin"
replace GRAFANA_ADMIN_PASSWORD "$GRAFANA_ADMIN_PASSWORD"
replace MLFLOW_ADMIN_USER   "admin"
replace MLFLOW_ADMIN_PASSWORD  "$MLFLOW_ADMIN_PASSWORD"
replace MLFLOW_TRACKING_USERNAME "admin"
replace MLFLOW_TRACKING_PASSWORD "$MLFLOW_ADMIN_PASSWORD"
replace MLFLOW_FLASK_SERVER_SECRET_KEY "$MLFLOW_FLASK_SERVER_SECRET_KEY"

_minio_internal_port=$(grep -E '^MINIO_INTERNAL_PORT=' "$ENV_FILE" | cut -d= -f2 | sed 's/#.*//' | tr -d '[:space:]')
_minio_internal_port="${_minio_internal_port:-9000}"
replace MLFLOW_S3_ENDPOINT_URL  "http://minio:${_minio_internal_port}"
replace AWS_ACCESS_KEY_ID       "$MINIO_USER"
replace AWS_SECRET_ACCESS_KEY   "$MINIO_ROOT_PASSWORD"
replace METRICS_TOKEN       "$METRICS_TOKEN"

# Les DATABASE_URL contiennent le mot de passe — on les réécrit entièrement.
# Ces valeurs sont pour le DEV LOCAL sans Docker (l'API accède à postgres directement).
# En Docker, docker-compose.yml reconstruit ces URLs via pgbouncer.
_pg_port=$(grep -E '^POSTGRES_PORT=' "$ENV_FILE" | cut -d= -f2 | sed 's/#.*//' | tr -d '[:space:]')
_pg_port="${_pg_port:-5433}"
_pg_db=$(grep -E '^POSTGRES_DB=' "$ENV_FILE" | cut -d= -f2 | sed 's/#.*//' | tr -d '[:space:]')
_pg_db="${_pg_db:-sklearn_api}"
_pg_user=$(grep -E '^POSTGRES_USER=' "$ENV_FILE" | cut -d= -f2 | sed 's/#.*//' | tr -d '[:space:]')
_pg_user="${_pg_user:-postgres}"

replace DATABASE_URL \
  "postgresql+asyncpg://${_pg_user}:${POSTGRES_PASSWORD}@localhost:${_pg_port}/${_pg_db}"
replace DATABASE_READ_REPLICA_URL \
  "postgresql+asyncpg://${_pg_user}:${POSTGRES_PASSWORD}@localhost:${_pg_port}/${_pg_db}"

# ── Génération de monitoring/prometheus.yml depuis le template ─────────────────
PROMETHEUS_TEMPLATE="$ROOT_DIR/monitoring/prometheus.yml.template"
PROMETHEUS_OUT="$ROOT_DIR/monitoring/prometheus.yml"
if [ -f "$PROMETHEUS_TEMPLATE" ]; then
  _api_port=$(grep -E '^API_PORT=' "$ENV_FILE" | cut -d= -f2 | sed 's/#.*//' | tr -d '[:space:]')
  _api_port="${_api_port:-8000}"
  export API_PORT="$_api_port"
  envsubst '${API_PORT}' < "$PROMETHEUS_TEMPLATE" > "$PROMETHEUS_OUT"
  echo "✅  monitoring/prometheus.yml généré (API_PORT=${_api_port})"
fi

# ── Résumé ─────────────────────────────────────────────────────────────────────
echo ""
echo "✅  .env généré avec succès dans : $ENV_FILE"
echo ""
echo "   ADMIN_TOKEN            : $ADMIN_TOKEN"
echo "   GRAFANA_ADMIN_PASSWORD : $GRAFANA_ADMIN_PASSWORD"
echo "   MLFLOW_ADMIN_PASSWORD  : $MLFLOW_ADMIN_PASSWORD"
echo ""
echo "⚠️  Ces valeurs ne s'affichent qu'une seule fois — conservez-les."
echo "   Ne committez jamais .env dans git."
