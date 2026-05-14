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
MINIO_SECRET_KEY=$(gen 32)
REDIS_PASSWORD=$(gen 24)
GRAFANA_ADMIN_PASSWORD=$(gen 24)

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
replace MINIO_SECRET_KEY    "$MINIO_SECRET_KEY"
replace REDIS_PASSWORD      "$REDIS_PASSWORD"
replace GRAFANA_ADMIN_USER  "admin"
replace GRAFANA_ADMIN_PASSWORD "$GRAFANA_ADMIN_PASSWORD"

# Les DATABASE_URL contiennent le mot de passe — on les réécrit entièrement
replace DATABASE_URL \
  "postgresql+asyncpg://postgres:${POSTGRES_PASSWORD}@pgbouncer:5432/sklearn_api"
replace DATABASE_READ_REPLICA_URL \
  "postgresql+asyncpg://postgres:${POSTGRES_PASSWORD}@pgbouncer-read:5432/sklearn_api"

# ── Résumé ─────────────────────────────────────────────────────────────────────
echo ""
echo "✅  .env généré avec succès dans : $ENV_FILE"
echo ""
echo "   ADMIN_TOKEN          : $ADMIN_TOKEN"
echo "   GRAFANA_ADMIN_PASSWORD : $GRAFANA_ADMIN_PASSWORD"
echo ""
echo "⚠️  Ces valeurs ne s'affichent qu'une seule fois — conservez-les."
echo "   Ne committez jamais .env dans git."
