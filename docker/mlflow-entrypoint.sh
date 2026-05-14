#!/bin/sh
# Installs deps, generates /tmp/mlflow-auth/auth.ini from env vars and starts MLflow
# with basic-auth enabled. MLFLOW_ADMIN_PASSWORD must be set in the container environment.
set -e

pip install psycopg2-binary boto3 "mlflow[auth]" --quiet

mkdir -p /tmp/mlflow-auth

cat > /tmp/mlflow-auth/auth.ini << EOF
[mlflow]
default_permission = NO_PERMISSIONS
database_uri = sqlite:////tmp/mlflow-auth/basic_auth.db
admin_username = ${MLFLOW_ADMIN_USER:-admin}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF

exec env MLFLOW_AUTH_CONFIG_PATH=/tmp/mlflow-auth/auth.ini \
  mlflow server \
  --app-name basic-auth \
  --backend-store-uri "postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD:-postgres}@postgres:${POSTGRES_INTERNAL_PORT:-5432}/mlflow" \
  --default-artifact-root s3://mlflow/ \
  --host 0.0.0.0 \
  --port "${MLFLOW_PORT:-5000}"
