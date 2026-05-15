#!/bin/sh
# Installs deps, generates /tmp/mlflow-auth/auth.ini from env vars and starts MLflow
# with basic-auth enabled. MLFLOW_ADMIN_PASSWORD must be set in the container environment.
set -e

pip install psycopg2-binary boto3 "mlflow[auth]" --quiet

# Créer le bucket MLflow dans MinIO s'il n'existe pas
python - <<'PYEOF'
import os, boto3, botocore
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
)
try:
    s3.head_bucket(Bucket="mlflow")
    print("[mlflow-entrypoint] Bucket 'mlflow' existe déjà.")
except botocore.exceptions.ClientError:
    s3.create_bucket(Bucket="mlflow")
    print("[mlflow-entrypoint] Bucket 'mlflow' créé dans MinIO.")
PYEOF

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
