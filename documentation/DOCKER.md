# Docker — PredictML API

## Daily Commands

```bash
# Start all services
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f api
docker-compose logs -f streamlit
docker-compose logs -f nginx
docker-compose logs -f redis-master
docker-compose logs -f prediction-writer

# Rebuild after modifying API code
docker-compose up -d --build api prediction-writer

# Rebuild after modifying the Streamlit dashboard
docker-compose up -d --build streamlit
```

## Services and Ports

The API is publicly exposed via **Nginx on port 80** (reverse proxy + load balancer).  
Port 8000 of the API is not directly exposed to the host.

| Service | External URL | Description |
|---|---|---|
| **API** (via Nginx) | http://localhost | Nginx reverse proxy → 3 FastAPI replicas |
| API docs | http://localhost/docs | Swagger UI |
| Dashboard | http://localhost:8501 | Streamlit Admin Dashboard |
| MLflow | http://localhost:5000 | Experiment tracking |
| MinIO console | http://localhost:9001 | Model storage management |
| Redis (master) | localhost:6379 | Distributed cache (auth required) |
| Grafana | http://localhost:3000 | Observability (Prometheus + Loki + Tempo) |
| PostgreSQL | localhost:5433 | Main database (write) |

## Docker Service Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │              frontend network                 │
                       │                                               │
Client ─── port 80 ──► │ nginx (reverse proxy, least_conn)            │
                       │   └─► api:8000 ×3 replicas                  │
                       └──────────────────────────────────────────────┘
                                        │
                       ┌────────────────▼─────────────────────────────┐
                       │              internal network                 │
                       │                                               │
                       │ pgbouncer (write pooler)                      │
                       │   └─► postgres:5432 (primary)                │
                       │         └─► postgres-replica (streaming)      │
                       │               └─► pgbouncer-read (pooler)    │
                       │                                               │
                       │ redis-master ← redis-replica-1/2             │
                       │ redis-sentinel-1/2/3 (quorum: 2)             │
                       │                                               │
                       │ prediction-writer (Redis Stream consumer)     │
                       │ minio · mlflow · grafana · streamlit          │
                       └──────────────────────────────────────────────┘
```

### Services by Role

| Service | Role |
|---|---|
| `nginx` | Reverse proxy, least_conn load balancer, single port 80 entry point |
| `api` (×3 replicas) | FastAPI — predictions, models, users |
| `migrate` | One-shot Alembic — runs before the API, does not restart |
| `prediction-writer` | Redis Streams worker — batch INSERT of predictions to DB |
| `postgres` | PostgreSQL primary — all writes |
| `postgres-replica` | PostgreSQL replica via streaming — analytical queries |
| `pgbouncer` | Connection pooler (transaction mode) for the primary |
| `pgbouncer-read` | Connection pooler for the replica |
| `redis-master` | Redis master — model cache (DB 0) + rate limiting (DB 1) |
| `redis-replica-1/2` | Redis replicas — high availability |
| `redis-sentinel-1/2/3` | Sentinel — automatic failover < 10 s (quorum: 2) |
| `minio` | S3-compatible object storage (models .joblib, train.py scripts) |
| `mlflow` | Experiment tracking (metrics, artifacts) |
| `grafana` | LGTM stack — Prometheus + Loki + Tempo + Grafana |
| `streamlit` | Multi-page admin dashboard |

## PostgreSQL Access

```bash
# Primary (stable container_name)
docker exec -it predictml-postgres psql -U postgres -d sklearn_api

# Or via local psql
psql -h localhost -p 5433 -U postgres -d sklearn_api
```

## Redis Cache (Sentinel)

```bash
# Connect to the master (with auth)
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD"

# Check the master role
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" INFO replication | grep role

# List cached keys
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" KEYS "*"

# Clear the model cache (forces reload from MinIO)
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" FLUSHDB

# Check Sentinel status
docker exec predictml-redis-sentinel-1 redis-cli -p 26379 SENTINEL masters
```

> **Note**: `REDIS_PASSWORD` is **required** in `.env`. The Sentinel uses the same
> variable to authenticate against the master and replicas.

## Prediction Queue (Redis Streams)

POST /predict predictions are written asynchronously via Redis Streams
to decouple DB writes from the critical inference path.

```bash
# Length of the pending stream
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" XLEN predictions:new

# Dead Letter Queue (messages that failed after MAX_RETRIES)
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" XLEN predictions:dlq

# Worker logs
docker-compose logs -f prediction-writer
```

If Redis is unavailable, the writer automatically falls back to synchronous mode.

## Alembic Migrations

Migrations are executed by the `migrate` service **before** the API starts.
In a multi-replica environment, this avoids simultaneous migration conflicts.

```bash
# Run migrations manually (if necessary)
docker-compose run --rm migrate

# View migration history
docker-compose run --rm migrate alembic history

# Create a new migration
docker-compose run --rm migrate alembic revision --autogenerate -m "description"
```

## Initialisation (First Deployment)

> The API no longer has `container_name: predictml-api` (3 replicas). Use
> `docker-compose exec` instead of `docker exec predictml-api`.

```bash
# Initialise the database and admin user
docker-compose exec api python init_data/init_db.py
```

## Complete Reset

```bash
# Removes all volumes (data loss)
docker-compose down -v
docker-compose up -d --build
docker-compose exec api python init_data/init_db.py
```

## Prometheus Metrics

The API exposes a standard scrape endpoint on `GET /metrics` (format `text/plain 0.0.4`).

> In multi-replica mode, Nginx distributes Prometheus requests between replicas.
> `PROMETHEUS_MULTIPROC_DIR` aggregates counters from all workers.

### HTTP Metrics (prometheus-fastapi-instrumentator)

| Metric | Type | Labels |
|---|---|---|
| `http_requests_total` | Counter | `method`, `handler`, `status_code` |
| `http_request_duration_seconds` | Histogram | `method`, `handler`, `status_code` |
| Python process metrics | Gauge | — |

### Business ML Metrics (src/core/ml_metrics.py)

| Metric | Type | Labels | Description |
|---|---|---|---|
| `predictml_predictions_total` | Counter | `model_name`, `version`, `mode`, `status` | Predictions by model and status |
| `predictml_inference_duration_seconds` | Histogram | `model_name`, `version` | Pure ML inference duration |
| `predictml_retrain_total` | Counter | `model_name`, `status` | Triggered retrains |
| `predictml_drift_detected_total` | Counter | `model_name`, `drift_type`, `severity` | Drift alerts |

### Verify the Endpoint

```bash
# Via Nginx (port 80)
curl http://localhost/metrics

# If METRICS_TOKEN is set
curl -H "Authorization: Bearer $METRICS_TOKEN" http://localhost/metrics
```

### Securing the Endpoint (recommended in production)

```bash
# In .env
METRICS_TOKEN=my-secret-token
```

Then update `monitoring/prometheus.yml`:

```yaml
- job_name: predictml-api
  static_configs:
    - targets: ['api:8000']
  authorization:
    credentials: "my-secret-token"
```

## Troubleshooting

```bash
# Check service status
docker-compose ps

# Check ports in use
netstat -ano | grep -E "80|8501|5433|9000|5000|6379|3000"

# Restart the API (all replicas)
docker-compose restart api

# Restart Nginx
docker-compose restart nginx

# Restart the Redis master
docker-compose restart redis-master

# Inspect service logs
docker-compose logs --tail=50 api
docker-compose logs --tail=50 nginx
docker-compose logs --tail=50 prediction-writer

# If the API is no longer responding via Nginx
docker-compose logs --tail=50 nginx
docker-compose ps api  # check the health check

# If Redis is unreachable
docker-compose ps redis-master redis-sentinel-1
docker-compose logs redis-sentinel-1

# If predictions are not being recorded in DB
docker-compose logs prediction-writer
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" XLEN predictions:dlq
```

## Environment Variables (`.env`)

> **Required variables** — the API or docker-compose refuses to start without them.

| Variable | Default | Required | Description |
|---|---|---|---|
| `SECRET_KEY` | — | **Yes** | HMAC key for model signing. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `REDIS_PASSWORD` | — | **Yes** | Redis password (master + replicas + sentinels). |
| `MINIO_ROOT_USER` | — | **Yes** | MinIO root account login. |
| `MINIO_ROOT_PASSWORD` | — | **Yes** | MinIO root account password. |
| `GRAFANA_ADMIN_PASSWORD` | — | **Yes** | Grafana admin password. |
| `TOKEN_LIFETIME_DAYS` | `90` | No | Bearer token validity in days. |
| `API_PORT` | `8000` | No | Internal API port (Nginx listens on 80) |
| `STREAMLIT_PORT` | `8501` | No | Dashboard port |
| `POSTGRES_PORT` | `5433` | No | PostgreSQL port exposed to host |
| `MINIO_PORT` | `9000` | No | MinIO API port |
| `MINIO_CONSOLE_PORT` | `9001` | No | MinIO console port |
| `MLFLOW_URL` | `http://localhost:5000` | No | MLflow URL visible from the browser |
| `REDIS_URL` | `redis://:$REDIS_PASSWORD@redis-master:6379/0` | No | Redis master URL (internal Docker) |
| `REDIS_SENTINEL_HOSTS` | (auto in compose) | No | Sentinel addresses `host:port,...` |
| `REDIS_CACHE_TTL` | `3600` | No | Model cache duration in seconds |
| `ENABLE_OTEL` | `true` | No | Enable OpenTelemetry to Grafana (enabled by default) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://grafana:4317` | No | Grafana OTLP endpoint |
| `METRICS_TOKEN` | `` | No | Bearer token to protect `GET /metrics` (empty = public) |
| `PROMETHEUS_MULTIPROC_DIR` | `/tmp/prometheus_multiproc` | No | Shared metrics directory between workers |
| `PREDICTION_STREAM_ENABLED` | `true` | No | Enable async Redis Streams queue for prediction writes |
| `PREDICTION_STREAM_BATCH_SIZE` | `100` | No | Rows per worker commit |
| `PREDICTION_STREAM_FLUSH_MS` | `500` | No | Max worker flush delay in ms |
| `PREDICTION_STREAM_MAX_RETRIES` | `3` | No | Attempts before sending to DLQ |
| `MAX_ROWS_ANALYTICS` | `50000` | No | Row limit for analytical queries |
| `ANALYTICS_MAX_DAYS` | `90` | No | Max time window for aggregations |
| `DATABASE_READ_REPLICA_URL` | (auto in compose) | No | PostgreSQL replica URL for analytical reads |
| `ADMIN_TOKEN` | `` | No | Custom admin token (auto-generated if empty) |
| `ENABLE_EMAIL_ALERTS` | `false` | No | Enable email alerts |
| `SMTP_HOST` | `` | No | SMTP server |
| `SMTP_PORT` | `587` | No | SMTP port |
| `SMTP_USER` | `` | No | SMTP user |
| `SMTP_PASSWORD` | `` | No | SMTP password |
| `SMTP_FROM` | `` | No | Sender address |
| `ALERT_EMAIL_TO` | `` | No | Alert recipients (comma-separated) |
| `WEEKLY_REPORT_ENABLED` | `false` | No | Enable weekly report |
| `PERFORMANCE_DRIFT_ALERT_THRESHOLD` | `0.10` | No | Accuracy drop threshold triggering an alert |
| `ERROR_RATE_ALERT_THRESHOLD` | `0.10` | No | Error rate triggering an alert |
| `MAX_MODEL_SIZE_MB` | `500` | No | Max upload size for a `.joblib` file |
