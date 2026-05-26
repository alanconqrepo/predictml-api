# Architecture — PredictML API

## Stack

| Service | Technology | External Port |
|---|---|---|
| Reverse proxy / Load balancer | Nginx 1.27 | **80** |
| API (3 replicas) | FastAPI (async) | — (internal only) |
| Admin Dashboard | Streamlit | 8501 |
| Database (primary) | PostgreSQL 16 | 5433 |
| Database (replica) | PostgreSQL 16 (streaming replication) | — |
| Connection pooler (write) | PgBouncer 1.23 (transaction mode) | — |
| Connection pooler (read) | PgBouncer 1.23 (transaction mode) | — |
| Model storage | MinIO (S3-compatible) | 9000 / console 9001 |
| Experiment tracking | MLflow | 5000 |
| Distributed cache | Redis 7 Sentinel (1 master + 2 replicas + 3 sentinels) | 6379 (master) |
| Prediction queue | Redis Streams + `prediction-writer` worker | — |
| Observability | Grafana LGTM (Loki + Tempo + Prometheus + OTLP) | 3000 |

---

## Project Structure

```
predictml-api/
├── src/
│   ├── api/                        # HTTP Endpoints
│   │   ├── models.py               # CRUD + drift + history + retrain + A/B
│   │   ├── predict.py              # POST /predict, /predict-batch, /explain, GET /predictions, /predictions/{id}, /predictions/{id}/explain, /stats, DELETE /predictions/purge
│   │   ├── users.py                # CRUD /users
│   │   ├── observed_results.py     # /observed-results
│   │   └── monitoring.py           # /monitoring/overview, /monitoring/model/{name}
│   ├── core/
│   │   ├── config.py               # Settings (env variables via dotenv)
│   │   ├── security.py             # Bearer token auth
│   │   ├── rate_limit.py           # Rate limiting per-IP via slowapi (Redis backend)
│   │   ├── ml_metrics.py           # Business ML Prometheus metrics (predictions, inference, drift, retrain)
│   │   └── telemetry.py            # OpenTelemetry → Grafana LGTM
│   ├── workers/
│   │   └── prediction_writer.py    # Redis Streams consumer — batch INSERT predictions
│   ├── db/
│   │   ├── models/                 # SQLAlchemy ORM
│   │   │   ├── user.py             # Table users
│   │   │   ├── prediction.py       # Table predictions
│   │   │   ├── model_metadata.py   # Table model_metadata
│   │   │   ├── observed_result.py  # Table observed_results
│   │   │   ├── golden_test.py      # Table golden_tests
│   │   │   └── model_history.py    # Table model_history
│   │   └── database.py             # Async session (asyncpg)
│   ├── services/
│   │   ├── db_service.py               # All DB queries
│   │   ├── model_service.py            # Loading, Redis cache, A/B/shadow routing
│   │   ├── minio_service.py            # Upload/download MinIO
│   │   ├── drift_service.py            # Drift computation Z-score + PSI + null rate (input + output)
│   │   ├── shap_service.py             # Local SHAP explanations (tree + linear)
│   │   ├── ab_significance_service.py  # A/B statistical tests (Chi-², Mann-Whitney U)
│   │   ├── auto_promotion_service.py   # Auto-promotion, auto-demotion (circuit breaker)
│   │   ├── golden_test_service.py      # Golden regression tests (CRUD + run)
│   │   ├── input_validation_service.py # Input feature schema validation
│   │   ├── supervision_reporter.py     # Supervision every 6h: drift, alerts, reactive retrain
│   │   ├── email_service.py            # Email alerts & weekly reports
│   │   └── webhook_service.py          # HTTP webhooks post-prediction
│   ├── schemas/                    # Pydantic schemas (I/O validation)
│   │   ├── model.py
│   │   ├── prediction.py
│   │   ├── user.py
│   │   ├── observed_result.py
│   │   └── monitoring.py
│   └── main.py                     # FastAPI app + lifecycle hooks
├── streamlit_app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                      # Login + home page
│   ├── utils/
│   │   ├── api_client.py           # HTTP client to the API
│   │   └── auth.py                 # session_state helpers
│   └── pages/
│       ├── 1_Users.py
│       ├── 2_Models.py
│       ├── 3_Predictions.py
│       ├── 4_Stats.py
│       ├── 5_Code_Example.py
│       ├── 6_AB_Testing.py      # A/B testing, shadow mode, statistical comparison
│       ├── 7_Supervision.py     # Global monitoring, drift, alerts, performance
│       ├── 8_Retrain.py         # Centralised retrain management (manual + scheduled)
│       └── 9_Golden_Tests.py    # CRUD golden test cases, run by version, CSV import
├── tests/                          # Pytest tests (automated, no Docker required)
├── smoke-tests/                    # Manual tests (live Docker)
├── init_data/                      # One-shot scripts (init_db, create_multiple_models)
│   └── example_train.py            # Example train.py script compatible with retrain
├── Models/                         # Local .joblib files
├── notebooks/                      # Jupyter notebooks
├── alembic/                        # DB migrations
├── docker-compose.yml
├── nginx.conf                          # Reverse proxy / load balancer configuration
├── docker/
│   ├── pg_hba.conf                     # PostgreSQL authentication (replication)
│   ├── postgres-replica-entrypoint.sh  # Init replica via pg_basebackup
│   └── sentinel-entrypoint.sh          # Generates sentinel.conf at startup
└── .env
```

---

## Data Flow — Prediction

```
Client
  │  POST /predict + Bearer Token
  ▼
Nginx (port 80, least_conn)
  → distributes to one of the 3 API replicas
  ▼
security.py
  → verifies token in DB
  → checks per-IP rate limit via slowapi (Redis DB 1)
  → checks daily quota (DB)
  ▼
predict.py
  → validates request (Pydantic)
  ▼
model_service.py
  → select_routing_versions() : A/B test / shadow / production
  → loads model (Redis DB 0 cache → MinIO)
  ▼
model.predict(X)  ← sklearn
  ▼
[if shadow] background_task → shadow prediction logged separately
  ▼
[PREDICTION_STREAM_ENABLED=true]
  → publishes to Redis Stream "predictions:new" (<1 ms)
  → prediction-writer consumes in batch (100 rows / 500 ms)
  → INSERT into PostgreSQL (via pgbouncer)
[synchronous fallback if Redis unavailable]
  ▼
[if webhook_url configured] webhook_service.send_webhook()
  ▼
Client ← JSON { prediction, probability, low_confidence, selected_version }
```

### PostgreSQL read/write routing

- **Writes** (predictions, models, users) → `pgbouncer` → `postgres` (primary)
- **Analytical reads** (`/predictions/stats`, `/monitoring`, aggregations) → `pgbouncer-read` → `postgres-replica`

Safety caps on analytical queries: `MAX_ROWS_ANALYTICS` (50,000 rows) and `ANALYTICS_MAX_DAYS` (90 days) to prevent unbounded aggregations.

### A/B / Shadow Routing

- `deployment_mode="production"`: main version, receives all traffic
- `deployment_mode="ab_test"`: receives `traffic_weight` fraction of traffic (e.g. 0.2 = 20%)
- `deployment_mode="shadow"`: executed in the background on each production request, result not returned to the client

The `model_service.select_routing_versions()` service determines which version responds to the request and which shadow versions to run in parallel.

---

## Data Flow — Streamlit Dashboard

```
Browser
  │  HTTP + API Token (session_state)
  ▼
streamlit_app/utils/api_client.py
  │  HTTP requests
  ▼
API FastAPI (http://api:8000)
  ▼
PostgreSQL / MinIO / Redis / MLflow
```

The Streamlit dashboard **never** talks directly to the DB or MinIO — the FastAPI API is the sole backend.

---

## Database

| Table | Role |
|---|---|
| `users` | Auth, roles (admin/user/readonly), rate limiting, Bearer token |
| `model_metadata` | Model registry (versioning, MinIO/MLflow location, tags, A/B config) |
| `predictions` | Complete log of each API call (features, result, response time, shadow flag) |
| `observed_results` | Actual observed results (to compare against predictions) |
| `golden_tests` | Golden test cases per model (expected features, expected output, description) |
| `model_history` | Log of model state changes (for rollback) |

### Table `predictions` — notable columns

| Column | Type | Description |
|---|---|---|
| `id_obs` | VARCHAR(255), nullable | Business identifier of the observation |
| `input_features` | JSON | Sent features |
| `prediction_result` | JSON | Model result |
| `probabilities` | JSON, nullable | Per-class probabilities |
| `response_time_ms` | Float | Inference time in milliseconds |
| `status` | VARCHAR(20) | `success` or `error` |
| `is_shadow` | Boolean | `true` if shadow prediction (not returned to client) |

### Table `model_history` — notable columns

| Column | Type | Description |
|---|---|---|
| `action` | str | Type of change (`set_production`, `update_metadata`, `rollback`…) |
| `snapshot` | JSON | Complete metadata state at the time of the change |
| `changed_fields` | JSON | List of modified fields |
| `changed_by_user_id` | int | Author of the change |

### Table `model_metadata` — notable columns

| Column | Type | Description |
|---|---|---|
| `deployment_mode` | str | `production`, `ab_test`, `shadow` |
| `traffic_weight` | float | Traffic share (0.0–1.0) |
| `confidence_threshold` | float | Min confidence threshold |
| `feature_baseline` | JSON | Per-feature stats for drift detection |
| `tags` | JSON | List of free tags |
| `webhook_url` | str | Post-prediction callback URL |
| `train_script_object_key` | str | MinIO key of the train.py script |
| `parent_version` | str | Source version of the retrain (lineage traceability) |
| `promotion_policy` | JSON | Post-retrain auto-promotion policy (`min_accuracy`, `max_latency_p95_ms`, etc.) |
| `retrain_schedule` | JSON | Automatic retraining cron schedule (`cron`, `lookback_days`, `enabled`, etc.) |
| `alert_thresholds` | JSON | Model-specific alert thresholds (overrides global thresholds) |
| `training_stats` | JSON | Snapshot of training data from the last retrain (`n_rows`, `feature_stats`, etc.) |

---

## Training Data Storage

### Strategy: MinIO (storage) + MLflow (traceability)

Training data is **not** stored as MLflow artifacts (avoids duplication on each run). They live in a dedicated MinIO bucket, and MLflow keeps only the **reference**.

```
MinIO bucket: datasets/
└── loan_model/
    └── v1/
        ├── train.parquet    ← training data
        └── test.parquet

MLflow Run:
  params:  dataset_uri = "s3://datasets/loan_model/v1"
  inputs:  mlflow.log_input(dataset, context="training")  ← reference, not a copy
```

**Advantages:**
- No duplication: 10 runs → 1 single copy of the dataset
- Large files handled natively by MinIO (multipart, streaming)
- Full traceability via MLflow

### Complete Workflow

```python
import os, mlflow, mlflow.sklearn, pandas as pd
from minio import Minio
from io import BytesIO

minio = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

if not minio.bucket_exists("datasets"):
    minio.make_bucket("datasets")

parquet_bytes = df_train.to_parquet(index=False)
minio.put_object("datasets", "loan_model/v1/train.parquet",
                 BytesIO(parquet_bytes), length=len(parquet_bytes))

DATASET_URI = "s3://datasets/loan_model/v1"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "minioadmin"
mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run() as run:
    dataset = mlflow.data.from_numpy(X_train.to_numpy(),
                                     source=f"{DATASET_URI}/train.parquet")
    mlflow.log_input(dataset, context="training")
    mlflow.log_param("dataset_uri", DATASET_URI)
    mlflow.sklearn.log_model(pipeline, "model")
    RUN_ID = run.info.run_id

requests.post("http://localhost:8000/models", headers=HEADERS, data={
    "name": "loan_model", "version": "1.0.0",
    "mlflow_run_id": RUN_ID, "training_dataset": DATASET_URI,
})
```

---

## Redis High Availability (Sentinel)

Redis runs in Sentinel mode: 1 master + 2 replicas + 3 sentinels (quorum: 2).

```
redis-master (write + read)
  ├── redis-replica-1  (asynchronous replication)
  └── redis-replica-2

redis-sentinel-1 }
redis-sentinel-2 } → quorum of 2 is sufficient to elect a new master
redis-sentinel-3 }   automatic failover in < 10 s
```

- **DB 0** — model instance cache (TTL configurable via `REDIS_CACHE_TTL`)
- **DB 1** — per-IP rate limiting counters (`slowapi`)
- **Stream** `predictions:new` — async prediction write queue
- **Stream** `predictions:dlq` — dead letter queue (after `MAX_RETRIES` failures)

The lock `retrain_lock:{name}:{version}` (SET NX EX 700) ensures that only one API replica
runs the scheduled retraining job at a time.

## Authentication

- **Mechanism**: HTTP Bearer token (`Authorization: Bearer <token>`)
- **Token**: `secrets.token_urlsafe(32)` stored in `users.api_token`
- **Roles**: `admin` (full access), `user` (predictions + read), `readonly`
- **Rate limiting**: daily quota per user (`rate_limit_per_day`)
- **Renewal**: `PATCH /users/{id}` with `{"regenerate_token": true}` (admin)

---

## Observability

### Structured Logging

The API uses `structlog` to produce structured JSON logs. Each logged request contains: `model_name`, `model_version`, `response_time_ms`, `status`, `user_id`.

### Prometheus Endpoint `/metrics`

The API exposes `GET /metrics` in Prometheus text 0.0.4 format via `prometheus-fastapi-instrumentator`.

Automatically collected metrics:
- `http_requests_total` (counter) — labels `method`, `handler`, `status_code`
- `http_request_duration_seconds` (histogram) — latency per endpoint and response code
- Standard Python process metrics (memory, CPU, file descriptors)

The file `monitoring/prometheus.yml` configures a `predictml-api` scrape job targeting `api:8000/metrics`. It is mounted in the `grafana/otel-lgtm` container at startup.

Optional security via `METRICS_TOKEN` (Bearer token). Multi-worker mode is supported via `PROMETHEUS_MULTIPROC_DIR` — counters are aggregated per worker in a shared directory.

### OpenTelemetry → Grafana LGTM

When `ENABLE_OTEL=true`, traces and metrics are sent to the OTLP collector (Grafana LGTM on port 4317).

Grafana LGTM bundles:
- **Loki** — log aggregation
- **Tempo** — distributed traces
- **Prometheus** — metrics (including scraping `/metrics`)
- **Grafana** — unified visualisation (http://localhost:3000)

### Email Alerts

Triggered automatically by `email_service.py` (APScheduler scheduler) when:
- Accuracy drift exceeds `PERFORMANCE_DRIFT_ALERT_THRESHOLD` (default: 10%)
- Error rate exceeds `ERROR_RATE_ALERT_THRESHOLD` (default: 10%)
- Weekly report if `WEEKLY_REPORT_ENABLED=true`

### Scheduled Retraining Scheduler

`src/tasks/retrain_scheduler.py` — second APScheduler (AsyncIOScheduler) dedicated to automatic retrains.

**Lifecycle:**
1. At startup (`lifespan` in `main.py`), loads all active `ModelMetadata` with `retrain_schedule.enabled=True`.
2. Creates one cron job per version (ID: `retrain_schedule:{name}:{version}`).
3. On each trigger, acquires a **Redis lock** `SET NX EX 700` to prevent concurrent executions in multi-replica environments.
4. Executes the retrain logic in a subprocess (timeout 600 s), creates a new version, updates `last_run_at` / `next_run_at`.

**Modifying a live schedule:** the `PATCH /models/{name}/{version}/schedule` endpoint updates the DB *and* the running scheduler (`add_retrain_job` / `remove_retrain_job`).

**Reactive retrain (drift-triggered):** if `trigger_on_drift` is configured in `retrain_schedule` (`"warning"` or `"critical"`), `run_alert_check()` evaluates `_max_input_drift` and `_max_output_drift` after each 6-hour cycle. If the detected drift level reaches or exceeds the configured threshold and the `drift_retrain_cooldown_hours` cooldown has expired, a retrain is triggered immediately via `_run_retrain_job()` — without waiting for the next cron cycle.
