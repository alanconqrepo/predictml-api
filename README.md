# PredictML API

Machine Learning deployment and prediction API, built with FastAPI.  
Designed to deploy scikit-learn models to production, version them, track every prediction, and measure model drift over time.

**Version 2.0** — Multi-user, distributed storage, experiment tracking, admin dashboard.

[![Tests](https://github.com/alanconqrepo/predictml-api/actions/workflows/tests.yml/badge.svg)](https://github.com/alanconqrepo/predictml-api/actions/workflows/tests.yml)

---

## Why this project?

Training an ML model is one thing. Making it production-ready, secured, versioned, and observable is another.

PredictML API solves this problem:

- **One-command deployment** — upload a `.joblib`, the API is ready
- **Multi-model and multi-version** — each model has its own lifecycle (active, production, deprecated)
- **Complete traceability** — every prediction is logged with its features, result, latency, and the user
- **Continuous evaluation** — observed results can be reported to measure actual model accuracy
- **A/B testing & shadow deployment** — route traffic between versions and compare silently
- **Drift detection** — detect feature drift in production (Z-score + PSI)
- **SHAP explainability** — understand why the model made a prediction, and measure the global feature importance on recent predictions
- **Automatic retraining** — trigger a retrain from the API with a `train.py` script, or schedule it automatically via a cron expression
- **Supervision & alerts** — global monitoring, email alerts, weekly reports
- **Multi-user management** — Bearer tokens, roles (admin/user/readonly), daily quotas
- **Admin dashboard** — Streamlit interface to manage everything without code

---

## Target audience

| Profile | Use case |
|---|---|
| Data Scientist | Deploy a `.joblib` model without writing server code |
| Back-end Developer | Consume the API in an application |
| MLOps | Version, monitor, compare, and retrain models in production |
| Administrator | Manage users, quotas, and access via the dashboard |

---

## Tech stack

| Component | Technology | Port |
|---|---|---|
| Reverse proxy | Nginx 1.27 | **80** |
| API (3 replicas) | FastAPI (async) | — (internal) |
| Admin dashboard | Streamlit | 8501 |
| Database | PostgreSQL 16 (primary + replica) | 5433 |
| Connection pooling | PgBouncer 1.23 | — |
| Model storage | MinIO (S3-compatible) | 9000 / console 9001 |
| Experiment tracking | MLflow | 5000 |
| Distributed cache | Redis 7 Sentinel (1 master + 2 replicas + 3 sentinels) | 6379 |
| Predictions queue | Redis Streams + dedicated worker | — |
| Observability | Grafana LGTM (Loki + Tempo + Prometheus) | 3000 |

---

## Prerequisites & Installation

```bash
# Prerequisites: Git, Docker Desktop (with Docker Compose v2)
git clone https://github.com/alanconqrepo/predictml-api.git
cd predictml-api
```

---

## Quick start

```bash
# 1. Generate secrets in .env (open a Git Bash terminal)
bash scripts/init_env.sh

# 2. (Optional) Remove existing Postgres volumes before first deployment
#    Do this if starting from scratch or if the password changed
docker-compose -p predictml-api down -v 2>&1

# 3. Start all services
docker-compose -p predictml-api up -d --build

# 4. Access the admin dashboard
open http://localhost:8501

# 5. Test the API (via Nginx port 80)
curl http://localhost/health
```

> The admin user is automatically created at startup using `ADMIN_TOKEN` in `.env`.
> If you want to upload local `.joblib` models from `Models/`, run:
> `docker-compose -p predictml-api exec api python init_data/init_db.py`

**Credentials**

| Service | Credentials |
|---|---|
| API admin token | value of `ADMIN_TOKEN` in `.env` |
| PostgreSQL | `postgres / <POSTGRES_PASSWORD>` (see `.env`) |
| MinIO | values of `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` in `.env` |
| Redis | value of `REDIS_PASSWORD` in `.env` |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3000 (`admin` / value of `GRAFANA_ADMIN_PASSWORD`) |

---

## Main features

### Model management
- Upload a `.joblib` file via API or reference an MLflow run
- Versioning (`name` + `version`)
- `is_production` flag to automatically route predictions
- Rich metadata: algorithm, accuracy, f1_score, features, classes, dataset
- Custom tags and notification webhooks
- Configurable confidence threshold (`confidence_threshold`)
- Feature baseline for drift detection

### Input schema validation
- `POST /models/{name}/{version}/validate-input` — validate features before predicting (without consuming quota)
- Detects **missing features**, **unexpected features**, and **type coercions** (`string` → `float`)
- Source of truth: `feature_names_in_` from the sklearn model (priority) or `feature_baseline` stored in DB
- `POST /predict?strict_validation=true` — strict mode: rejects with 422 unexpected features

### Predictions
- `POST /predict` — single prediction with smart routing (A/B, shadow); `?strict_validation=true` to reject unexpected features
- `POST /predict-batch` — batch predictions (model loaded once); `?strict_validation=true` also available
- Class probability output when available (`predict_proba`)
- `low_confidence` flag when the max probability is below the configured threshold
- `id_obs` identifier to link a prediction to an observed result
- Distributed Redis cache for model instances

### A/B Testing & Shadow Deployment
- `deployment_mode`: `"production"`, `"ab_test"`, or `"shadow"`
- `traffic_weight`: fraction of traffic routed to a version (0.0 – 1.0)
- Shadow mode: predictions run in the background without client impact
- `GET /models/{name}/ab-compare`: side-by-side comparison report with automatic **statistical significance test** (Chi-² on error rate, Mann-Whitney U on latency) — includes `p_value`, `winner`, and `min_samples_needed` to avoid promoting noise
- `GET /models/{name}/shadow-compare`: enriched report comparing the shadow model and the production model — compared accuracy, confidence delta, latency delta, disagreement rate, and automatic promotion recommendation

### SHAP explainability
- `POST /explain`: local SHAP values for an observation
- `GET /models/{name}/feature-importance`: global importance aggregated over the last N predictions
  - Parameters: `version`, `last_n` (default 100, max 500), `days` (default 7)
  - Returns `mean(|SHAP|)` per feature + ranking — ideal for detecting behavioral drifts
- Support for tree (TreeExplainer) and linear (LinearExplainer) models
- Interpretation: contribution of each feature to the prediction

### Data drift (input + output)
- `GET /models/{name}/drift`: drift report per feature (Z-score + PSI + null rate)
- `GET /models/{name}/output-drift`: output distribution drift (**label shift**) via PSI — compares recent prediction distribution to the reference distribution from `training_stats`
- Statuses: `ok`, `warning`, `critical`, `no_baseline`, `insufficient_data`
- **5 monitoring dimensions**: feature distribution drift, performance drift (accuracy/MAE), error rate drift, null rate per feature, and output label shift

### Per-model alert thresholds

- `alert_thresholds` configurable via `PATCH /models/{name}/{version}` to define model-specific thresholds (drift, error rate, null rate)
- Overrides global thresholds defined by environment variables
- Used by the supervision service to trigger targeted alerts

### Confidence trend & Lineage

- `GET /models/{name}/confidence-trend`: evolution of average prediction confidence over time (by time window)
- `parent_version`: each new version from a retrain stores a reference to its source version (complete lineage traceability)
- `training_stats`: training data snapshot saved automatically at each retrain (n_rows, feature_stats, label_distribution)

### Cache warm-up

- `POST /models/{name}/{version}/warmup`: preloads a model into memory (Redis cache) without waiting for the first prediction request
- Reduces cold-start latency during deployments

### Real-world performance
- `GET /models/{name}/performance`: metrics calculated from observed results
- Classification: accuracy, precision, recall, F1, confusion matrix, per-class metrics
- Regression: MAE, MSE, RMSE, R²
- Configurable time aggregation

### Automatic retraining
- Upload a `train.py` at model registration
- `POST /models/{name}/{version}/retrain`: triggers training over a date range
- `PATCH /models/{name}/{version}/schedule`: schedules automatic retraining via a cron expression (e.g. `"0 3 * * 1"` = every Monday at 3am UTC)
- **Drift-triggered retrain**: set `trigger_on_drift: "critical"` (or `"warning"`) in the schedule so that a retrain launches automatically when the detected drift reaches the threshold — `drift_retrain_cooldown_hours` prevents retrain loops
- Full stdout/stderr logs returned in the response
- New version automatically registered in MinIO
- Redis lock to prevent simultaneous executions in multi-replica deployments

### Auto-promotion & Auto-demotion (circuit breaker)
- `PATCH /models/{name}/policy`: define the post-retrain auto-promotion policy (`min_accuracy`, `max_latency_p95_ms`, `min_sample_validation`, `auto_promote`, `min_golden_test_pass_rate`)
- **Auto-demotion**: set `auto_demote: true` with `demote_on_drift` (`"warning"` or `"critical"`) and/or `demote_on_accuracy_below` (float threshold) — when criteria are crossed, the production model is automatically moved to inactive status by the supervisor (every 6h)
- `demote_cooldown_hours`: minimum delay between two automatic demotions

### Model Card
- `GET /models/{name}/{version}/card`: summary card for a model in a single call — metadata, real-world performance metrics, drift status, calibration, top-5 SHAP features, retrain info, and ground truth coverage
- Accepts `Accept: text/markdown` to return a shareable `.md` file

### Regression tests (Golden Tests)
- Test case CRUD per model: `POST /models/{name}/golden-tests`, `GET /models/{name}/golden-tests`, `DELETE /models/{name}/golden-tests/{id}`
- Bulk import from CSV: `POST /models/{name}/golden-tests/upload-csv`
- `POST /models/{name}/{version}/run-golden-tests`: run all test cases on a version — returns PASS/FAIL per case with expected/received diff
- Integrated into the auto-promotion policy via `min_golden_test_pass_rate`
- Full interface in **Streamlit dashboard page 9**

### Anomaly detection
- `GET /predictions/anomalies`: recent predictions where at least one feature has an abnormal z-score relative to the baseline (configurable `z_threshold`, default 3.0)
- Parameters: `model_name`, `days` (default 7), `z_threshold`, `limit`

### Leaderboard & Multi-model comparison
- `GET /models/leaderboard`: ranking of production models by metric (`accuracy`, `f1_score`, `latency_p95_ms`, `predictions_count`) over a configurable window, result cached (TTL)
- Displayed in the dashboard Stats page: Leaderboard tab + accuracy vs latency P95 scatter plot

### Probability calibration
- `GET /models/{name}/calibration`: measures whether predicted probabilities are reliable (Brier score, reliability curve)
- A model with `brier_score < 0.1` is well-calibrated; a positive overconfidence gap signals the model overestimates its certainty

### Confidence distribution
- `GET /models/{name}/confidence-distribution`: histogram of confidence level (`max(probabilities)`) on recent predictions
- Helps identify the proportion of uncertain predictions before adjusting the `confidence_threshold`

### Consolidated performance report
- `GET /models/{name}/performance-report`: aggregates in a single call real-world performance + drift + feature importance + calibration + A/B
- Ideal for automated monitoring scripts, programmatic alerts, or Grafana integrations

### Deprecation & Full lifecycle
- `PATCH /models/{name}/{version}/deprecate`: marks a version as deprecated — new predictions return HTTP 410 Gone
- `GET /models/{name}/readiness`: verifies a model meets all prerequisites before production (MinIO file accessible, baseline computed, no critical drift)
- `GET /models/{name}/retrain-history`: structured log of all retrains (source version → new version, accuracy, auto_promoted, trained_by, training window)

### History & Rollback
- `GET /models/{name}/history`: log of all changes
- `POST /models/{name}/{version}/rollback/{history_id}`: restore a previous state

### Supervision & Alerts
- `GET /monitoring/overview`: global dashboard (all errors, drifts, performance)
- `GET /monitoring/model/{name}`: per-model detail (timeseries, A/B, drift, recent errors)
- `GET /predictions/stats`: aggregated statistics (volume, error rate, p50/p95 response time)
- Configurable email alerts (drift, error rate)
- Automatic weekly reports
- `GET /metrics` — Prometheus endpoint (automatically scraped by Grafana LGTM, optional auth via `METRICS_TOKEN`)
- OpenTelemetry traces to Grafana LGTM (optional)

### User management
- Created by an admin with role and daily quota
- Unique Bearer token per user, valid for `TOKEN_LIFETIME_DAYS` days (default 90)
- Token renewal via `PATCH /users/{id}` with `{"regenerate_token": true}`
- Automatic rate limiting (HTTP 429 if daily quota exceeded)

### Security
- **HMAC-SHA256** — each `.joblib` is signed on upload and verified before `joblib.load()` (key: `SECRET_KEY`)
- **Audit logging** — sensitive admin operations (model creation/deletion, retrain, user management) logged as JSON via `structlog` with `user_id`, IP, and action
- **Token expiration** — Bearer tokens expire after `TOKEN_LIFETIME_DAYS` days (HTTP 401 beyond); renewal possible via `PATCH /users/{id}`
- **Per-IP rate limiting** — limit per IP and per minute via `slowapi` with Redis backend shared across replicas (HTTP 429 if exceeded), in addition to the per-user daily quota
- **Name validation** — `name` and `version` formats are validated (path traversal prevention)
- **`/health/dependencies`** — protected by admin auth (internal dependency health, not publicly exposed)
- **Required variables** — `SECRET_KEY`, `REDIS_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `GRAFANA_ADMIN_PASSWORD` — the API or docker-compose refuses to start if absent

---

## API Endpoints

| Method | Route | Auth | Description |
|---|---|---|---|
| GET | `/` | No | API status and available models |
| GET | `/health` | No | Health check (DB + Redis cache) |
| **Models** | | | |
| GET | `/models` | No | List of active models (filter by tag) |
| GET | `/models/cached` | No | Models loaded in memory |
| GET | `/models/{name}/{version}` | No | Full model details |
| POST | `/models` | Yes | Upload a model (.joblib or MLflow) |
| PATCH | `/models/{name}/{version}` | Yes | Update (production, A/B, tags, webhook…) |
| DELETE | `/models/{name}/{version}` | Yes | Delete a version |
| DELETE | `/models/{name}` | Yes | Delete all versions |
| GET | `/models/{name}/performance` | Yes | Real-world metrics via observed results |
| GET | `/models/{name}/drift` | Yes | Feature drift report |
| GET | `/models/{name}/feature-importance` | Yes | Aggregated global SHAP importance |
| GET | `/models/{name}/history` | Yes | Full change history |
| GET | `/models/{name}/{version}/history` | Yes | History of a specific version |
| POST | `/models/{name}/{version}/rollback/{history_id}` | Admin | Rollback to a previous state |
| POST | `/models/{name}/{version}/retrain` | Admin | Retrain with train.py |
| PATCH | `/models/{name}/{version}/schedule` | Admin | Configure the retraining cron schedule |
| PATCH | `/models/{name}/policy` | Admin | Define the post-retrain auto-promotion policy |
| GET | `/models/leaderboard` | No | Ranking of production models by metric |
| GET | `/models/{name}/performance-timeline` | Yes | Performance timeline by deployed version |
| GET | `/models/{name}/calibration` | Yes | Probability calibration (Brier score, reliability diagram) |
| GET | `/models/{name}/confidence-distribution` | Yes | Confidence distribution (histogram by bins) |
| GET | `/models/{name}/performance-report` | Yes | Consolidated report: performance + drift + SHAP + calibration |
| GET | `/models/{name}/readiness` | Yes | Readiness check before going to production |
| GET | `/models/{name}/retrain-history` | Yes | Retraining event history |
| PATCH | `/models/{name}/{version}/deprecate` | Admin | Deprecate a version (blocks predictions with HTTP 410) |
| POST | `/models/{name}/{version}/validate-input` | Yes | Validate feature schema without consuming quota |
| GET | `/models/{name}/{version}/download` | Yes | Download the .joblib file from MinIO |
| GET | `/models/{name}/ab-compare` | Yes | A/B comparison report with statistical significance |
| GET | `/models/{name}/shadow-compare` | Yes | Enriched shadow vs production report (accuracy, latency, disagreement) |
| GET | `/models/{name}/output-drift` | Yes | Output distribution drift (label shift via PSI) |
| GET | `/models/{name}/{version}/card` | Yes | Consolidated model card (JSON or Markdown) |
| GET | `/models/{name}/confidence-trend` | Yes | Confidence trend over time |
| POST | `/models/{name}/{version}/warmup` | Yes | Warm up the model in the Redis cache |
| GET | `/models/{name}/golden-tests` | Yes | List of golden test cases |
| POST | `/models/{name}/golden-tests` | Yes | Create a golden test case |
| DELETE | `/models/{name}/golden-tests/{id}` | Admin | Delete a golden test case |
| POST | `/models/{name}/golden-tests/upload-csv` | Admin | Bulk import of test cases from CSV |
| POST | `/models/{name}/{version}/run-golden-tests` | Yes | Run golden tests on a version |
| **Predictions** | | | |
| POST | `/predict` | Yes | Single prediction (`?explain=true` for inline SHAP) |
| POST | `/predict-batch` | Yes | Batch predictions |
| POST | `/explain` | Yes | Local SHAP explainability |
| GET | `/predictions` | Yes | Prediction history (cursor pagination) |
| GET | `/predictions/{id}` | Yes | Get a prediction by ID |
| GET | `/predictions/{id}/explain` | Yes | Post-hoc SHAP explanation of an existing prediction |
| GET | `/predictions/stats` | Yes | Aggregated statistics per model |
| GET | `/predictions/anomalies` | Yes | Predictions with anomalous features (z-score vs baseline) |
| GET | `/predictions/export` | Admin | Streaming export CSV / JSONL / Parquet |
| DELETE | `/predictions/purge` | Admin | GDPR purge of old predictions (`dry_run` by default) |
| **Observed results** | | | |
| POST | `/observed-results` | Yes | Record real-world results |
| GET | `/observed-results` | Yes | View observed results |
| GET | `/observed-results/export` | Yes | Export observed results (CSV/JSON) |
| GET | `/observed-results/stats` | Yes | Ground truth coverage statistics |
| POST | `/observed-results/upload-csv` | Yes | Bulk import from a CSV file |
| **Users** | | | |
| GET | `/users/me` | Yes | Current user profile |
| GET | `/users/me/quota` | Yes | Daily quota (used, remaining, reset time) |
| GET | `/users/{id}/usage` | Yes | Usage statistics per model and per day |
| POST | `/users` | Admin | Create a user |
| GET | `/users` | Admin | List all users |
| GET | `/users/{id}` | Yes | User details |
| PATCH | `/users/{id}` | Admin | Modify role, status, quota, token |
| DELETE | `/users/{id}` | Admin | Delete a user |
| **Monitoring** | | | |
| GET | `/monitoring/overview` | Yes | Global dashboard |
| GET | `/monitoring/model/{name}` | Yes | Model monitoring detail |
| GET | `/health/dependencies` | Admin | Detailed health of each dependency (DB, Redis, MinIO, MLflow) |
| GET | `/metrics` | Optional | Prometheus metrics (scraped by Grafana LGTM) |

---

## Minimal example

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "<ADMIN_TOKEN>"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-001",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
print(response.json())
# {"id": 42, "model_name": "iris_model", "model_version": "1.0", "prediction": 0,
#  "probability": [0.97, 0.02, 0.01], "low_confidence": false, "selected_version": null}

# SHAP explainability
explain = requests.post(
    f"{BASE_URL}/explain",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                     "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    }
)
print(explain.json()["shap_values"])
# {"petal length (cm)": -1.32, "petal width (cm)": -0.87, ...}
```

---

## Tests

```bash
# Automated tests (no Docker — uses FastAPI TestClient)
pytest tests/ -v

# Test a specific file
pytest tests/test_api.py -v

# Smoke tests (require Docker running)
python smoke-tests/test_multimodel_api.py
```

---

## Full documentation

| Document | Contents |
|---|---|
| [documentation/BEGINNER_GUIDE.md](documentation/BEGINNER_GUIDE.md) | Complete beginner guide — step-by-step tutorial with Python |
| [documentation/QUICKSTART.md](documentation/QUICKSTART.md) | Getting started guide and complete workflow |
| [documentation/API_REFERENCE.md](documentation/API_REFERENCE.md) | Complete reference for all endpoints, schemas, Python examples |
| [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) | Project structure, services, and data flows |
| [documentation/DOCKER.md](documentation/DOCKER.md) | Docker commands, services, environment variables |
| [documentation/DATABASE.md](documentation/DATABASE.md) | SQL schema, useful queries, Python connection |
| [documentation/FAQ.md](documentation/FAQ.md) | Frequently asked questions — common scenarios and troubleshooting |
| [documentation/DASHBOARD_GUIDE.md](documentation/DASHBOARD_GUIDE.md) | Streamlit admin dashboard guide — all pages |
| [documentation/TRAIN_SCRIPT_GUIDE.md](documentation/TRAIN_SCRIPT_GUIDE.md) | Writing `train.py` scripts — contract, templates, and security |
| [documentation/KPIS_REFERENCE.md](documentation/KPIS_REFERENCE.md) | KPI definitions and metric calculation methods |
| [documentation/GRAFANA_OTEL.md](documentation/GRAFANA_OTEL.md) | Grafana LGTM setup and OpenTelemetry integration |
| [documentation/SECURITY_MODEL_VALIDATION.md](documentation/SECURITY_MODEL_VALIDATION.md) | Model name/version format validation and security rules |

---

## Project structure

```
src/
├── api/                    # FastAPI endpoints
│   ├── models.py           # Model CRUD + drift + history + retrain + A/B
│   ├── predict.py          # Single, batch predictions, SHAP, stats
│   ├── users.py            # User management
│   ├── observed_results.py # Observed results
│   └── monitoring.py       # Global and per-model dashboard
├── core/                   # Config, auth, telemetry
│   ├── config.py
│   ├── security.py
│   └── telemetry.py
├── db/                     # SQLAlchemy ORM
│   ├── database.py
│   └── models/             # User, Prediction, ModelMetadata, ObservedResult, ModelHistory
├── services/               # Business logic
│   ├── db_service.py               # All DB queries
│   ├── model_service.py            # Loading, Redis cache, A/B/shadow routing
│   ├── minio_service.py            # MinIO upload/download
│   ├── drift_service.py            # Z-score + PSI + null rate drift computation (input + output)
│   ├── shap_service.py             # Local SHAP explanations
│   ├── ab_significance_service.py  # A/B statistical tests (Chi-², Mann-Whitney U)
│   ├── auto_promotion_service.py   # Auto-promotion + auto-demotion (circuit breaker)
│   ├── golden_test_service.py      # Golden regression tests (CRUD + run)
│   ├── input_validation_service.py # Input feature schema validation
│   ├── supervision_reporter.py     # 6h supervision: drift, alerts, reactive retrain
│   ├── email_service.py            # Email alerts & weekly reports
│   └── webhook_service.py          # HTTP webhooks post-prediction
├── schemas/                # Pydantic schemas (I/O validation)
└── main.py

streamlit_app/              # Multi-page admin dashboard (9 pages: Users, Models, Predictions, Stats, Code, A/B, Supervision, Retrain, Golden Tests)
tests/                      # Automated tests (pytest)
smoke-tests/                # Manual tests against live Docker
init_data/                  # One-shot initialization scripts
alembic/                    # Database migrations
```

---

## Code quality

```bash
# Lint
ruff check src/

# Formatting
black --check src/

# Auto-fix
ruff check src/ --fix && black src/
```
