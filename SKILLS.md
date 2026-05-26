# PredictML API — AI agent guide

This document explains how to interact with **PredictML API** as a user or AI agent. It covers the project objective, code structure, all available routes, model lifecycle, drift detection, and retraining.

---

## Project objective

PredictML API is an ML-as-a-Service platform that enables:

- **Deploy** scikit-learn models (`.joblib`) via a REST API
- **Serve predictions** in real time (single or batch)
- **Version** models with A/B traffic management and shadow deployment
- **Monitor** performance and detect feature drift
- **Automatically retrain** a model from a `train.py` script
- **Audit** all changes via a full history with rollback

Models are stored in **MinIO**, experiments tracked in **MLflow**, metadata in **PostgreSQL**, and loaded models are cached in **Redis**.

---

## Project structure

```
src/
├── api/
│   ├── models.py          # Model CRUD, retrain, drift, A/B, history
│   ├── predict.py         # Single, batch predictions, SHAP
│   ├── users.py           # User management (admin)
│   ├── observed_results.py# Observed results (ground truth)
│   └── monitoring.py      # Global and per-model monitoring dashboard
├── core/
│   ├── config.py          # Environment variables and settings
│   └── security.py        # Token verification, rate limiting
├── db/
│   ├── models.py          # SQLAlchemy ORM models
│   └── database.py        # Async PostgreSQL session
├── services/
│   ├── model_service.py   # Loading, Redis cache, A/B routing
│   ├── db_service.py      # Data access (predictions, users, models)
│   └── minio_service.py   # MinIO artifact upload/download
├── schemas/               # Pydantic schemas (requests / responses)
└── main.py                # FastAPI app, router mounting

streamlit_app/             # Streamlit admin dashboard (port 8501)
tests/                     # pytest tests (no Docker)
smoke-tests/               # Manual tests against the live API
init_data/                 # Initialization scripts and example_train.py
alembic/                   # Database migrations
```

---

## Authentication

All protected routes require a **Bearer token** in the header:

```
Authorization: Bearer <api_token>
```

Tokens are generated at user creation (and upon renewal). They are validated in the database on each request.

### User roles

| Role | Access |
|------|-------|
| `admin` | Full access (user management, models, retrain) |
| `user` | Predictions, model reading and history |
| `readonly` | Read-only |

### Rate limiting

The `/predict` and `/predict-batch` routes are subject to a daily quota (`rate_limit_per_day`, default 1000). A request exceeding the quota returns `429 Too Many Requests`.

---

## Model lifecycle

### 1. Create a model (first version)

```bash
POST /models
Content-Type: multipart/form-data
Authorization: Bearer <token>

Required fields:
  name        (string)   : Unique model name
  version     (string)   : Version (e.g. "1.0.0")
  file        (file)     : .joblib file (OR mlflow_run_id)

Optional fields:
  description, algorithm, accuracy, f1_score
  features_count, classes (JSON array)
  training_params (JSON object)
  training_dataset (string)
  feature_baseline (JSON)   : baseline profile for drift
  tags (JSON array)         : e.g. ["production", "finance"]
  webhook_url               : POST URL called after each prediction
  train_file (file)         : train.py script for retraining
```

**curl example:**
```bash
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer <token>" \
  -F "name=churn_model" \
  -F "version=1.0.0" \
  -F "file=@models/churn_v1.joblib" \
  -F "algorithm=RandomForest" \
  -F "accuracy=0.92" \
  -F "classes=[0,1]" \
  -F "train_file=@init_data/train_churn.py"
```

### 2. Set a version to production

```bash
PATCH /models/{name}/{version}
Content-Type: application/json
Authorization: Bearer <token>

{
  "is_production": true
}
```

> Setting `is_production: true` on a version **automatically demotes** all other versions of the same model.

### 3. Manually create a new version

Same call as creation, with a new `version` (e.g. `"1.1.0"`). The new version is not in production by default.

```bash
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer <token>" \
  -F "name=churn_model" \
  -F "version=1.1.0" \
  -F "file=@models/churn_v2.joblib" \
  -F "accuracy=0.94"
```

### 4. Update metadata

```bash
PATCH /models/{name}/{version}
Content-Type: application/json
Authorization: Bearer <token>

Available fields (all optional):
{
  "description": "string",
  "is_production": true/false,
  "accuracy": 0.94,
  "features_count": 12,
  "classes": [0, 1],
  "confidence_threshold": 0.75,
  "feature_baseline": {"feature_name": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0}},
  "tags": ["v2", "retrained"],
  "webhook_url": "https://...",
  "traffic_weight": 0.3,
  "deployment_mode": "production|ab_test|shadow"
}
```

### 5. Delete a version

```bash
DELETE /models/{name}/{version}
Authorization: Bearer <token>
```

### 6. Delete all versions of a model

```bash
DELETE /models/{name}
Authorization: Bearer <token>
```

---

## Predictions

### Single prediction

```bash
POST /predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_name": "churn_model",
  "model_version": "1.0.0",   # optional — uses production version if omitted
  "id_obs": "client_42",       # optional — observation identifier
  "features": {
    "age": 35,
    "tenure_months": 24,
    "monthly_charges": 79.5
  }
}
```

**Response:**
```json
{
  "model_name": "churn_model",
  "model_version": "1.0.0",
  "id_obs": "client_42",
  "prediction": 1,
  "probability": [0.12, 0.88],
  "low_confidence": false,
  "selected_version": null
}
```

> If `model_version` is omitted, A/B/Shadow routing applies automatically. `selected_version` indicates the chosen version.

### Batch prediction

```bash
POST /predict-batch
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_name": "churn_model",
  "model_version": null,
  "inputs": [
    {"id_obs": "c1", "features": {"age": 35, "tenure_months": 24, "monthly_charges": 79.5}},
    {"id_obs": "c2", "features": {"age": 58, "tenure_months": 6,  "monthly_charges": 120.0}}
  ]
}
```

> The batch consumes as many API calls as there are elements in `inputs` against the daily quota.

### Prediction history

```bash
GET /predictions?name=churn_model&start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
  &version=1.0.0    # optional
  &user=alice       # optional
  &id_obs=client_42 # optional
  &limit=100        # 1-1000, default 100
  &cursor=450       # cursor pagination (id of last seen prediction)
Authorization: Bearer <token>
```

### SHAP explainability

```bash
POST /explain
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_name": "churn_model",
  "model_version": "1.0.0",
  "features": {"age": 35, "tenure_months": 24, "monthly_charges": 79.5}
}
```

**Response:** `shap_values` per feature + `base_value` (E[f(X)]). Works with tree models (RandomForest, GradientBoosting…) and linear models (LogisticRegression, Ridge…).

---

## Observed results (ground truth)

Send real labels to calculate actual performance:

```bash
POST /observed-results
Content-Type: application/json
Authorization: Bearer <token>

{
  "data": [
    {
      "id_obs": "client_42",
      "model_name": "churn_model",
      "date_time": "2025-03-15T10:00:00",
      "observed_result": 1
    }
  ]
}
```

> The call is an **upsert**: if `id_obs` + `model_name` already exists, the value is updated.

**Retrieve results:**
```bash
GET /observed-results?model_name=churn_model&start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
  &id_obs=client_42  # optional
  &limit=100&offset=0
```

---

## Drift and monitoring

### Prerequisite: provide a `feature_baseline`

For drift detection to work, the model must have a baseline profile. It can be provided at creation or via PATCH:

```json
"feature_baseline": {
  "age":              {"mean": 42.3, "std": 12.1, "min": 18.0, "max": 90.0},
  "tenure_months":    {"mean": 18.5, "std": 9.4,  "min": 0.0,  "max": 72.0},
  "monthly_charges":  {"mean": 65.2, "std": 20.8, "min": 20.0, "max": 200.0}
}
```

### Per-model drift report

```bash
GET /models/{name}/drift?version=1.0.0&days=7&min_predictions=30
```

**Response per feature:**
- `z_score`: |mean_prod - mean_baseline| / std_baseline
- `psi`: Population Stability Index (binning on prod data vs baseline)
- `drift_status`: `ok` | `warning` | `critical` | `insufficient_data` | `no_baseline`

**Indicative thresholds:**
- Z-score > 2 → warning; > 3 → critical
- PSI > 0.1 → warning; > 0.2 → critical

### Real-world performance (after observed_results)

```bash
GET /models/{name}/performance?start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
  &version=1.0.0    # optional
  &granularity=day  # optional: day | week | month
```

Returns accuracy/F1 (classification) or MAE/RMSE/R² (regression) calculated on prediction/observed result pairs.

### Global monitoring

```bash
GET /monitoring/overview?start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
```

Returns a global dashboard: total predictions, error rate, p95 latency, drift and health status per model.

### Detailed per-model monitoring

```bash
GET /monitoring/model/{name}?start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
```

Returns: stats per version, daily timeseries, performance per day, feature drift, A/B comparison, latest errors.

---

## Retraining (retrain on drift)

### Complete flow

1. Detect drift via `GET /models/{name}/drift` or observe performance degradation via `GET /models/{name}/performance`
2. Trigger retraining via `POST /models/{name}/{version}/retrain`
3. The API downloads the `train.py` stored in MinIO, runs it in an isolated subprocess (timeout 600s), and creates a new version
4. Optionally, promote the new version to production

### Trigger a retrain

```bash
POST /models/{name}/{version}/retrain
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "start_date": "2025-01-01",
  "end_date":   "2025-12-31",
  "new_version": "1.2.0",      # optional — auto-generated if null
  "set_production": false       # automatically promote to production
}
```

**Response:**
```json
{
  "model_name": "churn_model",
  "source_version": "1.0.0",
  "new_version": "1.2.0",
  "success": true,
  "stdout": "...",
  "stderr": "",
  "error": null,
  "new_model_metadata": { ... }
}
```

### `train.py` script contract

The script provided at upload must respect the following constraints (verified statically at upload time):

| Constraint | Detail |
|---|---|
| Valid Python syntax | Verified via `ast.parse()` |
| Read `TRAIN_START_DATE` | `os.environ["TRAIN_START_DATE"]` (format YYYY-MM-DD) |
| Read `TRAIN_END_DATE` | `os.environ["TRAIN_END_DATE"]` |
| Read `OUTPUT_MODEL_PATH` | Path where to write the `.joblib` |
| Save the model | `joblib.dump` or `save_model` |
| Return metrics | Last JSON line on stdout: `{"accuracy": 0.94, "f1_score": 0.93}` |

**Automatically injected environment variables:**

| Variable | Description |
|---|---|
| `TRAIN_START_DATE` | Start date (YYYY-MM-DD) |
| `TRAIN_END_DATE` | End date (YYYY-MM-DD) |
| `OUTPUT_MODEL_PATH` | Absolute path of the `.joblib` to produce |
| `MLFLOW_TRACKING_URI` | MLflow URI (optional) |
| `MODEL_NAME` | Source model name |

**Minimal `train.py` skeleton:**
```python
import os, joblib, json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_START_DATE  = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE    = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

# Load date-filtered data
df = pd.read_csv("data.csv")
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

accuracy = model.score(X, y)

joblib.dump(model, OUTPUT_MODEL_PATH)

# Return metrics — must be the last JSON line on stdout
print(json.dumps({"accuracy": accuracy, "f1_score": accuracy}))
```

---

## A/B and Shadow deployment

### A/B testing

Configure two versions with `deployment_mode: "ab_test"` and a `traffic_weight` (0.0–1.0, sum ≤ 1.0):

```bash
PATCH /models/churn_model/1.0.0
{ "deployment_mode": "ab_test", "traffic_weight": 0.7 }

PATCH /models/churn_model/1.1.0
{ "deployment_mode": "ab_test", "traffic_weight": 0.3 }
```

On a `/predict` call without `model_version`, routing is weighted random. `selected_version` in the response indicates the chosen version.

### Shadow deployment

Test a new version in parallel without exposing it:

```bash
PATCH /models/churn_model/1.1.0
{ "deployment_mode": "shadow" }
```

Shadow predictions are recorded (`is_shadow: true`) but not returned to the client. Comparison is done via `GET /models/{name}/ab-compare`.

### A/B comparison

```bash
GET /models/{name}/ab-compare?days=30
```

Returns for each version: prediction count, error rate, p50/p95 latencies, prediction distribution, shadow/production agreement rate.

---

## History and rollback

### View model history

```bash
GET /models/{name}/history?limit=50&offset=0
GET /models/{name}/{version}/history?limit=50&offset=0
```

Each entry contains: `action` (CREATED, UPDATED, SET_PRODUCTION, ROLLBACK, DELETED), `snapshot` (full metadata state), `changed_fields`, `timestamp`, `changed_by_username`.

### Rollback to a previous state

```bash
POST /models/{name}/{version}/rollback/{history_id}
Authorization: Bearer <admin_token>
```

Restores model metadata to the state captured in `history_id`. A new ROLLBACK record is created in the history.

---

## User management (admin)

### Create a user

```bash
POST /users
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "username": "alice",
  "email": "alice@example.com",
  "role": "user",        # user | admin | readonly
  "rate_limit": 500      # daily quota, 1-100000
}
```

The `api_token` is returned **only** at creation.

### List, view, modify, delete

```bash
GET    /users                   # all users (admin)
GET    /users/{id}              # a user (admin or self)
PATCH  /users/{id}              # modify role, status, quota, renew token
DELETE /users/{id}              # delete (not possible on own account)
```

### Renew a token

```bash
PATCH /users/{id}
{ "regenerate_token": true }
```

The new token is returned in the response.

---

## Service routes

```bash
GET /         # API information and available models (no auth)
GET /health   # DB and Redis cache status (no auth)
GET /models/cached  # Models currently in Redis cache
```

---

## Prediction statistics

```bash
GET /predictions/stats?model_name=churn_model&days=30
```

Returns per model: total predictions, error rate, average p50/p95 latencies.

---

## Tips for an AI agent

1. **Always provide a `feature_baseline`** at model creation to enable drift detection.
2. **Provide a `train_file`** at creation if automatic retraining is planned.
3. **Use `id_obs`** in each prediction to be able to link observed results to predictions.
4. **Send `observed_results`** regularly so that real-world performance metrics can be calculated.
5. **Monitor `drift_status`** via `GET /models/{name}/drift` before deciding on a retrain.
6. **Do not set `set_production: true`** on a retrain without having verified the metrics returned in the response.
7. **Use `model_version: null`** in `/predict` to benefit from automatic A/B or shadow routing.
8. **The admin token** for this project is `<ADMIN_TOKEN>` (never expose publicly).
