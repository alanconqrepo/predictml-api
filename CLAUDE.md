# predictml-api

FastAPI ML prediction API with PostgreSQL, MinIO and MLflow. Streamlit admin dashboard.

## Stack

- **API**: FastAPI async — port 8000
- **Dashboard**: Streamlit Admin — port 8501
- **DB**: PostgreSQL 16 — port 5433
- **Model storage**: MinIO — port 9000 / console 9001
- **Experiment tracking**: MLflow — port 5000

## Structure

```
src/
├── api/            # Endpoints (models.py, predict.py, users.py, observed_results.py)
├── core/           # Config & auth (config.py, security.py)
├── db/             # SQLAlchemy ORM + DB service
├── services/       # Business logic (db_service, model_service, minio_service)
├── schemas/        # Pydantic
└── main.py

streamlit_app/      # Multi-page admin dashboard
├── app.py          # Login + home
├── utils/          # api_client.py, auth.py
└── pages/          # 1_Users, 2_Models, 3_Predictions, 4_Stats, 5_Code_Example

tests/              # Pytest — automated tests
smoke-tests/        # Manual tests against live Docker
init_data/          # One-shot scripts (create_multiple_models, init_db)
Models/             # Local .joblib files
notebooks/          # Jupyter
alembic/            # DB migrations
```

## Code quality

Coding rules are defined in **[CODING_STANDARDS.md](./CODING_STANDARDS.md)**.

```bash
# Check lint
ruff check src/

# Check formatting
black --check src/

# Auto-fix
ruff check src/ --fix && black src/
```

## Key commands

```bash
# Start
docker-compose up -d

# Initialize (first deployment only)
docker exec predictml-api python init_data/init_db.py

# Automated tests (see Tests section for prerequisites)
pytest tests/ -v

# Smoke tests (Docker required)
python smoke-tests/test_multimodel_api.py

# Logs
docker-compose logs -f api
docker-compose logs -f streamlit

# PostgreSQL
docker exec -it predictml-postgres psql -U postgres -d sklearn_api
```

## Credentials

| Service | Value |
|---|---|
| Admin token | `<ADMIN_TOKEN>` |
| DB | `postgres / postgres` |
| MinIO | `minioadmin / minioadmin` |

## Dependencies

Any new dependency must be added in **both files**:
- `requirements.txt` — used by the API Dockerfile
- `pyproject.toml` — used by the CI GitHub Actions

For the Streamlit dashboard, dependencies are in `streamlit_app/requirements.txt`.

## Tests

Tests in `tests/` use FastAPI's `TestClient` — no Docker required.
They cover: auth, public endpoints, `ModelService` logic.

### Prerequisites

`pytest` is installed via **uv tool** (not in the project venv):

```bash
# Initial install (once only)
WITH_ARGS=$(cat requirements.txt | grep -v '^#' | grep -v '^$' | sed 's/^/--with /' | tr '\n' ' ')
uv tool install pytest --with fakeredis --with aiosqlite --with asyncpg $WITH_ARGS
```

> The system `pytest` (`/root/.local/bin/pytest`) is separate from the project Python.
> Do not use `python -m pytest` — the module is not installed in this Python.

### Run tests

```bash
pytest tests/ -v                              # all tests
pytest tests/test_api.py                      # basic endpoints only
pytest tests/test_predictions_purge.py -v     # GDPR purge only
pytest tests/test_ab_significance.py -v       # A/B significance
pytest tests/test_auto_promotion_policy.py -v # post-retrain auto-promotion

# Filter by keyword
pytest tests/ -k "purge" -v
pytest tests/ -k "admin" -v

# Stop at first failure
pytest tests/ -x -v

# Without warnings
pytest tests/ -v -p no:warnings
```

### Test files by feature

| File | Feature |
|---|---|
| `test_api.py` | Basic endpoints, auth, health |
| `test_predict_post.py` | POST /predict |
| `test_predictions_get.py` | GET /predictions |
| `test_predictions_purge.py` | DELETE /predictions/purge |
| `test_export_endpoint.py` | GET /predictions/export |
| `test_prediction_stats.py` | GET /predictions/stats |
| `test_models_create.py` | POST /models |
| `test_models_get.py` | GET /models |
| `test_models_update.py` | PATCH /models |
| `test_models_delete.py` | DELETE /models |
| `test_retrain.py` | POST /models/{name}/{version}/retrain |
| `test_scheduled_retraining.py` | PATCH /models/{name}/{version}/schedule |
| `test_auto_promotion_policy.py` | PATCH /models/{name}/policy |
| `test_ab_shadow.py` | A/B and shadow routing |
| `test_ab_significance.py` | GET /models/{name}/ab-compare |
| `test_feature_importance.py` | GET /models/{name}/feature-importance |
| `test_observed_results.py` | POST/GET /observed-results |
| `test_users.py` | User management |
| `test_security.py` | Auth and tokens |
| `test_rate_limit.py` | Rate limiting |
| `test_drift.py` | Drift detection |
| `test_db_service_crud.py` | DBService (CRUD) |
| `test_monitoring_api.py` | Monitoring endpoints |
| `test_input_validation.py` | Input schema validation + strict mode /predict |

### Notes

- The test DB is in-memory SQLite — tables are recreated at each pytest session.
- MinIO and Redis are mocked (no servers required).
- Some `async def` tests require `pytest-asyncio` (not installed by default) — they fail with "async def functions are not natively supported"; these are pre-existing failures with no impact on functionality.

Smoke tests in `smoke-tests/` require Docker and hit the live API.

## Main endpoints

- `POST /predict` — Prediction (Bearer auth); `?strict_validation=true` for strict schema validation
- `GET /predictions` — History (Bearer auth)
- `GET/POST/PATCH/DELETE /models` — Model management
- `GET/POST/PATCH/DELETE /users` — User management (admin)
- `POST/GET /observed-results` — Observed results
- `PATCH /users/{id}` with `{"regenerate_token": true}` — Renew a token (admin)
- `POST /models/{name}/{version}/retrain` — Retrain a model (admin)
- `PATCH /models/{name}/{version}/schedule` — Configure the retraining cron schedule (admin)
- `PATCH /models/{name}/policy` — Define the post-retrain auto-promotion policy (admin)
- `POST /models/{name}/{version}/validate-input` — Validate input schema without predicting (Bearer auth)
- `GET /models/{name}/feature-importance` — Global feature importance (aggregated SHAP, Bearer auth)
- `GET /models/{name}/ab-compare` — A/B comparison with statistical significance test (Bearer auth)
- `DELETE /predictions/purge` — GDPR purge of old predictions (admin)

## Feature: Input schema validation

### Why

The most frequent silent failures in ML production come from pipelines sending missing, renamed, or wrongly-typed features. The `/predict` endpoint accepted any JSON — inconsistencies caused unexplained 500 errors or, worse, silently wrong predictions.

### Signature

```
POST /models/{name}/{version}/validate-input
Body: { "petal_length": 5.1, "petal_width": 1.8, "sepal_length": 6.3 }
```

### Response

```json
{
  "valid": false,
  "errors": [
    { "type": "missing_feature",    "feature": "sepal_width" },
    { "type": "unexpected_feature", "feature": "petal_width_squared" }
  ],
  "warnings": [
    { "type": "type_coercion", "feature": "petal_length", "from_type": "string", "to_type": "float" }
  ],
  "expected_features": ["petal_length", "petal_width", "sepal_length", "sepal_width"]
}
```

### Source of expected features (priority)

1. `feature_names_in_` from the loaded sklearn model (trained with a pandas DataFrame)
2. Keys of `feature_baseline` stored in model metadata
3. If no schema: `valid=true`, `expected_features=null`

### Strict mode on /predict

Add `?strict_validation=true` to `/predict` to reject with 422 if **unexpected** features are present (in addition to missing features already checked by default).

```bash
curl -X POST "http://localhost:8000/predict?strict_validation=true" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "iris", "features": {"sepal_length": 5.1, "extra_col": 99}}'
# → 422 with detail.errors listing the unexpected features
```

### Implementation

- Endpoint: `POST /models/{name}/{version}/validate-input` in `src/api/models.py`
- Service: `src/services/input_validation_service.py` (`validate_input_features`, `resolve_expected_features`)
- Schemas: `ValidateInputResponse`, `InputValidationError`, `InputValidationWarning` in `src/schemas/model.py`
- Strict param: `?strict_validation=true` on `POST /predict` in `src/api/predict.py`
- Tests: `tests/test_input_validation.py` (23 tests)

### Test files table

| File | Feature |
|---|---|
| `test_input_validation.py` | Input schema validation + strict mode /predict |

## Feature: A/B Statistical Significance

The `GET /models/{name}/ab-compare` endpoint enriches its response with an `ab_significance` block:

```json
{
  "ab_significance": {
    "metric": "error_rate",
    "test": "chi2",
    "p_value": 0.023,
    "significant": true,
    "confidence_level": 0.95,
    "winner": "v2.0.0",
    "min_samples_needed": 200,
    "current_samples": { "v1.0.0": 450, "v2.0.0": 380 }
  }
}
```

### Test selection logic

| Condition | Test used | Metric |
|---|---|---|
| ≥ 1 observed error in either group | Chi-² on contingency table | `error_rate` |
| 0 errors + response times available | Mann-Whitney U | `response_time_ms` |
| Insufficient data (< 2 active versions) | — | `ab_significance: null` |

### `min_samples_needed` calculation

- **Chi-²**: power formula based on Cohen h effect size (proportion comparison)
- **Mann-Whitney U**: formula based on Cohen d (continuous distributions)
- Target power: 80% — threshold: `confidence_level` (default 95%)

### Implementation

- Service: `src/services/ab_significance_service.py`
- Unit tests: `tests/test_ab_significance.py` (20 tests)

## Feature: Retrain (retraining)

### How it works

1. When uploading a model (`POST /models`), optionally provide a `train_file` script
   (Python file `train.py`).
2. If provided, the script is **statically validated** then stored in MinIO
   (`{name}/v{version}_train.py`).
3. The admin can trigger a retrain via `POST /models/{name}/{version}/retrain`
   by specifying a date range.
4. The script runs in an **isolated subprocess** (timeout 600s) with environment
   variables automatically injected.
5. The produced `.joblib` is uploaded to MinIO and registered as a **new version** of the model.
6. If `set_production: true`, the new version is automatically set to production.
7. The full `stdout`/`stderr` logs are returned in the response and displayed in
   the Streamlit dashboard.

### `train.py` script constraints (checked at upload)

The script must:

| Constraint | Detail |
|---|---|
| Valid Python syntax | Verified via `ast.parse()` |
| Reference `TRAIN_START_DATE` | Read `os.environ["TRAIN_START_DATE"]` |
| Reference `TRAIN_END_DATE` | Read `os.environ["TRAIN_END_DATE"]` |
| Reference `OUTPUT_MODEL_PATH` | Path where to save the `.joblib` |
| Save the model | Call `joblib.dump` or `save_model` |

### Environment variables injected by the API

| Variable | Description |
|---|---|
| `TRAIN_START_DATE` | Start date (YYYY-MM-DD) |
| `TRAIN_END_DATE` | End date (YYYY-MM-DD) |
| `OUTPUT_MODEL_PATH` | Absolute path for the produced `.joblib` |
| `MLFLOW_TRACKING_URI` | MLflow URI (optional) |
| `MODEL_NAME` | Source model name (optional) |

### Returning metrics

For the API to update `accuracy` and `f1_score` of the new version,
print on **stdout** a JSON on the **last JSON line** of the output:

```json
{"accuracy": 0.95, "f1_score": 0.94}
```

The following optional keys enrich the `training_stats` field of the new version
(automatic snapshot of training data):

```python
print(json.dumps({
    "accuracy": 0.95,
    "f1_score": 0.94,
    "n_rows": 12450,
    "feature_stats": {"sepal_length": {"mean": 5.8, "std": 0.83}},
    "label_distribution": {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
}))
```

`train_start_date`, `train_end_date` and `trained_at` are always automatically populated.
`n_rows`, `feature_stats` and `label_distribution` are `null` if absent from the stdout JSON.

### Upload with train.py

```bash
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer <token>" \
  -F "name=my_model" -F "version=1.0.0" \
  -F "file=@my_model.joblib" \
  -F "train_file=@init_data/example_train.py"
```

### Trigger a retrain

```bash
curl -X POST http://localhost:8000/models/my_model/1.0.0/retrain \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-01-01",
    "end_date":   "2025-12-31",
    "new_version": "1.1.0",
    "set_production": false
  }'
```

A complete `train.py` example respecting the contract is available in
`init_data/example_train.py`.

See also `documentation/TRAIN_SCRIPT_GUIDE.md` — section "Security & sandbox"
for details on the import whitelist and resource limits.

## Feature: Post-retrain auto-promotion

### How it works

1. The admin defines a policy via `PATCH /models/{name}/policy`.
2. The policy is stored in `ModelMetadata.promotion_policy` (JSON field)
   and propagated to **all active versions** of the model.
3. At the end of each retrain (`POST /models/{name}/{version}/retrain`),
   if `auto_promote: true`:
   - Retrieves historical (prediction, observed result) pairs for the model.
   - If `len(pairs) < min_sample_validation` → not promoted.
   - If `min_accuracy` defined: verifies accuracy on the last N pairs.
   - If `max_latency_p95_ms` defined: verifies P95 latency of predictions.
   - If all criteria are met → `is_production = true` automatically.
4. The retrain response includes `auto_promoted: true|false` and `auto_promote_reason`.

### Policy fields

| Field | Type | Default | Description |
|---|---|---|---|
| `min_accuracy` | float [0–1] | null | Minimum required accuracy |
| `max_latency_p95_ms` | float > 0 | null | Maximum P95 latency in ms |
| `min_sample_validation` | int ≥ 1 | 10 | Minimum number of validation pairs |
| `auto_promote` | bool | false | Enable auto-promotion |

### Semantics of `auto_promoted` in the retrain response

| Value | Meaning |
|---|---|
| `null` | No policy configured, or `set_production=True` (manual promotion) |
| `false` | Policy evaluated: criteria not met (see `auto_promote_reason`) |
| `true` | Policy evaluated: criteria met, version promoted to production |

### Define a policy

```bash
curl -X PATCH http://localhost:8000/models/my_model/policy \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "min_accuracy": 0.90,
    "max_latency_p95_ms": 200,
    "min_sample_validation": 50,
    "auto_promote": true
  }'
```

### Implementation

- Endpoint: `PATCH /models/{name}/policy` in `src/api/models.py`
- Evaluation service: `src/services/auto_promotion_service.py`
- DB field: `promotion_policy` (JSON) in `src/db/models/model_metadata.py`
- Migration: `alembic/versions/20260419_5ab8c1f0_add_promotion_policy.py`
- Tests: `tests/test_auto_promotion_policy.py` (25 tests)

## Feature: GDPR Purge (data retention)

### Why

The `predictions` table grows indefinitely. On an active deployment (1,000 predictions/day),
it reaches 365,000 rows/year. Without a retention policy, analytical query performance
degrades and GDPR compliance becomes an issue.

### Signature

```
DELETE /predictions/purge
  ?older_than_days=90    # delete predictions > 90 days old
  &model_name=iris       # optional: purge a single model
  &dry_run=true          # simulate without deleting (default: true)
```

### Response

```json
{
  "dry_run": false,
  "deleted_count": 12450,
  "oldest_remaining": "2026-01-15T08:32:00",
  "models_affected": ["iris", "wine"],
  "linked_observed_results_count": 3
}
```

- `linked_observed_results_count > 0` → warning: deleted predictions are linked to
  `observed_results` (loss of historical performance data).

### Behavior

- `dry_run=true` by default — no deletion without explicit confirmation (`dry_run=false`).
- SQL filter: `WHERE timestamp < now() - interval 'N days'` + optional `model_name` filter.
- Admin only.

### Implementation

- Endpoint: `DELETE /predictions/purge` in `src/api/predict.py`
- DB service: `DBService.purge_predictions()` in `src/services/db_service.py`
- Response schema: `PurgeResponse` in `src/schemas/prediction.py`
- Tests: `tests/test_predictions_purge.py` (16 tests)

### Example

```bash
# Simulate a purge (dry_run by default)
curl -X DELETE "http://localhost:8000/predictions/purge?older_than_days=90" \
  -H "Authorization: Bearer <admin_token>"

# Actually purge iris predictions > 90 days old
curl -X DELETE "http://localhost:8000/predictions/purge?older_than_days=90&model_name=iris&dry_run=false" \
  -H "Authorization: Bearer <admin_token>"
```

## Feature: Scheduled Retraining

### How it works

1. The admin configures a cron schedule via `PATCH /models/{name}/{version}/schedule`.
2. The schedule is stored in `ModelMetadata.retrain_schedule` (JSON field).
3. At API startup, the APScheduler scheduler loads all models with
   `retrain_schedule.enabled=True` from the DB and creates one cron job per version.
4. On each trigger, the job:
   - Acquires a **Redis lock** (`SET NX EX 700`) to prevent simultaneous executions
     in a multi-replica environment.
   - Computes `TRAIN_START_DATE = today - lookback_days` and `TRAIN_END_DATE = today`.
   - Executes the retrain logic (identical to the manual endpoint) in a subprocess
     (timeout 600s).
   - Creates a new version with `trained_by="scheduler"`.
   - If `auto_promote=True` and the model has a `promotion_policy`, evaluates auto-promotion.
   - Updates `last_run_at` and `next_run_at` on the source version.
5. To modify or disable a schedule, call the endpoint again with `enabled=false`.

### `retrain_schedule` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `cron` | string | null | 5-field cron expression (e.g. `"0 3 * * 1"`) |
| `lookback_days` | int ≥ 1 | 30 | History days → `TRAIN_START_DATE` |
| `auto_promote` | bool | false | Evaluate `promotion_policy` after each retrain |
| `enabled` | bool | true | `false` = pause without clearing the schedule |
| `last_run_at` | datetime | null | Last trigger timestamp (UTC, naive) |
| `next_run_at` | datetime | null | Next calculated trigger (UTC, naive) |

### Configure a schedule

```bash
# Every Monday at 03:00 UTC, 30-day window
curl -X PATCH http://localhost:8000/models/my_model/1.0.0/schedule \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "cron": "0 3 * * 1",
    "lookback_days": 30,
    "auto_promote": false,
    "enabled": true
  }'

# Disable the schedule
curl -X PATCH http://localhost:8000/models/my_model/1.0.0/schedule \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"cron": "0 3 * * 1", "enabled": false}'
```

### Multi-replica caveat

The Redis lock `retrain_lock:{name}:{version}` (TTL 700s) ensures that only one replica
runs the job at a time. If the API runs without Redis (test, dev), the job will still
execute — the lock is acquired via in-memory FakeRedis.

### Implementation

- Endpoint: `PATCH /models/{name}/{version}/schedule` in `src/api/models.py`
- Scheduler: `src/tasks/retrain_scheduler.py`
- DB field: `retrain_schedule` (JSON) in `src/db/models/model_metadata.py`
- Migration: `alembic/versions/20260419_6bc2d3e1_add_retrain_schedule.py`
- Tests: `tests/test_scheduled_retraining.py` (17 tests)
