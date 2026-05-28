# API Reference — PredictML API v2.0

Complete documentation of all endpoints, data schemas and Python code examples.

---

## Authentication

All protected routes use a Bearer token in the HTTP header.

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "your-token-here"

headers = {"Authorization": f"Bearer {TOKEN}"}
```

**Authentication error codes**

| Code | Reason |
|---|---|
| 401 | Token absent or invalid |
| 403 | Inactive account or insufficient role |
| 429 | Daily prediction quota exceeded |

**Available roles**

| Role | Access |
|---|---|
| `admin` | Full access — user management, models, predictions |
| `user` | Predictions, viewing own profile |
| `readonly` | Read-only, no predictions |

---

## Public Endpoints

### `GET /`

API status and list of active models.

```python
response = requests.get(f"{BASE_URL}/")
print(response.json())
```

```json
{
  "message": "PredictML API - Multi Models",
  "status": "ok",
  "models_available": ["iris_model", "fraud_detector"],
  "models_count": 2,
  "models_cached": ["iris_model:1.0"]
}
```

---

### `GET /health`

Checks database connectivity and Redis cache status.

```python
response = requests.get(f"{BASE_URL}/health")
```

```json
{
  "status": "ok",
  "models_available": 2,
  "models_cached": 1
}
```

---

## Models

### `GET /models`

Lists all active models. Filterable by tag.

```python
# All models
response = requests.get(f"{BASE_URL}/models")

# Filtered by tag
response = requests.get(f"{BASE_URL}/models", params={"tag": "production"})
models = response.json()
```

**Query parameter**

| Parameter | Type | Description |
|---|---|---|
| `tag` | str | Filter by tag (optional) |

---

### `GET /models/cached`

Lists models currently loaded in memory (Redis cache).

```python
response = requests.get(f"{BASE_URL}/models/cached")
```

```json
{
  "cached_models": ["iris_model:1.0", "fraud_detector:2.1"],
  "count": 2
}
```

---

### `GET /models/{name}/{version}`

Full details of a model, including metrics and loading information.

```python
response = requests.get(f"{BASE_URL}/models/iris_model/1.0.0")
model = response.json()
print(model["model_type"])        # "RandomForestClassifier"
print(model["feature_names"])     # ["sepal length (cm)", ...]
print(model["deployment_mode"])   # "production" | "ab_test" | "shadow"
print(model["traffic_weight"])    # 0.8
```

**Response schema `ModelGetResponse`**

```json
{
  "id": 1,
  "name": "iris_model",
  "version": "1.0.0",
  "description": "Iris Classifier — 3 species",
  "algorithm": "RandomForestClassifier",
  "mlflow_run_id": "abc123def456",
  "minio_bucket": "models",
  "minio_object_key": "iris_model/v1.0.0_model.joblib",
  "file_size_bytes": 24576,
  "file_hash": "sha256:...",
  "accuracy": 0.97,
  "precision": 0.97,
  "recall": 0.96,
  "f1_score": 0.97,
  "confidence_threshold": 0.7,
  "features_count": 4,
  "classes": ["setosa", "versicolor", "virginica"],
  "training_params": {"n_estimators": 100, "max_depth": 5},
  "training_dataset": "iris_train_2024.csv",
  "trained_by": "alice",
  "training_date": "2024-01-15T10:30:00",
  "tags": ["iris", "classification"],
  "webhook_url": "https://hooks.example.com/predictml",
  "feature_baseline": {
    "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}
  },
  "is_active": true,
  "is_production": true,
  "deployment_mode": "production",
  "traffic_weight": 1.0,
  "train_script_object_key": "iris_model/v1.0.0_train.py",
  "parent_version": null,
  "created_at": "2024-01-15T10:35:00",
  "updated_at": "2024-01-20T08:00:00",
  "deprecated_at": null,
  "creator_username": "alice",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "feature_names": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
}
```

---

### `POST /models` — Upload a Model

Upload of a `.joblib` file with its metadata. Uses `multipart/form-data`.

**Auth required** (role `user` or higher)

```python
import requests

with open("iris_model.joblib", "rb") as f:
    files = {"file": ("iris_model.joblib", f, "application/octet-stream")}
    data = {
        "name": "iris_model",
        "version": "1.0.0",
        "description": "Iris Classifier — 3 species",
        "algorithm": "RandomForestClassifier",
        "accuracy": "0.97",
        "f1_score": "0.97",
        "features_count": "4",
        "classes": '["setosa", "versicolor", "virginica"]',
        "training_params": '{"n_estimators": 100, "max_depth": 5}',
        "training_dataset": "iris_train_2024.csv",
        "tags": '["iris", "classification"]',
        "webhook_url": "https://hooks.example.com/predictml",
        "confidence_threshold": "0.7",
        "feature_baseline": '{"sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}}',
    }
    response = requests.post(
        f"{BASE_URL}/models",
        headers=headers,
        files=files,
        data=data,
    )
print(response.status_code)  # 201
```

**With an MLflow run (without `.joblib` file)**

```python
data = {
    "name": "iris_model",
    "version": "2.0.0",
    "mlflow_run_id": "abc123def456789",
}
response = requests.post(f"{BASE_URL}/models", headers=headers, data=data)
```

**With a retraining script**

```python
with open("iris_model.joblib", "rb") as f_model, open("train.py", "rb") as f_train:
    response = requests.post(
        f"{BASE_URL}/models",
        headers=headers,
        files={
            "file": ("iris_model.joblib", f_model, "application/octet-stream"),
            "train_file": ("train.py", f_train, "text/x-python"),
        },
        data={"name": "iris_model", "version": "1.0.0"},
    )
```

**Form fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | str | Yes | Unique model name |
| `version` | str | Yes | Version (e.g. "1.0.0") |
| `file` | `.joblib` file | If no MLflow | Serialised file |
| `train_file` | `.py` file | No | Retraining script |
| `description` | str | No | Human-readable description |
| `algorithm` | str | No | Algorithm name |
| `mlflow_run_id` | str | No | MLflow run ID |
| `accuracy` | float | No | Accuracy score |
| `f1_score` | float | No | F1 score |
| `features_count` | int | No | Number of features |
| `classes` | JSON str | No | Class labels `["A","B"]` |
| `training_params` | JSON str | No | Hyperparameters `{"n": 100}` |
| `training_dataset` | str | No | Dataset name/URI |
| `tags` | JSON str | No | Tags `["tag1","tag2"]` |
| `webhook_url` | str | No | Post-prediction webhook URL |
| `confidence_threshold` | float | No | Confidence threshold (0.0–1.0) |
| `feature_baseline` | JSON str | No | Per-feature baseline stats |

---

### `PATCH /models/{name}/{version}` — Update a Model

Updates metadata, deployment mode or production parameters.

**Auth required**

```python
# Set to production
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=headers,
               json={"is_production": True})

# Configure A/B testing
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=headers,
               json={"deployment_mode": "ab_test", "traffic_weight": 0.2})

# Shadow mode (silent testing without impacting responses)
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=headers,
               json={"deployment_mode": "shadow"})

# Configure a confidence threshold
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers,
               json={"confidence_threshold": 0.8})

# Add a drift baseline
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers,
               json={
                   "feature_baseline": {
                       "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}
                   }
               })

# Add tags and a webhook
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers,
               json={"tags": ["iris", "v1"], "webhook_url": "https://hooks.example.com/ml"})
```

**Schema `ModelUpdateInput`**

| Field | Type | Description |
|---|---|---|
| `description` | str | New description |
| `is_production` | bool | If `true`, other versions are set to `false` |
| `accuracy` | float | Updated score |
| `features_count` | int | Number of features |
| `classes` | list | Class labels |
| `confidence_threshold` | float (0–1) | Min confidence threshold for `low_confidence` |
| `feature_baseline` | dict | Per-feature stats `{name: {mean, std, min, max}}` |
| `tags` | list[str] | Free tags |
| `webhook_url` | str | URL called after each prediction |
| `deployment_mode` | str | `"production"`, `"ab_test"` or `"shadow"` |
| `traffic_weight` | float (0–1) | Traffic fraction routed to this version |
| `alert_thresholds` | dict | Model-specific alert thresholds (e.g. `{"error_rate": 0.05, "drift_zscore": 2.0}`) — overrides global thresholds |

---

### `DELETE /models/{name}/{version}` — Delete a Version

Removes the model from the database, MinIO and MLflow. Returns 204.

```python
response = requests.delete(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers)
assert response.status_code == 204
```

---

### `DELETE /models/{name}` — Delete All Versions

```python
response = requests.delete(f"{BASE_URL}/models/iris_model", headers=headers)
print(response.json())
# {"name": "iris_model", "deleted_versions": ["1.0.0", "2.0.0"],
#  "mlflow_runs_deleted": ["abc123"], "minio_objects_deleted": [...]}
```

---

### `GET /models/{name}/performance` — Real-World Performance

Calculates real-world metrics by joining predictions with observed results.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/performance",
    headers=headers,
    params={
        "start": "2025-01-01T00:00:00",
        "end": "2025-12-31T23:59:59",
        "version": "1.0.0",   # optional
        "period": "week",     # optional: "day", "week", "month"
    }
)
data = response.json()
```

**Schema `ModelPerformanceResponse`** (classification)

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "period_start": "2025-01-01T00:00:00",
  "period_end": "2025-12-31T23:59:59",
  "total_predictions": 500,
  "matched_predictions": 120,
  "model_type": "classification",
  "accuracy": 0.95,
  "precision_weighted": 0.95,
  "recall_weighted": 0.94,
  "f1_weighted": 0.94,
  "confusion_matrix": [[40, 1, 0], [2, 38, 1], [0, 1, 37]],
  "classes": ["setosa", "versicolor", "virginica"],
  "per_class_metrics": {
    "setosa":     {"precision": 0.95, "recall": 0.98, "f1_score": 0.97, "support": 41},
    "versicolor": {"precision": 0.95, "recall": 0.93, "f1_score": 0.94, "support": 41},
    "virginica":  {"precision": 0.97, "recall": 0.97, "f1_score": 0.97, "support": 38}
  },
  "by_period": [
    {"period": "2025-W01", "matched_count": 30, "accuracy": 0.97, "f1_weighted": 0.96}
  ]
}
```

**Schema `ModelPerformanceResponse`** (regression)

```json
{
  "model_type": "regression",
  "mae": 0.42,
  "mse": 0.31,
  "rmse": 0.56,
  "r2": 0.87
}
```

---

### `GET /models/{name}/drift` — Data Drift

Drift report for each feature (Z-score + PSI) compared to the baseline.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/drift",
    headers=headers,
    params={"days": 30, "version": "1.0.0"}
)
data = response.json()
print(data["drift_summary"])  # "ok" | "warning" | "critical" | "no_baseline"
```

**Schema `DriftReportResponse`**

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "period_days": 30,
  "predictions_analyzed": 500,
  "baseline_available": true,
  "drift_summary": "warning",
  "features": {
    "sepal length (cm)": {
      "baseline_mean": 5.84, "baseline_std": 0.83,
      "production_mean": 6.12, "production_std": 0.91,
      "production_count": 500,
      "z_score": 1.9,
      "psi": 0.08,
      "drift_status": "warning"
    },
    "petal length (cm)": {
      "drift_status": "ok",
      "z_score": 0.3,
      "psi": 0.01
    }
  }
}
```

The report covers **4 monitoring dimensions**:
1. **Distribution drift** (Z-score on mean, PSI on distribution)
2. **Performance drift** (accuracy/MAE vs baseline)
3. **Error rate drift** (HTTP 500 and prediction errors)
4. **Null rate** — rate of null values per feature (`null_rate_current` vs `null_rate_baseline`)

See also `GET /models/{name}/output-drift` for output distribution drift (label shift).

---

### `GET /models/{name}/output-drift` — Output Distribution Drift

Detects **label shift**: compares the recent prediction distribution to the reference distribution (`training_stats.label_distribution`).

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period_days` | int | 7 | Analysis window (max 90) |
| `model_version` | str | production | Target version |
| `min_predictions` | int | 30 | Minimum number of predictions to compute (otherwise `insufficient_data`) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/output-drift",
    headers=headers,
    params={"period_days": 7, "min_predictions": 50}
)
drift = response.json()

print(f"Status: {drift['status']}")  # ok | warning | critical | insufficient_data | no_reference
print(f"PSI: {drift['psi']:.4f}")
for cls, data in drift["class_distribution"].items():
    print(f"  {cls}: reference={data['reference']:.2%}, recent={data['recent']:.2%}")
```

**Schema `OutputDriftResponse`**

```json
{
  "model_name": "iris_model",
  "model_version": "2.0.0",
  "period_days": 7,
  "predictions_analyzed": 420,
  "status": "ok",
  "psi": 0.045,
  "class_distribution": {
    "setosa":     {"reference": 0.33, "recent": 0.35, "delta": 0.02},
    "versicolor": {"reference": 0.34, "recent": 0.32, "delta": -0.02},
    "virginica":  {"reference": 0.33, "recent": 0.33, "delta": 0.00}
  }
}
```

**PSI Thresholds**

| PSI | Status |
|---|---|
| < 0.1 | `ok` |
| 0.1 – 0.2 | `warning` |
| ≥ 0.2 | `critical` |

> A `status: critical` triggers a webhook alert (`output_drift_critical`) and can trigger a retrain if `trigger_on_drift` is configured in the schedule.

**Drift Statuses**

| Status | Meaning |
|---|---|
| `ok` | No drift detected |
| `warning` | Moderate drift (Z-score > 1.5 or PSI > 0.1) |
| `critical` | Strong drift (Z-score > 2 or PSI > 0.2) |
| `no_baseline` | No baseline configured for this model |
| `insufficient_data` | Not enough recent predictions |

---

### `GET /models/{name}/feature-importance` — Global Feature Importance (aggregated SHAP)

Computes the average `|SHAP|` per feature over a sample of recent predictions.
Allows identifying the most influential features in production and detecting behavioural drift before performance metrics degrade.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `version` | str | production | Target version; otherwise `is_production=True`, otherwise most recent |
| `last_n` | int | 100 | Number of predictions to sample (max 500) |
| `days` | int | 7 | Time window in days (max 90) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/feature-importance",
    headers=headers,
    params={"version": "1.0.0", "last_n": 200, "days": 14}
)
data = response.json()

print(f"Analysed over {data['sample_size']} predictions")
for feat, info in sorted(
    data["feature_importance"].items(), key=lambda x: x[1]["rank"]
):
    print(f"  #{info['rank']} {feat}: mean |SHAP| = {info['mean_abs_shap']:.4f}")
```

**Schema `FeatureImportanceResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "sample_size": 98,
  "feature_importance": {
    "petal length (cm)": { "mean_abs_shap": 0.42, "rank": 1 },
    "petal width (cm)":  { "mean_abs_shap": 0.31, "rank": 2 },
    "sepal length (cm)": { "mean_abs_shap": 0.18, "rank": 3 },
    "sepal width (cm)":  { "mean_abs_shap": 0.09, "rank": 4 }
  }
}
```

**Edge Cases**

| Situation | Behaviour |
|---|---|
| No predictions in the window | `sample_size: 0`, `feature_importance: {}` |
| Model without `feature_names_in_` | 422 — must be trained with a pandas DataFrame |
| Model type not supported by SHAP | 422 — see `POST /explain` for the list of types |

> **Typical use:** monitor weekly that the most important features remain stable. A ranking change often indicates model behavioural drift before accuracy drops.

---

### `GET /models/{name}/history` — Complete History

Log of all state changes for all versions of a model.

**Auth required**

```python
response = requests.get(f"{BASE_URL}/models/iris_model/history", headers=headers)
data = response.json()
for entry in data["entries"]:
    print(f"[{entry['timestamp']}] {entry['action']} by {entry['changed_by_username']}")
```

**Schema `ModelHistoryResponse`**

```json
{
  "model_name": "iris_model",
  "version": null,
  "total": 5,
  "entries": [
    {
      "id": 12,
      "model_name": "iris_model",
      "model_version": "1.0.0",
      "changed_by_username": "alice",
      "action": "set_production",
      "changed_fields": ["is_production"],
      "snapshot": {"is_production": true, "accuracy": 0.97},
      "timestamp": "2025-01-20T08:00:00"
    }
  ]
}
```

---

### `GET /models/{name}/{version}/history` — History of a Version

```python
response = requests.get(f"{BASE_URL}/models/iris_model/1.0.0/history", headers=headers)
```

Same schema as `/models/{name}/history` but filtered on the specified version.

---

### `POST /models/{name}/{version}/rollback/{history_id}` — Rollback

Restores a model's metadata to a previous state recorded in the history.

**Auth required: admin**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/2.0.0/rollback/12",
    headers=headers
)
data = response.json()
print(f"Restored to history_id={data['rolled_back_to_history_id']}")
print(f"Restored fields: {data['restored_fields']}")
```

**Schema `RollbackResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "rolled_back_to_history_id": 12,
  "new_history_id": 18,
  "restored_fields": ["is_production", "confidence_threshold"],
  "snapshot": {"is_production": false, "confidence_threshold": 0.7}
}
```

---

### `POST /models/{name}/{version}/retrain` — Retrain

Triggers retraining of a model via its `train.py` script stored in MinIO.

**Auth required: admin**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/retrain",
    headers=headers,
    json={
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "new_version": "1.1.0",      # optional, auto-generated if absent
        "set_production": false
    }
)
data = response.json()
print(f"Success: {data['success']}")
print(f"New version: {data['new_version']}")
print(data["stdout"])   # training logs
```

**Schema `RetrainRequest`**

| Field | Type | Required | Description |
|---|---|---|---|
| `start_date` | str (YYYY-MM-DD) | Yes | Data start date |
| `end_date` | str (YYYY-MM-DD) | Yes | Data end date |
| `new_version` | str | No | Output version (auto if absent) |
| `set_production` | bool | No | Set to production after (default: false) |

**Schema `RetrainResponse`**

```json
{
  "model_name": "iris_model",
  "source_version": "1.0.0",
  "new_version": "1.1.0",
  "success": true,
  "stdout": "Epoch 1/10 ... \n{\"accuracy\": 0.96, \"f1_score\": 0.95}",
  "stderr": "",
  "error": null,
  "auto_promoted": null,
  "auto_promote_reason": null,
  "new_model_metadata": {
    "id": 5,
    "name": "iris_model",
    "version": "1.1.0",
    "parent_version": "1.0.0",
    "training_stats": {
      "trained_at": "2026-04-25T03:00:00",
      "train_start_date": "2026-03-26",
      "train_end_date": "2026-04-25",
      "n_rows": 12450,
      "feature_stats": {"sepal_length": {"mean": 5.8, "std": 0.83}},
      "label_distribution": {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
    }
  }
}
```

> The `train.py` script must reference the env variables `TRAIN_START_DATE`, `TRAIN_END_DATE`, `OUTPUT_MODEL_PATH`. See [CLAUDE.md](../CLAUDE.md) for the complete contract.

---

### `PATCH /models/{name}/{version}/schedule` — Schedule Automatic Retraining

Configures a cron schedule to automatically trigger retraining of a version.
The APScheduler loads all active schedules at API startup.

**Auth required: admin**

```python
response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0/schedule",
    headers=headers,
    json={
        "cron": "0 3 * * 1",    # every Monday at 03:00 UTC
        "lookback_days": 30,     # TRAIN_START_DATE = today - 30d
        "auto_promote": False,   # evaluate promotion_policy after retrain
        "enabled": True
    }
)
data = response.json()
print(f"Next trigger: {data['retrain_schedule']['next_run_at']}")
```

**Schema `RetrainScheduleInput`**

| Field | Type | Default | Description |
|---|---|---|---|
| `cron` | str | null | 5-field cron expression (e.g. `"0 3 * * 1"`) |
| `lookback_days` | int ≥ 1 | 30 | Historical window passed to the script (days) |
| `auto_promote` | bool | false | Evaluate `promotion_policy` after each retrain |
| `enabled` | bool | true | `false` = pause without clearing the schedule |
| `trigger_on_drift` | `"warning"` \| `"critical"` \| null | null | Drift level triggering a reactive retrain (without waiting for the cron) |
| `drift_retrain_cooldown_hours` | int ≥ 1 | 24 | Minimum cooldown between two drift-triggered retrains (prevents loops) |

**Schema `ScheduleUpdateResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "retrain_schedule": {
    "cron": "0 3 * * 1",
    "lookback_days": 30,
    "auto_promote": false,
    "enabled": true,
    "last_run_at": null,
    "next_run_at": "2026-04-21T03:00:00"
  }
}
```

> If `cron` is invalid, the API returns HTTP 422 with error details.  
> If `enabled=True` without `cron`, HTTP 422 is returned.  
> To disable without clearing the schedule: `{"cron": "0 3 * * 1", "enabled": false}`.

---

### `GET /models/{name}/ab-compare` — A/B Report with Statistical Significance

Side-by-side comparison of A/B test or shadow versions over a period, enriched with an automatic statistical significance test between the two most active versions.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | 30 | Analysis window in days (1–90) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/ab-compare",
    headers=headers,
    params={"days": 7}
)
data = response.json()

# Raw metrics per version
for v in data["versions"]:
    print(f"{v['version']} ({v['deployment_mode']}): "
          f"{v['total_predictions']} preds, error_rate={v['error_rate']:.2%}, "
          f"agreement={v['agreement_rate']}")

# Statistical significance
sig = data.get("ab_significance")
if sig:
    status = "✅ significant" if sig["significant"] else "⚠️  not significant"
    print(f"\nDifference {status} (p={sig['p_value']:.4f}, test={sig['test']})")
    if sig["winner"]:
        print(f"Best version: {sig['winner']}")
    if sig["current_samples"][sig["winner"] or list(sig["current_samples"])[0]] < sig["min_samples_needed"]:
        print(f"⚠️  Insufficient data — {sig['min_samples_needed']} observations/version recommended")
```

**Schema `ABCompareResponse`**

```json
{
  "model_name": "iris_model",
  "period_days": 7,
  "versions": [
    {
      "version": "1.0.0",
      "deployment_mode": "ab_test",
      "traffic_weight": 0.8,
      "total_predictions": 800,
      "shadow_predictions": 0,
      "error_rate": 0.05,
      "avg_response_time_ms": 12.5,
      "p95_response_time_ms": 45.0,
      "prediction_distribution": {"0": 450, "1": 200, "2": 150},
      "agreement_rate": null
    },
    {
      "version": "2.0.0",
      "deployment_mode": "ab_test",
      "traffic_weight": 0.2,
      "total_predictions": 200,
      "shadow_predictions": 0,
      "error_rate": 0.01,
      "avg_response_time_ms": 9.8,
      "p95_response_time_ms": 38.0,
      "prediction_distribution": {"0": 115, "1": 50, "2": 35},
      "agreement_rate": null
    }
  ],
  "ab_significance": {
    "metric": "error_rate",
    "test": "chi2",
    "p_value": 0.008,
    "significant": true,
    "confidence_level": 0.95,
    "winner": "2.0.0",
    "min_samples_needed": 120,
    "current_samples": {"1.0.0": 800, "2.0.0": 200}
  }
}
```

**`ab_significance` fields**

| Field | Type | Description |
|---|---|---|
| `metric` | str | Metric tested: `"error_rate"` or `"response_time_ms"` |
| `test` | str | Test used: `"chi2"` (error) or `"mann_whitney_u"` (latency) |
| `p_value` | float | P-value of the statistical test |
| `significant` | bool | `true` if `p_value < 1 - confidence_level` |
| `confidence_level` | float | Confidence threshold (default `0.95`) |
| `winner` | str \| null | Version with the best metric, `null` if exact tie |
| `min_samples_needed` | int | Observations/version recommended to detect this effect (80% power) |
| `current_samples` | dict | Number of available observations per version |

> **Test selection logic:**  
> Chi-² if at least one error is observed in either group (success/error contingency table).  
> Fallback to Mann-Whitney U on response times if no errors are present.  
> `ab_significance: null` if fewer than 2 active versions or insufficient data.

---

### `GET /models/{name}/shadow-compare` — Shadow vs Production Report

Enriched comparison between the shadow model and the production model: accuracy, latency, disagreement rate and promotion recommendation.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | 7 | Analysis window |
| `shadow_version` | str | auto | Target shadow version (otherwise the first active `shadow` version) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/shadow-compare",
    headers=headers,
    params={"days": 14}
)
cmp = response.json()

print(f"Production v{cmp['production_version']}: accuracy={cmp['production_accuracy']:.2%}")
print(f"Shadow v{cmp['shadow_version']}: accuracy={cmp['shadow_accuracy']:.2%}")
print(f"Accuracy delta: {cmp['accuracy_delta']:+.2%}")
print(f"Disagreement rate: {cmp['disagreement_rate']:.2%}")
print(f"Recommendation: {cmp['recommendation']}")
```

**Schema `ShadowCompareResponse`**

```json
{
  "model_name": "iris_model",
  "period_days": 14,
  "production_version": "1.0.0",
  "shadow_version": "2.0.0",
  "predictions_analyzed": 1240,
  "production_accuracy": 0.94,
  "shadow_accuracy": 0.97,
  "accuracy_delta": 0.03,
  "production_avg_latency_ms": 14.2,
  "shadow_avg_latency_ms": 12.8,
  "latency_delta_ms": -1.4,
  "confidence_delta": 0.02,
  "disagreement_rate": 0.08,
  "recommendation": "promote"
}
```

| `recommendation` | Meaning |
|---|---|
| `promote` | Shadow is better on all metrics — promote to production |
| `keep_shadow` | Mixed metrics or insufficient data — continue observation |
| `no_shadow` | No active shadow model |

---

### `GET /models/{name}/{version}/card` — Model Card

Summary sheet of a model in a single call: metadata, real performance, drift, calibration, top SHAP features, retrain info and ground truth coverage.

**Auth required**

```python
# JSON (default)
response = requests.get(
    f"{BASE_URL}/models/iris_model/2.0.0/card",
    headers=headers
)
card = response.json()
print(f"Model: {card['name']} v{card['version']}")
print(f"Real accuracy: {card['performance']['accuracy']:.2%}")
print(f"Drift: {card['drift']['summary']}")
print(f"Top feature: {card['feature_importance'][0]['feature']}")

# Markdown — ready to share or insert into a PR
response_md = requests.get(
    f"{BASE_URL}/models/iris_model/2.0.0/card",
    headers={"Authorization": f"Bearer {TOKEN}", "Accept": "text/markdown"}
)
with open("model_card.md", "w") as f:
    f.write(response_md.text)
```

**Schema `ModelCardResponse`**

```json
{
  "name": "iris_model",
  "version": "2.0.0",
  "description": "Iris Classifier — 3 species",
  "algorithm": "RandomForestClassifier",
  "trained_by": "alice",
  "training_date": "2026-03-01T03:00:00",
  "parent_version": "1.0.0",
  "performance": {
    "accuracy": 0.97,
    "f1_score": 0.96,
    "matched_predictions": 920,
    "total_predictions": 1240
  },
  "drift": {
    "summary": "ok",
    "features_in_warning": [],
    "features_in_critical": []
  },
  "calibration": {
    "brier_score": 0.042,
    "overconfidence_gap": 0.031
  },
  "feature_importance": [
    {"feature": "petal length (cm)", "mean_abs_shap": 0.42, "rank": 1},
    {"feature": "petal width (cm)",  "mean_abs_shap": 0.31, "rank": 2}
  ],
  "retrain": {
    "last_retrain_at": "2026-04-01T03:00:00",
    "trained_by": "scheduler",
    "n_rows": 12450
  },
  "ground_truth_coverage": 0.74
}
```

---

### `GET /models/{name}/confidence-trend` — Confidence Trend

Returns the evolution of the average maximum confidence probability over a period, by time window.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `version` | str | production | Target version |
| `days` | int | 30 | Analysis window in days |
| `granularity` | str | `"day"` | Granularity: `"hour"`, `"day"`, `"week"` |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/confidence-trend",
    headers=headers,
    params={"days": 14, "granularity": "day"}
)
data = response.json()

for point in data["trend"]:
    print(f"{point['period']}: conf_avg={point['avg_confidence']:.3f}, "
          f"low_conf_rate={point['low_confidence_rate']:.1%}")
```

**Schema `ConfidenceTrendResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "days": 14,
  "granularity": "day",
  "trend": [
    {
      "period": "2026-04-11",
      "prediction_count": 142,
      "avg_confidence": 0.91,
      "low_confidence_rate": 0.04
    },
    {
      "period": "2026-04-12",
      "prediction_count": 158,
      "avg_confidence": 0.88,
      "low_confidence_rate": 0.07
    }
  ]
}
```

> A gradual drop in `avg_confidence` without accuracy degradation can indicate that the model is encountering observations increasingly close to decision boundaries — an early sign of drift.

---

### `POST /models/{name}/{version}/warmup` — Cache Warmup

Preloads the model into Redis cache without waiting for the first prediction request. Reduces cold-start latency during deployments.

**Auth required**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/warmup",
    headers=headers
)
data = response.json()
print(f"Loaded in {data['load_time_ms']:.1f} ms — {data['status']}")
```

**Schema `WarmupResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "status": "loaded",
  "load_time_ms": 42.3,
  "cached": true
}
```

| `status` | Meaning |
|---|---|
| `loaded` | Model loaded and cached |
| `already_cached` | Model already present in Redis cache |
| `error` | Loading failed (see `detail`) |

---

## Predictions

### `POST /models/{name}/{version}/validate-input` — Input Schema Validation

Validates input features against the expected schema of a model version, **without making a prediction**.

**Auth required**

Detects:
- **missing features** — present in the model, absent in the request
- **unexpected features** — present in the request, absent in the model
- **type coercions** — `string` values convertible to `float` (non-blocking warning)

The source of truth is, in priority order: `feature_names_in_` from the loaded sklearn model, then the keys of `feature_baseline` stored in DB.

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/validate-input",
    headers=headers,
    json={
        "petal_length": 5.1,
        "petal_width": 1.8,
        "sepal_length": 6.3
        # sepal_width missing
    }
)
print(response.json())
```

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

| Field | Description |
|---|---|
| `valid` | `true` only if `errors` is empty |
| `errors` | List of blocking errors (`missing_feature`, `unexpected_feature`) |
| `warnings` | Non-blocking warnings (`type_coercion`) |
| `expected_features` | Sorted list of expected features; `null` if no schema available |

---

### `POST /predict`

Makes a prediction with intelligent routing (A/B test, shadow).

**Auth required** — counts towards the daily quota.

**Version selection** (priority order):
1. `model_version` if provided
2. A/B routing if `ab_test` versions are configured
3. Version with `is_production=true`
4. Most recently created version

```python
response = requests.post(
    f"{BASE_URL}/predict",
    headers=headers,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-2025-001",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
result = response.json()
```

**Query parameter `explain`** (optional, default `false`):

Add `?explain=true` to receive SHAP values directly in the response, without a separate call to `POST /explain`.

```python
response = requests.post(
    f"{BASE_URL}/predict?explain=true",
    headers=headers,
    json={
        "model_name": "iris_model",
        "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                     "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    }
)
data = response.json()
# data["explanation"]["shap_values"] contains the local SHAP values
if data.get("explanation"):
    for feat, val in data["explanation"]["shap_values"].items():
        print(f"  {feat}: {val:+.4f}")
```

The `explanation` field follows the `ExplainOutput` schema (see `POST /explain`). If the model does not support SHAP, `explanation` is `null`.

**Query parameter `strict_validation`** (optional, default `false`):

Add `?strict_validation=true` to reject requests with **unexpected** features (in addition to missing features already checked by default). Returns a structured `422` if validation fails.

```python
# Rejects if unexpected features are present
response = requests.post(
    f"{BASE_URL}/predict?strict_validation=true",
    headers=headers,
    json={
        "model_name": "iris_model",
        "features": {"sepal_length": 5.1, "extra_col": 99.0, ...}
    }
)
# → 422 with detail.errors listing unexpected features
```

**Schema `PredictionInput`**

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | str | Yes | Model name |
| `model_version` | str | No | Specific version; otherwise auto-selection |
| `id_obs` | str | No | Observation identifier |
| `features` | dict | Yes | `{"feature_name": value}` |

**Schema `PredictionOutput`**

```json
{
  "id": 42,
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "id_obs": "obs-2025-001",
  "prediction": 0,
  "probability": [0.97, 0.02, 0.01],
  "low_confidence": false,
  "selected_version": "1.0.0"
}
```

| Field | Description |
|---|---|
| `id` | Database ID of the logged prediction (`null` if `store=false`) |
| `prediction` | Model result (class or value) |
| `probability` | Per-class probabilities (if `predict_proba` available) |
| `low_confidence` | `true` if max prob < model's `confidence_threshold` |
| `selected_version` | Version chosen by A/B routing (if applicable) |

---

### `POST /predict-batch`

Batch predictions: the model is loaded once, all predictions are persisted in one transaction.

**Auth required**

```python
response = requests.post(
    f"{BASE_URL}/predict-batch",
    headers=headers,
    json={
        "model_name": "iris_model",
        "model_version": "1.0.0",   # optional
        "inputs": [
            {
                "id_obs": "obs-001",
                "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                             "petal length (cm)": 1.4, "petal width (cm)": 0.2}
            },
            {
                "id_obs": "obs-002",
                "features": {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0,
                             "petal length (cm)": 5.2, "petal width (cm)": 2.3}
            }
        ]
    }
)
data = response.json()
for item in data["predictions"]:
    print(f"{item['id_obs']} → {item['prediction']} (conf: {item['probability']})")
```

**Query parameter `strict_validation`** (optional, default `false`):

Add `?strict_validation=true` to reject requests containing **unexpected** features in any item of the batch. Returns a `422` with error details if validation fails, without executing any predictions.

```python
# Strict mode: rejects if an unexpected feature is present in the batch
response = requests.post(
    f"{BASE_URL}/predict-batch?strict_validation=true",
    headers=headers,
    json={"model_name": "iris_model", "inputs": [
        {"features": {"sepal length (cm)": 5.1, "extra_col": 99}}
    ]}
)
# → 422 with detail.errors listing unexpected features
```

**Schema `BatchPredictionInput`**

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | str | Yes | Model name |
| `model_version` | str | No | Version; otherwise auto-selection |
| `inputs` | list | Yes | List of items `{features, id_obs}` (min 1, max limited by rate limiting) |

**Schema `BatchPredictionOutput`**

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "predictions": [
    {"id_obs": "obs-001", "prediction": 0, "probability": [0.97, 0.02, 0.01], "low_confidence": false},
    {"id_obs": "obs-002", "prediction": 2, "probability": [0.01, 0.05, 0.94], "low_confidence": false}
  ]
}
```

---

### `POST /explain`

Calculates local SHAP values to explain a prediction.

**Auth required**

```python
response = requests.post(
    f"{BASE_URL}/explain",
    headers=headers,
    json={
        "model_name": "iris_model",
        "model_version": "1.0.0",   # optional
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
data = response.json()
print(f"Prediction: {data['prediction']}")
print(f"Base value: {data['base_value']}")
for feat, val in sorted(data["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑" if val > 0 else "↓"
    print(f"  {direction} {feat}: {val:+.4f}")
```

**Schema `ExplainInput`**

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | str | Yes | Model name |
| `model_version` | str | No | Version; otherwise auto-selection |
| `features` | dict | Yes | Features to explain |

**Schema `ExplainOutput`**

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "prediction": 0,
  "shap_values": {
    "petal length (cm)": -1.32,
    "petal width (cm)": -0.87,
    "sepal length (cm)": -0.12,
    "sepal width (cm)": 0.05
  },
  "base_value": 1.0,
  "model_type": "tree"
}
```

> **Interpretation:** positive SHAP value = feature pushes towards the predicted class, negative = pushes away. The sum of SHAP values + `base_value` = raw prediction.

> **Supported models:** tree-based (`RandomForest`, `GradientBoosting`, `XGBoost`) via `TreeExplainer`; linear (`LogisticRegression`, `LinearRegression`) via `LinearExplainer`.

---

### `GET /predictions`

Filterable prediction history with cursor pagination.

**Auth required**

```python
from datetime import datetime, timedelta

params = {
    "name": "iris_model",
    "start": (datetime.now() - timedelta(days=7)).isoformat(),
    "end": datetime.now().isoformat(),
    "version": "1.0.0",        # optional
    "user": "alice",           # optional
    "limit": 50,
    "cursor": None,            # id of the last entry for the next page
}
response = requests.get(f"{BASE_URL}/predictions", headers=headers, params=params)
data = response.json()

# Next page
if data["next_cursor"]:
    params["cursor"] = data["next_cursor"]
    next_page = requests.get(f"{BASE_URL}/predictions", headers=headers, params=params)
```

**Query parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | str | Yes | Model name |
| `start` | datetime | Yes | Start date (ISO 8601) |
| `end` | datetime | Yes | End date (ISO 8601) |
| `version` | str | No | Filter by version |
| `user` | str | No | Filter by username |
| `id_obs` | str | No | Filter by observation identifier |
| `min_confidence` | float (0–1) | No | Minimum confidence (max of probabilities) |
| `max_confidence` | float (0–1) | No | Maximum confidence (max of probabilities) |
| `limit` | int | No | Max results (1-1000, default 100) |
| `cursor` | int | No | ID of the last seen prediction (cursor pagination) |

**Schema `PredictionsListResponse`**

```json
{
  "total": 142,
  "limit": 50,
  "next_cursor": 1089,
  "predictions": [
    {
      "id": 1040,
      "model_name": "iris_model",
      "model_version": "1.0.0",
      "id_obs": "obs-2025-001",
      "input_features": {"sepal length (cm)": 5.1},
      "prediction_result": 0,
      "probabilities": [0.97, 0.02, 0.01],
      "response_time_ms": 12.5,
      "timestamp": "2025-01-15T14:32:00",
      "status": "success",
      "error_message": null,
      "username": "alice",
      "is_shadow": false
    }
  ]
}
```

> **Cursor pagination:** use `next_cursor` (id of the last returned prediction) as the `cursor` parameter of the next request. More efficient than `offset` on large volumes.

---

### `GET /predictions/{prediction_id}` — Look Up a Prediction by ID

Returns the full details of a prediction from its internal identifier.

**Auth required**

```python
prediction_id = 1040

response = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}",
    headers=headers
)
data = response.json()
print(f"Model: {data['model_name']} v{data['model_version']}")
print(f"Result: {data['prediction_result']}")
print(f"Latency: {data['response_time_ms']} ms")
```

Returns the same schema as an element of `GET /predictions` (see above). Returns `404` if the prediction does not exist or belongs to another user (non-admin).

---

### `GET /predictions/{prediction_id}/explain` — Post-hoc SHAP Explanation

Generates a post-hoc SHAP explanation for an existing prediction, by reloading the features from the database.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}/explain",
    headers=headers
)
data = response.json()
print(f"Prediction: {data['prediction']}")
for feat, val in sorted(data["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {'↑' if val > 0 else '↓'} {feat}: {val:+.4f}")
```

Returns the same schema as `POST /explain`. Returns `404` if the prediction does not exist, `422` if the model does not support SHAP.

---

### `GET /predictions/stats`

Aggregated prediction statistics per model.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/predictions/stats",
    headers=headers,
    params={"days": 7, "model_name": "iris_model"}  # model_name optional
)
data = response.json()
for stat in data["stats"]:
    print(f"{stat['model_name']}: {stat['total_predictions']} predictions, "
          f"error={stat['error_rate']:.1%}, p95={stat['p95_response_time_ms']}ms")
```

**Schema `PredictionStatsResponse`**

```json
{
  "days": 7,
  "model_name": "iris_model",
  "stats": [
    {
      "model_name": "iris_model",
      "total_predictions": 1250,
      "error_count": 5,
      "error_rate": 0.004,
      "avg_response_time_ms": 14.2,
      "p50_response_time_ms": 11.0,
      "p95_response_time_ms": 38.5
    }
  ]
}
```

---

### `GET /predictions/anomalies` — Anomaly Detection

Returns recent predictions where at least one feature has an abnormal z-score compared to the model baseline.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | str | Yes | Model name |
| `days` | int | 7 | Analysis window (max 90) |
| `z_threshold` | float | 3.0 | Z-score threshold above which a feature is considered anomalous |
| `limit` | int | 200 | Max results (max 1000) |

```python
response = requests.get(
    f"{BASE_URL}/predictions/anomalies",
    headers=headers,
    params={"model_name": "iris_model", "days": 7, "z_threshold": 3.0}
)
data = response.json()

print(f"{data['anomaly_count']} anomalous predictions out of {data['total_analyzed']} analysed")
for pred in data["anomalies"]:
    print(f"  ID {pred['prediction_id']} ({pred['timestamp'][:10]}):")
    for feat in pred["anomalous_features"]:
        print(f"    {feat['feature']}: z={feat['z_score']:.1f} "
              f"(value={feat['value']}, baseline_mean={feat['baseline_mean']:.2f})")
```

**Schema `AnomaliesResponse`**

```json
{
  "model_name": "iris_model",
  "period_days": 7,
  "z_threshold": 3.0,
  "total_analyzed": 840,
  "anomaly_count": 5,
  "anomalies": [
    {
      "prediction_id": 1082,
      "timestamp": "2026-04-25T14:32:00",
      "model_version": "2.0.0",
      "prediction_result": 1,
      "anomalous_features": [
        {
          "feature": "sepal length (cm)",
          "value": 12.4,
          "z_score": 7.8,
          "baseline_mean": 5.84,
          "baseline_std": 0.83
        }
      ]
    }
  ]
}
```

> Returns `{"error": "no_baseline"}` if `feature_baseline` is not configured for this model.

---

### `DELETE /predictions/purge` — GDPR Purge

Deletes predictions older than N days. `dry_run=true` by default — no deletion without explicit confirmation.

**Auth required: admin**

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `older_than_days` | int | Yes | Delete predictions > N days old |
| `model_name` | str | No | Limit the purge to a single model |
| `dry_run` | bool | `true` | Simulate without deleting |

```python
# Simulate (dry_run by default)
response = requests.delete(
    f"{BASE_URL}/predictions/purge",
    headers=headers,
    params={"older_than_days": 90}
)
data = response.json()
print(f"Would be deleted: {data['deleted_count']} predictions")
print(f"Linked observed results: {data['linked_observed_results_count']}")

# Actually purge
response = requests.delete(
    f"{BASE_URL}/predictions/purge",
    headers=headers,
    params={"older_than_days": 90, "model_name": "iris_model", "dry_run": "false"}
)
```

**Schema `PurgeResponse`**

```json
{
  "dry_run": false,
  "deleted_count": 12450,
  "oldest_remaining": "2026-01-15T08:32:00",
  "models_affected": ["iris_model", "wine"],
  "linked_observed_results_count": 3
}
```

> `linked_observed_results_count > 0` indicates that predictions linked to `observed_results` will be deleted — loss of historical performance data.

---

## Golden Tests

Regression tests to validate that a model always produces the expected outputs on reference cases. Particularly useful after a retrain.

### `GET /models/{name}/golden-tests` — List Test Cases

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/golden-tests",
    headers=headers
)
tests = response.json()

for t in tests:
    print(f"#{t['id']} [{t.get('description', '—')}] "
          f"→ expected: {t['expected_output']}")
```

**Schema `GoldenTestResponse`**

```json
[
  {
    "id": 1,
    "model_name": "iris_model",
    "description": "Typical Iris setosa",
    "input_features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                       "petal length (cm)": 1.4, "petal width (cm)": 0.2},
    "expected_output": "setosa",
    "created_at": "2026-04-01T10:00:00",
    "created_by": "alice"
  }
]
```

---

### `POST /models/{name}/golden-tests` — Create a Test Case

**Auth required**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/golden-tests",
    headers=headers,
    json={
        "input_features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        },
        "expected_output": "setosa",
        "description": "Typical Iris setosa — all nominal features"
    }
)
print(response.json()["id"])  # id of the new case
```

| Field | Type | Required | Description |
|---|---|---|---|
| `input_features` | dict | Yes | Input features for the case |
| `expected_output` | str/int/float | Yes | Expected model output |
| `description` | str | No | Test case description |

---

### `DELETE /models/{name}/golden-tests/{test_id}` — Delete a Case

**Auth required: admin**

```python
response = requests.delete(
    f"{BASE_URL}/models/iris_model/golden-tests/1",
    headers=headers
)
assert response.status_code == 204
```

---

### `POST /models/{name}/golden-tests/upload-csv` — Bulk Import from CSV

**Auth required: admin**

**CSV Format**

```
description,input_features,expected_output
"Typical Iris setosa","{""sepal length (cm)"": 5.1, ""sepal width (cm)"": 3.5, ""petal length (cm)"": 1.4, ""petal width (cm)"": 0.2}",setosa
"Robust Iris virginica","{""sepal length (cm)"": 6.7, ""sepal width (cm)"": 3.0, ""petal length (cm)"": 5.2, ""petal width (cm)"": 2.3}",virginica
```

```python
import io, csv

rows = [
    {"description": "Iris setosa",
     "input_features": '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}',
     "expected_output": "setosa"},
    {"description": "Iris virginica",
     "input_features": '{"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}',
     "expected_output": "virginica"},
]

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=["description", "input_features", "expected_output"])
writer.writeheader()
writer.writerows(rows)
buf.seek(0)

response = requests.post(
    f"{BASE_URL}/models/iris_model/golden-tests/upload-csv",
    headers=headers,
    files={"file": ("tests.csv", buf, "text/csv")}
)
result = response.json()
print(f"{result['imported']} cases imported")
if result.get("errors"):
    for err in result["errors"]:
        print(f"  ❌ Row {err['row']}: {err['reason']}")
```

---

### `POST /models/{name}/{version}/run-golden-tests` — Run Tests

Runs all registered test cases for a model on a given version.

**Auth required**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/2.0.0/run-golden-tests",
    headers=headers
)
result = response.json()

print(f"✅ {result['passed']} / {result['total_tests']} tests passed "
      f"({result['pass_rate']:.1%})")

for detail in result["details"]:
    icon = "✅" if detail["passed"] else "❌"
    desc = detail.get("description", f"Test #{detail['test_id']}")
    print(f"  {icon} {desc}")
    if not detail["passed"]:
        print(f"      expected: {detail['expected']!r} | received: {detail['actual']!r}")
```

**Schema `GoldenTestRunResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "total_tests": 5,
  "passed": 4,
  "failed": 1,
  "pass_rate": 0.8,
  "details": [
    {
      "test_id": 1,
      "description": "Typical Iris setosa",
      "input": {"sepal length (cm)": 5.1},
      "expected": "setosa",
      "actual": "setosa",
      "passed": true
    },
    {
      "test_id": 3,
      "description": "Borderline versicolor case",
      "input": {"sepal length (cm)": 6.0},
      "expected": "versicolor",
      "actual": "virginica",
      "passed": false
    }
  ]
}
```

> Integration with `promotion_policy.min_golden_test_pass_rate` allows blocking auto-promotion if the pass rate is insufficient.

---

## Observed Results

Allows recording real results after prediction, to evaluate models in production.

### `POST /observed-results`

Records or updates observed results. Idempotent on `(id_obs, model_name)`.

**Auth required**

```python
response = requests.post(
    f"{BASE_URL}/observed-results",
    headers=headers,
    json={
        "data": [
            {
                "id_obs": "obs-2025-001",
                "model_name": "iris_model",
                "date_time": "2025-01-16T08:00:00",
                "observed_result": 0
            },
            {
                "id_obs": "obs-2025-002",
                "model_name": "iris_model",
                "date_time": "2025-01-16T08:00:00",
                "observed_result": 1
            }
        ]
    }
)
print(response.json())  # {"upserted": 2}
```

**Schema `ObservedResultInput`**

| Field | Type | Description |
|---|---|---|
| `id_obs` | str | Observation identifier (linked to the prediction) |
| `model_name` | str | Model name |
| `date_time` | datetime | Timestamp of the real result |
| `observed_result` | float/int/str | Actual observed result |

---

### `GET /observed-results`

Retrieves observed results with optional filters.

```python
params = {
    "model_name": "iris_model",  # optional
    "id_obs": "obs-2025-001",    # optional
    "start": "2025-01-01T00:00:00",
    "end": "2025-01-31T23:59:59",
    "limit": 100,
    "offset": 0,
}
response = requests.get(f"{BASE_URL}/observed-results", headers=headers, params=params)
data = response.json()
```

```json
{
  "total": 50,
  "limit": 100,
  "offset": 0,
  "results": [
    {
      "id": 1,
      "id_obs": "obs-2025-001",
      "model_name": "iris_model",
      "observed_result": 0,
      "date_time": "2025-01-16T08:00:00",
      "username": "alice"
    }
  ]
}
```

---

### `GET /observed-results/export` — CSV/JSON Export

Exports filtered observed results to a downloadable file.

**Auth required**

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | str | No | Filter by model |
| `start` | datetime | No | Start date |
| `end` | datetime | No | End date |
| `format` | str | `"csv"` | `"csv"` or `"json"` |

```python
response = requests.get(
    f"{BASE_URL}/observed-results/export",
    headers=headers,
    params={"model_name": "iris_model", "format": "csv"}
)

with open("observed_results.csv", "wb") as f:
    f.write(response.content)
# Columns: id_obs, model_name, observed_result, date_time, username
```

---

### `GET /observed-results/stats` — Coverage Statistics

Returns statistics on ground truth coverage: how many predictions have an associated observed result.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/observed-results/stats",
    headers=headers,
    params={"model_name": "iris_model", "days": 30}
)
data = response.json()
print(f"Predictions: {data['total_predictions']}")
print(f"With ground truth: {data['matched_count']}")
print(f"Coverage: {data['coverage_rate']:.1%}")
```

**Schema `ObservedResultsStatsResponse`**

```json
{
  "model_name": "iris_model",
  "days": 30,
  "total_predictions": 1500,
  "matched_count": 420,
  "coverage_rate": 0.28,
  "unmatched_count": 1080
}
```

---

### `POST /observed-results/upload-csv` — Bulk CSV Import

Imports a CSV file of observed results. Idempotent on `(id_obs, model_name)`.

**Auth required**

**CSV Format**

```
id_obs,model_name,date_time,observed_result
obs-001,iris_model,2026-01-15T10:00:00,0
obs-002,iris_model,2026-01-15T10:01:00,2
```

```python
import io

csv_content = """id_obs,model_name,date_time,observed_result
obs-001,iris_model,2026-01-15T10:00:00,0
obs-002,iris_model,2026-01-15T10:01:00,2
obs-003,iris_model,2026-01-15T10:02:00,1
"""

response = requests.post(
    f"{BASE_URL}/observed-results/upload-csv",
    headers=headers,
    files={"file": ("results.csv", io.StringIO(csv_content), "text/csv")}
)
print(response.json())
```

**Schema `CSVUploadResponse`**

```json
{
  "upserted": 3,
  "errors": []
}
```

If some rows are invalid, they are listed in `errors` and valid rows are still inserted.

---

## Users

### `POST /users` — Create a User

**Auth required: admin**

```python
response = requests.post(
    f"{BASE_URL}/users",
    headers=headers,
    json={
        "username": "alice",
        "email": "alice@example.com",
        "role": "user",
        "rate_limit": 5000
    }
)
user = response.json()
print(user["api_token"])  # Token to keep — displayed only once
```

**Schema `UserCreateInput`**

| Field | Type | Description |
|---|---|---|
| `username` | str (3-50 chars) | Unique name |
| `email` | EmailStr | Unique email |
| `role` | str | `"admin"`, `"user"` or `"readonly"` (default: `"user"`) |
| `rate_limit` | int (1-100000) | Predictions per day (default: 1000) |

**Schema `UserResponse`**

```json
{
  "id": 3,
  "username": "alice",
  "email": "alice@example.com",
  "role": "user",
  "is_active": true,
  "rate_limit_per_day": 5000,
  "api_token": "eyJhbGciOiJIUzI1...",
  "created_at": "2025-01-15T10:00:00",
  "last_login": null
}
```

---

### `GET /users` — List Users

**Auth required: admin**

```python
response = requests.get(f"{BASE_URL}/users", headers=headers)
users = response.json()
```

---

### `GET /users/{user_id}` — User Details

A user can view their own profile. An admin can view all profiles.

```python
response = requests.get(f"{BASE_URL}/users/3", headers=headers)
```

---

### `PATCH /users/{user_id}` — Modify a User

**Auth required: admin**

```python
# Deactivate an account
requests.patch(f"{BASE_URL}/users/3", headers=headers, json={"is_active": False})

# Change role and quota
requests.patch(f"{BASE_URL}/users/3", headers=headers,
               json={"role": "readonly", "rate_limit": 100})

# Renew the token
response = requests.patch(
    f"{BASE_URL}/users/3",
    headers=headers,
    json={"regenerate_token": True}
)
new_token = response.json()["api_token"]
```

**Schema `UserUpdateInput`**

| Field | Type | Description |
|---|---|---|
| `is_active` | bool | Activate/deactivate the account |
| `role` | str | New role |
| `rate_limit` | int | New daily quota |
| `regenerate_token` | bool | Generates a new Bearer token |

---

### `DELETE /users/{user_id}` — Delete a User

Deletes the user and all their predictions in cascade. Returns 204.

**Auth required: admin**

```python
response = requests.delete(f"{BASE_URL}/users/3", headers=headers)
assert response.status_code == 204
```

---

## Monitoring

### `GET /monitoring/overview`

Global health dashboard for all models over a period.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/monitoring/overview",
    headers=headers,
    params={"days": 7}
)
data = response.json()
print(f"Global errors: {data['global_stats']['error_rate']:.1%}")
for model in data["models"]:
    print(f"  {model['model_name']}: drift={model['drift_summary']}, "
          f"error_rate={model['error_rate']:.1%}")
```

**Schema `GlobalDashboard`**

```json
{
  "period_days": 7,
  "global_stats": {
    "total_predictions": 5420,
    "error_count": 23,
    "error_rate": 0.004,
    "avg_response_time_ms": 13.2
  },
  "models": [
    {
      "model_name": "iris_model",
      "versions_count": 2,
      "total_predictions": 3200,
      "error_rate": 0.003,
      "drift_summary": "ok",
      "has_production_version": true
    }
  ]
}
```

---

### `GET /monitoring/model/{name}`

Full monitoring detail for a model: timeseries, drift, A/B, recent errors.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/monitoring/model/iris_model",
    headers=headers,
    params={"days": 30}
)
data = response.json()
# data["timeseries"] — prediction points per period
# data["drift"]      — drift report
# data["ab_compare"] — A/B comparison if applicable
```

**Schema `ModelDetailDashboard`**

```json
{
  "model_name": "iris_model",
  "period": {"days": 30, "start": "2025-01-01", "end": "2025-01-31"},
  "versions": [
    {
      "version": "1.0.0",
      "deployment_mode": "production",
      "total_predictions": 3200,
      "error_rate": 0.003,
      "avg_response_time_ms": 12.1
    }
  ],
  "timeseries": [
    {"timestamp": "2025-01-01T00:00:00", "count": 105, "error_count": 0, "avg_ms": 11.5}
  ],
  "drift_summary": "ok",
  "recent_errors": [
    {"timestamp": "2025-01-10T14:22:00", "error_message": "Missing feature: petal length"}
  ]
}
```

---

## Models — Additional Endpoints

### `GET /models/leaderboard` — Model Ranking

Ranks production models by metric over a rolling window. Result is cached (configurable TTL).

**Auth not required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric` | str | `accuracy` | Ranking metric: `accuracy`, `f1_score`, `latency_p95_ms`, `predictions_count` |
| `days` | int | 30 | Time window |
| `top_n` | int | 10 | Number of models returned |

```python
response = requests.get(
    f"{BASE_URL}/models/leaderboard",
    params={"metric": "accuracy", "days": 30, "top_n": 5}
)
leaderboard = response.json()

for i, entry in enumerate(leaderboard["entries"], 1):
    print(f"#{i} {entry['model_name']} v{entry['version']} — "
          f"accuracy={entry.get('accuracy')}, p95={entry.get('latency_p95_ms')}ms")
```

**Schema `LeaderboardResponse`**

```json
{
  "metric": "accuracy",
  "days": 30,
  "entries": [
    {
      "model_name": "iris_model",
      "version": "2.0.0",
      "is_production": true,
      "accuracy": 0.97,
      "f1_score": 0.96,
      "latency_p95_ms": 14.2,
      "predictions_count": 8450,
      "last_prediction_at": "2026-04-27T18:32:00"
    }
  ]
}
```

---

### `GET /models/{name}/performance-timeline` — Performance Timeline

Evolution of performance metrics version by version, sorted by deployment date.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | 90 | Time window |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/performance-timeline",
    headers=headers,
    params={"days": 90}
)
timeline = response.json()

for entry in timeline["versions"]:
    print(f"v{entry['version']} deployed on {entry['deployed_at']} — "
          f"accuracy={entry.get('accuracy')}, MAE={entry.get('mae')}")
```

**Schema**

```json
{
  "model_name": "iris_model",
  "period_days": 90,
  "versions": [
    {
      "version": "1.0.0",
      "deployed_at": "2026-01-15T00:00:00",
      "accuracy": 0.94,
      "f1_score": 0.93,
      "mae": null,
      "predictions_count": 4200
    },
    {
      "version": "2.0.0",
      "deployed_at": "2026-03-01T00:00:00",
      "accuracy": 0.97,
      "f1_score": 0.96,
      "mae": null,
      "predictions_count": 8450
    }
  ]
}
```

---

### `GET /models/{name}/calibration` — Probability Calibration

Measures the calibration quality of predicted probabilities: a perfectly calibrated model returns 70% confidence when it is right 70% of the time.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | 30 | Time window |
| `version` | str | production | Target version |
| `bins` | int | 10 | Number of bins for the reliability curve |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/calibration",
    headers=headers,
    params={"days": 30, "version": "2.0.0"}
)
cal = response.json()

print(f"Brier score: {cal['brier_score']:.4f}")  # 0 = perfect, 1 = worst
print(f"Overconfidence gap: {cal['overconfidence_gap']:+.3f}")
# Positive = model too confident, negative = not confident enough

for b in cal["reliability_diagram"]:
    print(f"  conf={b['bin_center']:.1f} → reality={b['fraction_positive']:.2f} "
          f"(n={b['count']})")
```

**Schema `CalibrationResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "period_days": 30,
  "sample_size": 920,
  "brier_score": 0.042,
  "overconfidence_gap": 0.031,
  "reliability_diagram": [
    {"bin_center": 0.05, "mean_predicted": 0.04, "fraction_positive": 0.02, "count": 48},
    {"bin_center": 0.15, "mean_predicted": 0.14, "fraction_positive": 0.11, "count": 72},
    {"bin_center": 0.85, "mean_predicted": 0.86, "fraction_positive": 0.91, "count": 235},
    {"bin_center": 0.95, "mean_predicted": 0.96, "fraction_positive": 0.94, "count": 187}
  ]
}
```

> **Interpretation:** A `brier_score` < 0.1 is good for classification. An `overconfidence_gap` > 0.05 signals that the model overestimates its certainty — to monitor before high-criticality deployment.

---

### `GET /models/{name}/confidence-distribution` — Confidence Distribution

Histogram of the confidence level (`max(probabilities)`) over recent predictions. Allows identifying the proportion of uncertain predictions.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | 7 | Time window |
| `version` | str | production | Target version |
| `bins` | int | 10 | Histogram resolution |
| `high_threshold` | float | 0.9 | High confidence threshold |
| `uncertain_threshold` | float | 0.6 | Uncertainty threshold |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/confidence-distribution",
    headers=headers,
    params={"days": 7, "uncertain_threshold": 0.7}
)
dist = response.json()

print(f"Highly confident predictions (>{dist['high_threshold']}): "
      f"{dist['high_confidence_pct']:.1%}")
print(f"Uncertain predictions (<{dist['uncertain_threshold']}): "
      f"{dist['uncertain_pct']:.1%}")
```

**Schema `ConfidenceDistributionResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "period_days": 7,
  "total_predictions": 1240,
  "high_threshold": 0.9,
  "uncertain_threshold": 0.6,
  "high_confidence_pct": 0.82,
  "uncertain_pct": 0.04,
  "bins": [
    {"lower": 0.0, "upper": 0.1, "count": 3, "pct": 0.002},
    {"lower": 0.9, "upper": 1.0, "count": 1018, "pct": 0.821}
  ]
}
```

---

### `GET /models/{name}/performance-report` — Consolidated Report

Aggregates in a single call: performance, drift, feature importance, calibration, and A/B comparison.
Ideal for automatic monitoring scripts or programmatic alerts.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | 30 | Common time window for all components |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/performance-report",
    headers=headers,
    params={"days": 30}
)
report = response.json()

# Global check in one line
if (report["drift"]["drift_summary"] == "critical" or
        report["performance"].get("accuracy", 1.0) < 0.85):
    print("⚠️  Action required on iris_model")
```

**Schema `PerformanceReportResponse`**

```json
{
  "model_name": "iris_model",
  "generated_at": "2026-04-28T10:00:00",
  "period_days": 30,
  "performance": {"accuracy": 0.97, "f1_score": 0.96, "matched_predictions": 920},
  "drift": {"drift_summary": "ok", "features": {}},
  "feature_importance": {"petal length (cm)": {"mean_abs_shap": 0.42, "rank": 1}},
  "calibration": {"brier_score": 0.042, "overconfidence_gap": 0.031},
  "ab_compare": null
}
```

---

### `GET /models/{name}/readiness` — Readiness Check

Verifies that a model satisfies all prerequisites before being set to production.

**Auth required**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/readiness",
    headers=headers
)
check = response.json()

if check["ready"]:
    print("✅ Model ready for production")
else:
    for name, ok in check["checks"].items():
        if not ok:
            print(f"❌ {name}: not satisfied")
```

**Schema `ReadinessResponse`**

```json
{
  "model_name": "iris_model",
  "ready": false,
  "checks": {
    "is_production": false,
    "file_accessible": true,
    "baseline_computed": false,
    "no_critical_drift": true
  }
}
```

| Check | Description |
|---|---|
| `is_production` | `is_production=True` on at least one version |
| `file_accessible` | `.joblib` file accessible in MinIO |
| `baseline_computed` | `feature_baseline` computed (required for drift) |
| `no_critical_drift` | No critical drift detected in the recent window |

---

### `GET /models/{name}/retrain-history` — Retrain History

Structured log of all retrain events for a model: manual or scheduled.

**Auth required**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 20 | Number of entries |
| `offset` | int | 0 | Pagination |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/retrain-history",
    headers=headers,
    params={"limit": 10}
)
history = response.json()

print(f"Total retrains: {history['total']}")
for entry in history["history"]:
    promoted = "✅" if entry["auto_promoted"] else "⏭️"
    print(f"{promoted} {entry['timestamp'][:10]} "
          f"v{entry['source_version']} → v{entry['new_version']} "
          f"(accuracy={entry.get('accuracy')}, by={entry['trained_by']})")
```

**Schema `RetrainHistoryResponse`**

```json
{
  "model_name": "iris_model",
  "total": 8,
  "history": [
    {
      "timestamp": "2026-04-01T03:00:00",
      "source_version": "1.0.0",
      "new_version": "1.1.0",
      "trained_by": "scheduler",
      "accuracy": 0.95,
      "f1_score": 0.94,
      "auto_promoted": true,
      "auto_promote_reason": "all criteria met",
      "n_rows": 12450,
      "train_start_date": "2026-03-01",
      "train_end_date": "2026-04-01"
    }
  ]
}
```

---

### `PATCH /models/{name}/{version}/deprecate` — Deprecate a Version

Marks a version as deprecated. New predictions on this version return **HTTP 410 Gone**.

**Auth required: admin**

```python
response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0/deprecate",
    headers=headers
)
print(response.json())
# {"model_name": "iris_model", "version": "1.0.0",
#  "deprecated_at": "2026-04-28T10:00:00",
#  "message": "Version deprecated. New predictions are blocked."}
```

> **Note:** Deprecation is irreversible via this endpoint. To restore a version, use `POST /models/{name}/{version}/rollback/{history_id}`.

---

### `PATCH /models/{name}/policy` — Post-Retrain Auto-Promotion Policy

Defines the criteria a retrained model must satisfy to be automatically promoted to production.

**Auth required: admin**

**Request body**

```json
{
  "min_accuracy": 0.90,
  "max_latency_p95_ms": 200.0,
  "min_sample_validation": 50,
  "auto_promote": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `min_accuracy` | float [0–1] | null | Minimum accuracy (on recent observed_results) |
| `max_latency_p95_ms` | float > 0 | null | Maximum P95 latency in ms |
| `min_sample_validation` | int ≥ 1 | 10 | Minimum number of (prediction, result) pairs to evaluate |
| `auto_promote` | bool | false | Enable post-retrain auto-promotion |
| `min_golden_test_pass_rate` | float [0–1] | null | Minimum golden test pass rate before promotion |
| `auto_demote` | bool | false | Enable auto-demotion (circuit breaker) |
| `demote_on_drift` | `"warning"` \| `"critical"` \| null | null | Drift level triggering automatic demotion |
| `demote_on_accuracy_below` | float [0–1] | null | Minimum accuracy below which the model is demoted |
| `demote_cooldown_hours` | int ≥ 1 | 24 | Minimum delay between two automatic demotions |

```python
# Post-retrain auto-promotion
response = requests.patch(
    f"{BASE_URL}/models/iris_model/policy",
    headers=headers,
    json={
        "min_accuracy": 0.90,
        "max_latency_p95_ms": 200,
        "min_sample_validation": 50,
        "min_golden_test_pass_rate": 0.95,
        "auto_promote": True
    }
)

# Circuit breaker — auto-demotion if critical drift or accuracy too low
response = requests.patch(
    f"{BASE_URL}/models/iris_model/policy",
    headers=headers,
    json={
        "auto_demote": True,
        "demote_on_drift": "critical",
        "demote_on_accuracy_below": 0.80,
        "demote_cooldown_hours": 48
    }
)
print(f"Policy enabled: {response.json()['auto_promote']}")
```

> The policy is evaluated automatically at the end of each retrain (auto-promotion) and during each supervision cycle every 6h (auto-demotion). The result is returned in the `POST /models/{name}/{version}/retrain` response via the `auto_promoted` and `auto_promote_reason` fields.

---

### `GET /models/{name}/{version}/download` — Download the .joblib File

Downloads the serialised model file from MinIO.

**Auth required**

```python
import pathlib

response = requests.get(
    f"{BASE_URL}/models/iris_model/2.0.0/download",
    headers=headers,
    stream=True
)
response.raise_for_status()

output_path = pathlib.Path("iris_model_v2.0.0.joblib")
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Model downloaded: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
```

The response is a binary stream (`Content-Type: application/octet-stream`) with header `Content-Disposition: attachment; filename=iris_model_2.0.0.joblib`.

---

## Predictions — Additional Endpoints

### `GET /predictions/export` — Streaming Export

Exports prediction history as CSV, JSONL, or Parquet. Uses cursor pagination internally to handle large volumes without overloading memory.

**Auth required: admin**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `format` | str | `csv` | Format: `csv`, `jsonl`, `parquet` |
| `model_name` | str | — | Filter by model (optional) |
| `start` | datetime | — | Start of range (ISO 8601) |
| `end` | datetime | — | End of range (ISO 8601) |
| `version` | str | — | Filter by version (optional) |

```python
from datetime import datetime, timedelta

response = requests.get(
    f"{BASE_URL}/predictions/export",
    headers=headers,
    params={
        "format": "csv",
        "model_name": "iris_model",
        "start": (datetime.now() - timedelta(days=30)).isoformat(),
        "end": datetime.now().isoformat()
    },
    stream=True
)
response.raise_for_status()

with open("predictions_export.csv", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# For Parquet (binary, compatible with pandas/polars)
response_parquet = requests.get(
    f"{BASE_URL}/predictions/export",
    headers=headers,
    params={"format": "parquet", "model_name": "iris_model"},
    stream=True
)
with open("predictions.parquet", "wb") as f:
    for chunk in response_parquet.iter_content(chunk_size=8192):
        f.write(chunk)

import pandas as pd
df = pd.read_parquet("predictions.parquet")
print(df.head())
```

---

## Users — Additional Endpoints

### `GET /users/me` — Current User Profile

Returns the profile of the user who owns the Bearer token being used.

**Auth required**

```python
response = requests.get(f"{BASE_URL}/users/me", headers=headers)
me = response.json()
print(f"Logged in as {me['username']} (role: {me['role']})")
```

**Schema**

```json
{
  "id": 3,
  "username": "alice",
  "email": "alice@example.com",
  "role": "user",
  "is_active": true,
  "rate_limit_per_day": 5000,
  "created_at": "2026-01-10T09:00:00",
  "last_login": "2026-04-28T08:12:00"
}
```

---

### `GET /users/me/quota` — Daily Quota

Returns the current user's quota consumption for the current day.

**Auth required**

```python
response = requests.get(f"{BASE_URL}/users/me/quota", headers=headers)
quota = response.json()

print(f"Quota: {quota['used_today']} / {quota['rate_limit_per_day']} predictions")
print(f"Remaining: {quota['remaining']} — Reset at {quota['reset_at']}")
```

**Schema `QuotaResponse`**

```json
{
  "rate_limit_per_day": 5000,
  "used_today": 342,
  "remaining": 4658,
  "reset_at": "2026-04-29T00:00:00"
}
```

---

### `GET /users/{user_id}/usage` — Usage Statistics

Usage statistics for a user: volume by model and by day. Accessible by the user themselves or by an admin.

**Auth required (self or admin)**

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | 30 | Time window |

```python
response = requests.get(
    f"{BASE_URL}/users/3/usage",
    headers=headers,
    params={"days": 30}
)
usage = response.json()

print(f"Total 30d: {usage['total_predictions']} predictions")
for model in usage["by_model"]:
    print(f"  {model['model_name']}: {model['predictions']} pred, "
          f"{model['errors']} errors")
```

**Schema `UserUsageResponse`**

```json
{
  "user_id": 3,
  "username": "alice",
  "period_days": 30,
  "total_predictions": 8420,
  "by_model": [
    {"model_name": "iris_model", "predictions": 6200, "errors": 12},
    {"model_name": "fraud_detector", "predictions": 2220, "errors": 3}
  ],
  "by_day": [
    {"date": "2026-04-28", "predictions": 342},
    {"date": "2026-04-27", "predictions": 489}
  ]
}
```

---

## Infrastructure

### `GET /health/dependencies` — Detailed Dependency Health

Checks connectivity and latency of each dependent service. Useful for production diagnosis and orchestrator health checks (K8s readiness probe).

**Auth not required**

```python
response = requests.get(f"{BASE_URL}/health/dependencies")
health = response.json()

print(f"Overall status: {health['status']}")
for service, info in health["dependencies"].items():
    status_icon = "✅" if info["status"] == "ok" else "❌"
    latency = f" ({info.get('latency_ms', '?')}ms)" if "latency_ms" in info else ""
    print(f"  {status_icon} {service}{latency}")
```

**Schema `DependencyHealthResponse`**

```json
{
  "status": "ok",
  "dependencies": {
    "database": {"status": "ok", "latency_ms": 2.1},
    "redis":    {"status": "ok", "latency_ms": 0.4},
    "minio":    {"status": "ok", "latency_ms": 5.8},
    "mlflow":   {"status": "degraded", "latency_ms": null, "error": "Connection timeout"}
  }
}
```

| Status | Meaning |
|---|---|
| `ok` | Service reachable and functional |
| `degraded` | Service reachable but slow or partial error |
| `unavailable` | Service unreachable |

---

### `GET /metrics` — Prometheus Metrics

Exposes API metrics in Prometheus text format. Automatically scraped by Grafana LGTM via the dashboard at `http://localhost:3000`.

**Optional auth** — if `METRICS_TOKEN` is defined in the environment variables, the Bearer token is required.

```python
# Without token (if METRICS_TOKEN not configured)
response = requests.get(f"{BASE_URL}/metrics")
print(response.text[:500])  # Prometheus text format

# With token
response = requests.get(
    f"{BASE_URL}/metrics",
    headers={"Authorization": "Bearer <METRICS_TOKEN>"}
)
```

**Exposed metrics (examples)**

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/predict",status="200"} 18420

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.05",endpoint="/predict"} 17832

# HELP predictions_total Total predictions by model
# TYPE predictions_total counter
predictions_total{model="iris_model",version="2.0.0"} 12540
```

> To configure Grafana scraping, add the endpoint in `prometheus.yml`:
> ```yaml
> - job_name: predictml
>   static_configs:
>     - targets: ['api:8000']
>   metrics_path: /metrics
> ```

---

## Complete Python Client

```python
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List


class PredictMLClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def _get(self, path: str, params: dict = None):
        r = requests.get(f"{self.base_url}{path}", headers=self.headers, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict = None, **kwargs):
        r = requests.post(f"{self.base_url}{path}", headers=self.headers, json=json, **kwargs)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, json: dict):
        r = requests.patch(f"{self.base_url}{path}", headers=self.headers, json=json)
        r.raise_for_status()
        return r.json()

    # ── Predictions ──────────────────────────────────────────────────────────

    def predict(self, model_name: str, features: Dict[str, Any],
                model_version: Optional[str] = None, id_obs: Optional[str] = None) -> dict:
        return self._post("/predict", json={
            "model_name": model_name, "model_version": model_version,
            "id_obs": id_obs, "features": features,
        })

    def predict_batch(self, model_name: str, inputs: List[dict],
                      model_version: Optional[str] = None) -> dict:
        return self._post("/predict-batch", json={
            "model_name": model_name, "model_version": model_version, "inputs": inputs,
        })

    def explain(self, model_name: str, features: Dict[str, Any],
                model_version: Optional[str] = None) -> dict:
        return self._post("/explain", json={
            "model_name": model_name, "model_version": model_version, "features": features,
        })

    def get_predictions(self, model_name: str, days: int = 7,
                        version: Optional[str] = None, limit: int = 100,
                        cursor: Optional[int] = None) -> dict:
        return self._get("/predictions", params={
            "name": model_name,
            "start": (datetime.now() - timedelta(days=days)).isoformat(),
            "end": datetime.now().isoformat(),
            "version": version, "limit": limit, "cursor": cursor,
        })

    def get_stats(self, days: int = 7, model_name: Optional[str] = None) -> dict:
        return self._get("/predictions/stats", params={"days": days, "model_name": model_name})

    # ── Models ───────────────────────────────────────────────────────────────

    def upload_model(self, pkl_path: str, name: str, version: str, **metadata) -> dict:
        with open(pkl_path, "rb") as f:
            files = {"file": (pkl_path, f, "application/octet-stream")}
            data = {"name": name, "version": version,
                    **{k: str(v) for k, v in metadata.items()}}
            r = requests.post(f"{self.base_url}/models", headers=self.headers,
                              files=files, data=data)
            r.raise_for_status()
            return r.json()

    def set_production(self, name: str, version: str) -> dict:
        return self._patch(f"/models/{name}/{version}", json={"is_production": True})

    def configure_ab_test(self, name: str, version: str, traffic_weight: float) -> dict:
        return self._patch(f"/models/{name}/{version}",
                           json={"deployment_mode": "ab_test", "traffic_weight": traffic_weight})

    def set_shadow(self, name: str, version: str) -> dict:
        return self._patch(f"/models/{name}/{version}", json={"deployment_mode": "shadow"})

    def get_performance(self, name: str, start: str, end: str,
                        version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/performance",
                         params={"start": start, "end": end, "version": version})

    def get_drift(self, name: str, days: int = 30, version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/drift", params={"days": days, "version": version})

    def get_ab_compare(self, name: str, days: int = 7) -> dict:
        return self._get(f"/models/{name}/ab-compare", params={"days": days})

    def get_shadow_compare(self, name: str, days: int = 7,
                           shadow_version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/shadow-compare",
                         params={"days": days, "shadow_version": shadow_version})

    def get_output_drift(self, name: str, period_days: int = 7,
                         model_version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/output-drift",
                         params={"period_days": period_days, "model_version": model_version})

    def get_model_card(self, name: str, version: str) -> dict:
        return self._get(f"/models/{name}/{version}/card")

    def get_anomalies(self, model_name: str, days: int = 7,
                      z_threshold: float = 3.0, limit: int = 200) -> dict:
        return self._get("/predictions/anomalies",
                         params={"model_name": model_name, "days": days,
                                 "z_threshold": z_threshold, "limit": limit})

    def retrain(self, name: str, version: str, start_date: str, end_date: str,
                new_version: Optional[str] = None, set_production: bool = False) -> dict:
        return self._post(f"/models/{name}/{version}/retrain", json={
            "start_date": start_date, "end_date": end_date,
            "new_version": new_version, "set_production": set_production,
        })

    # ── Golden Tests ──────────────────────────────────────────────────────────

    def list_golden_tests(self, model_name: str) -> list:
        return self._get(f"/models/{model_name}/golden-tests")

    def create_golden_test(self, model_name: str, input_features: Dict[str, Any],
                           expected_output, description: Optional[str] = None) -> dict:
        return self._post(f"/models/{model_name}/golden-tests", json={
            "input_features": input_features,
            "expected_output": expected_output,
            "description": description,
        })

    def run_golden_tests(self, model_name: str, version: str) -> dict:
        return self._post(f"/models/{model_name}/{version}/run-golden-tests")

    # ── Observed Results ──────────────────────────────────────────────────────

    def submit_observed_results(self, records: List[dict]) -> dict:
        return self._post("/observed-results", json={"data": records})

    # ── Monitoring ────────────────────────────────────────────────────────────

    def get_overview(self, days: int = 7) -> dict:
        return self._get("/monitoring/overview", params={"days": days})

    def get_model_dashboard(self, name: str, days: int = 30) -> dict:
        return self._get(f"/monitoring/model/{name}", params={"days": days})
```
