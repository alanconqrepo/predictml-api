# FAQ — Frequently Asked Questions about PredictML

---

## Installation & Startup

### How do I start PredictML for the first time?

```bash
git clone https://github.com/alanconqrepo/predictml-api.git
cd predictml-api

# Create the .env with required variables
touch .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env
python -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(24))" >> .env
echo "MINIO_ROOT_USER=minioadmin" >> .env
python -c "import secrets; print('MINIO_ROOT_PASSWORD=' + secrets.token_urlsafe(24))" >> .env
echo "GRAFANA_ADMIN_PASSWORD=admin" >> .env

docker-compose up -d --build

# The API runs on 3 replicas — use docker-compose exec
docker-compose exec api python init_data/init_db.py
```

Verify everything is working (the API is accessible via Nginx on port 80):
```bash
curl http://localhost/health
# {"status": "ok", "models_available": 0, "models_cached": 0}
```

The dashboard is at http://localhost:8501.

---

### Containers won't start. What should I do?

```bash
# View detailed logs
docker-compose logs api
docker-compose logs postgres
docker-compose logs nginx
docker-compose logs redis-master

# Check that ports are not already in use
netstat -tlnp | grep -E '80|8501|5433|9000|5000|6379|3000'

# If a required variable is missing from .env, docker-compose will show an error
# (e.g. "Set REDIS_PASSWORD in .env")

# Full rebuild
docker-compose down && docker-compose up -d --build
```

---

### How do I rebuild the Streamlit container after modifying a page?

```bash
docker-compose up -d --build streamlit
```

---

## Authentication & Users

### How do I get my API token?

The default admin token is `<ADMIN_TOKEN>`.

To create a token for another user:
```python
response = requests.post(
    "http://localhost:8000/users",
    headers={"Authorization": "Bearer <ADMIN_TOKEN>"},
    json={"username": "alice", "email": "alice@example.com", "role": "user", "rate_limit": 1000}
)
token = response.json()["api_token"]
```

---

### I lost my admin token. How do I retrieve it?

```bash
docker exec -it predictml-postgres psql -U postgres -d sklearn_api \
  -c "SELECT username, api_token FROM users WHERE role='admin';"
```

---

### How do I renew a token?

```python
# Via the API (admin required)
requests.patch(
    "http://localhost:8000/users/2",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={"regenerate_token": True}
)
```

Via the dashboard: **Users** page → **Regenerate** button on the relevant user.

---

### Why am I getting HTTP 429?

Your daily prediction quota is exhausted. Check:
```python
r = requests.get("http://localhost:8000/users/me/quota",
                 headers={"Authorization": f"Bearer {TOKEN}"})
print(r.json())
# {"daily_limit": 1000, "used_today": 1000, "remaining": 0, "reset_at": "2026-01-16T00:00:00"}
```

Solution: increase the quota via `PATCH /users/{id}` (admin) or wait for the reset at midnight UTC.

---

## Model Management

### How do I upload my first model?

```python
import joblib, requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Train
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Save
joblib.dump(model, "my_model.joblib")

# Upload
with open("my_model.joblib", "rb") as f:
    r = requests.post(
        "http://localhost:8000/models",
        headers={"Authorization": "Bearer <ADMIN_TOKEN>"},
        files={"file": ("my_model.joblib", f, "application/octet-stream")},
        data={
            "name": "my_model", "version": "1.0.0",
            "accuracy": str(accuracy_score(y_test, y_pred)),
            "f1_score": str(f1_score(y_test, y_pred, average="weighted")),
        }
    )
print(r.json())  # {"id": 1, "name": "my_model", "version": "1.0.0", ...}
```

---

### How do I set a model to production?

```python
requests.patch(
    "http://localhost:8000/models/my_model/1.0.0",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"is_production": True}
)
```

Only one model per `name` can be in production at a time. Setting v2.0 to production automatically removes v1.0's production status.

---

### What is the difference between `is_active` and `is_production`?

- `is_active: true` → the model exists and can receive predictions (if explicitly targeted or in A/B/shadow)
- `is_production: true` → the model receives default traffic when `POST /predict` does not specify a version

---

### How do I compare two versions of a model?

```python
# 1. Configure the A/B test
requests.patch("http://localhost:8000/models/my_model/2.0.0",
               headers={"Authorization": f"Bearer {TOKEN}"},
               json={"deployment_mode": "ab_test", "traffic_weight": 0.2})

# 2. After accumulating traffic, compare
r = requests.get("http://localhost:8000/models/my_model/ab-compare",
                 headers={"Authorization": f"Bearer {TOKEN}"})
data = r.json()
sig = data.get("ab_significance")
if sig and sig["significant"]:
    print(f"Winner: {sig['winner']} (p={sig['p_value']:.4f})")
```

---

### How do I detect data drift?

1. First, configure a baseline (feature stats at training time):
```python
requests.patch(
    "http://localhost:8000/models/my_model/1.0.0",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"feature_baseline": {
        "age":    {"mean": 35.5, "std": 10.2, "min": 18, "max": 80},
        "income": {"mean": 45000, "std": 15000, "min": 10000, "max": 200000},
    }}
)
```

2. After accumulating production predictions, consult the drift:
```python
r = requests.get("http://localhost:8000/models/my_model/drift",
                 headers={"Authorization": f"Bearer {TOKEN}"},
                 params={"days": 7})
```

3. Via the dashboard: **Supervision** page → Drift section.

---

## Predictions

### How do I make a prediction?

```python
r = requests.post(
    "http://localhost:8000/predict",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "model_name": "my_model",
        "id_obs":     "obs-001",
        "features":   {"age": 35, "income": 50000, "score": 720}
    }
)
print(r.json())
# {"prediction": 1, "probability": [0.12, 0.88], "low_confidence": false, ...}
```

---

### What is `id_obs`?

An identifier of your choice to link the prediction to an observed result later. Example: customer ID, case number, session UUID.

If you don't need it, omit it (the value will be `null`).

---

### How do I record the real result (feedback)?

```python
requests.post(
    "http://localhost:8000/observed-results",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"data": [
        {"id_obs": "obs-001", "model_name": "my_model",
         "date_time": "2026-01-15T10:00:00", "observed_result": 1}
    ]}
)
```

This allows computing the real-world performance of the model (accuracy, F1, confusion matrix).

---

### How do I make batch predictions?

```python
r = requests.post(
    "http://localhost:8000/predict-batch",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "model_name": "my_model",
        "inputs": [
            {"id_obs": "obs-001", "features": {"age": 35, "income": 50000}},
            {"id_obs": "obs-002", "features": {"age": 28, "income": 35000}},
            {"id_obs": "obs-003", "features": {"age": 52, "income": 85000}},
        ]
    }
)
for item in r.json()["predictions"]:
    print(f"{item['id_obs']} → {item['prediction']}")
```

More efficient than individual calls: the model is loaded only once.

---

### How do I get a SHAP explanation?

```python
# Explanation of a single prediction
r = requests.post(
    "http://localhost:8000/explain",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"model_name": "my_model", "features": {"age": 35, "income": 50000}}
)
for feat, val in r.json()["shap_values"].items():
    print(f"  {feat}: {val:+.4f}")

# Inline SHAP in /predict
r = requests.post(
    "http://localhost:8000/predict",
    headers={"Authorization": f"Bearer {TOKEN}"},
    params={"explain": "true"},
    json={"model_name": "my_model", "features": {"age": 35, "income": 50000}}
)
print(r.json()["explanation"]["shap_values"])
```

---

## Retraining

### My train.py is rejected at upload. Why?

The API checks that your script:
1. Has valid Python syntax
2. Reads `os.environ["TRAIN_START_DATE"]`
3. Reads `os.environ["TRAIN_END_DATE"]`
4. Reads `os.environ["OUTPUT_MODEL_PATH"]`
5. Calls `joblib.dump()` or `save_model()`

Check that all 5 elements are present.

---

### Retraining fails with "No data for date range". What should I do?

Your script cannot find data for the requested date range. Solutions:
1. Check your data source (CSV, DB) and the date filtering
2. Test manually: `TRAIN_START_DATE=2025-01-01 TRAIN_END_DATE=2025-12-31 OUTPUT_MODEL_PATH=/tmp/test.joblib python train.py`
3. Add a check in your script and exit with `sys.exit(1)` + error JSON message if no data

---

### How do I view the logs of the last retrain?

Via the API:
```python
r = requests.get("http://localhost:8000/models/my_model/retrain-history",
                 headers={"Authorization": f"Bearer {TOKEN}"})
for entry in r.json():
    print(f"{entry['trained_at']} v{entry['source_version']} → v{entry['new_version']}")
```

Via the dashboard: **Retrain** page → **History** tab.

---

## Streamlit Dashboard

### I can't see all menus in the sidebar. Why?

Some pages (notably **Users** and admin actions on other pages) are only visible to users with the `admin` role. Log in with an admin token.

---

### How do I export my predictions?

**Predictions** page → **Export** button → choose the format (CSV, JSONL, Parquet).

Or via the API:
```python
r = requests.get(
    "http://localhost:8000/predictions/export",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    params={"model_name": "my_model", "format": "csv"}
)
with open("export.csv", "wb") as f:
    f.write(r.content)
```

---

### What does the "What-If Explorer" on the Models page do?

It lets you modify feature values with sliders and see the prediction change in real time, without creating a logged prediction. Useful for understanding the model's decision boundaries.

---

## Important Environment Variables

**Required** (docker-compose or the API refuse to start if missing):

| Variable | Description |
|---|---|
| `SECRET_KEY` | HMAC key for model signing. `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `REDIS_PASSWORD` | Redis password (master + replicas + sentinels) |
| `MINIO_ROOT_USER` | MinIO login |
| `MINIO_ROOT_PASSWORD` | MinIO password |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password |

**Optional**:

| Variable | Description | Default |
|---|---|---|
| `ADMIN_TOKEN` | Custom admin token | auto-generated |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | PostgreSQL credentials | `postgres` / `postgres` |
| `REDIS_CACHE_TTL` | Model cache TTL in seconds | `3600` |
| `ENABLE_OTEL` | Enable OpenTelemetry to Grafana | `true` |
| `ENABLE_EMAIL_ALERTS` | Enable email alerts | `false` |
| `PREDICTION_STREAM_ENABLED` | Async Redis Streams queue for predictions | `true` |
| `MAX_ROWS_ANALYTICS` | Row limit per analytical query | `50000` |
| `ANALYTICS_MAX_DAYS` | Max aggregation window | `90` |
| `TOKEN_LIFETIME_DAYS` | Bearer token validity in days | `90` |
| `ANTHROPIC_API_KEY` | Key for the Claude chatbot (Help page) | `` |

These variables are defined in the `.env` file at the project root:
```bash
SECRET_KEY=<generate with secrets.token_urlsafe(32)>
REDIS_PASSWORD=<generate with secrets.token_urlsafe(24)>
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=<generate with secrets.token_urlsafe(24)>
GRAFANA_ADMIN_PASSWORD=admin
```

---

## HTTP Error Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 201 | Resource created (new model, user) |
| 400 | Invalid request (missing or incorrect parameter) |
| 401 | Token absent or invalid |
| 403 | Insufficient role (admin required) |
| 404 | Resource not found (model, user) |
| 409 | Conflict (model name+version already exists) |
| 410 | Gone — deprecated model (predictions blocked) |
| 422 | Pydantic validation error (feature schema) |
| 429 | Daily quota exhausted |
| 500 | Internal server error |
