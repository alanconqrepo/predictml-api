# Beginner Guide — PredictML API

This guide is intended for someone discovering PredictML API for the first time. It explains the concept, the architecture, and walks step-by-step through a complete ML workflow in Python.

---

## What is PredictML API?

PredictML API is a platform that lets you **put a Machine Learning model into production in just a few minutes**.

The problem it solves: you have trained a scikit-learn model on your computer. How do you make it usable by a web application? How do you version it, trace every prediction, measure its real-world accuracy, and update it? PredictML API answers all of this.

**What you do:**
1. Train a model locally (`model.fit(...)`)
2. Save it as `.joblib` (`joblib.dump(...)`)
3. Upload it via the API
4. Make predictions via HTTP from any application

**What the API handles for you:**
- Storing the model in MinIO (S3-compatible object storage)
- Versioning models (v1.0, v2.0…)
- Logging every prediction with its features, result and latency
- Computing real-world metrics when you send observed results
- Detecting whether production data drifts from training data
- Explaining why the model made a given prediction (SHAP)
- Silently testing a new version before putting it into production (shadow)

---

## Architecture Explained Simply

```
Your Python script  ──upload .joblib──▶  PredictML API  ──stores──▶  MinIO
                                            │
                                            │  ──logs──▶  PostgreSQL
                                            │
Your application    ──POST /predict──▶  API  ──returns──▶  { prediction: 0 }
```

The API runs in Docker with 7 services:
- **FastAPI** (port 8000) — the main API
- **PostgreSQL** (port 5433) — stores predictions and metadata
- **MinIO** (port 9000) — stores `.joblib` files
- **Redis** (port 6379) — caches models in memory for fast predictions
- **MLflow** (port 5000) — optional experiment tracking
- **Streamlit** (port 8501) — administration dashboard
- **Grafana** (port 3000) — observability (logs, traces, metrics)

---

## Installation

```bash
# Prerequisites: Git + Docker Desktop
git clone https://github.com/alanconqrepo/predictml-api.git
cd predictml-api

# Generate secrets in .env (open a Git Bash terminal)
bash scripts/init_env.sh

# (Optional) Start from scratch — removes existing Postgres volumes
# Useful if you had already launched the project with a different password
docker-compose -p predictml-api down -v 2>&1 && echo "=== Volumes deleted ==="

# Start all services
docker-compose -p predictml-api up -d --build

# Verify (via Nginx port 80)
curl http://localhost/health
# {"status": "healthy", "database": "connected", "models_available": 0, "models_cached": 0}
```

> The admin user is created automatically at startup if `ADMIN_TOKEN` is defined in `.env`.
> The admin token is the value of `ADMIN_TOKEN` in your `.env` file.

Install Python dependencies for the examples below:

```bash
pip install requests scikit-learn pandas numpy shap
```

---

## Complete Tutorial — Iris Classifier

### Step 1: Train and Save a Model

```python
# 1_train.py
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load the Iris dataset (150 flowers, 4 features, 3 classes)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

X = df[iris.feature_names]
y = df["target"]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Save
joblib.dump(model, "iris_model.joblib")
print("Model saved to iris_model.joblib")
```

```bash
python 1_train.py
# Accuracy: 1.0000
# F1 Score: 1.0000
# Model saved to iris_model.joblib
```

### Step 2: Upload the Model

```python
# 2_upload.py
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "<ADMIN_TOKEN>"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

with open("iris_model.joblib", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/models",
        headers=HEADERS,
        files={"file": ("iris_model.joblib", f, "application/octet-stream")},
        data={
            "name": "iris_model",
            "version": "1.0.0",
            "description": "Iris Classifier — RandomForest 100 trees",
            "algorithm": "RandomForestClassifier",
            "accuracy": "1.0",
            "f1_score": "1.0",
            "features_count": "4",
            "classes": '["setosa", "versicolor", "virginica"]',
        },
    )

print(f"Status: {response.status_code}")   # 201
model = response.json()
print(f"ID: {model['id']}, version: {model['version']}")
```

### Step 3: Set to Production

```python
# 3_set_production.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0",
    headers=HEADERS,
    json={"is_production": True}
)
print(f"In production: {response.json()['is_production']}")  # True
```

### Step 4: Make a Prediction

```python
# 4_predict.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

# A flower to identify
response = requests.post(
    f"{BASE_URL}/predict",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-001",          # identifier to link to the real result later
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)

result = response.json()
classes = ["setosa", "versicolor", "virginica"]
print(f"Prediction: {classes[result['prediction']]} (class {result['prediction']})")
print(f"Probabilities: {result['probability']}")
# Prediction: setosa (class 0)
# Probabilities: [0.97, 0.02, 0.01]
```

### Step 5: Batch Predictions

More efficient than individual calls: the model is loaded only once.

```python
# 5_batch.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

observations = [
    {"id_obs": "obs-001", "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}},
    {"id_obs": "obs-002", "features": {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}},
    {"id_obs": "obs-003", "features": {"sepal length (cm)": 5.8, "sepal width (cm)": 2.7, "petal length (cm)": 4.1, "petal width (cm)": 1.0}},
]

response = requests.post(
    f"{BASE_URL}/predict-batch",
    headers=HEADERS,
    json={"model_name": "iris_model", "inputs": observations}
)

classes = ["setosa", "versicolor", "virginica"]
for item in response.json()["predictions"]:
    print(f"  {item['id_obs']} → {classes[item['prediction']]} "
          f"(conf: {max(item['probability']):.0%})")
# obs-001 → setosa (conf: 97%)
# obs-002 → virginica (conf: 94%)
# obs-003 → versicolor (conf: 71%)
```

### Step 6: SHAP Explanation

Understand why the model made this prediction.

```python
# 6_explain.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

response = requests.post(
    f"{BASE_URL}/explain",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
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
print(f"Base value: {data['base_value']:.4f}")
print("\nSHAP contributions (+ = towards predicted class, - = away from it):")
for feat, val in sorted(data["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
    bar = "█" * int(abs(val) * 5) if abs(val) > 0.05 else "·"
    print(f"  {'↑' if val > 0 else '↓'} {feat:<25} {val:+.4f}  {bar}")
```

### Step 7: Record Observed Results

After obtaining the true result, send it to compute real-world performance.

```python
# 7_feedback.py
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

# The true results (what you observed after the prediction)
observed = [
    {"id_obs": "obs-001", "model_name": "iris_model",
     "date_time": datetime.now().isoformat(), "observed_result": 0},   # setosa ✓
    {"id_obs": "obs-002", "model_name": "iris_model",
     "date_time": datetime.now().isoformat(), "observed_result": 2},   # virginica ✓
    {"id_obs": "obs-003", "model_name": "iris_model",
     "date_time": datetime.now().isoformat(), "observed_result": 1},   # versicolor ✓
]

response = requests.post(
    f"{BASE_URL}/observed-results",
    headers=HEADERS,
    json={"data": observed}
)
print(response.json())  # {"upserted": 3}
```

### Step 8: Check Real-World Performance

```python
# 8_performance.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

response = requests.get(
    f"{BASE_URL}/models/iris_model/performance",
    headers=HEADERS,
    params={"start": "2025-01-01T00:00:00", "end": "2025-12-31T23:59:59"}
)

data = response.json()
print(f"Total predictions: {data['total_predictions']}")
print(f"With real result: {data['matched_predictions']}")
print(f"Real accuracy: {data.get('accuracy', 'N/A')}")
print(f"Real F1: {data.get('f1_weighted', 'N/A')}")
```

### Step 9: Detect Data Drift

First configure a baseline (feature stats at training time):

```python
# 9_drift.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

# 1. Define the baseline (feature stats at training time)
requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0",
    headers=HEADERS,
    json={
        "feature_baseline": {
            "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9},
            "sepal width (cm)":  {"mean": 3.05, "std": 0.43, "min": 2.0, "max": 4.4},
            "petal length (cm)": {"mean": 3.76, "std": 1.76, "min": 1.0, "max": 6.9},
            "petal width (cm)":  {"mean": 1.20, "std": 0.76, "min": 0.1, "max": 2.5},
        }
    }
)

# 2. Consult the drift report (after predictions have accumulated)
response = requests.get(
    f"{BASE_URL}/models/iris_model/drift",
    headers=HEADERS,
    params={"days": 30}
)
data = response.json()
print(f"Drift summary: {data['drift_summary']}")
for feat, info in data["features"].items():
    status = info["drift_status"]
    z = info.get("z_score")
    print(f"  {feat:<25} {status:<15} z={z:.2f}" if z else f"  {feat:<25} {status}")
```

**Status interpretation:**
- `ok` — no drift detected
- `warning` — moderate drift, monitor closely
- `critical` — strong drift, retraining probably required
- `no_baseline` — you have not yet configured a baseline

---

## Advanced Features

### A/B Testing

Test a new version on 20% of traffic without risk:

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

# 1. Upload the new version
with open("iris_model_v2.joblib", "rb") as f:
    requests.post(f"{BASE_URL}/models", headers=HEADERS,
                  files={"file": ("iris_model_v2.joblib", f, "application/octet-stream")},
                  data={"name": "iris_model", "version": "2.0.0"})

# 2. Configure the A/B test (20% of traffic to v2)
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=HEADERS,
               json={"deployment_mode": "ab_test", "traffic_weight": 0.2})

# 3. Normal predictions will be routed automatically (80% v1 / 20% v2)
response = requests.post(f"{BASE_URL}/predict", headers=HEADERS,
                         json={"model_name": "iris_model",
                               "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                                            "petal length (cm)": 1.4, "petal width (cm)": 0.2}})
print(f"Version used: {response.json()['selected_version']}")

# 4. Compare results after a few days
compare = requests.get(f"{BASE_URL}/models/iris_model/ab-compare",
                       headers=HEADERS, params={"days": 7})
data = compare.json()

for v in data["versions"]:
    print(f"  v{v['version']}: {v['total_predictions']} preds, error={v['error_rate']:.1%}")

# 5. Interpret statistical significance
sig = data.get("ab_significance")
if sig:
    if sig["significant"]:
        print(f"\n✅ Statistically significant difference (p={sig['p_value']:.4f})")
        print(f"   Winner: {sig['winner']} — based on {sig['metric']}")
    else:
        print(f"\n⚠️  NON-significant difference (p={sig['p_value']:.4f})")
        print(f"   Need ~{sig['min_samples_needed']} observations/version to conclude")
        print("   Do not promote yet — accumulate more data")
```

### Shadow Deployment

Silently test a new version without ever exposing it to clients:

```python
# v2 receives the same inputs as v1 in the background
# but its results are never returned to clients
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=HEADERS,
               json={"deployment_mode": "shadow"})

# After accumulation, compare concordance rates
compare = requests.get(f"{BASE_URL}/models/iris_model/ab-compare", headers=HEADERS)
for v in compare.json()["versions"]:
    if v["deployment_mode"] == "shadow":
        print(f"Shadow vs prod concordance: {v['agreement_rate']:.1%}")
```

### Confidence Threshold

Mark uncertain predictions with `low_confidence: true`:

```python
# Configure a confidence threshold of 80%
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=HEADERS,
               json={"confidence_threshold": 0.8})

# Predictions with max probability < 80% will have low_confidence: true
result = requests.post(f"{BASE_URL}/predict", headers=HEADERS,
                       json={"model_name": "iris_model",
                             "features": {"sepal length (cm)": 5.8, "sepal width (cm)": 2.7,
                                          "petal length (cm)": 4.1, "petal width (cm)": 1.0}})
data = result.json()
if data.get("low_confidence"):
    print("⚠ Uncertain prediction — manual review recommended")
```

### Automatic Retraining

If you provided a `train.py` script at upload, you can trigger a retrain:

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/retrain",
    headers=HEADERS,
    json={
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "new_version": "1.1.0",
        "set_production": False
    }
)
data = response.json()
if data["success"]:
    print(f"New version created: {data['new_version']}")
    print(data["stdout"][-500:])   # last logs
else:
    print(f"Error: {data['error']}")
```

### Inline SHAP Explanation on /predict

Get SHAP values directly in the prediction response without an extra call:

```python
response = requests.post(
    f"{BASE_URL}/predict",
    headers=HEADERS,
    params={"explain": "true"},          # query parameter
    json={
        "model_name": "iris_model",
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
# The response contains an "explanation" field with SHAP values
if data.get("explanation"):
    for feat, val in data["explanation"]["shap_values"].items():
        print(f"  {feat}: {val:+.4f}")
```

### Look Up a Prediction by ID

Retrieve the detail of a past prediction (features, result, latency) from its internal identifier:

```python
prediction_id = 42   # ID returned by /predict or found in /predictions

response = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}",
    headers=HEADERS
)
data = response.json()
print(f"Model: {data['model_name']} v{data['model_version']}")
print(f"Result: {data['prediction_result']}")
print(f"Latency: {data['response_time_ms']} ms")

# Retrieve the post-hoc SHAP explanation (if the model supports SHAP)
explain = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}/explain",
    headers=HEADERS
)
print(explain.json()["shap_values"])
```

### CSV Import of Observed Results

Rather than sending results one by one, import them in bulk from a CSV file:

```python
import io
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

# Expected format: columns id_obs, model_name, date_time, observed_result
csv_content = """id_obs,model_name,date_time,observed_result
obs-001,iris_model,2026-01-15T10:00:00,0
obs-002,iris_model,2026-01-15T10:01:00,2
obs-003,iris_model,2026-01-15T10:02:00,1
"""

response = requests.post(
    f"{BASE_URL}/observed-results/upload-csv",
    headers=HEADERS,
    files={"file": ("results.csv", io.StringIO(csv_content), "text/csv")}
)
print(response.json())  # {"upserted": 3, "errors": []}
```

You can also export observed results and check coverage:

```python
# Coverage statistics (how many predictions have an observed result)
stats = requests.get(f"{BASE_URL}/observed-results/stats", headers=HEADERS,
                     params={"model_name": "iris_model"})
data = stats.json()
print(f"Predictions with ground truth: {data['matched_count']} / {data['total_predictions']}")
print(f"Coverage: {data['coverage_rate']:.1%}")

# CSV export of observed results
export = requests.get(f"{BASE_URL}/observed-results/export", headers=HEADERS,
                      params={"model_name": "iris_model", "format": "csv"})
with open("observed_results_export.csv", "wb") as f:
    f.write(export.content)
```

### Webhooks

Receive a notification after each prediction:

```python
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=HEADERS,
               json={"webhook_url": "https://your-app.com/hooks/predictions"})
# The API will send a JSON POST after each prediction on this model
```

---

## Streamlit Dashboard

Open **http://localhost:8501** and log in with the admin token.

**Available pages:**

| Page | Features |
|---|---|
| Home | System overview, links to services |
| Users | Create/deactivate accounts, renew tokens, usage analytics by model |
| Models | Details, set to production, What-if Explorer (sliders → live prediction), confusion matrix, SHAP feature importance, schema validation, .joblib download, confidence threshold, MLflow link |
| Predictions | History filterable by model, date, version; inline observed result submission; CSV/JSONL/Parquet export; GDPR purge |
| Stats | Volume, response time, distribution charts; leaderboard; accuracy vs latency P95 scatter plot |
| Code Example | Dynamically generated Python, curl/bash and JavaScript examples |
| A/B Testing | Shadow mode, statistical comparison, one-click winner promotion |
| Supervision | Global monitoring, drift, alerts, per-model threshold configuration, CSV/Markdown report export |
| Retrain | Schedule, trigger and monitor retrains; history with metrics; feature importance delta before/after retrain |

---

## User Management

```python
import requests

BASE_URL = "http://localhost:8000"
ADMIN_HEADERS = {"Authorization": "Bearer <ADMIN_TOKEN>"}

# Create a user for your application
response = requests.post(
    f"{BASE_URL}/users",
    headers=ADMIN_HEADERS,
    json={
        "username": "my_app",
        "email": "app@example.com",
        "role": "user",
        "rate_limit": 10000    # 10,000 predictions/day
    }
)
user = response.json()
app_token = user["api_token"]
print(f"App token: {app_token}")  # Store securely

# Use the app token for predictions
app_headers = {"Authorization": f"Bearer {app_token}"}
result = requests.post(f"{BASE_URL}/predict", headers=app_headers,
                       json={"model_name": "iris_model",
                             "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                                          "petal length (cm)": 1.4, "petal width (cm)": 0.2}})
```

**Roles:**
- `admin` — full access (user management, models, everything)
- `user` — can make predictions and view their own data
- `readonly` — can only read (no predictions)

---

## Summary of Important URLs

| Service | URL |
|---|---|
| API (via Nginx) | http://localhost |
| Interactive documentation | http://localhost/docs |
| Admin dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| MinIO (file management) | http://localhost:9001 |
| Grafana (observability) | http://localhost:3000 |

---

## Next Steps

- [QUICKSTART.md](QUICKSTART.md) — concise summary of essential commands
- [API_REFERENCE.md](API_REFERENCE.md) — complete reference of all endpoints
- [ARCHITECTURE.md](ARCHITECTURE.md) — how services interact
- `http://localhost:8000/docs` — interactive Swagger interface to test the API live
