"""
Full code example: MLflow + API
"""

import os

import streamlit as st
from utils.auth import require_auth
from utils.i18n import t

st.set_page_config(page_title=t("code_example.page_title"), page_icon="💡", layout="wide")
require_auth()

st.title(t("code_example.title"))
st.caption(t("code_example.intro"))

API_URL = st.session_state.get("api_url", "http://localhost:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")

# ── Mode toggle ──────────────────────────────────────────────────────────────
mode = st.radio(
    t("code_example.mode_label"),
    options=[t("code_example.mode_minimal"), t("code_example.mode_complete")],
    horizontal=True,
    help=t("code_example.mode_help"),
)
is_complete = mode == t("code_example.mode_complete")

st.divider()

tab_python, tab_curl, tab_js = st.tabs([
    t("code_example.tab_python"),
    t("code_example.tab_curl"),
    t("code_example.tab_js"),
])

# ============================================================
# TAB PYTHON
# ============================================================
with tab_python:
    # SECTION 1 — Train and track with MLflow
    with st.expander(t("code_example.python.section1_title"), expanded=False):
        st.markdown(t("code_example.python.section1_install"))
        if not is_complete:
            code_mlflow = f"""\
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

MLFLOW_TRACKING_URI = "{MLFLOW_URL}"
MODEL_NAME = "iris-classifier"
MODEL_VERSION = "1.0.0"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris_classification")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    params = {{"n_estimators": 100, "max_depth": 5, "random_state": 42}}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

joblib.dump(model, f"{{MODEL_NAME}}_v{{MODEL_VERSION}}.joblib")
print(f"Accuracy: {{acc:.4f}} | F1: {{f1:.4f}} | MLflow run: {{run_id}}")
"""
        else:
            code_mlflow = f"""\
# Production-grade train.py — mirrors documentation/Scripts/iris/train_iris.py
# This script can be uploaded as train_file when creating a model to enable
# automatic / scheduled retraining directly from the dashboard.
import json, os, sys, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import mlflow, mlflow.sklearn
    _MLFLOW_OK = True
except ImportError:
    _MLFLOW_OK = False

# ── 1. Environment variables injected by the API during automated retraining ──
TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE", "2025-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE",   "2025-12-31")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "iris-classifier.joblib")
MODEL_NAME        = os.environ.get("MODEL_NAME", "iris-classifier")
MLFLOW_URI        = os.environ.get("MLFLOW_TRACKING_URI", "{MLFLOW_URL}")

# ── 2. Data — replace with your own source filtered by date range ─────────────
iris = load_iris()
X_full = pd.DataFrame(iris.data, columns=iris.feature_names)
y_full = iris.target

start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end   = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
n_samples = min(len(X_full), max(30, (end - start).days * 2))
idx = np.random.default_rng(42).choice(len(X_full), n_samples, replace=False)
X, y = X_full.iloc[idx], y_full[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if n_samples >= 30 else None,
)

# ── 3. Training ────────────────────────────────────────────────────────────────
HYPERPARAMS = {{
    "n_estimators": 200, "max_depth": 10, "min_samples_split": 4,
    "min_samples_leaf": 2, "max_features": "sqrt",
    "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
}}
model = RandomForestClassifier(**HYPERPARAMS)
model.fit(X_train, y_train)

# ── 4. Evaluation ──────────────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
acc       = float(accuracy_score(y_test, y_pred))
f1        = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
recall    = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

# ── 5. Save model at OUTPUT_MODEL_PATH (REQUIRED) ─────────────────────────────
joblib.dump(model, OUTPUT_MODEL_PATH)

# ── 6. Feature stats for drift detection ──────────────────────────────────────
feature_stats = {{
    name: {{
        "mean": round(float(X_train[name].mean()), 4),
        "std":  round(float(X_train[name].std()),  4),
        "min":  round(float(X_train[name].min()),  4),
        "max":  round(float(X_train[name].max()),  4),
        "null_rate": 0.0,
    }}
    for name in iris.feature_names
}}

label_distribution = {{
    str(int(cls)): round(float(np.sum(y_train == cls)) / len(y_train), 4)
    for cls in np.unique(y_train)
}}

# ── 7. Feature importances ──────────────────────────────────────────────────────
_imp = model.feature_importances_
feature_importances = dict(sorted(
    {{n: round(float(v), 6) for n, v in zip(iris.feature_names, _imp / _imp.sum())}}.items(),
    key=lambda kv: kv[1], reverse=True
))

# ── 8. Library versions → requirements.txt stored in MinIO at upload ──────────
import importlib.metadata as _imeta
dependencies = {{}}
for pkg in ["scikit-learn", "numpy", "pandas", "mlflow"]:
    try:
        dependencies[pkg] = _imeta.version(pkg)
    except _imeta.PackageNotFoundError:
        pass

# ── 9. MLflow tracking (graceful degradation if unreachable) ──────────────────
mlflow_run_id = None
if _MLFLOW_OK and MLFLOW_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(f"predictml/{{MODEL_NAME}}")
        with mlflow.start_run(run_name=f"{{MODEL_NAME}}_{{TRAIN_START_DATE}}_{{TRAIN_END_DATE}}") as run:
            mlflow.log_params({{**HYPERPARAMS, "train_start_date": TRAIN_START_DATE, "train_end_date": TRAIN_END_DATE}})
            mlflow.log_metrics({{"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}})
            mlflow.set_tags({{"model_name": MODEL_NAME, "algorithm": "RandomForest"}})
            mlflow.sklearn.log_model(model, "model")
            mlflow_run_id = run.info.run_id
        print(f"[MLflow] run_id={{mlflow_run_id}}", file=sys.stderr)
    except Exception as e:
        print(f"[MLflow] unavailable — continuing: {{e}}", file=sys.stderr)

# ── 10. JSON stdout — LAST LINE (parsed by the API during automated retraining)
output = {{
    "accuracy":             round(acc, 4),
    "f1_score":             round(f1, 4),
    "precision":            round(precision, 4),
    "recall":               round(recall, 4),
    "n_rows":               len(X_train),
    "features_count":       len(iris.feature_names),
    "classes":              list(iris.target_names),
    "hyperparameters":      HYPERPARAMS,
    "confidence_threshold": 0.60,       # low_confidence=True when max(proba) < 0.60
    "feature_importances":  feature_importances,
    "feature_stats":        feature_stats,
    "label_distribution":   label_distribution,
    "dependencies":         dependencies,
    "training_dataset":     "sklearn Iris (150 obs, 3 classes)",
}}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

print(json.dumps(output))   # MUST be the very last print
"""
        st.code(code_mlflow, language="python")

    # SECTION 2 — Upload via the API
    with st.expander(t("code_example.python.section2_title"), expanded=False):
        if not is_complete:
            code_upload = f"""\
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "{API_URL}"
API_TOKEN = os.getenv("PREDICTML_API_TOKEN")
MODEL_NAME = "iris-classifier"
MODEL_VERSION = "1.0.0"

headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

with open(f"{{MODEL_NAME}}_v{{MODEL_VERSION}}.joblib", "rb") as f:
    response = requests.post(
        f"{{API_URL}}/models",
        headers=headers,
        files={{"file": (f"{{MODEL_NAME}}.joblib", f, "application/octet-stream")}},
        data={{
            "name":          MODEL_NAME,
            "version":       MODEL_VERSION,
            "description":   "Random Forest on Iris dataset",
            "algorithm":     "RandomForestClassifier",
            "accuracy":      str(acc),       # from step 1
            "f1_score":      str(f1),        # from step 1
            "mlflow_run_id": run_id,         # from step 1
            "features_count": "4",
            "classes":       "[0, 1, 2]",
        }},
    )

response.raise_for_status()
res = response.json()
print(f"Uploaded: {{res['name']}} v{{res['version']}} (id={{res['id']}})")

# Set to production
requests.patch(
    f"{{API_URL}}/models/{{MODEL_NAME}}/{{MODEL_VERSION}}",
    headers={{**headers, "Content-Type": "application/json"}},
    json={{"is_production": True}},
).raise_for_status()
print("Version set to production.")
"""
        else:
            code_upload = f"""\
import json, os, requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "{API_URL}"
API_TOKEN = os.getenv("PREDICTML_API_TOKEN")
MODEL_NAME = "iris-classifier"
MODEL_VERSION = "1.0.0"

headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

# Variables from step 1 (acc, f1, precision, recall, mlflow_run_id,
# HYPERPARAMS, feature_stats, feature_importances, label_distribution, dependencies)

with open("iris-classifier.joblib", "rb") as model_fh, \\
     open("train_iris.py", "rb") as train_fh:
    # train_file enables automatic / scheduled retraining from the dashboard
    files = {{
        "file":       ("iris-classifier.joblib", model_fh, "application/octet-stream"),
        "train_file": ("train_iris.py",           train_fh, "text/plain"),
    }}
    data = {{
        "name":             MODEL_NAME,
        "version":          MODEL_VERSION,
        "description":      "RandomForest on Iris — production example",
        "algorithm":        "RandomForest",
        "accuracy":         str(round(acc, 4)),
        "f1_score":         str(round(f1, 4)),
        "features_count":   "4",
        "classes":          json.dumps(list(iris.target_names)),
        "mlflow_run_id":    mlflow_run_id or "",
        # All training metrics stored for display and regression tracking
        "training_metrics": json.dumps({{
            "accuracy": round(acc, 4), "f1_score": round(f1, 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
        }}),
        # Hyperparameters shown in the dashboard and propagated to retrained versions
        "hyperparameters":    json.dumps(HYPERPARAMS),
        # Feature stats power the drift detection charts
        "feature_baseline":   json.dumps(feature_stats),
        # Local library versions → requirements.txt stored in MinIO
        "local_dependencies": json.dumps(dependencies),
        "training_dataset":   "sklearn Iris (150 obs, 3 classes)",
        "tags":               json.dumps(["Example", "iris", "classification"]),
        # Webhook POSTed after every prediction (model_name, id_obs, prediction, probability…)
        "webhook_url":        "https://webhook.site/00000000-0000-0000-0000-000000000000",
    }}
    response = requests.post(
        f"{{API_URL}}/models", headers=headers, files=files, data=data, timeout=180
    )

response.raise_for_status()
res = response.json()
print(f"Uploaded: {{res['name']}} v{{res['version']}} (id={{res['id']}})")
if res.get("requirements_object_key"):
    print(f"  requirements.txt → MinIO: {{res['requirements_object_key']}}")

# ── PATCH: production + A/B routing + confidence threshold ────────────────────
requests.patch(
    f"{{API_URL}}/models/{{MODEL_NAME}}/{{MODEL_VERSION}}",
    headers={{**headers, "Content-Type": "application/json"}},
    json={{
        "is_production":        True,
        "deployment_mode":      "ab_test",   # "production" | "ab_test" | "shadow"
        "traffic_weight":       0.5,         # 50 % of traffic in A/B mode
        "confidence_threshold": 0.60,        # low_confidence=True when max(proba) < 0.60
        "feature_importances":  feature_importances,
        "training_stats": {{
            "label_distribution": label_distribution,
            "n_rows":             len(X_train),
            "train_start_date":   TRAIN_START_DATE,
            "train_end_date":     TRAIN_END_DATE,
        }},
    }},
    timeout=30,
).raise_for_status()
print("Set to production (ab_test, 50 % traffic, confidence_threshold=0.60).")
"""
        st.code(code_upload, language="python")

    # SECTION 3 — Make a prediction
    with st.expander(t("code_example.python.section3_title"), expanded=False):
        if not is_complete:
            code_predict = f"""\
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "{API_URL}"
API_TOKEN = os.getenv("PREDICTML_API_TOKEN")
headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

payload = {{
    "model_name": "iris-classifier",
    "features": {{
        "sepal length (cm)": 5.1,
        "sepal width (cm)":  3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)":  0.2,
    }},
    "id_obs": "obs_001",   # optional — join key for /observed-results
}}

response = requests.post(f"{{API_URL}}/predict", headers=headers, json=payload)
response.raise_for_status()

result = response.json()
print(f"Prediction  : {{result['prediction']}}")
print(f"Probabilities: {{result.get('probability')}}")
"""
        else:
            code_predict = f"""\
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "{API_URL}"
API_TOKEN = os.getenv("PREDICTML_API_TOKEN")
headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

# ── Single prediction — specific version + SHAP explanation ──────────────────
payload = {{
    "model_name":    "iris-classifier",
    "model_version": "1.0.0",              # omit → uses is_production=True version
    "id_obs":        "obs_001",            # join key for /observed-results
    "timestamp":     "2025-06-15T10:30:00",  # optional — inject a historical timestamp
    "features": {{
        "sepal length (cm)": 5.1,
        "sepal width (cm)":  3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)":  0.2,
    }},
}}

# ?explain=true returns SHAP values per feature
result = requests.post(
    f"{{API_URL}}/predict?explain=true", headers=headers, json=payload
).raise_for_status() or requests.post(
    f"{{API_URL}}/predict?explain=true", headers=headers, json=payload
).json()

response = requests.post(f"{{API_URL}}/predict?explain=true", headers=headers, json=payload)
response.raise_for_status()
result = response.json()
print(f"Prediction  : {{result['prediction']}}")
print(f"Probabilities: {{result.get('probability')}}")
print(f"SHAP values : {{result.get('shap_values')}}")
if result.get("low_confidence"):
    print("Warning: max proba < confidence_threshold")

# ── Batch prediction ──────────────────────────────────────────────────────────
batch_payload = {{
    "model_name": "iris-classifier",
    "inputs": [
        {{"id_obs": "obs_001", "features": {{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}}}},
        {{"id_obs": "obs_002", "features": {{"sepal length (cm)": 6.3, "sepal width (cm)": 2.5, "petal length (cm)": 5.0, "petal width (cm)": 1.9}}}},
        {{"id_obs": "obs_003", "features": {{"sepal length (cm)": 7.0, "sepal width (cm)": 3.2, "petal length (cm)": 4.7, "petal width (cm)": 1.4}}}},
    ],
}}
batch = requests.post(f"{{API_URL}}/predict-batch", headers=headers, json=batch_payload)
batch.raise_for_status()
for item in batch.json()["predictions"]:
    low = " ⚠️" if item.get("low_confidence") else ""
    print(f"  {{item['id_obs']}} → {{item['prediction']}}{{low}}")

# ── Validate input schema without predicting ──────────────────────────────────
val = requests.post(
    f"{{API_URL}}/models/iris-classifier/1.0.0/validate-input",
    headers=headers,
    json={{"sepal length (cm)": 5.1, "petal length (cm)": 1.4, "extra_col": 99}},
)
val_result = val.json()
# {{"valid": false, "errors": [{{"type": "missing_feature", "feature": "sepal width (cm)"}}],
#  "warnings": [], "expected_features": ["sepal length (cm)", "sepal width (cm)", ...]}}
print(f"valid={{val_result['valid']}}  errors={{val_result['errors']}}")
"""
        st.code(code_predict, language="python")

    # SECTION 4 — Record observed results
    with st.expander(t("code_example.python.section4_title"), expanded=False):
        st.markdown(t("code_example.python.section4_caption"))
        if not is_complete:
            code_observed = f"""\
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_URL = "{API_URL}"
API_TOKEN = os.getenv("PREDICTML_API_TOKEN")
headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

payload = {{
    "data": [
        {{
            "id_obs":          "obs_001",
            "model_name":      "iris-classifier",
            "date_time":       datetime.utcnow().isoformat(),
            "observed_result": 0,    # true observed class
        }}
    ]
}}

response = requests.post(f"{{API_URL}}/observed-results", headers=headers, json=payload)
response.raise_for_status()
print(f"{{response.json()['upserted']}} result(s) recorded.")
"""
        else:
            code_observed = f"""\
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "{API_URL}"
API_TOKEN = os.getenv("PREDICTML_API_TOKEN")
headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

# ── Batch submit observed results ─────────────────────────────────────────────
# Matched against predictions by id_obs — powers live accuracy, AUC and confusion matrix.
# Upsert: safe to call multiple times with the same id_obs.
payload = {{
    "data": [
        {{"id_obs": "obs_001", "model_name": "iris-classifier", "date_time": "2025-06-15T10:30:00", "observed_result": 0}},
        {{"id_obs": "obs_002", "model_name": "iris-classifier", "date_time": "2025-06-15T10:30:05", "observed_result": 2}},
        {{"id_obs": "obs_003", "model_name": "iris-classifier", "date_time": "2025-06-15T10:30:10", "observed_result": 1}},
    ]
}}
result = requests.post(f"{{API_URL}}/observed-results", headers=headers, json=payload)
result.raise_for_status()
print(f"{{result.json()['upserted']}} result(s) recorded.")

# ── Retrieve unlabeled predictions to annotate ────────────────────────────────
# Strategies: uncertainty (lowest confidence first) | recent | random
unlabeled = requests.get(
    f"{{API_URL}}/predictions/unlabeled",
    headers=headers,
    params={{"model_name": "iris-classifier", "strategy": "uncertainty", "limit": 20}},
)
unlabeled.raise_for_status()
queue = unlabeled.json()
print(f"{{queue['total_unlabeled']}} predictions pending annotation")
for pred in queue["predictions"][:3]:
    conf = pred.get("max_confidence")
    print(f"  id_obs={{pred['id_obs']}}  result={{pred['prediction_result']}}  confidence={{conf:.2f if conf else 'n/a'}}")

# ── Live performance report (after labeling) ──────────────────────────────────
perf = requests.get(
    f"{{API_URL}}/models/iris-classifier/performance",
    headers=headers,
    params={{"days": 30}},
)
perf.raise_for_status()
p = perf.json()
print(f"Live accuracy={{p.get('accuracy'):.4f}}  F1={{p.get('f1_weighted'):.4f}}  "
      f"({{p['matched_predictions']}}/{{p['total_predictions']}} labeled)")
"""
        st.code(code_observed, language="python")

# ============================================================
# TAB CURL / BASH
# ============================================================
with tab_curl:
    with st.expander(t("code_example.curl.section1_title"), expanded=False):
        if not is_complete:
            code_curl_upload = f"""\
#!/usr/bin/env bash
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

curl -X POST "$API_URL/models" \\
  -H "Authorization: Bearer $TOKEN" \\
  -F "file=@iris-classifier.joblib;type=application/octet-stream" \\
  -F "name=iris-classifier" \\
  -F "version=1.0.0" \\
  -F "description=Random Forest on Iris dataset" \\
  -F "algorithm=RandomForestClassifier" \\
  -F "accuracy=0.9667" \\
  -F "f1_score=0.9667" \\
  -F "features_count=4" \\
  -F 'classes=["setosa","versicolor","virginica"]'
"""
        else:
            code_curl_upload = f"""\
#!/usr/bin/env bash
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

# ── Upload model + training script ────────────────────────────────────────────
# train_file enables automatic / scheduled retraining from the dashboard
curl -X POST "$API_URL/models" \\
  -H "Authorization: Bearer $TOKEN" \\
  -F "file=@iris-classifier.joblib;type=application/octet-stream" \\
  -F "train_file=@train_iris.py;type=text/plain" \\
  -F "name=iris-classifier" \\
  -F "version=1.0.0" \\
  -F "description=RandomForest on Iris — production example" \\
  -F "algorithm=RandomForest" \\
  -F "accuracy=0.9667" \\
  -F "f1_score=0.9667" \\
  -F "features_count=4" \\
  -F 'classes=["setosa","versicolor","virginica"]' \\
  -F 'hyperparameters={{"n_estimators":200,"max_depth":10,"class_weight":"balanced"}}' \\
  -F 'training_metrics={{"accuracy":0.9667,"f1_score":0.9667,"precision":0.9667,"recall":0.9667}}' \\
  -F 'feature_baseline={{"sepal length (cm)":{{"mean":5.84,"std":0.83,"min":4.3,"max":7.9,"null_rate":0.0}}}}' \\
  -F 'tags=["Example","iris","classification"]' \\
  -F "webhook_url=https://webhook.site/00000000-0000-0000-0000-000000000000" \\
  -F "training_dataset=sklearn Iris (150 obs, 3 classes)"

# ── Set to production + A/B mode + confidence threshold ───────────────────────
curl -X PATCH "$API_URL/models/iris-classifier/1.0.0" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "is_production":        true,
    "deployment_mode":      "ab_test",
    "traffic_weight":       0.5,
    "confidence_threshold": 0.60
  }}'
"""
        st.code(code_curl_upload, language="bash")

    with st.expander(t("code_example.curl.section2_title"), expanded=False):
        if not is_complete:
            code_curl_predict = f"""\
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

curl -X POST "$API_URL/predict" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model_name": "iris-classifier",
    "features": {{
      "sepal length (cm)": 5.1,
      "sepal width (cm)":  3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)":  0.2
    }},
    "id_obs": "obs_001"
  }}'
"""
        else:
            code_curl_predict = f"""\
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

# ── Single prediction — specific version + SHAP explanation ──────────────────
curl -X POST "$API_URL/predict?explain=true" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model_name":    "iris-classifier",
    "model_version": "1.0.0",
    "id_obs":        "obs_001",
    "timestamp":     "2025-06-15T10:30:00",
    "features": {{
      "sepal length (cm)": 5.1,
      "sepal width (cm)":  3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)":  0.2
    }}
  }}'
# Response includes shap_values, low_confidence, selected_version (A/B routing)

# ── Batch prediction ──────────────────────────────────────────────────────────
curl -X POST "$API_URL/predict-batch" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model_name": "iris-classifier",
    "inputs": [
      {{"id_obs": "obs_001", "features": {{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}}}},
      {{"id_obs": "obs_002", "features": {{"sepal length (cm)": 6.3, "sepal width (cm)": 2.5, "petal length (cm)": 5.0, "petal width (cm)": 1.9}}}}
    ]
  }}'

# ── Validate input schema without predicting ──────────────────────────────────
curl -X POST "$API_URL/models/iris-classifier/1.0.0/validate-input" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"sepal length (cm)": 5.1, "petal length (cm)": 1.4}}'
# → {{"valid": false, "errors": [{{"type":"missing_feature","feature":"sepal width (cm)"}}], ...}}

# ── Strict mode: reject requests with unexpected features ─────────────────────
curl -X POST "$API_URL/predict?strict_validation=true" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"model_name": "iris-classifier", "features": {{"sepal length (cm)": 5.1, "extra_col": 99}}}}'
# → 422 Unprocessable Entity
"""
        st.code(code_curl_predict, language="bash")

    with st.expander(t("code_example.curl.section3_title"), expanded=False):
        if not is_complete:
            code_curl_history = f"""\
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

curl -G "$API_URL/predictions" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "model_name=iris-classifier" \\
  --data-urlencode "limit=10"
"""
        else:
            code_curl_history = f"""\
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

# ── Prediction history — id-based cursor pagination ───────────────────────────
curl -G "$API_URL/predictions" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "model_name=iris-classifier" \\
  --data-urlencode "model_version=1.0.0" \\
  --data-urlencode "limit=50" \\
  --data-urlencode "cursor=0"       # use next_cursor from the previous response

# ── Prediction statistics (error rate, latency P50 / P95) ────────────────────
curl -G "$API_URL/predictions/stats" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "model_name=iris-classifier" \\
  --data-urlencode "days=30"

# ── Export predictions as CSV ─────────────────────────────────────────────────
curl -G "$API_URL/predictions/export" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "model_name=iris-classifier" \\
  --data-urlencode "format=csv" \\
  -o predictions_iris.csv

# ── GDPR purge — dry_run=true by default (no deletion without explicit opt-in) ─
curl -X DELETE -G "$API_URL/predictions/purge" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "older_than_days=90" \\
  --data-urlencode "model_name=iris-classifier" \\
  --data-urlencode "dry_run=true"
# → {{"dry_run":true,"deleted_count":1240,"oldest_remaining":"..."}}
"""
        st.code(code_curl_history, language="bash")

    with st.expander(t("code_example.curl.section4_title"), expanded=False):
        if not is_complete:
            code_curl_observed = f"""\
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

curl -X POST "$API_URL/observed-results" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "data": [
      {{
        "id_obs":          "obs_001",
        "model_name":      "iris-classifier",
        "date_time":       "2025-06-15T10:30:00",
        "observed_result": 0
      }}
    ]
  }}'
"""
        else:
            code_curl_observed = f"""\
source .env  # charge PREDICTML_API_TOKEN depuis .env
API_URL="{API_URL}"
TOKEN="${{PREDICTML_API_TOKEN}}"

# ── Batch submit observed results ─────────────────────────────────────────────
curl -X POST "$API_URL/observed-results" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "data": [
      {{"id_obs":"obs_001","model_name":"iris-classifier","date_time":"2025-06-15T10:30:00","observed_result":0}},
      {{"id_obs":"obs_002","model_name":"iris-classifier","date_time":"2025-06-15T10:30:05","observed_result":2}},
      {{"id_obs":"obs_003","model_name":"iris-classifier","date_time":"2025-06-15T10:30:10","observed_result":1}}
    ]
  }}'

# ── Unlabeled predictions — prioritized by lowest confidence ──────────────────
# Strategies: uncertainty | recent | random
curl -G "$API_URL/predictions/unlabeled" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "model_name=iris-classifier" \\
  --data-urlencode "strategy=uncertainty" \\
  --data-urlencode "limit=20"

# ── Live performance report (after labeling) ──────────────────────────────────
curl -G "$API_URL/models/iris-classifier/performance" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "days=30"
# → accuracy, f1_weighted, confusion_matrix, per_class_metrics, matched_predictions…
"""
        st.code(code_curl_observed, language="bash")

# ============================================================
# TAB JAVASCRIPT
# ============================================================
with tab_js:
    with st.expander(t("code_example.js.section1_title"), expanded=False):
        st.markdown(t("code_example.js.section1_caption"))
        if not is_complete:
            code_js_upload = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

// fileInput is an <input type="file"> element in your HTML page
const modelFile = fileInput.files[0];

const formData = new FormData();
formData.append("file", modelFile, "iris-classifier.joblib");
formData.append("name",           "iris-classifier");
formData.append("version",        "1.0.0");
formData.append("description",    "Random Forest on Iris dataset");
formData.append("algorithm",      "RandomForestClassifier");
formData.append("accuracy",       "0.9667");
formData.append("f1_score",       "0.9667");
formData.append("features_count", "4");
formData.append("classes",        '["setosa","versicolor","virginica"]');

const response = await fetch(`${{API_URL}}/models`, {{
  method: "POST",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}` }},
  body: formData,
}});
const model = await response.json();
console.log(`Uploaded: ${{model.name}} v${{model.version}}`);
"""
        else:
            code_js_upload = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

// modelInput / trainInput are <input type="file"> elements
const modelFile = modelInput.files[0];
const trainFile = trainInput.files[0];   // optional — enables retraining

const hyperparameters = {{ n_estimators: 200, max_depth: 10, class_weight: "balanced" }};
const featureBaseline = {{
  "sepal length (cm)": {{ mean: 5.84, std: 0.83, min: 4.3, max: 7.9, null_rate: 0.0 }},
  "sepal width (cm)":  {{ mean: 3.06, std: 0.44, min: 2.0, max: 4.4, null_rate: 0.0 }},
  "petal length (cm)": {{ mean: 3.76, std: 1.77, min: 1.0, max: 6.9, null_rate: 0.0 }},
  "petal width (cm)":  {{ mean: 1.20, std: 0.76, min: 0.1, max: 2.5, null_rate: 0.0 }},
}};

const formData = new FormData();
formData.append("file", modelFile, "iris-classifier.joblib");
if (trainFile) formData.append("train_file", trainFile, "train_iris.py");
formData.append("name",             "iris-classifier");
formData.append("version",          "1.0.0");
formData.append("description",      "RandomForest on Iris — production example");
formData.append("algorithm",        "RandomForest");
formData.append("accuracy",         "0.9667");
formData.append("f1_score",         "0.9667");
formData.append("features_count",   "4");
formData.append("classes",          JSON.stringify(["setosa", "versicolor", "virginica"]));
formData.append("hyperparameters",  JSON.stringify(hyperparameters));
formData.append("training_metrics", JSON.stringify({{ accuracy: 0.9667, f1_score: 0.9667, precision: 0.9667, recall: 0.9667 }}));
formData.append("feature_baseline", JSON.stringify(featureBaseline));
formData.append("tags",             JSON.stringify(["Example", "iris", "classification"]));
formData.append("webhook_url",      "https://webhook.site/00000000-0000-0000-0000-000000000000");
formData.append("training_dataset", "sklearn Iris (150 obs, 3 classes)");

const uploadResp = await fetch(`${{API_URL}}/models`, {{
  method: "POST",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}` }},
  body: formData,
}});
const model = await uploadResp.json();
console.log(`Uploaded: ${{model.name}} v${{model.version}} (id=${{model.id}})`);

// ── Set to production + A/B routing + confidence threshold ────────────────────
await fetch(`${{API_URL}}/models/${{model.name}}/${{model.version}}`, {{
  method: "PATCH",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}`, "Content-Type": "application/json" }},
  body: JSON.stringify({{
    is_production:        true,
    deployment_mode:      "ab_test",   // "production" | "ab_test" | "shadow"
    traffic_weight:       0.5,
    confidence_threshold: 0.60,
  }}),
}});
console.log("Set to production (ab_test, 50 % traffic).");
"""
        st.code(code_js_upload, language="javascript")

    with st.expander(t("code_example.js.section2_title"), expanded=False):
        if not is_complete:
            code_js_predict = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

const response = await fetch(`${{API_URL}}/predict`, {{
  method: "POST",
  headers: {{
    "Authorization": `Bearer ${{TOKEN}}`,
    "Content-Type": "application/json",
  }},
  body: JSON.stringify({{
    model_name: "iris-classifier",
    features: {{
      "sepal length (cm)": 5.1,
      "sepal width (cm)":  3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)":  0.2,
    }},
    id_obs: "obs_001",
  }}),
}});
const result = await response.json();
console.log(`Prediction  : ${{result.prediction}}`);
console.log(`Probabilities:`, result.probability);
"""
        else:
            code_js_predict = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

// ── Single prediction — specific version + SHAP explanation ──────────────────
const response = await fetch(`${{API_URL}}/predict?explain=true`, {{
  method: "POST",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}`, "Content-Type": "application/json" }},
  body: JSON.stringify({{
    model_name:    "iris-classifier",
    model_version: "1.0.0",            // omit → production version
    id_obs:        "obs_001",
    timestamp:     "2025-06-15T10:30:00",  // optional — inject historical timestamp
    features: {{
      "sepal length (cm)": 5.1,
      "sepal width (cm)":  3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)":  0.2,
    }},
  }}),
}});
const result = await response.json();
console.log(`Prediction  : ${{result.prediction}}`);
console.log(`Probabilities:`, result.probability);
console.log(`SHAP values :`, result.shap_values);   // null if explain=false
if (result.low_confidence) console.warn("Warning: low confidence prediction");

// ── Batch prediction ──────────────────────────────────────────────────────────
const batchResp = await fetch(`${{API_URL}}/predict-batch`, {{
  method: "POST",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}`, "Content-Type": "application/json" }},
  body: JSON.stringify({{
    model_name: "iris-classifier",
    inputs: [
      {{ id_obs: "obs_001", features: {{ "sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2 }} }},
      {{ id_obs: "obs_002", features: {{ "sepal length (cm)": 6.3, "sepal width (cm)": 2.5, "petal length (cm)": 5.0, "petal width (cm)": 1.9 }} }},
    ],
  }}),
}});
const batch = await batchResp.json();
batch.predictions.forEach(p => {{
  const low = p.low_confidence ? " ⚠️" : "";
  console.log(`${{p.id_obs}} → ${{p.prediction}}${{low}}`);
}});

// ── Validate input without predicting ────────────────────────────────────────
const valResp = await fetch(`${{API_URL}}/models/iris-classifier/1.0.0/validate-input`, {{
  method: "POST",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}`, "Content-Type": "application/json" }},
  body: JSON.stringify({{ "sepal length (cm)": 5.1, "petal length (cm)": 1.4 }}),
}});
const val = await valResp.json();
// {{"valid":false,"errors":[{{"type":"missing_feature","feature":"sepal width (cm)"}}],...}}
console.log(`valid=${{val.valid}}`, val.errors);
"""
        st.code(code_js_predict, language="javascript")

    with st.expander(t("code_example.js.section3_title"), expanded=False):
        if not is_complete:
            code_js_history = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

const params = new URLSearchParams({{ model_name: "iris-classifier", limit: "10" }});
const response = await fetch(`${{API_URL}}/predictions?${{params}}`, {{
  headers: {{ "Authorization": `Bearer ${{TOKEN}}` }},
}});
const history = await response.json();
console.log(`${{history.total}} prediction(s) total`);
"""
        else:
            code_js_history = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

// ── Prediction history — id-based cursor pagination ───────────────────────────
const params = new URLSearchParams({{
  model_name:    "iris-classifier",
  model_version: "1.0.0",
  limit:         "50",
  cursor:        "0",      // use next_cursor from the previous response
}});
const resp = await fetch(`${{API_URL}}/predictions?${{params}}`, {{
  headers: {{ "Authorization": `Bearer ${{TOKEN}}` }},
}});
const history = await resp.json();
console.log(`${{history.total}} prediction(s) total`);
history.predictions.forEach(p => console.log(p.id_obs, p.prediction_result));
if (history.next_cursor) console.log(`Next cursor: ${{history.next_cursor}}`);

// ── Prediction statistics (error rate, latency P50 / P95) ────────────────────
const statsResp = await fetch(
  `${{API_URL}}/predictions/stats?model_name=iris-classifier&days=30`,
  {{ headers: {{ "Authorization": `Bearer ${{TOKEN}}` }} }},
);
const stats = await statsResp.json();
stats.stats.forEach(s => {{
  console.log(`${{s.model_name}} — error_rate=${{s.error_rate}} p95=${{s.p95_response_time_ms}}ms`);
}});
"""
        st.code(code_js_history, language="javascript")

    with st.expander(t("code_example.js.section4_title"), expanded=False):
        if not is_complete:
            code_js_observed = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

const response = await fetch(`${{API_URL}}/observed-results`, {{
  method: "POST",
  headers: {{
    "Authorization": `Bearer ${{TOKEN}}`,
    "Content-Type": "application/json",
  }},
  body: JSON.stringify({{
    data: [
      {{
        id_obs:          "obs_001",
        model_name:      "iris-classifier",
        date_time:       new Date().toISOString(),
        observed_result: 0,
      }},
    ],
  }}),
}});
const result = await response.json();
console.log(`${{result.upserted}} result(s) recorded.`);
"""
        else:
            code_js_observed = f"""\
import 'dotenv/config'; // npm install dotenv
const API_URL = "{API_URL}";
const TOKEN = process.env.PREDICTML_API_TOKEN;

// ── Batch submit observed results ─────────────────────────────────────────────
// Matched by id_obs — powers live accuracy, AUC and confusion matrix.
// Upsert: safe to call multiple times with the same id_obs.
const response = await fetch(`${{API_URL}}/observed-results`, {{
  method: "POST",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}`, "Content-Type": "application/json" }},
  body: JSON.stringify({{
    data: [
      {{ id_obs: "obs_001", model_name: "iris-classifier", date_time: "2025-06-15T10:30:00", observed_result: 0 }},
      {{ id_obs: "obs_002", model_name: "iris-classifier", date_time: "2025-06-15T10:30:05", observed_result: 2 }},
      {{ id_obs: "obs_003", model_name: "iris-classifier", date_time: "2025-06-15T10:30:10", observed_result: 1 }},
    ],
  }}),
}});
const result = await response.json();
console.log(`${{result.upserted}} result(s) recorded.`);

// ── Unlabeled predictions — lowest confidence first ───────────────────────────
const unlabeled = await fetch(
  `${{API_URL}}/predictions/unlabeled?model_name=iris-classifier&strategy=uncertainty&limit=20`,
  {{ headers: {{ "Authorization": `Bearer ${{TOKEN}}` }} }},
);
const queue = await unlabeled.json();
console.log(`${{queue.total_unlabeled}} unlabeled predictions`);
queue.predictions.forEach(p => {{
  const conf = p.max_confidence != null ? p.max_confidence.toFixed(2) : "n/a";
  console.log(`id_obs=${{p.id_obs}}  result=${{p.prediction_result}}  confidence=${{conf}}`);
}});

// ── Live performance report (after labeling) ──────────────────────────────────
const perf = await fetch(
  `${{API_URL}}/models/iris-classifier/performance?days=30`,
  {{ headers: {{ "Authorization": `Bearer ${{TOKEN}}` }} }},
);
const p = await perf.json();
console.log(`Live accuracy=${{p.accuracy?.toFixed(4)}}  F1=${{p.f1_weighted?.toFixed(4)}}  `
  + `(${{p.matched_predictions}}/${{p.total_predictions}} labeled)`);
"""
        st.code(code_js_observed, language="javascript")

st.divider()
st.caption(t("code_example.footer_caption", api_url=API_URL, mlflow_url=MLFLOW_URL))
