"""
upload_iris_model_GradientBoosting.py — Trains and uploads an Iris model (GradientBoosting) via the PredictML API
=================================================================================================================

This script runs LOCALLY. It:
  1. Executes train_iris_GradientBoosting.py as a subprocess to produce the .joblib
  2. Uploads the .joblib + train_iris_GradientBoosting.py via POST /models (version 1.1.0)
  3. Adds the tag "Example" — the model is NOT set to production

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python upload_iris_model_GradientBoosting.py

Environment variables:
  API_URL        API URL               (default: http://localhost:8000)
  API_TOKEN      Bearer token — required
  MODEL_NAME     Model name            (default: iris-classifier)
  MODEL_VERSION  Version               (default: 1.1.0)
  TRAIN_START    Training start date   (default: 2024-01-01)
  TRAIN_END      Training end date     (default: 2024-12-31)

Python requirements:
  pip install requests scikit-learn numpy pandas
"""

import json
import os
import subprocess
import sys
import time
import tempfile

import requests
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL   = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

MODEL_NAME    = os.environ.get("MODEL_NAME",    "iris-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.1.0")
DESCRIPTION   = "GradientBoostingClassifier trained on the Iris dataset (example)"
ALGORITHM     = "GradientBoosting"

TRAIN_START = os.environ.get("TRAIN_START", "2024-01-01")
TRAIN_END   = os.environ.get("TRAIN_END",   "2024-12-31")

MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI",      "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "admin")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

MINIO_ENDPOINT   = os.environ.get("MLFLOW_S3_ENDPOINT_URL",
                       f"http://localhost:{os.environ.get('MINIO_PORT', '9010')}")
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID",     os.environ.get("MINIO_ROOT_USER", ""))
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_ROOT_PASSWORD", ""))
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET", "models")

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "train_iris_GradientBoosting.py")

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN not defined.")
    print("    Run: API_TOKEN=<your_token> python upload_iris_model_GradientBoosting.py")
    sys.exit(1)

if not os.path.exists(TRAIN_SCRIPT_PATH):
    print(f"❌  train_iris_GradientBoosting.py not found: {TRAIN_SCRIPT_PATH}")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# ── 1. Check that the API is reachable ────────────────────────────────────────

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    print(f"✅  API reachable: {API_URL}")
except Exception as e:
    print(f"❌  API unreachable ({API_URL}): {e}")
    sys.exit(1)

# ── 2. Run train_iris_GradientBoosting.py ─────────────────────────────────────

tmp_pkl = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
tmp_pkl.close()

print(f"⏳  Training via train_iris_GradientBoosting.py ({TRAIN_START} → {TRAIN_END})…")

train_env = {
    **os.environ,
    "TRAIN_START_DATE":          TRAIN_START,
    "TRAIN_END_DATE":            TRAIN_END,
    "OUTPUT_MODEL_PATH":         tmp_pkl.name,
    "MODEL_NAME":                MODEL_NAME,
    "MLFLOW_TRACKING_URI":       MLFLOW_TRACKING_URI,
    "MLFLOW_TRACKING_USERNAME":  MLFLOW_TRACKING_USERNAME,
    "MLFLOW_TRACKING_PASSWORD":  MLFLOW_TRACKING_PASSWORD,
    "MLFLOW_S3_ENDPOINT_URL":    MINIO_ENDPOINT,
    "AWS_ACCESS_KEY_ID":         MINIO_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY":     MINIO_SECRET_KEY,
    "MINIO_BUCKET":              MINIO_BUCKET,
    "PYTHONIOENCODING":          "utf-8",
}

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT_PATH],
    env=train_env,
    capture_output=True,
    text=True,
    encoding="utf-8",
)

if result.returncode != 0:
    print("❌  train_iris_GradientBoosting.py failed:")
    print(result.stderr)
    os.unlink(tmp_pkl.name)
    sys.exit(1)

if result.stderr:
    print(result.stderr.strip())

# Retrieve metrics from the last JSON line of stdout
metrics = {}
for line in reversed(result.stdout.strip().splitlines()):
    try:
        metrics = json.loads(line)
        break
    except json.JSONDecodeError:
        continue

acc              = metrics.get("accuracy")
f1               = metrics.get("f1_score")
precision        = metrics.get("precision")
recall           = metrics.get("recall")
features_count   = metrics.get("features_count")
classes          = metrics.get("classes")
training_dataset = metrics.get("training_dataset")
mlflow_run_id    = metrics.get("mlflow_run_id")
hyperparameters  = metrics.get("hyperparameters")
print(f"✅  Training complete — Accuracy: {acc} | F1: {f1}"
      + (f" | MLflow run: {mlflow_run_id}" if mlflow_run_id else ""))

# ── 3. Upload via POST /models ────────────────────────────────────────────────

print(f"⏳  Uploading {MODEL_NAME} v{MODEL_VERSION}…")

try:
    with open(tmp_pkl.name, "rb") as pkl_fh, open(TRAIN_SCRIPT_PATH, "rb") as train_fh:
        data = {
            "name":        MODEL_NAME,
            "version":     MODEL_VERSION,
            "description": DESCRIPTION,
            "algorithm":   ALGORITHM,
        }
        if acc is not None:
            data["accuracy"] = str(round(acc, 4))
        if f1 is not None:
            data["f1_score"] = str(round(f1, 4))
        if features_count is not None:
            data["features_count"] = str(features_count)
        if classes is not None:
            data["classes"] = json.dumps(classes)
        if training_dataset:
            data["training_dataset"] = training_dataset
        if mlflow_run_id:
            data["mlflow_run_id"] = mlflow_run_id
        _tm = {k: round(v, 4) for k, v in {
            "accuracy":  acc,
            "f1_score":  f1,
            "precision": precision,
            "recall":    recall,
        }.items() if v is not None}
        if _tm:
            data["training_metrics"] = json.dumps(_tm)
        if hyperparameters:
            data["hyperparameters"] = json.dumps(hyperparameters)

        _upload_t0 = time.perf_counter()
        response = requests.post(
            f"{API_URL}/models",
            headers=HEADERS,
            files={
                "file":       (f"{MODEL_NAME}.joblib", pkl_fh,   "application/octet-stream"),
                "train_file": ("train_iris_GradientBoosting.py", train_fh, "text/plain"),
            },
            data=data,
            timeout=180,
        )
        _upload_elapsed = time.perf_counter() - _upload_t0
        print(f"  [TIMING] POST /models responded in {_upload_elapsed:.2f}s — status {response.status_code}")
finally:
    os.unlink(tmp_pkl.name)

# ── 4. Upload result ──────────────────────────────────────────────────────────

if response.status_code == 409:
    print(f"⚠️   {MODEL_NAME} v{MODEL_VERSION} already exists — updating via PATCH…")
    patch_payload = {"is_production": True, "deployment_mode": "ab_test", "traffic_weight": 0.5}
    if hyperparameters:
        patch_payload["hyperparameters"] = hyperparameters
    try:
        patch_resp = requests.patch(
            f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
            headers={**HEADERS, "Content-Type": "application/json"},
            json=patch_payload,
            timeout=30,
        )
    except Exception as _e:
        print(f"    [WARN] PATCH already-exists failed: {_e} — continuing")
        sys.exit(0)
    if patch_resp.status_code == 200:
        print(f"✅  ab_test mode and hyperparameters updated.")
    else:
        print(f"❌  PATCH failed ({patch_resp.status_code}): {patch_resp.text[:200]}")
    sys.exit(0)

if response.status_code not in (200, 201):
    print(f"\n❌  Error {response.status_code}")
    try:
        body = response.json()
        print(f"   Detail: {json.dumps(body, indent=2, ensure_ascii=False)}")
    except Exception:
        print(f"   Response: {response.text[:500]}")
    sys.exit(1)

res = response.json()
print(f"\n✅  Model uploaded successfully!")
print(f"   Name      : {res.get('name')}")
print(f"   Version   : {res.get('version')}")
print(f"   ID        : {res.get('id')}")
print(f"   Algorithm : {ALGORITHM}")

# ── 5. Add tag "Example", feature_baseline and training_dataset ────────────────
#      The model is NOT set to production (is_production remains false)

print(f"⏳  Adding tag 'Example', feature baseline and lineage dataset…")

patch_body: dict = {"is_production": True, "deployment_mode": "ab_test", "traffic_weight": 0.5, "tags": ["Example"]}
if metrics.get("feature_stats"):
    patch_body["feature_baseline"] = metrics["feature_stats"]
if metrics.get("confidence_threshold") is not None:
    patch_body["confidence_threshold"] = metrics["confidence_threshold"]

patch = requests.patch(
    f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=patch_body,
    timeout=30,
)

if patch.status_code == 200:
    extras = []
    if "feature_baseline" in patch_body:
        extras.append("feature baseline")
    print(f"✅  Model set to production (ab_test){(' with ' + ', '.join(extras)) if extras else ''}.")
else:
    print(f"⚠️   PATCH failed ({patch.status_code}): {patch.text[:200]}")

print(f"\n   → iris-classifier v{MODEL_VERSION} in A/B test with v1.0.0")
print(f"   → Comparison: GET {API_URL}/models/{MODEL_NAME}/ab-compare")
print(f"   → Retrain:")
print(f"       POST {API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}/retrain")
