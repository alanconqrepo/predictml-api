"""
upload_wine_model.py — Trains and uploads a Wine model (Regression) via the PredictML API
===========================================================================================

This script runs LOCALLY. It:
  1. Executes train_wine.py in a subprocess to produce the .joblib
  2. Uploads the .joblib + train_wine.py via POST /models
  3. Sets the model to production with the "Example" tag and the feature baseline

Model: GradientBoostingRegressor — predicts alcohol content (continuous target)
       from the 12 other chemical measurements in the sklearn Wine dataset.

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python upload_wine_model.py

Environment variables:
  API_URL        API URL               (default: http://localhost:80)
  API_TOKEN      Bearer token — required
  MODEL_NAME     Model name            (default: wine-regressor)
  MODEL_VERSION  Version               (default: 1.0.0)
  TRAIN_START    Training start date   (default: 2024-01-01)
  TRAIN_END      Training end date     (default: 2024-12-31)

Python prerequisites:
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

MODEL_NAME    = os.environ.get("MODEL_NAME",    "wine-regressor")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
DESCRIPTION   = "GradientBoostingRegressor trained on the Wine dataset (regression — target: alcohol content)"
ALGORITHM     = "GradientBoosting"

TRAIN_START = os.environ.get("TRAIN_START", "2024-01-01")
TRAIN_END   = os.environ.get("TRAIN_END",   "2024-12-31")

RETRAIN_IN_API = False  # True = re-run train.py on the API side (server libraries)

MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI",      "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "admin")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

MINIO_ENDPOINT   = os.environ.get("MLFLOW_S3_ENDPOINT_URL",
                       f"http://localhost:{os.environ.get('MINIO_PORT', '9010')}")
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID",     os.environ.get("MINIO_ROOT_USER", ""))
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_ROOT_PASSWORD", ""))
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET", "models")

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "train_wine.py")

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN not defined.")
    print("    Run: API_TOKEN=<your_token> python upload_wine_model.py")
    sys.exit(1)

if not os.path.exists(TRAIN_SCRIPT_PATH):
    print(f"❌  train_wine.py not found: {TRAIN_SCRIPT_PATH}")
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

# ── 2. Execute train_wine.py ──────────────────────────────────────────────────

tmp_pkl = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
tmp_pkl.close()

print(f"⏳  Training via train_wine.py ({TRAIN_START} → {TRAIN_END})…")

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
    print("❌  train_wine.py failed:")
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

r2               = metrics.get("r2")
mae              = metrics.get("mae")
rmse             = metrics.get("rmse")
features_count   = metrics.get("features_count")
training_dataset = metrics.get("training_dataset")
mlflow_run_id    = metrics.get("mlflow_run_id")
hyperparameters  = metrics.get("hyperparameters")
dependencies     = metrics.get("dependencies", {})
print(
    f"✅  Training complete — R²: {r2} | MAE: {mae} | RMSE: {rmse}"
    + (f" | MLflow run: {mlflow_run_id}" if mlflow_run_id else "")
)

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
        # For regression: accuracy = R² (best proxy for dashboard display)
        if r2 is not None:
            data["accuracy"] = str(round(r2, 4))
        if features_count is not None:
            data["features_count"] = str(features_count)
        if training_dataset:
            data["training_dataset"] = training_dataset
        if mlflow_run_id:
            data["mlflow_run_id"] = mlflow_run_id
        # training_metrics: full regression metrics
        _tm = {k: round(v, 4) for k, v in {
            "mae":  mae,
            "rmse": rmse,
            "r2":   r2,
        }.items() if v is not None}
        if _tm:
            data["training_metrics"] = json.dumps(_tm)
        if hyperparameters:
            data["hyperparameters"] = json.dumps(hyperparameters)
        data["run_training"] = str(RETRAIN_IN_API).lower()
        if dependencies:
            data["local_dependencies"] = json.dumps(dependencies)

        _upload_t0 = time.perf_counter()
        response = requests.post(
            f"{API_URL}/models",
            headers=HEADERS,
            files={
                "file":       (f"{MODEL_NAME}.joblib", pkl_fh,   "application/octet-stream"),
                "train_file": ("train_wine.py",      train_fh, "text/plain"),
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
        print(f"✅  ab_test mode updated.")
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

# ── 5. Set to production + tag "Example" + feature_baseline ──────────────────

print(f"⏳  Setting to production, adding 'Example' tag and feature baseline…")

patch_body: dict = {"is_production": True, "deployment_mode": "ab_test", "traffic_weight": 0.5, "tags": ["Example"]}
if metrics.get("feature_stats"):
    patch_body["feature_baseline"] = metrics["feature_stats"]
training_stats: dict = {}
if metrics.get("label_distribution"):
    training_stats["label_distribution"] = metrics["label_distribution"]
if metrics.get("regression_bins"):
    training_stats["regression_bins"] = metrics["regression_bins"]
if metrics.get("n_rows") is not None:
    training_stats["n_rows"] = metrics["n_rows"]
if TRAIN_START:
    training_stats["train_start_date"] = TRAIN_START
if TRAIN_END:
    training_stats["train_end_date"] = TRAIN_END
if training_stats:
    patch_body["training_stats"] = training_stats

patch = requests.patch(
    f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=patch_body,
    timeout=30,
)

if patch.status_code == 200:
    baseline_ok = "feature_baseline" in patch_body
    print(f"✅  Model set to production (ab_test)"
          f"{' with feature baseline' if baseline_ok else ''}.")
else:
    print(f"⚠️   PATCH failed ({patch.status_code}): {patch.text[:200]}")

print(f"\n   → wine-regressor v{MODEL_VERSION} in A/B test (ready for v1.1.0)")
print(f"   → Retrain:")
print(f"       POST {API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}/retrain")
