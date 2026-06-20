"""
upload_titanic_LogisticRegression_uploaded.py — Titanic v1.3.0 (LogisticRegression) — uploaded only
=====================================================================================================

Upload titanic-survival v1.3.0 with LogisticRegression Pipeline.
This model is simply uploaded, without being set to production or assigned a deployment mode.

Deployment:
  - is_production = False  (not set to production)
  - deployment_mode = None  (no routing)
  - tag "Example"

Usage:
  API_TOKEN=<token> python upload_titanic_LogisticRegression_uploaded.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

API_URL   = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

MODEL_NAME    = os.environ.get("MODEL_NAME",    "titanic-survival")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.3.0")
DESCRIPTION   = (
    "sklearn Pipeline (OneHotEncoder + StandardScaler + LogisticRegression) "
    "trained on synthetic Titanic data — v1.3.0, linear model, maximum interpretability. "
    "Features: age, fare, parch, sibsp (numerical) + pclass, sex, embarked (categorical)."
)
ALGORITHM = "LogisticRegression"

TRAIN_START = os.environ.get("TRAIN_START", "2024-01-01")
TRAIN_END   = os.environ.get("TRAIN_END",   "2024-12-31")

MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI",      "http://localhost:5000")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "admin")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
MINIO_ENDPOINT   = os.environ.get("MLFLOW_S3_ENDPOINT_URL",
                       f"http://localhost:{os.environ.get('MINIO_PORT', '9010')}")
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID",     os.environ.get("MINIO_ROOT_USER", ""))
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_ROOT_PASSWORD", ""))

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "train_titanic_LogisticRegression.py")

if not API_TOKEN:
    print("❌  API_TOKEN not defined.")
    sys.exit(1)
if not os.path.exists(TRAIN_SCRIPT_PATH):
    print(f"❌  train_titanic_LogisticRegression.py not found: {TRAIN_SCRIPT_PATH}")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# ── 1. Health check ───────────────────────────────────────────────────────────

try:
    requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    print(f"✅  API reachable: {API_URL}")
except Exception as e:
    print(f"❌  API unreachable: {e}")
    sys.exit(1)

# ── 2. Training ───────────────────────────────────────────────────────────────

tmp_pkl = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
tmp_pkl.close()

print(f"⏳  Training LogisticRegression ({TRAIN_START} → {TRAIN_END})…")

train_env = {
    **os.environ,
    "TRAIN_START_DATE": TRAIN_START, "TRAIN_END_DATE": TRAIN_END,
    "OUTPUT_MODEL_PATH": tmp_pkl.name, "MODEL_NAME": MODEL_NAME,
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "MLFLOW_TRACKING_USERNAME": MLFLOW_TRACKING_USERNAME,
    "MLFLOW_TRACKING_PASSWORD": MLFLOW_TRACKING_PASSWORD,
    "MLFLOW_S3_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_ACCESS_KEY_ID": MINIO_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": MINIO_SECRET_KEY,
    "PYTHONIOENCODING": "utf-8",
}

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT_PATH], env=train_env,
    capture_output=True, text=True, encoding="utf-8",
)

if result.returncode != 0:
    print("❌  train_titanic_LogisticRegression.py failed:")
    print(result.stderr)
    os.unlink(tmp_pkl.name)
    sys.exit(1)

if result.stderr:
    print(result.stderr.strip())

metrics = {}
for line in reversed(result.stdout.strip().splitlines()):
    try:
        metrics = json.loads(line)
        break
    except json.JSONDecodeError:
        continue

acc           = metrics.get("accuracy")
f1            = metrics.get("f1_score")
precision     = metrics.get("precision")
recall        = metrics.get("recall")
roc_auc       = metrics.get("roc_auc")
hyperparams   = metrics.get("hyperparameters")
mlflow_run_id = metrics.get("mlflow_run_id")
dependencies  = metrics.get("dependencies", {})
print(
    f"✅  Training — Acc={acc} F1={f1} AUC={roc_auc}"
    + (f" | MLflow={mlflow_run_id}" if mlflow_run_id else "")
)

# ── 3. Upload ─────────────────────────────────────────────────────────────────

print(f"⏳  Uploading {MODEL_NAME} v{MODEL_VERSION} (uploaded only)…")

try:
    with open(tmp_pkl.name, "rb") as pkl_fh, open(TRAIN_SCRIPT_PATH, "rb") as train_fh:
        data = {
            "name": MODEL_NAME, "version": MODEL_VERSION,
            "description": DESCRIPTION, "algorithm": ALGORITHM,
        }
        if acc is not None:
            data["accuracy"] = str(round(acc, 4))
        if f1 is not None:
            data["f1_score"] = str(round(f1, 4))
        if roc_auc is not None:
            data["auc"] = str(round(roc_auc, 4))
        if metrics.get("features_count") is not None:
            data["features_count"] = str(metrics["features_count"])
        if metrics.get("classes"):
            data["classes"] = json.dumps(metrics["classes"])
        if metrics.get("training_dataset"):
            data["training_dataset"] = metrics["training_dataset"]
        if mlflow_run_id:
            data["mlflow_run_id"] = mlflow_run_id
        _tm = {k: round(v, 4) for k, v in {
            "accuracy": acc, "f1_score": f1, "precision": precision,
            "recall": recall, "roc_auc": roc_auc,
        }.items() if v is not None}
        if _tm:
            data["training_metrics"] = json.dumps(_tm)
        if hyperparams:
            data["hyperparameters"] = json.dumps(hyperparams)
        if dependencies:
            data["local_dependencies"] = json.dumps(dependencies)

        _upload_t0 = time.perf_counter()
        response = requests.post(
            f"{API_URL}/models", headers=HEADERS,
            files={
                "file":       (f"{MODEL_NAME}.joblib", pkl_fh, "application/octet-stream"),
                "train_file": ("train_titanic_LogisticRegression.py", train_fh, "text/plain"),
            },
            data=data, timeout=180,
        )
        _upload_elapsed = time.perf_counter() - _upload_t0
        print(f"  [TIMING] POST /models responded in {_upload_elapsed:.2f}s — status {response.status_code}")
finally:
    os.unlink(tmp_pkl.name)

if response.status_code == 409:
    print(f"⚠️   {MODEL_NAME} v{MODEL_VERSION} already exists — PATCH tag only…")
    patch_payload = {"tags": ["Example"]}
    if hyperparams:
        patch_payload["hyperparameters"] = hyperparams
    try:
        patch_resp = requests.patch(
            f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
            headers={**HEADERS, "Content-Type": "application/json"},
            json=patch_payload, timeout=30,
        )
    except Exception as _e:
        print(f"    [WARN] PATCH already-exists failed: {_e} — continuing")
        sys.exit(0)
    print("✅  PATCH OK" if patch_resp.status_code == 200
          else f"❌  PATCH {patch_resp.status_code}: {patch_resp.text[:200]}")
    sys.exit(0)

if response.status_code not in (200, 201):
    print(f"❌  Error {response.status_code}: {response.text[:300]}")
    sys.exit(1)

res = response.json()
print(f"✅  Model uploaded — ID={res.get('id')} version={res.get('version')}")

# ── 4. PATCH → tag only (no production, no deployment_mode) ──────────────────

print("⏳  Adding tag 'Example' and feature baseline…")

patch_body: dict = {"tags": ["Example"]}
if metrics.get("feature_stats"):
    patch_body["feature_baseline"] = metrics["feature_stats"]
if metrics.get("cat_distributions"):
    patch_body["categorical_baseline"] = metrics["cat_distributions"]
if metrics.get("confidence_threshold") is not None:
    patch_body["confidence_threshold"] = metrics["confidence_threshold"]

patch = requests.patch(
    f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=patch_body, timeout=30,
)

if patch.status_code == 200:
    print("✅  Tag 'Example' added (is_production=False, no deployment_mode).")
else:
    print(f"⚠️   PATCH failed ({patch.status_code}): {patch.text[:200]}")

print(f"\n   → titanic-survival v{MODEL_VERSION} uploaded but inactive")
print(f"   → Activate: PATCH /models/{MODEL_NAME}/{MODEL_VERSION}  {{\"is_production\": true}}")
