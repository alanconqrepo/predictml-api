"""
upload_cancer_ExtraTrees_shadow.py — cancer-classifier v1.2.0 (ExtraTrees) — mode shadow
==========================================================================================

Upload cancer-classifier v1.2.0 avec ExtraTreesClassifier en mode shadow.

Déploiement :
  - is_production = False
  - deployment_mode = "shadow" (prédictions en parallèle, non exposées au client)
  - tag "Example"

Usage :
  API_TOKEN=<token> python upload_cancer_ExtraTrees_shadow.py
"""

import json
import os
import subprocess
import sys
import tempfile

import requests
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

API_URL   = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

MODEL_NAME    = os.environ.get("MODEL_NAME",    "cancer-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.2.0")
DESCRIPTION   = "ExtraTreesClassifier cancer — shadow (v1.2.0) — splits aléatoires, max_depth=8"
ALGORITHM     = "ExtraTrees"

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
TRAIN_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "train_cancer_ExtraTrees.py")

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    sys.exit(1)
if not os.path.exists(TRAIN_SCRIPT_PATH):
    print(f"❌  train_cancer_ExtraTrees.py introuvable : {TRAIN_SCRIPT_PATH}")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

try:
    requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    print(f"✅  API accessible : {API_URL}")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
tmp_pkl.close()

print(f"⏳  Entraînement ExtraTrees ({TRAIN_START} → {TRAIN_END})…")

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT_PATH],
    env={
        **os.environ,
        "TRAIN_START_DATE":         TRAIN_START,
        "TRAIN_END_DATE":           TRAIN_END,
        "OUTPUT_MODEL_PATH":        tmp_pkl.name,
        "MLFLOW_TRACKING_URI":      MLFLOW_TRACKING_URI,
        "MLFLOW_TRACKING_USERNAME": MLFLOW_TRACKING_USERNAME,
        "MLFLOW_TRACKING_PASSWORD": MLFLOW_TRACKING_PASSWORD,
        "MLFLOW_S3_ENDPOINT_URL":   MINIO_ENDPOINT,
        "AWS_ACCESS_KEY_ID":        MINIO_ACCESS_KEY,
        "AWS_SECRET_ACCESS_KEY":    MINIO_SECRET_KEY,
        "PYTHONIOENCODING":         "utf-8",
    },
    capture_output=True, text=True, encoding="utf-8",
)

if result.returncode != 0:
    print("❌  Entraînement échoué :")
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

acc             = metrics.get("accuracy")
f1              = metrics.get("f1_score")
precision       = metrics.get("precision")
recall          = metrics.get("recall")
roc_auc         = metrics.get("roc_auc")
features_count  = metrics.get("features_count")
classes         = metrics.get("classes")
training_dataset = metrics.get("training_dataset")
mlflow_run_id   = metrics.get("mlflow_run_id")
hyperparameters = metrics.get("hyperparameters")
print(f"✅  Acc={acc} | F1={f1} | ROC-AUC={roc_auc}")

try:
    with open(tmp_pkl.name, "rb") as pkl_fh, open(TRAIN_SCRIPT_PATH, "rb") as train_fh:
        data = {
            "name":        MODEL_NAME,
            "version":     MODEL_VERSION,
            "description": DESCRIPTION,
            "algorithm":   ALGORITHM,
        }
        if acc is not None:             data["accuracy"]         = str(round(acc, 4))
        if f1 is not None:              data["f1_score"]         = str(round(f1, 4))
        if features_count is not None:  data["features_count"]   = str(features_count)
        if classes is not None:         data["classes"]          = json.dumps(classes)
        if training_dataset:            data["training_dataset"] = training_dataset
        _tm = {k: round(v, 4) for k, v in {
            "accuracy": acc, "f1_score": f1, "precision": precision,
            "recall": recall, "roc_auc": roc_auc,
        }.items() if v is not None}
        if _tm:             data["training_metrics"] = json.dumps(_tm)
        if mlflow_run_id:   data["mlflow_run_id"]    = mlflow_run_id
        if hyperparameters: data["hyperparameters"]  = json.dumps(hyperparameters)

        response = requests.post(
            f"{API_URL}/models",
            headers=HEADERS,
            files={
                "file":       (f"{MODEL_NAME}.pkl",          pkl_fh,   "application/octet-stream"),
                "train_file": ("train_cancer_ExtraTrees.py",  train_fh, "text/plain"),
            },
            data=data, timeout=30,
        )
finally:
    os.unlink(tmp_pkl.name)

if response.status_code == 409:
    print(f"⚠️   {MODEL_NAME} v{MODEL_VERSION} existe déjà — PATCH shadow…")
    patch_resp = requests.patch(
        f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"is_production": False, "deployment_mode": "shadow"},
        timeout=10,
    )
    print("✅  Mode shadow mis à jour." if patch_resp.status_code == 200 else f"❌  PATCH échoué ({patch_resp.status_code})")
    sys.exit(0)

if response.status_code not in (200, 201):
    print(f"❌  Erreur {response.status_code} : {response.text[:300]}")
    sys.exit(1)

res = response.json()
print(f"✅  {res.get('name')} v{res.get('version')} uploadé (id={res.get('id')})")

patch_body = {"is_production": False, "deployment_mode": "shadow", "tags": ["Example"]}
if metrics.get("feature_stats"):
    patch_body["feature_baseline"] = metrics["feature_stats"]
if metrics.get("confidence_threshold") is not None:
    patch_body["confidence_threshold"] = metrics["confidence_threshold"]

patch = requests.patch(
    f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=patch_body, timeout=10,
)
print("✅  Mode shadow configuré." if patch.status_code == 200 else f"⚠️   PATCH échoué ({patch.status_code})")
