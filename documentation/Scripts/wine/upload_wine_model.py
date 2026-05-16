"""
upload_wine_model.py — Entraîne et uploade un modèle Wine (Régression) via l'API PredictML
===========================================================================================

Ce script tourne LOCALEMENT. Il :
  1. Exécute train_wine.py en subprocess pour produire le .pkl
  2. Uploade le .pkl + train_wine.py via POST /models
  3. Met le modèle en production avec le tag "Example" et la baseline des features

Modèle : GradientBoostingRegressor — prédit la teneur en alcool (variable continue)
         à partir des 12 autres mesures chimiques du dataset Wine sklearn.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python upload_wine_model.py

Variables d'environnement :
  API_URL        URL de l'API          (défaut : http://localhost:80)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle         (défaut : wine-regressor)
  MODEL_VERSION  Version               (défaut : 1.0.0)
  TRAIN_START    Date début training   (défaut : 2024-01-01)
  TRAIN_END      Date fin training     (défaut : 2024-12-31)

Prérequis Python :
  pip install requests scikit-learn numpy pandas
"""

import json
import os
import subprocess
import sys
import tempfile

import requests
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL   = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

MODEL_NAME    = os.environ.get("MODEL_NAME",    "wine-regressor")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
DESCRIPTION   = "GradientBoostingRegressor entraîné sur le dataset Wine (régression — cible : teneur en alcool)"
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
TRAIN_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "train_wine.py")

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python upload_wine_model.py")
    sys.exit(1)

if not os.path.exists(TRAIN_SCRIPT_PATH):
    print(f"❌  train_wine.py introuvable : {TRAIN_SCRIPT_PATH}")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# ── 1. Vérification que l'API est accessible ──────────────────────────────────

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    print(f"✅  API accessible : {API_URL}")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

# ── 2. Exécution de train_wine.py ─────────────────────────────────────────────

tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
tmp_pkl.close()

print(f"⏳  Entraînement via train_wine.py ({TRAIN_START} → {TRAIN_END})…")

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
    print("❌  train_wine.py a échoué :")
    print(result.stderr)
    os.unlink(tmp_pkl.name)
    sys.exit(1)

if result.stderr:
    print(result.stderr.strip())

# Récupération des métriques depuis la dernière ligne JSON de stdout
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
print(
    f"✅  Entraînement terminé — R² : {r2} | MAE : {mae} | RMSE : {rmse}"
    + (f" | MLflow run : {mlflow_run_id}" if mlflow_run_id else "")
)

# ── 3. Upload via POST /models ────────────────────────────────────────────────

print(f"⏳  Upload de {MODEL_NAME} v{MODEL_VERSION}…")

try:
    with open(tmp_pkl.name, "rb") as pkl_fh, open(TRAIN_SCRIPT_PATH, "rb") as train_fh:
        data = {
            "name":        MODEL_NAME,
            "version":     MODEL_VERSION,
            "description": DESCRIPTION,
            "algorithm":   ALGORITHM,
        }
        # Pour la régression : accuracy = R² (meilleur proxy pour l'affichage dashboard)
        if r2 is not None:
            data["accuracy"] = str(round(r2, 4))
        if features_count is not None:
            data["features_count"] = str(features_count)
        if training_dataset:
            data["training_dataset"] = training_dataset
        if mlflow_run_id:
            data["mlflow_run_id"] = mlflow_run_id
        # training_metrics : métriques de régression complètes
        _tm = {k: round(v, 4) for k, v in {
            "mae":  mae,
            "rmse": rmse,
            "r2":   r2,
        }.items() if v is not None}
        if _tm:
            data["training_metrics"] = json.dumps(_tm)
        if hyperparameters:
            data["hyperparameters"] = json.dumps(hyperparameters)

        response = requests.post(
            f"{API_URL}/models",
            headers=HEADERS,
            files={
                "file":       (f"{MODEL_NAME}.pkl", pkl_fh,   "application/octet-stream"),
                "train_file": ("train_wine.py",      train_fh, "text/plain"),
            },
            data=data,
            timeout=30,
        )
finally:
    os.unlink(tmp_pkl.name)

# ── 4. Résultat upload ────────────────────────────────────────────────────────

if response.status_code == 409:
    print(f"⚠️   {MODEL_NAME} v{MODEL_VERSION} existe déjà — mise à jour des hyperparamètres via PATCH…")
    patch_payload = {}
    if hyperparameters:
        patch_payload["hyperparameters"] = hyperparameters
    if patch_payload:
        patch_resp = requests.patch(
            f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
            headers={**HEADERS, "Content-Type": "application/json"},
            json=patch_payload,
            timeout=10,
        )
        if patch_resp.status_code == 200:
            print(f"✅  Hyperparamètres mis à jour.")
        else:
            print(f"❌  PATCH échoué ({patch_resp.status_code}) : {patch_resp.text[:200]}")
    sys.exit(0)

if response.status_code not in (200, 201):
    print(f"\n❌  Erreur {response.status_code}")
    try:
        body = response.json()
        print(f"   Détail : {json.dumps(body, indent=2, ensure_ascii=False)}")
    except Exception:
        print(f"   Réponse : {response.text[:500]}")
    sys.exit(1)

res = response.json()
print(f"\n✅  Modèle uploadé avec succès !")
print(f"   Nom       : {res.get('name')}")
print(f"   Version   : {res.get('version')}")
print(f"   ID        : {res.get('id')}")
print(f"   Algorithme: {ALGORITHM}")

# ── 5. Mise en production + tag "Example" + feature_baseline ─────────────────

print(f"⏳  Mise en production, tag 'Example' et baseline des features…")

patch_body: dict = {"is_production": True, "tags": ["Example"]}
if metrics.get("feature_stats"):
    patch_body["feature_baseline"] = metrics["feature_stats"]

patch = requests.patch(
    f"{API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=patch_body,
    timeout=10,
)

if patch.status_code == 200:
    baseline_ok = "feature_baseline" in patch_body
    print(f"✅  Modèle passé en production avec le tag 'Example'"
          f"{' et baseline des features' if baseline_ok else ''}.")
else:
    print(f"⚠️   PATCH échoué ({patch.status_code}) : {patch.text[:200]}")

print(f"\n   → Dashboard : {API_URL.replace(':8000', ':8501')}/Models")
print(f"   → Ré-entraîner :")
print(f"       POST {API_URL}/models/{MODEL_NAME}/{MODEL_VERSION}/retrain")
