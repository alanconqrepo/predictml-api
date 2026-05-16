"""
send_predictions_iris.py — Envoi de prédictions Iris via l'API PredictML
=========================================================================

Démontre deux modes d'appel :
  1. Prédictions unitaires   — POST /predict         (une requête par observation)
  2. Prédictions en lot      — POST /predict-batch   (une seule requête, plus efficace)

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_iris.py

Variables d'environnement :
  API_URL        URL de l'API           (défaut : http://localhost:8000)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle cible    (défaut : iris-classifier)
  MODEL_VERSION  Version (optionnel)    (défaut : version en production)

Prérequis Python :
  pip install requests
"""

import json
import os
import sys
import uuid

import requests
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
# ── Configuration ─────────────────────────────────────────────────────────────

API_URL   = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

MODEL_NAME    = os.environ.get("MODEL_NAME",    "iris-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)   # None = version en production

HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

# Données Iris — quelques observations représentatives des 3 classes
IRIS_SAMPLES = [
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "setosa (attendu)",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)":  3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)":  0.2,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "versicolor (attendu)",
        "features": {
            "sepal length (cm)": 6.4,
            "sepal width (cm)":  3.2,
            "petal length (cm)": 4.5,
            "petal width (cm)":  1.5,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "virginica (attendu)",
        "features": {
            "sepal length (cm)": 6.3,
            "sepal width (cm)":  2.8,
            "petal length (cm)": 5.1,
            "petal width (cm)":  1.5,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "setosa (attendu)",
        "features": {
            "sepal length (cm)": 4.6,
            "sepal width (cm)":  3.4,
            "petal length (cm)": 1.4,
            "petal width (cm)":  0.3,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "virginica (attendu)",
        "features": {
            "sepal length (cm)": 7.7,
            "sepal width (cm)":  3.8,
            "petal length (cm)": 6.7,
            "petal width (cm)":  2.2,
        },
    },
]

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python send_predictions_iris.py")
    sys.exit(1)

# ── 0. Vérification API ───────────────────────────────────────────────────────

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}\n")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

# ── 1. Prédictions unitaires — POST /predict ──────────────────────────────────

print("=" * 60)
print("  MODE 1 — Prédictions unitaires (POST /predict)")
print("=" * 60)

for sample in IRIS_SAMPLES[:3]:
    payload = {
        "model_name": MODEL_NAME,
        "id_obs":     sample["id_obs"],
        "features":   sample["features"],
    }
    if MODEL_VERSION:
        payload["model_version"] = MODEL_VERSION

    r = requests.post(f"{API_URL}/predict", headers=HEADERS, json=payload, timeout=10)

    if r.status_code == 200:
        data = r.json()
        print(
            f"  ✅  {sample['id_obs']:<25}"
            f"  prédit : {str(data.get('prediction')):<12}"
            f"  [{sample['label']}]"
        )
    else:
        print(f"  ❌  {sample['id_obs']} — Erreur {r.status_code} : {r.text[:120]}")

# ── 2. Prédictions en lot — POST /predict-batch ───────────────────────────────

print()
print("=" * 60)
print("  MODE 2 — Prédictions en lot (POST /predict-batch)")
print("=" * 60)

batch_inputs = [
    {"id_obs": s["id_obs"], "features": s["features"]}
    for s in IRIS_SAMPLES
]
batch_payload = {
    "model_name": MODEL_NAME,
    "inputs":     batch_inputs,
}
if MODEL_VERSION:
    batch_payload["model_version"] = MODEL_VERSION

r = requests.post(f"{API_URL}/predict-batch", headers=HEADERS, json=batch_payload, timeout=30)

if r.status_code == 200:
    results = r.json()
    predictions = results if isinstance(results, list) else results.get("predictions", [])
    print(f"  ✅  {len(predictions)} prédictions reçues :\n")
    for pred, sample in zip(predictions, IRIS_SAMPLES):
        print(
            f"     {pred.get('id_obs', ''):<30}"
            f"  prédit : {str(pred.get('prediction')):<12}"
            f"  [{sample['label']}]"
        )
else:
    print(f"  ❌  Erreur {r.status_code} : {r.text[:200]}")

# ── 3. Résumé ─────────────────────────────────────────────────────────────────

print()
print("=" * 60)
version_label = f"v{MODEL_VERSION}" if MODEL_VERSION else "production"
print(f"  Modèle utilisé : {MODEL_NAME} ({version_label})")
print(f"  Historique     : {API_URL.replace(':8000', ':8501')}/Predictions")
print("=" * 60)
