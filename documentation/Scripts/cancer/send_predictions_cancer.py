"""
send_predictions_cancer.py — Envoi de prédictions Breast Cancer via l'API PredictML
=====================================================================================

Démontre deux modes d'appel :
  1. Prédictions unitaires   — POST /predict         (une requête par observation)
  2. Prédictions en lot      — POST /predict-batch   (une seule requête, plus efficace)

Dataset : Breast Cancer Wisconsin (classification binaire : malignant vs benign)
Features : 30 mesures nucléaires (mean radius, mean texture, mean perimeter, etc.)

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_cancer.py

Variables d'environnement :
  API_URL        URL de l'API           (défaut : http://localhost:8000)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle cible    (défaut : cancer-classifier)
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

MODEL_NAME    = os.environ.get("MODEL_NAME",    "cancer-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)

HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

# Observations représentatives du dataset Breast Cancer Wisconsin
# Source : sklearn.datasets.load_breast_cancer() — quelques exemples typiques
CANCER_SAMPLES = [
    {
        "id_obs": str(uuid.uuid4()),
        "label":  "malignant (attendu)",
        "features": {
            "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8,
            "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776,
            "mean concavity": 0.3001, "mean concave points": 0.1471,
            "mean symmetry": 0.2419, "mean fractal dimension": 0.07871,
            "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589,
            "area error": 153.4, "smoothness error": 0.006399, "compactness error": 0.04904,
            "concavity error": 0.05373, "concave points error": 0.01587,
            "symmetry error": 0.03003, "fractal dimension error": 0.006193,
            "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6,
            "worst area": 2019.0, "worst smoothness": 0.1622, "worst compactness": 0.6656,
            "worst concavity": 0.7119, "worst concave points": 0.2654,
            "worst symmetry": 0.4601, "worst fractal dimension": 0.1189,
        },
    },
    {
        "id_obs": str(uuid.uuid4()),
        "label":  "malignant (attendu)",
        "features": {
            "mean radius": 20.57, "mean texture": 17.77, "mean perimeter": 132.9,
            "mean area": 1326.0, "mean smoothness": 0.08474, "mean compactness": 0.07864,
            "mean concavity": 0.0869, "mean concave points": 0.07017,
            "mean symmetry": 0.1812, "mean fractal dimension": 0.05667,
            "radius error": 0.5435, "texture error": 0.7339, "perimeter error": 3.398,
            "area error": 74.08, "smoothness error": 0.005225, "compactness error": 0.01308,
            "concavity error": 0.01860, "concave points error": 0.01340,
            "symmetry error": 0.01389, "fractal dimension error": 0.003532,
            "worst radius": 24.99, "worst texture": 23.41, "worst perimeter": 158.8,
            "worst area": 1956.0, "worst smoothness": 0.1238, "worst compactness": 0.1866,
            "worst concavity": 0.2416, "worst concave points": 0.1860,
            "worst symmetry": 0.2750, "worst fractal dimension": 0.08902,
        },
    },
    {
        "id_obs": str(uuid.uuid4()),
        "label":  "benign (attendu)",
        "features": {
            "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46,
            "mean area": 566.3, "mean smoothness": 0.09779, "mean compactness": 0.08129,
            "mean concavity": 0.06664, "mean concave points": 0.04781,
            "mean symmetry": 0.1885, "mean fractal dimension": 0.05766,
            "radius error": 0.2699, "texture error": 0.7886, "perimeter error": 2.058,
            "area error": 23.56, "smoothness error": 0.008462, "compactness error": 0.01460,
            "concavity error": 0.02387, "concave points error": 0.01315,
            "symmetry error": 0.01980, "fractal dimension error": 0.002300,
            "worst radius": 15.11, "worst texture": 19.26, "worst perimeter": 99.70,
            "worst area": 711.2, "worst smoothness": 0.1440, "worst compactness": 0.1773,
            "worst concavity": 0.2390, "worst concave points": 0.1288,
            "worst symmetry": 0.2977, "worst fractal dimension": 0.07259,
        },
    },
    {
        "id_obs": str(uuid.uuid4()),
        "label":  "benign (attendu)",
        "features": {
            "mean radius": 9.463, "mean texture": 19.07, "mean perimeter": 60.45,
            "mean area": 271.7, "mean smoothness": 0.1065, "mean compactness": 0.1068,
            "mean concavity": 0.06812, "mean concave points": 0.03404,
            "mean symmetry": 0.1854, "mean fractal dimension": 0.07130,
            "radius error": 0.3276, "texture error": 1.432, "perimeter error": 2.158,
            "area error": 17.54, "smoothness error": 0.006735, "compactness error": 0.02505,
            "concavity error": 0.02426, "concave points error": 0.009044,
            "symmetry error": 0.01716, "fractal dimension error": 0.004124,
            "worst radius": 10.56, "worst texture": 24.98, "worst perimeter": 68.17,
            "worst area": 331.9, "worst smoothness": 0.1403, "worst compactness": 0.2536,
            "worst concavity": 0.2262, "worst concave points": 0.09585,
            "worst symmetry": 0.2948, "worst fractal dimension": 0.09519,
        },
    },
    {
        "id_obs": str(uuid.uuid4()),
        "label":  "benign (attendu)",
        "features": {
            "mean radius": 12.45, "mean texture": 15.70, "mean perimeter": 82.57,
            "mean area": 477.1, "mean smoothness": 0.1278, "mean compactness": 0.17000,
            "mean concavity": 0.1578, "mean concave points": 0.08089,
            "mean symmetry": 0.2087, "mean fractal dimension": 0.07613,
            "radius error": 0.3345, "texture error": 0.8902, "perimeter error": 2.217,
            "area error": 27.19, "smoothness error": 0.007491, "compactness error": 0.03323,
            "concavity error": 0.03564, "concave points error": 0.01769,
            "symmetry error": 0.01827, "fractal dimension error": 0.004972,
            "worst radius": 14.50, "worst texture": 20.47, "worst perimeter": 94.28,
            "worst area": 640.2, "worst smoothness": 0.1703, "worst compactness": 0.3179,
            "worst concavity": 0.3530, "worst concave points": 0.1661,
            "worst symmetry": 0.3414, "worst fractal dimension": 0.1005,
        },
    },
]

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python send_predictions_cancer.py")
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

for sample in CANCER_SAMPLES[:3]:
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
    for s in CANCER_SAMPLES
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
    for pred, sample in zip(predictions, CANCER_SAMPLES):
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
