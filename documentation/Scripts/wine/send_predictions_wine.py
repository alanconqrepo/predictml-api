"""
send_predictions_wine.py — Envoi de prédictions Wine Regressor via l'API PredictML
====================================================================================

Démontre deux modes d'appel :
  1. Prédictions unitaires   — POST /predict         (une requête par observation)
  2. Prédictions en lot      — POST /predict-batch   (une seule requête, plus efficace)

Le modèle wine-regressor prédit la teneur en alcool (variable continue, ~11–15)
à partir des 12 autres mesures chimiques du dataset Wine sklearn.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_wine.py

Variables d'environnement :
  API_URL        URL de l'API           (défaut : http://localhost:80)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle cible    (défaut : wine-regressor)
  MODEL_VERSION  Version (optionnel)    (défaut : version en production)

Prérequis Python :
  pip install requests
"""

import os
import sys
import uuid

import requests
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL   = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

MODEL_NAME    = os.environ.get("MODEL_NAME",    "wine-regressor")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)   # None = version en production

HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

# Données Wine — observations représentatives (cible : teneur en alcool)
# Features = les 12 mesures chimiques hors "alcohol"
WINE_SAMPLES = [
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "alcool ~14.2 (classe 0 — vin rouge riche)",
        "features": {
            "malic_acid":                     1.71,
            "ash":                            2.43,
            "alcalinity_of_ash":             15.6,
            "magnesium":                     127.0,
            "total_phenols":                  2.80,
            "flavanoids":                     3.06,
            "nonflavanoid_phenols":           0.28,
            "proanthocyanins":                2.29,
            "color_intensity":                5.64,
            "hue":                            1.04,
            "od280/od315_of_diluted_wines":   3.92,
            "proline":                     1065.0,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "alcool ~12.4 (classe 1 — vin équilibré)",
        "features": {
            "malic_acid":                     1.63,
            "ash":                            2.34,
            "alcalinity_of_ash":             14.0,
            "magnesium":                      84.0,
            "total_phenols":                  1.95,
            "flavanoids":                     1.65,
            "nonflavanoid_phenols":           0.34,
            "proanthocyanins":                1.15,
            "color_intensity":                2.40,
            "hue":                            1.01,
            "od280/od315_of_diluted_wines":   1.58,
            "proline":                       285.0,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "alcool ~13.1 (classe 1 — vin corsé)",
        "features": {
            "malic_acid":                     2.36,
            "ash":                            2.67,
            "alcalinity_of_ash":             18.6,
            "magnesium":                     101.0,
            "total_phenols":                  2.80,
            "flavanoids":                     3.24,
            "nonflavanoid_phenols":           0.30,
            "proanthocyanins":                2.81,
            "color_intensity":                5.68,
            "hue":                            1.03,
            "od280/od315_of_diluted_wines":   3.17,
            "proline":                     1185.0,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "alcool ~12.2 (classe 2 — vin léger)",
        "features": {
            "malic_acid":                     3.88,
            "ash":                            2.48,
            "alcalinity_of_ash":             21.2,
            "magnesium":                      88.0,
            "total_phenols":                  1.85,
            "flavanoids":                     0.87,
            "nonflavanoid_phenols":           0.47,
            "proanthocyanins":                1.04,
            "color_intensity":                4.25,
            "hue":                            0.51,
            "od280/od315_of_diluted_wines":   1.42,
            "proline":                       462.0,
        },
    },
    {
        "id_obs":  str(uuid.uuid4()),
        "label":   "alcool ~14.1 (classe 0 — vin puissant)",
        "features": {
            "malic_acid":                     2.02,
            "ash":                            2.40,
            "alcalinity_of_ash":             18.8,
            "magnesium":                     103.0,
            "total_phenols":                  2.75,
            "flavanoids":                     2.92,
            "nonflavanoid_phenols":           0.32,
            "proanthocyanins":                2.38,
            "color_intensity":                6.22,
            "hue":                            1.06,
            "od280/od315_of_diluted_wines":   2.28,
            "proline":                       570.0,
        },
    },
]

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python send_predictions_wine.py")
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

print("=" * 65)
print("  MODE 1 — Prédictions unitaires (POST /predict)")
print("=" * 65)

for sample in WINE_SAMPLES[:3]:
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
        pred_val = data.get("prediction")
        print(
            f"  ✅  {sample['id_obs']:<36}"
            f"  prédit : {str(pred_val):<8}"
            f"  [{sample['label']}]"
        )
    else:
        print(f"  ❌  {sample['id_obs']} — Erreur {r.status_code} : {r.text[:120]}")

# ── 2. Prédictions en lot — POST /predict-batch ───────────────────────────────

print()
print("=" * 65)
print("  MODE 2 — Prédictions en lot (POST /predict-batch)")
print("=" * 65)

batch_inputs = [
    {"id_obs": s["id_obs"], "features": s["features"]}
    for s in WINE_SAMPLES
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
    for pred, sample in zip(predictions, WINE_SAMPLES):
        print(
            f"     {pred.get('id_obs', ''):<36}"
            f"  prédit : {str(pred.get('prediction')):<8}"
            f"  [{sample['label']}]"
        )
else:
    print(f"  ❌  Erreur {r.status_code} : {r.text[:200]}")

# ── 3. Résumé ─────────────────────────────────────────────────────────────────

print()
print("=" * 65)
version_label = f"v{MODEL_VERSION}" if MODEL_VERSION else "production"
print(f"  Modèle utilisé : {MODEL_NAME} ({version_label})")
print(f"  Historique     : {API_URL.replace(':8000', ':8501')}/Predictions")
print("=" * 65)
