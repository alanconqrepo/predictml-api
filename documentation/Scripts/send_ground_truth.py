"""
send_ground_truth.py — Envoi des résultats observés (ground truth) via l'API PredictML
========================================================================================

Ce script illustre le workflow complet pour alimenter la performance réelle d'un modèle :
  1. Envoie des prédictions avec des id_obs déterministes  (POST /predict)
  2. Envoie les labels réels correspondants               (POST /observed-results)
  3. Vérifie la couverture ground truth                   (GET /observed-results/stats)

Le lien prédiction ↔ résultat se fait via id_obs — utilisez les mêmes identifiants
que ceux passés lors de l'appel /predict.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth.py

Variables d'environnement :
  API_URL        URL de l'API           (défaut : http://localhost:8000)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle cible    (défaut : iris-classifier)
  MODEL_VERSION  Version (optionnel)    (défaut : version en production)

Prérequis Python :
  pip install requests
"""

import os
import sys
from datetime import datetime, timezone

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL   = os.environ.get("API_URL",   "http://localhost:8000")
API_TOKEN = os.environ.get("API_TOKEN", "")

MODEL_NAME    = os.environ.get("MODEL_NAME",    "iris-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)

HEADERS      = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
NOW_ISO      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

# Observations avec leurs features et leur label réel (ground truth)
# observed_result = index de classe : 0=setosa, 1=versicolor, 2=virginica
SAMPLES = [
    {
        "id_obs":          "gt-iris-setosa-01",
        "features":        {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                            "petal length (cm)": 1.4, "petal width (cm)": 0.2},
        "observed_result": 0,
        "label":           "setosa",
    },
    {
        "id_obs":          "gt-iris-versicolor-01",
        "features":        {"sepal length (cm)": 6.4, "sepal width (cm)": 3.2,
                            "petal length (cm)": 4.5, "petal width (cm)": 1.5},
        "observed_result": 1,
        "label":           "versicolor",
    },
    {
        "id_obs":          "gt-iris-virginica-01",
        "features":        {"sepal length (cm)": 6.3, "sepal width (cm)": 2.8,
                            "petal length (cm)": 5.1, "petal width (cm)": 1.5},
        "observed_result": 2,
        "label":           "virginica",
    },
    {
        "id_obs":          "gt-iris-setosa-02",
        "features":        {"sepal length (cm)": 4.6, "sepal width (cm)": 3.4,
                            "petal length (cm)": 1.4, "petal width (cm)": 0.3},
        "observed_result": 0,
        "label":           "setosa",
    },
    {
        "id_obs":          "gt-iris-virginica-02",
        "features":        {"sepal length (cm)": 7.7, "sepal width (cm)": 3.8,
                            "petal length (cm)": 6.7, "petal width (cm)": 2.2},
        "observed_result": 2,
        "label":           "virginica",
    },
]

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python send_ground_truth.py")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}\n")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

# ── 1. Envoi des prédictions — crée les observations en DB ───────────────────

print("=" * 62)
print("  ÉTAPE 1 — Envoi des prédictions (POST /predict)")
print("=" * 62)

predictions_sent = []
for s in SAMPLES:
    payload = {"model_name": MODEL_NAME, "id_obs": s["id_obs"], "features": s["features"]}
    if MODEL_VERSION:
        payload["model_version"] = MODEL_VERSION

    r = requests.post(f"{API_URL}/predict", headers=HEADERS, json=payload, timeout=10)

    if r.status_code == 200:
        pred = r.json().get("prediction")
        match = "✅" if str(pred) == str(s["observed_result"]) else "⚠️ "
        print(f"  {match}  {s['id_obs']:<28}  prédit={pred}  attendu={s['observed_result']} ({s['label']})")
        predictions_sent.append(s["id_obs"])
    else:
        print(f"  ❌  {s['id_obs']} — Erreur {r.status_code} : {r.text[:100]}")

print(f"\n  {len(predictions_sent)}/{len(SAMPLES)} prédictions enregistrées en DB.\n")

# ── 2. Envoi du ground truth — POST /observed-results ────────────────────────

print("=" * 62)
print("  ÉTAPE 2 — Envoi du ground truth (POST /observed-results)")
print("=" * 62)

# Seules les observations dont la prédiction a réussi sont envoyées
ground_truth_payload = {
    "data": [
        {
            "id_obs":          s["id_obs"],
            "model_name":      MODEL_NAME,
            "date_time":       NOW_ISO,
            "observed_result": s["observed_result"],
        }
        for s in SAMPLES
        if s["id_obs"] in predictions_sent
    ]
}

r = requests.post(f"{API_URL}/observed-results", headers=HEADERS, json=ground_truth_payload, timeout=10)

if r.status_code == 200:
    result = r.json()
    upserted = result.get("upserted", "?")
    print(f"\n  ✅  {upserted} résultats observés enregistrés (idempotent — re-exécutable sans doublon).\n")
else:
    print(f"\n  ❌  Erreur {r.status_code} : {r.text[:200]}")
    sys.exit(1)

# ── 3. Vérification de la couverture ─────────────────────────────────────────

print("=" * 62)
print("  ÉTAPE 3 — Couverture ground truth")
print("=" * 62)

r = requests.get(
    f"{API_URL}/observed-results/stats",
    headers=HEADERS,
    params={"model_name": MODEL_NAME},
    timeout=10,
)

if r.status_code == 200:
    stats = r.json()
    total   = stats.get("total_predictions", "?")
    labeled = stats.get("labeled_count",     "?")
    rate    = stats.get("coverage_rate",     0.0)
    print(f"\n  Modèle          : {MODEL_NAME}")
    print(f"  Total prédictions labellisées : {labeled} / {total}")
    print(f"  Taux de couverture            : {rate:.1%}")
else:
    print(f"  (stats indisponibles — {r.status_code})")

# ── Résumé ────────────────────────────────────────────────────────────────────

print()
print("=" * 62)
print(f"  → Historique : {API_URL.replace(':8000', ':8501')}/Predictions")
print(f"  → Performance réelle disponible sur la page Models")
print("=" * 62)
