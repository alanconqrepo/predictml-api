"""
send_ground_truth_iris.py — Envoi des résultats observés (ground truth) via l'API PredictML
============================================================================================

Ce script récupère les prédictions existantes en base (GET /predictions) et envoie
les labels réels correspondants (POST /observed-results) en simulant un taux d'erreur
réaliste :
  - 80 % des cas : observed_result = valeur prédite  (prédiction correcte)
  - 20 % des cas : observed_result = autre classe     (erreur de prédiction)

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth_iris.py

Variables d'environnement :
  API_URL        URL de l'API           (défaut : http://localhost:8000)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle cible    (défaut : iris-classifier)
  DAYS_BACK      Fenêtre de recherche   (défaut : 30 jours)
  ERROR_RATE     Taux d'erreur simulé   (défaut : 0.20)

Prérequis Python :
  pip install requests
"""

import os
import random
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import find_dotenv, load_dotenv

# Force UTF-8 sur Windows (terminal cp1252 ne supporte pas les emojis)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL   = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

MODEL_NAME  = os.environ.get("MODEL_NAME", "iris-classifier")
DAYS_BACK   = int(os.environ.get("DAYS_BACK",   "30"))
ERROR_RATE  = float(os.environ.get("ERROR_RATE", "0.20"))

HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
NOW     = datetime.now(timezone.utc)
NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%S")

# Classes connues pour iris (index entier ou label string)
IRIS_INT_CLASSES    = [0, 1, 2]
IRIS_STRING_CLASSES = ["setosa", "versicolor", "virginica"]

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python send_ground_truth_iris.py")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}\n")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

# ── 1. Récupération des prédictions existantes ────────────────────────────────

print("=" * 62)
print("  ÉTAPE 1 — Récupération des prédictions (GET /predictions)")
print("=" * 62)

start_dt = (NOW - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%dT%H:%M:%S")
end_dt   = NOW.strftime("%Y-%m-%dT%H:%M:%S")

all_predictions = []
cursor = None

while True:
    params = {
        "name":  MODEL_NAME,
        "start": start_dt,
        "end":   end_dt,
        "limit": 200,
    }
    if cursor:
        params["cursor"] = cursor

    r = requests.get(f"{API_URL}/predictions", headers=HEADERS, params=params, timeout=15)

    if r.status_code != 200:
        print(f"  ❌  Erreur {r.status_code} : {r.text[:200]}")
        sys.exit(1)

    data        = r.json()
    page        = data.get("predictions", [])
    next_cursor = data.get("next_cursor")

    # Garder uniquement les prédictions avec id_obs et un prediction_result scalar
    all_predictions.extend(
        p for p in page
        if p.get("id_obs") and isinstance(p.get("prediction_result"), (int, float, str))
    )

    if not next_cursor:
        break
    cursor = next_cursor

print(f"\n  {len(all_predictions)} prédictions avec id_obs trouvées sur les {DAYS_BACK} derniers jours.\n")

if not all_predictions:
    print("  ⚠️   Aucune prédiction disponible. Lancez d'abord send_predictions_iris.py.")
    sys.exit(0)

# ── 2. Construction des observed results ──────────────────────────────────────

print("=" * 62)
print("  ÉTAPE 2 — Construction du ground truth (80 % correct / 20 % erreur)")
print("=" * 62)


def pick_wrong_value(predicted):
    """Retourne une classe iris différente de la valeur prédite."""
    if isinstance(predicted, bool):
        # bool est sous-classe de int — traiter séparément
        others = [c for c in IRIS_INT_CLASSES if c != int(predicted)]
    elif isinstance(predicted, int):
        others = [c for c in IRIS_INT_CLASSES if c != predicted]
    elif isinstance(predicted, float):
        others = [c for c in IRIS_INT_CLASSES if c != int(predicted)]
    elif isinstance(predicted, str):
        others = [c for c in IRIS_STRING_CLASSES if c != predicted]
    else:
        others = IRIS_INT_CLASSES[:]
    return random.choice(others) if others else predicted


random.seed(42)  # Reproductibilité

# Dédupliquer par (id_obs, model_name) — garder la première occurrence
seen: set[tuple] = set()
unique_predictions = []
for pred in all_predictions:
    key = (pred["id_obs"], MODEL_NAME)
    if key not in seen:
        seen.add(key)
        unique_predictions.append(pred)

if len(unique_predictions) < len(all_predictions):
    print(
        f"  ℹ️   {len(all_predictions) - len(unique_predictions)} doublon(s) ignoré(s)"
        f" (même id_obs présent plusieurs fois en DB).\n"
    )

ground_truth_data = []
correct_count = 0
error_count   = 0

for pred in unique_predictions:
    predicted    = pred["prediction_result"]
    is_error     = random.random() < ERROR_RATE
    observed     = pick_wrong_value(predicted) if is_error else predicted

    if is_error:
        error_count += 1
        marker = "⚠️ "
    else:
        correct_count += 1
        marker = "✅"

    print(
        f"  {marker}  {pred['id_obs']:<36}"
        f"  prédit={str(predicted):<12}"
        f"  observé={str(observed)}"
    )

    ground_truth_data.append({
        "id_obs":          pred["id_obs"],
        "model_name":      MODEL_NAME,
        "date_time":       NOW_ISO,
        "observed_result": observed,
    })

print(
    f"\n  {correct_count} correctes · {error_count} erreurs"
    f"  (taux d'erreur réel : {error_count / len(ground_truth_data):.1%})\n"
)

# ── 3. Envoi du ground truth — POST /observed-results ────────────────────────

print("=" * 62)
print("  ÉTAPE 3 — Envoi du ground truth (POST /observed-results)")
print("=" * 62)

r = requests.post(
    f"{API_URL}/observed-results",
    headers=HEADERS,
    json={"data": ground_truth_data},
    timeout=30,
)

if r.status_code == 200:
    result   = r.json()
    upserted = result.get("upserted", "?")
    print(f"\n  ✅  {upserted} résultats observés enregistrés (idempotent — re-exécutable sans doublon).\n")
else:
    print(f"\n  ❌  Erreur {r.status_code} : {r.text[:200]}")
    sys.exit(1)

# ── 4. Vérification de la couverture ─────────────────────────────────────────

print("=" * 62)
print("  ÉTAPE 4 — Couverture ground truth")
print("=" * 62)

r = requests.get(
    f"{API_URL}/observed-results/stats",
    headers=HEADERS,
    params={"model_name": MODEL_NAME},
    timeout=10,
)

if r.status_code == 200:
    stats   = r.json()
    total   = stats.get("total_predictions", "?")
    labeled = stats.get("labeled_count",     "?")
    rate    = stats.get("coverage_rate",     0.0)
    print(f"\n  Modèle                        : {MODEL_NAME}")
    print(f"  Prédictions labellisées       : {labeled} / {total}")
    print(f"  Taux de couverture            : {rate:.1%}")
else:
    print(f"  (stats indisponibles — {r.status_code})")

# ── Résumé ────────────────────────────────────────────────────────────────────

print()
print("=" * 62)
print(f"  → Historique : {API_URL.replace(':8000', ':8501')}/Predictions")
print(f"  → Performance réelle disponible sur la page Models")
print("=" * 62)
