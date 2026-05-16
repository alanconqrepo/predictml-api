"""
send_ground_truth_wine.py — Envoi des résultats observés (ground truth) via l'API PredictML
=============================================================================================

Ce script récupère les prédictions existantes en base (GET /predictions) et envoie
les valeurs réelles correspondantes (POST /observed-results) en simulant un taux d'erreur
réaliste pour un modèle de régression :
  - 80 % des cas : observed_result ≈ valeur prédite  (bruit Gaussien faible ± 0.1)
  - 20 % des cas : observed_result = valeur prédite ± erreur significative (± 1.0–2.5)

La teneur en alcool du dataset Wine varie entre 11.0 et 14.8.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth_wine.py

Variables d'environnement :
  API_URL        URL de l'API           (défaut : http://localhost:80)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle cible    (défaut : wine-regressor)
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

MODEL_NAME = os.environ.get("MODEL_NAME", "wine-regressor")
DAYS_BACK  = int(os.environ.get("DAYS_BACK",   "30"))
ERROR_RATE = float(os.environ.get("ERROR_RATE", "0.20"))

# Bornes physiques de la teneur en alcool dans le dataset Wine
ALCOHOL_MIN = 11.0
ALCOHOL_MAX = 14.8

HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
NOW     = datetime.now(timezone.utc)
NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%S")

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python send_ground_truth_wine.py")
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

    # Garder uniquement les prédictions numériques avec id_obs (régression → float)
    all_predictions.extend(
        p for p in page
        if p.get("id_obs") and isinstance(p.get("prediction_result"), (int, float))
    )

    if not next_cursor:
        break
    cursor = next_cursor

print(f"\n  {len(all_predictions)} prédictions numériques trouvées sur les {DAYS_BACK} derniers jours.\n")

if not all_predictions:
    print("  ⚠️   Aucune prédiction disponible. Lancez d'abord send_predictions_wine.py.")
    sys.exit(0)

# ── 2. Construction des observed results ──────────────────────────────────────

print("=" * 62)
print("  ÉTAPE 2 — Construction du ground truth (80 % proche / 20 % erreur)")
print("=" * 62)


def pick_wrong_value(predicted: float) -> float:
    """Retourne une valeur d'alcool significativement différente de la prédiction."""
    error = random.uniform(1.0, 2.5) * random.choice([-1, 1])
    result = round(predicted + error, 2)
    # Maintenir dans les bornes physiques
    return round(max(ALCOHOL_MIN, min(ALCOHOL_MAX, result)), 2)


def pick_correct_value(predicted: float) -> float:
    """Retourne la valeur observée avec un léger bruit de mesure (± 0.1)."""
    noise = random.uniform(-0.1, 0.1)
    return round(predicted + noise, 2)


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
    predicted = float(pred["prediction_result"])
    is_error  = random.random() < ERROR_RATE

    if is_error:
        observed = pick_wrong_value(predicted)
        error_count += 1
        marker = "⚠️ "
    else:
        observed = pick_correct_value(predicted)
        correct_count += 1
        marker = "✅"

    print(
        f"  {marker}  {pred['id_obs']:<36}"
        f"  prédit={predicted:<8.3f}"
        f"  observé={observed:.3f}"
    )

    ground_truth_data.append({
        "id_obs":          pred["id_obs"],
        "model_name":      MODEL_NAME,
        "date_time":       NOW_ISO,
        "observed_result": observed,
    })

print(
    f"\n  {correct_count} proches · {error_count} erreurs"
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
