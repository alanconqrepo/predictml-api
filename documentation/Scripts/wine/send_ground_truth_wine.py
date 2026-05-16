"""
send_ground_truth_wine.py — Vérité terrain Wine avec dégradation temporelle
============================================================================

Charge wine_predictions_log.json (produit par send_predictions_wine.py) et
calcule la vraie teneur en alcool selon les règles de chaque phase, puis
envoie les observed_results via POST /observed-results.

Règles de vérité terrain (régression) :
  Phase 1-2 : true_alcohol = base_alcohol_du_type + bruit gaussien N(0, 0.35)
  Phase 3   : si proline > 1200 ET color_intensity > 8.0
              → true_alcohol += 1.8 (biais systématique — nouvelle technique)
              Le modèle ignore ce biais et sous-prédit massivement.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth_wine.py

Variables d'environnement :
  API_URL    URL de l'API  (défaut : http://localhost:80)
  API_TOKEN  Token Bearer — requis
  MODEL_NAME Nom du modèle (défaut : wine-regressor)
  NOISE_STD  Écart-type du bruit gaussien (défaut : 0.35)
"""

import json
import os
import random
import sys

import requests
from dotenv import find_dotenv, load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(find_dotenv())

# ── Configuration ──────────────────────────────────────────────────────────────

API_URL    = os.environ.get("API_URL",    "http://localhost:80")
API_TOKEN  = os.environ.get("API_TOKEN",  os.environ.get("ADMIN_TOKEN", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "wine-regressor")
NOISE_STD  = float(os.environ.get("NOISE_STD", "0.35"))

# Biais ajouté en phase 3 pour les vins concernés par la nouvelle règle
PHASE3_BIAS   = 1.8
PROLINE_THR   = 1200.0
COLOR_INT_THR = 8.0

HEADERS    = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE   = os.path.join(os.path.dirname(__file__), "wine_predictions_log.json")
BATCH_SIZE = 50

# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}\n")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

if not os.path.exists(LOG_FILE):
    print(f"❌  Log introuvable : {LOG_FILE}")
    print("    Lancez d'abord : python send_predictions_wine.py")
    sys.exit(1)

# ── Chargement du log ──────────────────────────────────────────────────────────

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

print(f"  {len(log)} entrées chargées depuis {LOG_FILE}\n")

# ── Règle de vérité terrain (régression) ──────────────────────────────────────

rng = random.Random(99)


def compute_true_alcohol(entry: dict) -> float:
    """Calcule la vraie teneur en alcool selon la phase et les features."""
    base         = entry["base_alcohol"]
    noise        = rng.gauss(0.0, NOISE_STD)
    true_alcohol = base + noise

    if entry["phase"] == 3:
        proline   = entry["features"].get("proline",         0.0)
        color_int = entry["features"].get("color_intensity", 0.0)
        if proline > PROLINE_THR and color_int > COLOR_INT_THR:
            true_alcohol += PHASE3_BIAS  # biais systématique non vu par le modèle

    return round(true_alcohol, 2)


# ── Construction des observed results ─────────────────────────────────────────

print("=" * 72)
print("  Calcul de la vérité terrain par phase")
print("=" * 72)

observed_data = []
phase_stats   = {p: {"total": 0, "biased": 0, "alcohol_values": []} for p in [1, 2, 3]}

for entry in log:
    true_alc = compute_true_alcohol(entry)
    phase    = entry["phase"]

    is_biased = (
        phase == 3
        and entry["features"].get("proline",         0.0) > PROLINE_THR
        and entry["features"].get("color_intensity", 0.0) > COLOR_INT_THR
    )
    phase_stats[phase]["total"] += 1
    phase_stats[phase]["alcohol_values"].append(true_alc)
    if is_biased:
        phase_stats[phase]["biased"] += 1

    observed_data.append({
        "id_obs":          entry["id_obs"],
        "model_name":      MODEL_NAME,
        "date_time":       entry["timestamp"],
        "observed_result": true_alc,
    })

for phase in [1, 2, 3]:
    s    = phase_stats[phase]
    vals = s["alcohol_values"]
    mean = sum(vals) / len(vals) if vals else 0.0
    line = f"  Phase {phase} : {s['total']:4d} obs  alcool moyen={mean:.2f}%"
    if s["biased"]:
        pct = s["biased"] / s["total"] * 100
        line += f"  ({s['biased']} avec biais +{PHASE3_BIAS}, {pct:.0f}%)"
    print(line)

print()

# ── Envoi en batches ───────────────────────────────────────────────────────────

print("=" * 72)
print("  Envoi des observed results (POST /observed-results)")
print("=" * 72)

total_upserted = 0
for i in range(0, len(observed_data), BATCH_SIZE):
    chunk = observed_data[i : i + BATCH_SIZE]
    r = requests.post(
        f"{API_URL}/observed-results",
        headers=HEADERS,
        json={"data": chunk},
        timeout=30,
    )
    if r.status_code == 200:
        n = r.json().get("upserted", len(chunk))
        total_upserted += n
        end_idx = min(i + BATCH_SIZE, len(observed_data))
        print(f"  [{i + 1:4d} – {end_idx:4d}]  ✅  {n} upserted")
    else:
        print(f"  [{i + 1:4d} – ...]  ❌  Erreur {r.status_code} : {r.text[:100]}")

# ── Vérification de couverture ─────────────────────────────────────────────────

print()
print("=" * 72)
print("  Couverture ground truth (GET /observed-results/stats)")
print("=" * 72)

r = requests.get(
    f"{API_URL}/observed-results/stats",
    headers=HEADERS,
    params={"model_name": MODEL_NAME},
    timeout=10,
)
if r.status_code == 200:
    stats = r.json()
    print(f"\n  Modèle      : {MODEL_NAME}")
    print(
        f"  Labellisées : {stats.get('labeled_count', '?')}"
        f" / {stats.get('total_predictions', '?')}"
    )
    print(f"  Couverture  : {stats.get('coverage_rate', 0.0):.1%}")
else:
    print(f"  (stats indisponibles — {r.status_code})")

# ── Résumé ─────────────────────────────────────────────────────────────────────

print()
print("=" * 72)
print(f"  Total upserted   : {total_upserted}")
print()
print("  Scénario de dégradation simulé :")
print("  Phase 1 (stable)  → RMSE ~0.50  (distribution normale)")
print("  Phase 2 (drift)   → RMSE ~0.85  (nouveau cépage, proline en hausse)")
print(
    f"  Phase 3 (règle)   → RMSE ~1.50"
    f"  (biais +{PHASE3_BIAS} pour proline>{PROLINE_THR:.0f} & color_intensity>{COLOR_INT_THR})"
)
print()
print("  Le modèle nécessite un retraining sur les données de production récentes.")
print()
print(f"  → Dashboard : {API_URL.replace(':8000', ':8501')}/Models")
print("=" * 72)
