"""
send_ground_truth_iris.py — Vérité terrain Iris avec dégradation temporelle
============================================================================

Charge iris_predictions_log.json (produit par send_predictions_iris.py) et
calcule la vraie classe selon les règles de chaque phase, puis envoie les
observed_results via POST /observed-results.

Règles de vérité terrain :
  Phase 1-2 : true_class = classe d'échantillonnage originale
  Phase 3   : si petal_length > 5.0 ET petal_width > 1.7
              → true_class = 2 (virginica) — nouvelle règle déterministe
              Simule un changement biologique : grandes fleurs = virginica,
              même si le modèle (entraîné avant) prédit versicolor.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth_iris.py

Variables d'environnement :
  API_URL    URL de l'API  (défaut : http://localhost:80)
  API_TOKEN  Token Bearer — requis
  MODEL_NAME Nom du modèle (défaut : iris-classifier)
"""

import json
import os
import sys

import requests
from dotenv import find_dotenv, load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(find_dotenv())

# ── Configuration ──────────────────────────────────────────────────────────────

API_URL    = os.environ.get("API_URL",    "http://localhost:80")
API_TOKEN  = os.environ.get("API_TOKEN",  os.environ.get("ADMIN_TOKEN", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "iris-classifier")

HEADERS    = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE   = os.path.join(os.path.dirname(__file__), "iris_predictions_log.json")
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
    print("    Lancez d'abord : python send_predictions_iris.py")
    sys.exit(1)

# ── Chargement du log ──────────────────────────────────────────────────────────

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

print(f"  {len(log)} entrées chargées depuis {LOG_FILE}\n")

# ── Règle de vérité terrain ────────────────────────────────────────────────────


def compute_true_class(entry: dict) -> int:
    """Retourne la vraie classe selon la phase et les features."""
    if entry["phase"] < 3:
        return entry["true_class"]

    # Phase 3 : nouvelle règle déterministe
    petal_len = entry["features"].get("petal length (cm)", 0.0)
    petal_wid = entry["features"].get("petal width (cm)", 0.0)
    if petal_len > 5.0 and petal_wid > 1.7:
        return 2  # virginica — nouvelle règle
    return entry["true_class"]


# ── Construction des observed results ─────────────────────────────────────────

print("=" * 68)
print("  Calcul de la vérité terrain par phase")
print("=" * 68)

observed_data = []
phase_stats   = {p: {"total": 0, "overridden": 0} for p in [1, 2, 3]}

for entry in log:
    true_cls = compute_true_class(entry)
    phase    = entry["phase"]
    phase_stats[phase]["total"] += 1
    if true_cls != entry["true_class"]:
        phase_stats[phase]["overridden"] += 1

    observed_data.append({
        "id_obs":          entry["id_obs"],
        "model_name":      MODEL_NAME,
        "date_time":       entry["timestamp"],
        "observed_result": true_cls,
    })

for phase in [1, 2, 3]:
    s = phase_stats[phase]
    line = f"  Phase {phase} : {s['total']:4d} obs"
    if s["overridden"]:
        pct = s["overridden"] / s["total"] * 100
        line += f"  ({s['overridden']} reclassifiées → virginica par nouvelle règle, {pct:.0f}%)"
    else:
        line += "  (distribution stable)"
    print(line)

print()

# ── Envoi en batches ───────────────────────────────────────────────────────────

print("=" * 68)
print("  Envoi des observed results (POST /observed-results)")
print("=" * 68)

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
print("=" * 68)
print("  Couverture ground truth (GET /observed-results/stats)")
print("=" * 68)

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
print("=" * 68)
print(f"  Total upserted   : {total_upserted}")
print()
print("  Scénario de dégradation simulé :")
print("  Phase 1 (stable) → ~93% accuracy  (population iris originale)")
print("  Phase 2 (drift)  → ~78% accuracy  (pétales grandissants, frontières floues)")
print("  Phase 3 (règle)  → ~62% accuracy  (nouvelle règle : grands pétales = virginica)")
print()
print("  Le modèle nécessite un retraining sur les données de production récentes.")
print()
print(f"  → Dashboard : {API_URL.replace(':8000', ':8501')}/Models")
print("=" * 68)
