"""
send_predictions_iris.py — 900 prédictions Iris étalées sur 45 jours
=====================================================================

Simule une dégradation progressive du modèle en production :

  Phase 1 (J-45 → J-31, 15 jours) : distribution stable, ~93% accuracy
  Phase 2 (J-30 → J-16, 15 jours) : drift des features (pétales grandissants)
  Phase 3 (J-15 → J-1,  15 jours) : nouvelle règle déterministe — toute fleur
    avec petal_length > 5.0 ET petal_width > 1.7 est en réalité virginica,
    même si le modèle (entraîné sur l'ancienne distribution) prédit versicolor.

Produit iris_predictions_log.json — utilisé par send_ground_truth_iris.py.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_iris.py

Variables d'environnement :
  API_URL        URL de l'API              (défaut : http://localhost:80)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle             (défaut : iris-classifier)
  MODEL_VERSION  Version (optionnel)
  TOTAL_DAYS     Nombre de jours           (défaut : 45)
  N_PER_DAY      Prédictions par jour      (défaut : 20)
  SLEEP_BETWEEN  Pause entre batches (s)   (défaut : 7)
"""

import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import requests
from dotenv import find_dotenv, load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(find_dotenv())

# ── Configuration ──────────────────────────────────────────────────────────────

API_URL       = os.environ.get("API_URL",       "http://localhost:80")
API_TOKEN     = os.environ.get("API_TOKEN",     os.environ.get("ADMIN_TOKEN", ""))
MODEL_NAME    = os.environ.get("MODEL_NAME",    "iris-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)
TOTAL_DAYS    = int(os.environ.get("TOTAL_DAYS",    "45"))
N_PER_DAY     = int(os.environ.get("N_PER_DAY",     "20"))
SLEEP_BETWEEN = float(os.environ.get("SLEEP_BETWEEN", "7"))

HEADERS  = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE = os.path.join(os.path.dirname(__file__), "iris_predictions_log.json")

# ── Distributions par classe (μ, σ) — source : sklearn.datasets.load_iris() ───

CLASS_PARAMS = {
    0: {  # setosa
        "sepal length (cm)": (5.006, 0.352),
        "sepal width (cm)":  (3.428, 0.379),
        "petal length (cm)": (1.462, 0.174),
        "petal width (cm)":  (0.246, 0.105),
    },
    1: {  # versicolor
        "sepal length (cm)": (5.936, 0.516),
        "sepal width (cm)":  (2.770, 0.314),
        "petal length (cm)": (4.260, 0.470),
        "petal width (cm)":  (1.326, 0.198),
    },
    2: {  # virginica
        "sepal length (cm)": (6.588, 0.636),
        "sepal width (cm)":  (2.974, 0.322),
        "petal length (cm)": (5.552, 0.552),
        "petal width (cm)":  (2.026, 0.275),
    },
}

# Proportions [setosa, versicolor, virginica] par phase
PHASE_PROPORTIONS = {
    1: [0.40, 0.40, 0.20],   # stable et équilibré
    2: [0.25, 0.40, 0.35],   # hausse progressive des virginica
    3: [0.15, 0.30, 0.55],   # dominance virginica (nouvelle population)
}

# Drift cumulatif par jour, actif à partir du jour 16 (début phase 2)
DRIFT_PER_DAY = {
    "petal length (cm)":  0.04,
    "petal width (cm)":   0.02,
    "sepal length (cm)":  0.01,
    "sepal width (cm)":  -0.01,
}

# Bornes botaniquement plausibles
FEATURE_BOUNDS = {
    "sepal length (cm)": (3.5, 9.0),
    "sepal width (cm)":  (1.5, 5.5),
    "petal length (cm)": (0.5, 9.0),
    "petal width (cm)":  (0.1, 3.5),
}

# ── Helpers ────────────────────────────────────────────────────────────────────


def get_phase(day_idx: int) -> int:
    if day_idx < 15:
        return 1
    elif day_idx < 30:
        return 2
    return 3


def drift_days(day_idx: int) -> int:
    return max(0, day_idx - 14)


def weighted_choice(classes: list, weights: list, rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for cls, w in zip(classes, weights):
        cum += w
        if r < cum:
            return cls
    return classes[-1]


def sample_features(cls: int, cum_drift: int, rng: random.Random) -> dict:
    features = {}
    for feat, (mu, sigma) in CLASS_PARAMS[cls].items():
        drift = DRIFT_PER_DAY.get(feat, 0.0) * cum_drift
        val   = rng.gauss(mu + drift, sigma)
        lo, hi = FEATURE_BOUNDS[feat]
        features[feat] = round(max(lo, min(hi, val)), 3)
    return features


# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<token> python send_predictions_iris.py")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}\n")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

# ── Génération et envoi ────────────────────────────────────────────────────────

rng  = random.Random(42)
now  = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
log  = []
phase_counts = {1: 0, 2: 0, 3: 0}

print("=" * 68)
print(f"  Envoi de {TOTAL_DAYS * N_PER_DAY} prédictions iris sur {TOTAL_DAYS} jours")
print(f"  Phases : 1=stable (j1-15)  2=drift (j16-30)  3=règle (j31-45)")
print("=" * 68)

for day_idx in range(TOTAL_DAYS):
    day_date    = now - timedelta(days=(TOTAL_DAYS - day_idx))
    timestamp   = day_date.strftime("%Y-%m-%dT%H:%M:%S")
    phase       = get_phase(day_idx)
    cum_drift   = drift_days(day_idx)
    proportions = PHASE_PROPORTIONS[phase]

    batch_items = []
    day_entries = []

    for _ in range(N_PER_DAY):
        cls      = weighted_choice([0, 1, 2], proportions, rng)
        features = sample_features(cls, cum_drift, rng)
        obs_id   = str(uuid.uuid4())

        batch_items.append({
            "id_obs":    obs_id,
            "timestamp": timestamp,
            "features":  features,
        })
        day_entries.append({
            "id_obs":     obs_id,
            "day_offset": day_idx,
            "phase":      phase,
            "true_class": cls,
            "features":   features,
            "timestamp":  timestamp,
        })

    payload = {"model_name": MODEL_NAME, "inputs": batch_items}
    if MODEL_VERSION:
        payload["model_version"] = MODEL_VERSION

    r = requests.post(
        f"{API_URL}/predict-batch", headers=HEADERS, json=payload, timeout=30
    )

    if r.status_code == 200:
        log.extend(day_entries)
        phase_counts[phase] += N_PER_DAY
        drift_label = f"+{cum_drift * DRIFT_PER_DAY['petal length (cm)']:.2f}cm" if cum_drift else "baseline"
        print(
            f"  [{timestamp[:10]}]  phase={phase}  petal_drift={drift_label:<10}"
            f"  ✅  {N_PER_DAY} prédictions"
        )
    else:
        print(
            f"  [{timestamp[:10]}]  ❌  Erreur {r.status_code} : {r.text[:120]}"
        )

    if day_idx < TOTAL_DAYS - 1:
        time.sleep(SLEEP_BETWEEN)

# ── Sauvegarde du log ──────────────────────────────────────────────────────────

with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)

# ── Résumé ─────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print(f"  Total envoyé     : {len(log)} prédictions")
print(f"  Phase 1 (stable) : {phase_counts[1]:4d} obs  — accuracy attendue ~93%")
print(f"  Phase 2 (drift)  : {phase_counts[2]:4d} obs  — accuracy attendue ~78%")
print(f"  Phase 3 (règle)  : {phase_counts[3]:4d} obs  — accuracy attendue ~62%")
print(f"  Log sauvegardé   : {LOG_FILE}")
print()
print("  → Lancez maintenant : python send_ground_truth_iris.py")
print("=" * 68)
