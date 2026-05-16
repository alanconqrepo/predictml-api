"""
send_predictions_wine.py — 900 prédictions Wine étalées sur 45 jours
=====================================================================

Simule une dégradation progressive du modèle wine-regressor (prédiction du
taux d'alcool) sur 45 jours :

  Phase 1 (J-45 → J-31, 15 jours) : distribution stable, RMSE ~0.50
  Phase 2 (J-30 → J-16, 15 jours) : drift des features — proline et
    color_intensity augmentent progressivement (nouveau cépage), RMSE ~0.85
  Phase 3 (J-15 → J-1,  15 jours) : nouvelle règle déterministe —
    vins avec proline > 1200 ET color_intensity > 8.0 ont un alcool réel
    systématiquement plus élevé de +1.8 (nouvelle technique de vinification).
    Le modèle sous-prédit massivement ces cas → RMSE ~1.5.

Produit wine_predictions_log.json — utilisé par send_ground_truth_wine.py.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_wine.py

Variables d'environnement :
  API_URL        URL de l'API              (défaut : http://localhost:80)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle             (défaut : wine-regressor)
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
MODEL_NAME    = os.environ.get("MODEL_NAME",    "wine-regressor")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)
TOTAL_DAYS    = int(os.environ.get("TOTAL_DAYS",    "45"))
N_PER_DAY     = int(os.environ.get("N_PER_DAY",     "20"))
SLEEP_BETWEEN = float(os.environ.get("SLEEP_BETWEEN", "7"))

HEADERS  = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE = os.path.join(os.path.dirname(__file__), "wine_predictions_log.json")

# ── Distributions par type de vin (μ, σ) ──────────────────────────────────────
# Types : 0=léger (~12.3%), 1=moyen (~13.0%), 2=corsé (~13.8%)
# Features = 12 mesures chimiques hors "alcohol"

TYPE_PARAMS = {
    0: {  # léger
        "base_alcohol":               12.3,
        "malic_acid":                 (2.8,  0.70),
        "ash":                        (2.3,  0.20),
        "alcalinity_of_ash":          (21.0, 2.50),
        "magnesium":                  (88.0, 10.0),
        "total_phenols":              (1.80, 0.40),
        "flavanoids":                 (0.70, 0.30),
        "nonflavanoid_phenols":       (0.42, 0.08),
        "proanthocyanins":            (1.20, 0.30),
        "color_intensity":            (4.50, 0.80),
        "hue":                        (0.78, 0.10),
        "od280/od315_of_diluted_wines": (1.70, 0.30),
        "proline":                    (480.0, 120.0),
    },
    1: {  # moyen
        "base_alcohol":               13.0,
        "malic_acid":                 (2.0,  0.60),
        "ash":                        (2.4,  0.20),
        "alcalinity_of_ash":          (17.0, 2.00),
        "magnesium":                  (95.0, 12.0),
        "total_phenols":              (2.30, 0.50),
        "flavanoids":                 (2.00, 0.60),
        "nonflavanoid_phenols":       (0.34, 0.08),
        "proanthocyanins":            (1.60, 0.40),
        "color_intensity":            (5.20, 1.20),
        "hue":                        (0.96, 0.10),
        "od280/od315_of_diluted_wines": (2.60, 0.50),
        "proline":                    (700.0, 150.0),
    },
    2: {  # corsé
        "base_alcohol":               13.8,
        "malic_acid":                 (1.6,  0.50),
        "ash":                        (2.5,  0.20),
        "alcalinity_of_ash":          (16.0, 2.00),
        "magnesium":                  (110.0, 15.0),
        "total_phenols":              (2.80, 0.40),
        "flavanoids":                 (3.00, 0.60),
        "nonflavanoid_phenols":       (0.28, 0.06),
        "proanthocyanins":            (2.30, 0.40),
        "color_intensity":            (6.50, 1.50),
        "hue":                        (1.05, 0.10),
        "od280/od315_of_diluted_wines": (3.20, 0.50),
        "proline":                    (1050.0, 200.0),
    },
}

# Proportions [léger, moyen, corsé] par phase
PHASE_PROPORTIONS = {
    1: [0.25, 0.50, 0.25],   # distribution équilibrée
    2: [0.15, 0.45, 0.40],   # hausse des vins corsés
    3: [0.10, 0.30, 0.60],   # dominance corsé (nouveau cépage)
}

# Drift cumulatif par jour, actif à partir du jour 16 (début phase 2)
DRIFT_PER_DAY = {
    "proline":          8.0,
    "color_intensity":  0.06,
    "flavanoids":      -0.02,
    "total_phenols":   -0.01,
}

# Bornes physiquement plausibles
FEATURE_BOUNDS = {
    "malic_acid":               (0.1,  6.0),
    "ash":                      (1.0,  4.0),
    "alcalinity_of_ash":        (10.0, 30.0),
    "magnesium":                (60.0, 160.0),
    "total_phenols":            (0.5,  4.0),
    "flavanoids":               (0.1,  5.5),
    "nonflavanoid_phenols":     (0.1,  0.7),
    "proanthocyanins":          (0.3,  3.6),
    "color_intensity":          (1.0,  14.0),
    "hue":                      (0.4,  1.7),
    "od280/od315_of_diluted_wines": (0.5, 4.5),
    "proline":                  (200.0, 2000.0),
}

FEATURE_NAMES = [
    "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline",
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def get_phase(day_idx: int) -> int:
    if day_idx < 15:
        return 1
    elif day_idx < 30:
        return 2
    return 3


def drift_days(day_idx: int) -> int:
    return max(0, day_idx - 14)


def weighted_choice(types: list, weights: list, rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for t, w in zip(types, weights):
        cum += w
        if r < cum:
            return t
    return types[-1]


def sample_features(wine_type: int, cum_drift: int, rng: random.Random) -> dict:
    params = TYPE_PARAMS[wine_type]
    features = {}
    for feat in FEATURE_NAMES:
        mu, sigma = params[feat]
        drift = DRIFT_PER_DAY.get(feat, 0.0) * cum_drift
        val   = rng.gauss(mu + drift, sigma)
        lo, hi = FEATURE_BOUNDS[feat]
        features[feat] = round(max(lo, min(hi, val)), 3)
    return features


# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<token> python send_predictions_wine.py")
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

print("=" * 72)
print(f"  Envoi de {TOTAL_DAYS * N_PER_DAY} prédictions wine sur {TOTAL_DAYS} jours")
print(f"  Phases : 1=stable (j1-15)  2=drift cépage (j16-30)  3=règle alcool (j31-45)")
print("=" * 72)

for day_idx in range(TOTAL_DAYS):
    day_date    = now - timedelta(days=(TOTAL_DAYS - day_idx))
    timestamp   = day_date.strftime("%Y-%m-%dT%H:%M:%S")
    phase       = get_phase(day_idx)
    cum_drift   = drift_days(day_idx)
    proportions = PHASE_PROPORTIONS[phase]

    batch_items = []
    day_entries = []

    for _ in range(N_PER_DAY):
        wine_type    = weighted_choice([0, 1, 2], proportions, rng)
        features     = sample_features(wine_type, cum_drift, rng)
        base_alcohol = TYPE_PARAMS[wine_type]["base_alcohol"]
        obs_id       = str(uuid.uuid4())

        batch_items.append({
            "id_obs":    obs_id,
            "timestamp": timestamp,
            "features":  features,
        })
        day_entries.append({
            "id_obs":       obs_id,
            "day_offset":   day_idx,
            "phase":        phase,
            "wine_type":    wine_type,
            "base_alcohol": base_alcohol,
            "features":     features,
            "timestamp":    timestamp,
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
        proline_drift = cum_drift * DRIFT_PER_DAY["proline"]
        drift_label   = f"+{proline_drift:.0f}" if proline_drift else "baseline"
        print(
            f"  [{timestamp[:10]}]  phase={phase}  proline_drift={drift_label:<7}"
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
print("=" * 72)
print(f"  Total envoyé     : {len(log)} prédictions")
print(f"  Phase 1 (stable) : {phase_counts[1]:4d} obs  — RMSE attendu ~0.50")
print(f"  Phase 2 (drift)  : {phase_counts[2]:4d} obs  — RMSE attendu ~0.85")
print(f"  Phase 3 (règle)  : {phase_counts[3]:4d} obs  — RMSE attendu ~1.50")
print(f"  Log sauvegardé   : {LOG_FILE}")
print()
print("  → Lancez maintenant : python send_ground_truth_wine.py")
print("=" * 72)
