"""
send_predictions_cancer.py — 900 prédictions Cancer étalées sur 45 jours
=========================================================================

Simule une dégradation progressive du modèle cancer-classifier (classification
binaire : 0=malignant, 1=benign) sur 45 jours :

  Phase 1 (J-45 → J-31, 15 jours) : distribution stable, ~94% accuracy
  Phase 2 (J-30 → J-16, 15 jours) : drift de calibration du scanner —
    radius, area et perimeter augmentent systématiquement, les cas bénins
    ressemblent de plus en plus à des cas malins → ~82% accuracy
  Phase 3 (J-15 → J-1,  15 jours) : nouveau protocole clinique —
    toute cellule avec worst_concavity > 0.30 ET worst_area > 1200 est
    reclassifiée maligne, même si le modèle prédit benign → ~70% accuracy

Produit cancer_predictions_log.json — utilisé par send_ground_truth_cancer.py.

Usage :
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_cancer.py

Variables d'environnement :
  API_URL        URL de l'API              (défaut : http://localhost:80)
  API_TOKEN      Token Bearer — requis
  MODEL_NAME     Nom du modèle             (défaut : cancer-classifier)
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
MODEL_NAME    = os.environ.get("MODEL_NAME",    "cancer-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)
TOTAL_DAYS    = int(os.environ.get("TOTAL_DAYS",    "45"))
N_PER_DAY     = int(os.environ.get("N_PER_DAY",     "20"))
SLEEP_BETWEEN = float(os.environ.get("SLEEP_BETWEEN", "7"))

HEADERS  = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE = os.path.join(os.path.dirname(__file__), "cancer_predictions_log.json")

# ── Distributions par classe (μ, σ) — source : sklearn breast_cancer ──────────
# 0 = malignant,  1 = benign

CLASS_PARAMS = {
    0: {  # malignant
        "mean radius":              (17.46, 3.20),
        "mean texture":             (21.60, 4.20),
        "mean perimeter":           (115.4, 22.0),
        "mean area":                (978.0, 368.0),
        "mean smoothness":          (0.103, 0.015),
        "mean compactness":         (0.145, 0.052),
        "mean concavity":           (0.161, 0.079),
        "mean concave points":      (0.088, 0.034),
        "mean symmetry":            (0.195, 0.027),
        "mean fractal dimension":   (0.063, 0.010),
        "radius error":             (1.215, 0.620),
        "texture error":            (1.210, 0.570),
        "perimeter error":          (8.834, 5.80),
        "area error":               (102.9, 85.0),
        "smoothness error":         (0.007, 0.003),
        "compactness error":        (0.032, 0.019),
        "concavity error":          (0.042, 0.030),
        "concave points error":     (0.015, 0.006),
        "symmetry error":           (0.021, 0.008),
        "fractal dimension error":  (0.004, 0.002),
        "worst radius":             (21.1,  4.50),
        "worst texture":            (29.3,  6.00),
        "worst perimeter":          (141.4, 31.0),
        "worst area":               (1422., 590.0),
        "worst smoothness":         (0.145, 0.023),
        "worst compactness":        (0.374, 0.167),
        "worst concavity":          (0.451, 0.198),
        "worst concave points":     (0.183, 0.057),
        "worst symmetry":           (0.324, 0.073),
        "worst fractal dimension":  (0.092, 0.020),
    },
    1: {  # benign
        "mean radius":              (12.15, 1.78),
        "mean texture":             (17.91, 4.00),
        "mean perimeter":           (78.07, 13.6),
        "mean area":                (462.8, 136.0),
        "mean smoothness":          (0.092, 0.013),
        "mean compactness":         (0.080, 0.025),
        "mean concavity":           (0.046, 0.042),
        "mean concave points":      (0.026, 0.017),
        "mean symmetry":            (0.175, 0.023),
        "mean fractal dimension":   (0.063, 0.008),
        "radius error":             (0.284, 0.102),
        "texture error":            (1.180, 0.568),
        "perimeter error":          (1.977, 0.764),
        "area error":               (21.1,  8.70),
        "smoothness error":         (0.007, 0.002),
        "compactness error":        (0.011, 0.006),
        "concavity error":          (0.013, 0.013),
        "concave points error":     (0.006, 0.003),
        "symmetry error":           (0.019, 0.006),
        "fractal dimension error":  (0.002, 0.001),
        "worst radius":             (13.38, 2.30),
        "worst texture":            (23.52, 5.50),
        "worst perimeter":          (86.54, 16.3),
        "worst area":               (558.9, 163.0),
        "worst smoothness":         (0.125, 0.020),
        "worst compactness":        (0.177, 0.072),
        "worst concavity":          (0.166, 0.124),
        "worst concave points":     (0.074, 0.035),
        "worst symmetry":           (0.270, 0.062),
        "worst fractal dimension":  (0.079, 0.014),
    },
}

# Proportions [malignant, benign] par phase
PHASE_PROPORTIONS = {
    1: [0.37, 0.63],   # distribution originale du dataset
    2: [0.45, 0.55],   # plus de cas malins
    3: [0.55, 0.45],   # majorité maligne (population plus agressive)
}

# Drift cumulatif par jour, actif à partir du jour 16 (début phase 2)
# Simule une dérive de calibration du scanner : mesures systématiquement plus grandes
DRIFT_PER_DAY = {
    "mean radius":       0.05,
    "mean perimeter":    0.30,
    "mean area":         6.00,
    "worst radius":      0.05,
    "worst perimeter":   0.30,
    "worst area":        6.00,
    "worst concavity":   0.003,
    "mean concavity":    0.002,
}

# Bornes physiquement plausibles
FEATURE_BOUNDS = {
    "mean radius":             (5.0,  30.0),
    "mean texture":            (8.0,  40.0),
    "mean perimeter":          (40.0, 200.0),
    "mean area":               (140.0, 2600.0),
    "mean smoothness":         (0.05,  0.20),
    "mean compactness":        (0.01,  0.40),
    "mean concavity":          (0.00,  0.50),
    "mean concave points":     (0.00,  0.25),
    "mean symmetry":           (0.10,  0.35),
    "mean fractal dimension":  (0.04,  0.10),
    "radius error":            (0.10,  4.00),
    "texture error":           (0.30,  5.00),
    "perimeter error":         (0.50, 30.00),
    "area error":              (5.00, 500.0),
    "smoothness error":        (0.001, 0.030),
    "compactness error":       (0.002, 0.130),
    "concavity error":         (0.000, 0.200),
    "concave points error":    (0.000, 0.050),
    "symmetry error":          (0.007, 0.070),
    "fractal dimension error": (0.001, 0.020),
    "worst radius":            (7.00,  40.00),
    "worst texture":           (12.0,  50.00),
    "worst perimeter":         (50.0, 260.00),
    "worst area":              (190.0, 4300.0),
    "worst smoothness":        (0.07,  0.30),
    "worst compactness":       (0.02,  1.20),
    "worst concavity":         (0.00,  1.30),
    "worst concave points":    (0.00,  0.30),
    "worst symmetry":          (0.15,  0.70),
    "worst fractal dimension": (0.05,  0.25),
}

FEATURE_NAMES = list(CLASS_PARAMS[0].keys())

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
    for feat in FEATURE_NAMES:
        mu, sigma = CLASS_PARAMS[cls][feat]
        drift = DRIFT_PER_DAY.get(feat, 0.0) * cum_drift
        val   = rng.gauss(mu + drift, sigma)
        lo, hi = FEATURE_BOUNDS[feat]
        features[feat] = round(max(lo, min(hi, val)), 4)
    return features


# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<token> python send_predictions_cancer.py")
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
print(f"  Envoi de {TOTAL_DAYS * N_PER_DAY} prédictions cancer sur {TOTAL_DAYS} jours")
print(f"  Phases : 1=stable (j1-15)  2=drift scanner (j16-30)  3=protocole (j31-45)")
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
        cls      = weighted_choice([0, 1], proportions, rng)
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
        area_drift  = cum_drift * DRIFT_PER_DAY.get("worst area", 6.0)
        drift_label = f"+{area_drift:.0f}mm²" if area_drift else "baseline"
        print(
            f"  [{timestamp[:10]}]  phase={phase}  area_drift={drift_label:<10}"
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
print(f"  Phase 1 (stable) : {phase_counts[1]:4d} obs  — accuracy attendue ~94%")
print(f"  Phase 2 (drift)  : {phase_counts[2]:4d} obs  — accuracy attendue ~82%")
print(f"  Phase 3 (règle)  : {phase_counts[3]:4d} obs  — accuracy attendue ~70%")
print(f"  Log sauvegardé   : {LOG_FILE}")
print()
print("  → Lancez maintenant : python send_ground_truth_cancer.py")
print("=" * 72)
