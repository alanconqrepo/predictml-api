"""
send_predictions_cancer.py — 900 Cancer predictions spread over 45 days
=========================================================================

Simulates a progressive degradation of the cancer-classifier model (binary
classification: 0=malignant, 1=benign) over 45 days:

  Phase 1 (D-45 → D-31, 15 days) : stable distribution, ~94% accuracy

  Phase 2 (D-30 → D-16, 15 days) : scanner calibration drift —
    the scanner overestimates radius, area and perimeter for BENIGN tissues only.
    Benign cases drift into the model's malignant zone → ~60% accuracy.
    (Drift accumulates: +30 mm²/day on worst_area, +0.25/day on worst_radius)

  Phase 3 (D-15 → D-1,  15 days) : new early detection protocol —
    some "mild" MALIGNANT cases (mean_radius < 18 AND mean_concavity < 0.15)
    are now reclassified as BENIGN by the new guidelines. The model
    (trained on the old protocol) still predicts them as malignant → ~38% accuracy.

Produces cancer_predictions_log.json — used by send_ground_truth_cancer.py.

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_cancer.py

Environment variables:
  API_URL        API URL                   (default: http://localhost:80)
  API_TOKEN      Bearer token — required
  MODEL_NAME     Model name                (default: cancer-classifier)
  MODEL_VERSION  Version (optional)
  TOTAL_DAYS     Number of days            (default: 45)
  N_PER_DAY      Predictions per day       (default: 20)
  SLEEP_BETWEEN  Pause between batches (s) (default: 7)
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

# ── Per-class distributions (μ, σ) — source: sklearn breast_cancer ───────────
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

# [malignant, benign] proportions per phase
PHASE_PROPORTIONS = {
    1: [0.37, 0.63],   # original dataset distribution
    2: [0.45, 0.55],   # more malignant cases
    3: [0.55, 0.45],   # malignant majority (more aggressive population)
}

# Cumulative drift per day, active from day 16 (start of phase 2)
# Applied ONLY to BENIGN cases (cls==1) — simulates a scanner calibration drift
# that overestimates measurements of benign tissues, gradually pushing them
# into the model's malignant decision zone.
DRIFT_PER_DAY = {
    "mean radius":       0.20,   # +3 mm over 15d → benign resemble malignant
    "mean perimeter":    1.30,   # +19.5 mm over 15d
    "mean area":         25.0,   # +375 mm² over 15d  (μ benign 559 → 934)
    "worst radius":      0.25,   # +3.75 mm over 15d
    "worst perimeter":   1.60,   # +24 mm over 15d
    "worst area":        30.0,   # +450 mm² over 15d  (μ benign 559 → 1009)
    "worst concavity":   0.010,  # +0.15 over 15d
    "mean concavity":    0.007,  # +0.105 over 15d
}

# Physically plausible bounds
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
        # Drift applied ONLY to benign cases (cls==1):
        # the scanner over-estimates measurements of benign tissue → they drift
        # progressively into the malignant decision zone of the model.
        drift = DRIFT_PER_DAY.get(feat, 0.0) * cum_drift if cls == 1 else 0.0
        val   = rng.gauss(mu + drift, sigma)
        lo, hi = FEATURE_BOUNDS[feat]
        features[feat] = round(max(lo, min(hi, val)), 4)
    return features


# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN not defined.")
    print("    Run: API_TOKEN=<token> python send_predictions_cancer.py")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}\n")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

# ── Generation and sending ────────────────────────────────────────────────────

rng  = random.Random(42)
now  = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
log  = []
phase_counts = {1: 0, 2: 0, 3: 0}

print("=" * 72)
print(f"  Sending {TOTAL_DAYS * N_PER_DAY} cancer predictions over {TOTAL_DAYS} days")
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

    try:
        r = requests.post(
            f"{API_URL}/predict-batch", headers=HEADERS, json=payload, timeout=120
        )
    except requests.exceptions.Timeout:
        print(f"  [{timestamp[:10]}]  ⏱  Timeout — batch skipped (retry with a larger SLEEP_BETWEEN)")
        if day_idx < TOTAL_DAYS - 1 and SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)
        continue
    except requests.exceptions.RequestException as exc:
        print(f"  [{timestamp[:10]}]  ❌  Network error: {exc}")
        if day_idx < TOTAL_DAYS - 1 and SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)
        continue

    if r.status_code == 200:
        log.extend(day_entries)
        phase_counts[phase] += N_PER_DAY
        area_drift  = cum_drift * DRIFT_PER_DAY.get("worst area", 30.0)
        drift_label = f"+{area_drift:.0f}mm²" if area_drift else "baseline"
        print(
            f"  [{timestamp[:10]}]  phase={phase}  area_drift={drift_label:<10}"
            f"  ✅  {N_PER_DAY} predictions"
        )
    else:
        print(
            f"  [{timestamp[:10]}]  ❌  Error {r.status_code}: {r.text[:120]}"
        )

    if day_idx < TOTAL_DAYS - 1 and SLEEP_BETWEEN > 0:
        time.sleep(SLEEP_BETWEEN)

# ── Sauvegarde du log ──────────────────────────────────────────────────────────

with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)

if not log:
    print("\n❌  No prediction was recorded — check the errors above.")
    sys.exit(1)

# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("=" * 72)
print(f"  Total sent       : {len(log)} predictions")
print(f"  Phase 1 (stable) : {phase_counts[1]:4d} obs  — expected accuracy ~94%")
print(f"  Phase 2 (drift)  : {phase_counts[2]:4d} obs  — expected accuracy ~60%")
print(f"  Phase 3 (rule)   : {phase_counts[3]:4d} obs  — expected accuracy ~38%")
print(f"  Log saved        : {LOG_FILE}")
print()
print("  → Run next: python send_ground_truth_cancer.py")
print("=" * 72)
