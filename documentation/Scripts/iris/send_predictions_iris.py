"""
send_predictions_iris.py — 900 Iris predictions spread over 45 days
=====================================================================

Simulates a progressive degradation of the model in production:

  Phase 1 (D-45 → D-31, 15 days) : stable distribution, ~93% accuracy
  Phase 2 (D-30 → D-16, 15 days) : feature drift (growing petals)
    — versicolor drifts toward the virginica zone, massive confusion → ~65% accuracy
  Phase 3 (D-15 → D-1,  15 days) : new deterministic rule — any flower
    with petal_length > 4.5 AND petal_width > 1.5 is actually virginica,
    even if the model (trained on the old distribution) predicts versicolor.
    Wider coverage than before → ~40% accuracy

Produces iris_predictions_log.json — used by send_ground_truth_iris.py.

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_iris.py

Environment variables:
  API_URL        API URL                   (default: http://localhost:80)
  API_TOKEN      Bearer token — required
  MODEL_NAME     Model name                (default: iris-classifier)
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
MODEL_NAME    = os.environ.get("MODEL_NAME",    "iris-classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)
TOTAL_DAYS    = int(os.environ.get("TOTAL_DAYS",    "45"))
N_PER_DAY     = int(os.environ.get("N_PER_DAY",     "20"))
SLEEP_BETWEEN = float(os.environ.get("SLEEP_BETWEEN", "7"))

HEADERS  = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE = os.path.join(os.path.dirname(__file__), "iris_predictions_log.json")

# ── Per-class distributions (μ, σ) — source: sklearn.datasets.load_iris() ─────

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

# Proportions [setosa, versicolor, virginica] per phase
PHASE_PROPORTIONS = {
    1: [0.40, 0.40, 0.20],   # stable and balanced
    2: [0.15, 0.50, 0.35],   # more versicolor drifting toward the virginica zone
    3: [0.05, 0.30, 0.65],   # virginica dominance → model massively wrong
}

# Cumulative drift per day, active from day 16 (start of phase 2)
# Values doubled for a visible degradation on the chart
DRIFT_PER_DAY = {
    "petal length (cm)":  0.08,   # +0.08 cm/day → +1.2 cm over 15 days
    "petal width (cm)":   0.04,   # +0.04 cm/day → +0.6 cm over 15 days
    "sepal length (cm)":  0.02,
    "sepal width (cm)":  -0.02,
}

# Botanically plausible bounds
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
    print("❌  API_TOKEN not defined.")
    print("    Run: API_TOKEN=<token> python send_predictions_iris.py")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible: {API_URL}\n")
except Exception as e:
    print(f"❌  API unreachable ({API_URL}): {e}")
    sys.exit(1)

# ── Generation and sending ────────────────────────────────────────────────────

rng  = random.Random(42)
now  = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
log  = []
phase_counts = {1: 0, 2: 0, 3: 0}

print("=" * 68)
print(f"  Sending {TOTAL_DAYS * N_PER_DAY} iris predictions over {TOTAL_DAYS} days")
print(f"  Phases: 1=stable (d1-15)  2=drift (d16-30)  3=rule (d31-45)")
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
        drift_label = f"+{cum_drift * DRIFT_PER_DAY['petal length (cm)']:.2f}cm" if cum_drift else "baseline"
        print(
            f"  [{timestamp[:10]}]  phase={phase}  petal_drift={drift_label:<10}"
            f"  ✅  {N_PER_DAY} predictions"
        )
    else:
        print(
            f"  [{timestamp[:10]}]  ❌  Erreur {r.status_code} : {r.text[:120]}"
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
print("=" * 68)
print(f"  Total sent       : {len(log)} predictions")
print(f"  Phase 1 (stable) : {phase_counts[1]:4d} obs  — expected accuracy ~93%")
print(f"  Phase 2 (drift)  : {phase_counts[2]:4d} obs  — expected accuracy ~65%")
print(f"  Phase 3 (rule)   : {phase_counts[3]:4d} obs  — expected accuracy ~40%")
print(f"  Log saved        : {LOG_FILE}")
print()
print("  → Run next: python send_ground_truth_iris.py")
print("=" * 68)
