"""
send_predictions_titanic.py — 900 Titanic predictions spread over 45 days
===========================================================================

Simulates a progressive degradation of the model in production:

  Phase 1 (D-45 → D-31, 15 days) : stable distribution, ~80% accuracy
    — realistic mix of passengers (all classes, male/female)

  Phase 2 (D-30 → D-16, 15 days) : feature drift
    — younger ages (+drift), higher fares, more women in 1st class
    — the model (trained on the old distribution) starts making errors → ~65%

  Phase 3 (D-15 → D-1, 15 days) : new survival rule
    — all 3rd class passengers > 40 years old die (deterministic rule)
    — the model sometimes predicts "survivor" for these passengers → ~50% accuracy

Produces titanic_predictions_log.json — used by send_ground_truth_titanic.py.

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_predictions_titanic.py

Environment variables:
  API_URL        API URL                   (default: http://localhost:80)
  API_TOKEN      Bearer token — required
  MODEL_NAME     Model name                (default: titanic-survival)
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

API_URL = os.environ.get("API_URL", "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "titanic-survival")
MODEL_VERSION = os.environ.get("MODEL_VERSION", None)
TOTAL_DAYS = int(os.environ.get("TOTAL_DAYS", "45"))
N_PER_DAY = int(os.environ.get("N_PER_DAY", "20"))
SLEEP_BETWEEN = float(os.environ.get("SLEEP_BETWEEN", "7"))

HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE = os.path.join(os.path.dirname(__file__), "titanic_predictions_log.json")

# ── Distributions by profile (phase 1: stable) ───────────────────────────────
#
# "Real" survival probabilities by profile (simulated):
#   female 1st → 97 %   male 1st → 37 %
#   female 2nd → 74 %   male 2nd → 17 %
#   female 3rd → 50 %   male 3rd → 14 %
#
# Phase 1 distributions: realistic proportions from the historical Titanic
PHASE1_PROFILES = [
    # (pclass, sex, age_mu, age_sigma, fare_mu, fare_sigma,
    #  sibsp_choices, parch_choices, embarked_probs, true_survival_prob, weight)
    ("1st", "female", 36, 12, 100, 55, [0, 0, 1], [0, 0, 1], [0.6, 0.3, 0.1], 0.97, 0.05),
    ("1st", "male", 42, 14, 95, 60, [0, 0, 0, 1], [0, 0, 1], [0.6, 0.3, 0.1], 0.37, 0.09),
    ("2nd", "female", 29, 11, 22, 10, [0, 0, 1], [0, 0, 1], [0.7, 0.2, 0.1], 0.74, 0.07),
    ("2nd", "male", 32, 12, 20, 10, [0, 0, 0, 1], [0, 0, 1], [0.7, 0.2, 0.1], 0.17, 0.13),
    ("3rd", "female", 24, 10, 13, 6, [0, 0, 1, 2], [0, 0, 1], [0.7, 0.2, 0.1], 0.50, 0.15),
    ("3rd", "male", 26, 11, 12, 6, [0, 0, 0, 1, 2], [0, 0, 0, 1], [0.7, 0.2, 0.1], 0.14, 0.51),
]
PHASE1_WEIGHTS = [p[-1] for p in PHASE1_PROFILES]
PHASE1_WEIGHT_TOTAL = sum(PHASE1_WEIGHTS)

# ── Helpers ────────────────────────────────────────────────────────────────────


def get_phase(day_idx: int) -> int:
    if day_idx < 15:
        return 1
    elif day_idx < 30:
        return 2
    return 3


def drift_days(day_idx: int) -> int:
    """Number of active drift days (0 in phase 1)."""
    return max(0, day_idx - 14)


def weighted_choice(items: list, weights: list, rng: random.Random):
    r = rng.random() * sum(weights)
    cum = 0.0
    for item, w in zip(items, weights):
        cum += w
        if r <= cum:
            return item
    return items[-1]


def sample_passenger(phase: int, cum_drift: int, rng: random.Random) -> dict:
    """Generates a passenger with their features and real survival probability."""

    # Select the base profile (phase 1 or adapted)
    profile = weighted_choice(PHASE1_PROFILES, PHASE1_WEIGHTS, rng)
    (
        pclass,
        sex,
        age_mu,
        age_sigma,
        fare_mu,
        fare_sigma,
        sibsp_pool,
        parch_pool,
        emb_p,
        base_surv,
        _,
    ) = profile

    # ── Phase 2: progressive drift ────────────────────────────────────────────
    # Younger ages, higher fares, overrepresentation of women in 1st class
    if phase >= 2:
        age_mu = max(18, age_mu - cum_drift * 0.4)
        fare_mu = fare_mu * (1 + cum_drift * 0.03)
        # Bias toward women in 1st class (fictional clientele shift)
        if rng.random() < min(0.30, cum_drift * 0.02):
            pclass = "1st"
            sex = "female"
            base_surv = 0.97

    # ── Phase 3: rule change (3rd class > 40 years old → certain death) ──────
    # The "real" rule changes but features do not directly indicate
    # the change → the model continues predicting on the old distribution
    # (handled in send_ground_truth_titanic.py)

    # Draw numerical features
    age = max(1.0, min(80.0, round(rng.gauss(age_mu, age_sigma), 1)))
    fare_raw = rng.lognormvariate(max(0.1, fare_mu * (1 + cum_drift * 0.02)), 0.6)
    fare = round(max(5.0, min(512.0, fare_raw)), 2)
    sibsp = rng.choice(sibsp_pool)
    parch = rng.choice(parch_pool)
    embarked = weighted_choice(["S", "C", "Q"], emb_p, rng)

    # "True" survival probability (before applying phase 3 rule)
    child_bonus = 0.15 if age < 12 else 0.0
    true_surv_prob = min(0.95, base_surv + child_bonus)

    return {
        "features": {
            "age": age,
            "fare": fare,
            "parch": float(parch),
            "sibsp": float(sibsp),
            "pclass": pclass,
            "sex": sex,
            "embarked": embarked,
        },
        "true_surv_prob": true_surv_prob,
        "phase": phase,
        "pclass_raw": pclass,
        "sex_raw": sex,
        "age_raw": age,
    }


# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN not defined.")
    print("    Run: API_TOKEN=<token> python send_predictions_titanic.py")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}\n")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

# ── Generation and sending ────────────────────────────────────────────────────

rng = random.Random(42)
now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
log = []
phase_counts = {1: 0, 2: 0, 3: 0}

print("=" * 72)
print(f"  Sending {TOTAL_DAYS * N_PER_DAY} Titanic predictions over {TOTAL_DAYS} days")
print("  Phases: 1=stable (d1-15)  2=drift (d16-30)  3=rule_3rd (d31-45)")
print("=" * 72)

for day_idx in range(TOTAL_DAYS):
    day_date = now - timedelta(days=(TOTAL_DAYS - day_idx))
    timestamp = day_date.strftime("%Y-%m-%dT%H:%M:%S")
    phase = get_phase(day_idx)
    cum_drift = drift_days(day_idx)

    batch_items = []
    day_entries = []

    for _ in range(N_PER_DAY):
        passenger = sample_passenger(phase, cum_drift, rng)
        obs_id = str(uuid.uuid4())

        # Draw true survival (Bernoulli)
        true_class = int(rng.random() < passenger["true_surv_prob"])

        batch_items.append(
            {
                "id_obs": obs_id,
                "timestamp": timestamp,
                "features": passenger["features"],
            }
        )
        day_entries.append(
            {
                "id_obs": obs_id,
                "day_offset": day_idx,
                "phase": phase,
                "true_class": true_class,
                "features": passenger["features"],
                "timestamp": timestamp,
            }
        )

    payload = {"model_name": MODEL_NAME, "inputs": batch_items}
    if MODEL_VERSION:
        payload["model_version"] = MODEL_VERSION

    try:
        r = requests.post(
            f"{API_URL}/predict-batch",
            headers=HEADERS,
            json=payload,
            timeout=120,
        )
    except requests.exceptions.Timeout:
        print(
            f"  [{timestamp[:10]}]  ⏱  Timeout — batch skipped"
            " (retry with a larger SLEEP_BETWEEN)"
        )
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
        drift_label = f"+{cum_drift * 0.4:.1f}yrs" if cum_drift else "baseline"
        print(
            f"  [{timestamp[:10]}]  phase={phase}  age_drift={drift_label:<10}"
            f"  ✅  {N_PER_DAY} predictions"
        )
    else:
        print(f"  [{timestamp[:10]}]  ❌  Erreur {r.status_code} : {r.text[:120]}")

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
print(f"  Total sent         : {len(log)} predictions")
print(f"  Phase 1 (stable)   : {phase_counts[1]:4d} obs  — expected accuracy ~80%")
print(f"  Phase 2 (drift)    : {phase_counts[2]:4d} obs  — expected accuracy ~65%")
print(f"  Phase 3 (rule)     : {phase_counts[3]:4d} obs  — expected accuracy ~50%")
print(f"  Log saved          : {LOG_FILE}")
print()
print("  → Run next: python send_ground_truth_titanic.py")
print("=" * 72)
