"""
send_ground_truth_wine.py — Wine ground truth with temporal degradation
============================================================================

Loads wine_predictions_log.json (produced by send_predictions_wine.py) and
computes the true alcohol content according to each phase's rules, then
sends the observed_results via POST /observed-results.

Ground truth rules (regression):
  Phase 1-2 : true_alcohol = base_alcohol_for_type + Gaussian noise N(0, 0.35)
  Phase 3   : if proline > 1200 AND color_intensity > 8.0
              → true_alcohol += 1.8 (systematic bias — new technique)
              The model ignores this bias and massively under-predicts.

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth_wine.py

Environment variables:
  API_URL    API URL       (default: http://localhost:80)
  API_TOKEN  Bearer token — required
  MODEL_NAME Model name    (default: wine-regressor)
  NOISE_STD  Gaussian noise standard deviation (default: 0.35)
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

# Bias added in phase 3 for wines affected by the new rule
PHASE3_BIAS   = 1.8
PROLINE_THR   = 1200.0
COLOR_INT_THR = 8.0

HEADERS    = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE   = os.path.join(os.path.dirname(__file__), "wine_predictions_log.json")
BATCH_SIZE = 50

# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN is not set.")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API reachable: {API_URL}\n")
except Exception as e:
    print(f"❌  API unreachable ({API_URL}): {e}")
    sys.exit(1)

if not os.path.exists(LOG_FILE):
    print(f"❌  Log not found: {LOG_FILE}")
    print("    Run first: python send_predictions_wine.py")
    sys.exit(1)

# ── Load the log ───────────────────────────────────────────────────────────────

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

if not log:
    print("❌  Log is empty: no predictions to label. Re-run send_predictions_wine.py.")
    sys.exit(1)

print(f"  {len(log)} entries loaded from {LOG_FILE}\n")

# ── Ground truth rule (regression) ────────────────────────────────────────────

rng = random.Random(99)


def compute_true_alcohol(entry: dict) -> float:
    """Computes the true alcohol content according to the phase and features."""
    base         = entry["base_alcohol"]
    noise        = rng.gauss(0.0, NOISE_STD)
    true_alcohol = base + noise

    if entry["phase"] == 3:
        proline   = entry["features"].get("proline",         0.0)
        color_int = entry["features"].get("color_intensity", 0.0)
        if proline > PROLINE_THR and color_int > COLOR_INT_THR:
            true_alcohol += PHASE3_BIAS  # systematic bias not seen by the model

    return round(true_alcohol, 2)


# ── Build observed results ─────────────────────────────────────────────────────

print("=" * 72)
print("  Computing ground truth by phase")
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
    line = f"  Phase {phase}: {s['total']:4d} obs  mean_alcohol={mean:.2f}%"
    if s["biased"]:
        pct = s["biased"] / s["total"] * 100
        line += f"  ({s['biased']} with bias +{PHASE3_BIAS}, {pct:.0f}%)"
    print(line)

print()

# ── Send in batches ────────────────────────────────────────────────────────────

print("=" * 72)
print("  Sending observed results (POST /observed-results)")
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
        print(f"  [{i + 1:4d} – ...]  ❌  Error {r.status_code}: {r.text[:100]}")

# ── Coverage check ─────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("  Ground truth coverage (GET /observed-results/stats)")
print("=" * 72)

r = requests.get(
    f"{API_URL}/observed-results/stats",
    headers=HEADERS,
    params={"model_name": MODEL_NAME},
    timeout=10,
)
if r.status_code == 200:
    stats = r.json()
    print(f"\n  Model       : {MODEL_NAME}")
    print(
        f"  Labeled     : {stats.get('labeled_count', '?')}"
        f" / {stats.get('total_predictions', '?')}"
    )
    print(f"  Coverage    : {stats.get('coverage_rate', 0.0):.1%}")
else:
    print(f"  (stats unavailable — {r.status_code})")

# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("=" * 72)
print(f"  Total upserted   : {total_upserted}")
print()
print("  Simulated degradation scenario:")
print("  Phase 1 (stable)  → RMSE ~0.50  (normal distribution)")
print("  Phase 2 (drift)   → RMSE ~0.85  (new grape variety, rising proline)")
print(
    f"  Phase 3 (rule)    → RMSE ~1.50"
    f"  (bias +{PHASE3_BIAS} for proline>{PROLINE_THR:.0f} & color_intensity>{COLOR_INT_THR})"
)
print()
print("  The model requires retraining on recent production data.")
print()
print(f"  → Dashboard: {API_URL.replace(':8000', ':8501')}/Models")
print("=" * 72)
