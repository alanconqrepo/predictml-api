"""
send_ground_truth_titanic.py — Titanic ground truth with temporal degradation
==================================================================================

Loads titanic_predictions_log.json (produced by send_predictions_titanic.py) and
computes the true class according to the rules for each phase, then sends the
observed_results via POST /observed-results.

Ground truth rules:
  Phase 1-2: true_class = result of the survival probability drawn at random
              (recorded in the log by send_predictions_titanic.py)
  Phase 3  : if pclass == "3rd" AND age > 40
              → true_class = 0 (deceased) — new deterministic rule
              Simulates a context change: in this fictional scenario,
              3rd class passengers over 40 have no chance,
              even if the model (trained before this change) predicts "survivor".

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth_titanic.py

Environment variables:
  API_URL    API URL   (default: http://localhost:80)
  API_TOKEN  Bearer token — required
  MODEL_NAME Model name (default: titanic-survival)
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

API_URL = os.environ.get("API_URL", "http://localhost:80")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "titanic-survival")

HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
LOG_FILE = os.path.join(os.path.dirname(__file__), "titanic_predictions_log.json")
BATCH_SIZE = 50

# ── Validation ─────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN not defined.")
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
    print("    Run first: python send_predictions_titanic.py")
    sys.exit(1)

# ── Load log ──────────────────────────────────────────────────────────────────

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

if not log:
    print("❌  Empty log: no predictions to label. Re-run send_predictions_titanic.py.")
    sys.exit(1)

print(f"  {len(log)} entries loaded from {LOG_FILE}\n")

# ── Ground truth rule ────────────────────────────────────────────────────────


def compute_true_class(entry: dict) -> int:
    """Returns the true class (survived) according to the phase and features.

    Phase 1-2: uses the true_class recorded at generation time
               (probabilistic draw based on real distributions).
    Phase 3  : new deterministic rule — 3rd class + age > 40 → deceased (0).
    """
    if entry["phase"] < 3:
        return entry["true_class"]

    # Phase 3: deterministic rule for senior 3rd class passengers
    pclass = entry["features"].get("pclass", "")
    age = entry["features"].get("age", 0.0)
    if pclass == "3rd" and age > 40:
        return 0  # deceased — fictional post-context-change rule
    return entry["true_class"]


# ── Build observed results ─────────────────────────────────────────────────

print("=" * 68)
print("  Computing ground truth per phase")
print("=" * 68)

observed_data = []
phase_stats = {p: {"total": 0, "overridden": 0} for p in [1, 2, 3]}

for entry in log:
    true_cls = compute_true_class(entry)
    phase = entry["phase"]
    phase_stats[phase]["total"] += 1
    if true_cls != entry["true_class"]:
        phase_stats[phase]["overridden"] += 1

    observed_data.append(
        {
            "id_obs": entry["id_obs"],
            "model_name": MODEL_NAME,
            "date_time": entry["timestamp"],
            "observed_result": true_cls,
        }
    )

for phase in [1, 2, 3]:
    s = phase_stats[phase]
    line = f"  Phase {phase}: {s['total']:4d} obs"
    if s["overridden"]:
        pct = s["overridden"] / s["total"] * 100
        line += f"  ({s['overridden']} reclassified → deceased" f" (3rd class > 40 yrs), {pct:.0f}%)"
    else:
        line += "  (stable distribution)"
    print(line)

print()

# ── Send in batches ───────────────────────────────────────────────────────────

print("=" * 68)
print("  Sending observed results (POST /observed-results)")
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
        print(f"  [{i + 1:4d} – ...]  ❌  Error {r.status_code}: {r.text[:100]}")

# ── Coverage check ─────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  Ground truth coverage (GET /observed-results/stats)")
print("=" * 68)

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

# ── Summary ─────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print(f"  Total upserted    : {total_upserted}")
print()
print("  Simulated degradation scenario:")
print("  Phase 1 (stable) → ~80% accuracy  (realistic Titanic distribution)")
print("  Phase 2 (drift)  → ~65% accuracy  (younger ages, higher fares, 1st class bias)")
print("  Phase 3 (rule)   → ~50% accuracy  (3rd class > 40 yrs = deceased, new rule)")
print()
print("  The model requires retraining on recent production data.")
print()
print(f"  → Dashboard: {API_URL.replace(':8000', ':8501')}/Models")
print("=" * 68)
