"""
send_ground_truth_iris.py — Iris ground truth with temporal degradation
============================================================================

Loads iris_predictions_log.json (produced by send_predictions_iris.py) and
computes the true class according to each phase's rules, then sends
observed_results via POST /observed-results.

Ground truth rules:
  Phase 1-2: true_class = original sampling class
  Phase 3:   if petal_length > 4.5 AND petal_width > 1.5
              → true_class = 2 (virginica) — new deterministic rule
              Lowered threshold (was 5.0/1.7) to cover more drifted versicolor.
              Simulates a biological change: medium-large petals = virginica,
              even if the model (trained earlier) predicts versicolor.

Usage:
  API_URL=http://localhost:8000 API_TOKEN=<token> python send_ground_truth_iris.py

Environment variables:
  API_URL    API URL       (default: http://localhost:80)
  API_TOKEN  Bearer token — required
  MODEL_NAME Model name    (default: iris-classifier)
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
    print("❌  API_TOKEN not defined.")
    sys.exit(1)

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible: {API_URL}\n")
except Exception as e:
    print(f"❌  API unreachable ({API_URL}): {e}")
    sys.exit(1)

if not os.path.exists(LOG_FILE):
    print(f"❌  Log not found: {LOG_FILE}")
    print("    Run first: python send_predictions_iris.py")
    sys.exit(1)

# ── Log loading ────────────────────────────────────────────────────────────────

with open(LOG_FILE, encoding="utf-8") as f:
    log = json.load(f)

if not log:
    print("❌  Empty log: no predictions to label. Re-run send_predictions_iris.py.")
    sys.exit(1)

print(f"  {len(log)} entries loaded from {LOG_FILE}\n")

# ── Ground truth rule ──────────────────────────────────────────────────────────


def compute_true_class(entry: dict) -> int:
    """Returns the true class according to the phase and features."""
    if entry["phase"] < 3:
        return entry["true_class"]

    # Phase 3: new deterministic rule (lowered threshold → more reclassifications)
    petal_len = entry["features"].get("petal length (cm)", 0.0)
    petal_wid = entry["features"].get("petal width (cm)", 0.0)
    if petal_len > 4.5 and petal_wid > 1.5:
        return 2  # virginica — expanded rule
    return entry["true_class"]


# ── Building observed results ─────────────────────────────────────────────────

print("=" * 68)
print("  Computing ground truth by phase")
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
    line = f"  Phase {phase}: {s['total']:4d} obs"
    if s["overridden"]:
        pct = s["overridden"] / s["total"] * 100
        line += f"  ({s['overridden']} reclassified → virginica by new rule, {pct:.0f}%)"
    else:
        line += "  (stable distribution)"
    print(line)

print()

# ── Sending in batches ────────────────────────────────────────────────────────

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
        print(f"  [{i + 1:4d} – ...]  ❌  Error {r.status_code} : {r.text[:100]}")

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

# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print(f"  Total upserted   : {total_upserted}")
print()
print("  Simulated degradation scenario:")
print("  Phase 1 (stable) → ~93% accuracy  (original iris population)")
print("  Phase 2 (drift)  → ~65% accuracy  (growing petals, blurred boundaries)")
print("  Phase 3 (rule)   → ~40% accuracy  (lowered threshold: medium petals already virginica)")
print()
print("  The model requires retraining on recent production data.")
print()
print(f"  → Dashboard : {API_URL.replace(':8000', ':8501')}/Models")
print("=" * 68)
