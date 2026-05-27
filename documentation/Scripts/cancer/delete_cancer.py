"""
delete_cancer.py — Full cleanup of the cancer-classifier model via the PredictML API
======================================================================================

This script deletes in order:
  1. All predictions for the model  (DELETE /predictions/purge)
  2. All versions of the model      (DELETE /models/{name})

It then displays the number of remaining observed results (ground truth) and
explains why they cannot be deleted via the API.

Usage:
  python delete_cancer.py                     # dry-run -> shows what will be deleted
  python delete_cancer.py --yes               # deletes without confirmation

Environment variables:
  API_URL        API URL               (default: http://localhost:80)
  API_TOKEN      Bearer admin token — required
  MODEL_NAME     Model name            (default: cancer-classifier)

Python prerequisites:
  pip install requests python-dotenv

This script deletes all data, including recent predictions
and associated observed_results (ground truth).
"""

import json
import os
import sys

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL    = os.environ.get("API_URL",   "http://localhost:80")
API_TOKEN  = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "cancer-classifier")

DRY_RUN = "--yes" not in sys.argv

HEADERS      = {"Authorization": f"Bearer {API_TOKEN}"}
HEADERS_JSON = {**HEADERS, "Content-Type": "application/json"}

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN not defined.")
    print("    Run: API_TOKEN=<your_token> python delete_cancer.py [--yes]")
    sys.exit(1)

# ── 0. API health check ───────────────────────────────────────────────────────

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible: {API_URL}")
except Exception as e:
    print(f"❌  API unreachable ({API_URL}): {e}")
    sys.exit(1)

print()
if DRY_RUN:
    print("=" * 62)
    print("  DRY-RUN MODE — no deletion (add --yes to execute)")
    print("=" * 62)
else:
    print("=" * 62)
    print(f"  REAL DELETION for model: {MODEL_NAME}")
    print("=" * 62)

# ── 1. Inventory — models ─────────────────────────────────────────────────────

print(f"\n[1/3] Retrieving versions of '{MODEL_NAME}'…")

r = requests.get(f"{API_URL}/models", headers=HEADERS, params={"name": MODEL_NAME}, timeout=10)

if r.status_code == 200:
    all_models = r.json()
    versions = [m for m in all_models if m.get("name") == MODEL_NAME]
    if versions:
        print(f"      {len(versions)} version(s) found:")
        for v in versions:
            prod = " [PRODUCTION]" if v.get("is_production") else ""
            print(f"        * v{v.get('version')}{prod}")
    else:
        print(f"      No version found for '{MODEL_NAME}'.")
elif r.status_code == 404:
    versions = []
    print(f"      No model '{MODEL_NAME}' found.")
else:
    print(f"      ⚠️  GET /models responded {r.status_code}: {r.text[:120]}")
    versions = []

# ── 2. Inventory — predictions ────────────────────────────────────────────────

print(f"\n[2/3] Simulating purge of predictions and observed_results (older_than_days=0)…")

r = requests.delete(
    f"{API_URL}/predictions/purge",
    headers=HEADERS,
    params={"older_than_days": 0, "model_name": MODEL_NAME, "dry_run": True},
    timeout=15,
)

predictions_to_delete = 0
linked_gt = 0
if r.status_code == 200:
    purge_preview = r.json()
    predictions_to_delete = purge_preview.get("deleted_count", 0)
    linked_gt             = purge_preview.get("linked_observed_results_count", 0)
    oldest_remaining      = purge_preview.get("oldest_remaining")
    print(f"      {predictions_to_delete} prediction(s) to delete")
    print(f"      {linked_gt} associated observed_result(s) to delete")
    if oldest_remaining:
        print(f"      Oldest remaining after purge: {oldest_remaining}")
else:
    print(f"      ⚠️  GET purge dry-run responded {r.status_code}: {r.text[:120]}")

# ── 3. Inventory — observed results ──────────────────────────────────────────

print(f"\n[3/3] Observed_results will be deleted in cascade with predictions.")
observed_total = linked_gt

# ── Confirmation / dry-run exit ───────────────────────────────────────────────

print()
print("─" * 62)
print(f"  Summary of what will be deleted:")
print(f"    * {len(versions)} version(s) of model '{MODEL_NAME}'")
print(f"    * {predictions_to_delete} prediction(s)")
print(f"    * {observed_total} observed_result(s) (deleted in cascade)")
print("─" * 62)

if DRY_RUN:
    print()
    print("  Re-run with --yes to execute the deletion:")
    print(f"    python delete_cancer.py --yes")
    sys.exit(0)

print()

# ── Execution 1: prediction purge ─────────────────────────────────────────────

print("  STEP 1 — Purging predictions…")

if predictions_to_delete == 0:
    print("  ✅  No predictions to delete.")
else:
    r = requests.delete(
        f"{API_URL}/predictions/purge",
        headers=HEADERS,
        params={"older_than_days": 0, "model_name": MODEL_NAME, "dry_run": False},
        timeout=30,
    )

    if r.status_code == 200:
        result = r.json()
        deleted     = result.get("deleted_count", 0)
        deleted_obs = result.get("deleted_observed_results_count", 0)
        print(f"  ✅  {deleted} prediction(s) deleted.")
        if deleted_obs:
            print(f"  ✅  {deleted_obs} observed_result(s) deleted in cascade.")
    else:
        print(f"  ❌  Error {r.status_code} during purge:")
        try:
            print(f"      {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
        except Exception:
            print(f"      {r.text[:300]}")

# ── Execution 2: model deletion ───────────────────────────────────────────────

print()
print(f"  STEP 2 — Deleting all versions of '{MODEL_NAME}'…")

if not versions:
    print(f"  ✅  No model '{MODEL_NAME}' to delete.")
else:
    r = requests.delete(
        f"{API_URL}/models/{MODEL_NAME}",
        headers=HEADERS,
        timeout=30,
    )

    if r.status_code == 200:
        result = r.json()
        deleted_v  = result.get("deleted_versions",     [])
        mlflow_del = result.get("mlflow_runs_deleted",   [])
        minio_del  = result.get("minio_objects_deleted", [])

        print(f"  ✅  {len(deleted_v)} version(s) deleted: {deleted_v}")
        if mlflow_del:
            print(f"      {len(mlflow_del)} MLflow run(s) deleted")
        if minio_del:
            print(f"      {len(minio_del)} MinIO object(s) deleted")
    elif r.status_code == 404:
        print(f"  ✅  Model '{MODEL_NAME}' already absent.")
    else:
        print(f"  ❌  Error {r.status_code} during model deletion:")
        try:
            print(f"      {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
        except Exception:
            print(f"      {r.text[:300]}")

# ── Final summary ─────────────────────────────────────────────────────────────

print()
print("=" * 62)
print("  Cleanup complete.")
print("=" * 62)
