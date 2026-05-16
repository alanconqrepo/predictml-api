"""
delete_wine.py — Nettoyage complet du modèle wine-regressor via l'API PredictML
=================================================================================

Ce script supprime dans l'ordre :
  1. Toutes les prédictions du modèle  (DELETE /predictions/purge)
  2. Toutes les versions du modèle     (DELETE /models/{name})

Il affiche ensuite le nombre de résultats observés restants (ground truth) et
explique pourquoi ils ne peuvent pas être supprimés via l'API.

Usage :
  python delete_wine.py                     # dry-run → affiche ce qui sera supprimé
  python delete_wine.py --yes               # supprime sans confirmation

Variables d'environnement :
  API_URL        URL de l'API          (défaut : http://localhost:80)
  API_TOKEN      Token Bearer admin — requis
  MODEL_NAME     Nom du modèle        (défaut : wine-regressor)

Prérequis Python :
  pip install requests python-dotenv

Ce script supprime la totalité des données, y compris les prédictions récentes
et les observed_results (ground truth) associés.
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
MODEL_NAME = os.environ.get("MODEL_NAME", "wine-regressor")

DRY_RUN = "--yes" not in sys.argv

HEADERS      = {"Authorization": f"Bearer {API_TOKEN}"}
HEADERS_JSON = {**HEADERS, "Content-Type": "application/json"}

# ── Validation ────────────────────────────────────────────────────────────────

if not API_TOKEN:
    print("❌  API_TOKEN non défini.")
    print("    Lancez : API_TOKEN=<votre_token> python delete_wine.py [--yes]")
    sys.exit(1)

# ── 0. Vérification API ───────────────────────────────────────────────────────

try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    print(f"✅  API accessible : {API_URL}")
except Exception as e:
    print(f"❌  API inaccessible ({API_URL}) : {e}")
    sys.exit(1)

print()
if DRY_RUN:
    print("=" * 62)
    print("  MODE DRY-RUN — aucune suppression (ajoutez --yes pour exécuter)")
    print("=" * 62)
else:
    print("=" * 62)
    print(f"  SUPPRESSION RÉELLE pour le modèle : {MODEL_NAME}")
    print("=" * 62)

# ── 1. Inventaire — modèles ───────────────────────────────────────────────────

print(f"\n[1/3] Récupération des versions de '{MODEL_NAME}'…")

r = requests.get(f"{API_URL}/models", headers=HEADERS, params={"name": MODEL_NAME}, timeout=10)

if r.status_code == 200:
    all_models = r.json()
    versions = [m for m in all_models if m.get("name") == MODEL_NAME]
    if versions:
        print(f"      {len(versions)} version(s) trouvée(s) :")
        for v in versions:
            prod = " [PRODUCTION]" if v.get("is_production") else ""
            print(f"        • v{v.get('version')}{prod}")
    else:
        print(f"      Aucune version trouvée pour '{MODEL_NAME}'.")
elif r.status_code == 404:
    versions = []
    print(f"      Aucun modèle '{MODEL_NAME}' trouvé.")
else:
    print(f"      ⚠️  GET /models a répondu {r.status_code} : {r.text[:120]}")
    versions = []

# ── 2. Inventaire — prédictions ───────────────────────────────────────────────

print(f"\n[2/3] Simulation de purge des prédictions et observed_results (older_than_days=0)…")

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
    print(f"      {predictions_to_delete} prédiction(s) à supprimer")
    print(f"      {linked_gt} observed_result(s) associé(s) à supprimer")
    if oldest_remaining:
        print(f"      Plus ancienne après purge : {oldest_remaining}")
else:
    print(f"      ⚠️  GET purge dry-run a répondu {r.status_code} : {r.text[:120]}")

# ── 3. Inventaire — observed results ─────────────────────────────────────────

print(f"\n[3/3] Les observed_results seront supprimés en cascade avec les prédictions.")
observed_total = linked_gt

# ── Confirmation / sortie dry-run ─────────────────────────────────────────────

print()
print("─" * 62)
print(f"  Résumé de ce qui sera supprimé :")
print(f"    • {len(versions)} version(s) du modèle '{MODEL_NAME}'")
print(f"    • {predictions_to_delete} prédiction(s)")
print(f"    • {observed_total} observed_result(s) (supprimés en cascade)")
print("─" * 62)

if DRY_RUN:
    print()
    print("  Relancez avec --yes pour exécuter la suppression :")
    print(f"    python delete_wine.py --yes")
    sys.exit(0)

print()

# ── Exécution 1 : purge des prédictions ──────────────────────────────────────

print("  ÉTAPE 1 — Purge des prédictions…")

if predictions_to_delete == 0:
    print("  ✅  Aucune prédiction à supprimer.")
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
        print(f"  ✅  {deleted} prédiction(s) supprimée(s).")
        if deleted_obs:
            print(f"  ✅  {deleted_obs} observed_result(s) supprimé(s) en cascade.")
    else:
        print(f"  ❌  Erreur {r.status_code} lors de la purge :")
        try:
            print(f"      {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
        except Exception:
            print(f"      {r.text[:300]}")

# ── Exécution 2 : suppression du modèle ──────────────────────────────────────

print()
print(f"  ÉTAPE 2 — Suppression de toutes les versions de '{MODEL_NAME}'…")

if not versions:
    print(f"  ✅  Aucun modèle '{MODEL_NAME}' à supprimer.")
else:
    r = requests.delete(
        f"{API_URL}/models/{MODEL_NAME}",
        headers=HEADERS,
        timeout=30,
    )

    if r.status_code == 200:
        result = r.json()
        deleted_v  = result.get("deleted_versions",    [])
        mlflow_del = result.get("mlflow_runs_deleted",  [])
        minio_del  = result.get("minio_objects_deleted", [])

        print(f"  ✅  {len(deleted_v)} version(s) supprimée(s) : {deleted_v}")
        if mlflow_del:
            print(f"      {len(mlflow_del)} run(s) MLflow supprimé(s)")
        if minio_del:
            print(f"      {len(minio_del)} objet(s) MinIO supprimé(s)")
    elif r.status_code == 404:
        print(f"  ✅  Modèle '{MODEL_NAME}' déjà absent.")
    else:
        print(f"  ❌  Erreur {r.status_code} lors de la suppression du modèle :")
        try:
            print(f"      {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
        except Exception:
            print(f"      {r.text[:300]}")

# ── Résumé final ──────────────────────────────────────────────────────────────

print()
print("=" * 62)
print("  Nettoyage terminé.")
print("=" * 62)
