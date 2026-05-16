"""
seed_sample_data.py — Seed des données exemples au premier démarrage.

Idempotent : vérifie si iris-classifier v1.0.0 existe déjà avant d'agir.
Lance les scripts de la documentation dans l'ordre :
  1. iris/upload_iris_model.py                    (RandomForest — mis en production)
  2. iris/upload_iris_model_GradientBoosting.py   (GradientBoosting — non en prod)
  3. wine/upload_wine_model.py                    (GradientBoostingRegressor — mis en production)
  4. iris/send_predictions_iris.py                (exemples de prédictions iris)
  5. iris/send_ground_truth_iris.py               (ground truth iris)
  6. wine/send_predictions_wine.py                (exemples de prédictions wine)
  7. wine/send_ground_truth_wine.py               (ground truth wine)

Variables d'environnement :
  API_URL    URL de l'API interne  (défaut : http://api:8000)
  API_TOKEN  Token admin           (ADMIN_TOKEN en fallback)
"""

import os
import subprocess
import sys
from pathlib import Path

import requests

API_URL   = os.environ.get("API_URL",   "http://api:8000")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

if not API_TOKEN:
    print("❌  API_TOKEN / ADMIN_TOKEN non défini — seed ignoré.")
    sys.exit(0)

SCRIPTS_DIR = Path(__file__).parent.parent / "documentation" / "Scripts"

SCRIPTS = [
    ("iris/upload_iris_model.py",                  {"MODEL_NAME": "iris-classifier", "MODEL_VERSION": "1.0.0"}),
    ("iris/upload_iris_model_GradientBoosting.py", {"MODEL_NAME": "iris-classifier", "MODEL_VERSION": "1.1.0"}),
    ("wine/upload_wine_model.py",                  {"MODEL_NAME": "wine-regressor",  "MODEL_VERSION": "1.0.0"}),
    ("iris/send_predictions_iris.py",              {"MODEL_NAME": "iris-classifier"}),
    ("iris/send_ground_truth_iris.py",             {"MODEL_NAME": "iris-classifier"}),
    ("wine/send_predictions_wine.py",              {"MODEL_NAME": "wine-regressor"}),
    ("wine/send_ground_truth_wine.py",             {"MODEL_NAME": "wine-regressor"}),
]


def api_healthy() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def model_exists(name: str, version: str) -> bool:
    try:
        r = requests.get(
            f"{API_URL}/models/{name}/{version}",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def run_script(script_path_rel: str, extra_env: dict) -> bool:
    script_path = SCRIPTS_DIR / script_path_rel
    if not script_path.exists():
        print(f"⚠️   Script introuvable : {script_path} — ignoré.")
        return True

    env = {
        **os.environ,
        "API_URL":   API_URL,
        "API_TOKEN": API_TOKEN,
        **extra_env,
    }

    print(f"\n▶  {script_path_rel}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"❌  {script_path_rel} a échoué (code {result.returncode})")
        return False
    return True


def main():
    print("=" * 62)
    print("  PredictML — Seed des données exemples")
    print("=" * 62)

    if not api_healthy():
        print("❌  API inaccessible — seed annulé.")
        sys.exit(1)

    if model_exists("iris-classifier", "1.0.0"):
        print("\n✅  iris-classifier v1.0.0 déjà présent — seed ignoré.")
        sys.exit(0)

    print("\n  Données exemples absentes — lancement du seed…\n")

    for script_path_rel, extra_env in SCRIPTS:
        ok = run_script(script_path_rel, extra_env)
        if not ok:
            print(f"\n❌  Arrêt du seed après l'échec de {script_path_rel}.")
            sys.exit(1)

    print("\n" + "=" * 62)
    print("  ✅  Seed terminé avec succès !")
    print("  → Dashboard : http://localhost:8501")
    print("=" * 62)


if __name__ == "__main__":
    main()
