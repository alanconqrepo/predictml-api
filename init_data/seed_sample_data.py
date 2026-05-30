r"""
seed_sample_data.py — Seed sample data on first startup.

$env:API_URL = "http://localhost:80"; $env:API_TOKEN = "18_YgH-4oOQjFe6Ph0FtgUzM_oMolrnz"; $env:PYTHONIOENCODING = "utf-8"; & ".venv\Scripts\python.exe" init_data\seed_sample_data.py

Idempotent: checks whether iris-classifier v1.0.0 already exists before acting.

Deployment schema after seeding:
  iris-classifier
    v1.0.0  RandomForestClassifier    → production + ab_test
    v1.1.0  GradientBoostingClassifier → production + ab_test
    v1.2.0  ExtraTreesClassifier       → shadow
    v1.3.0  LogisticRegression         → uploaded only

  wine-regressor
    v1.0.0  GradientBoostingRegressor → production + ab_test
    v1.1.0  RandomForestRegressor     → production + ab_test
    v1.2.0  ExtraTreesRegressor       → shadow
    v1.3.0  Ridge (Pipeline)          → uploaded only

  cancer-classifier  (binary classification: malignant vs benign)
    v1.0.0  RandomForestClassifier    → production + ab_test
    v1.1.0  GradientBoostingClassifier → production + ab_test
    v1.2.0  ExtraTreesClassifier       → shadow
    v1.3.0  LogisticRegression         → uploaded only

  titanic-survival  (binary classification: survived 0/1, mixed numerical + categorical)
    v1.0.0  GradientBoosting Pipeline (OneHotEncoder)   → production + ab_test
    v1.1.0  RandomForest Pipeline (OneHotEncoder)        → production + ab_test
    v1.2.0  ExtraTrees Pipeline (OneHotEncoder)          → shadow
    v1.3.0  LogisticRegression Pipeline (OneHotEncoder)  → uploaded only

Environment variables:
  API_URL    Internal API URL  (default: http://api:8000)
  API_TOKEN  Admin token       (ADMIN_TOKEN as fallback)
"""

import os
import subprocess
import sys
from pathlib import Path

import requests

API_URL = os.environ.get("API_URL", "http://api:8000")
API_TOKEN = os.environ.get("API_TOKEN", os.environ.get("ADMIN_TOKEN", ""))

if not API_TOKEN:
    print("❌  API_TOKEN / ADMIN_TOKEN not set — seed skipped.")
    sys.exit(0)

SCRIPTS_DIR = Path(__file__).parent.parent / "documentation" / "Scripts"

# Execution order:
#  1. Iris models (v1.0.0 → v1.3.0)
#  2. Wine models (v1.0.0 → v1.3.0)
#  3. Predictions and ground truth for both datasets
SCRIPTS = [
    # ── iris-classifier ──────────────────────────────────────────────────────
    # v1.0.0 : RandomForest — production + ab_test
    ("iris/upload_iris_model.py", {"MODEL_NAME": "iris-classifier", "MODEL_VERSION": "1.0.0"}),
    # v1.1.0 : GradientBoosting — production + ab_test
    (
        "iris/upload_iris_model_GradientBoosting.py",
        {"MODEL_NAME": "iris-classifier", "MODEL_VERSION": "1.1.0"},
    ),
    # v1.2.0 : ExtraTrees — shadow
    (
        "iris/upload_iris_ExtraTrees_shadow.py",
        {"MODEL_NAME": "iris-classifier", "MODEL_VERSION": "1.2.0"},
    ),
    # v1.3.0 : LogisticRegression — uploaded only
    (
        "iris/upload_iris_LogisticRegression_uploaded.py",
        {"MODEL_NAME": "iris-classifier", "MODEL_VERSION": "1.3.0"},
    ),
    # ── wine-regressor ───────────────────────────────────────────────────────
    # v1.0.0 : GradientBoostingRegressor — production + ab_test
    ("wine/upload_wine_model.py", {"MODEL_NAME": "wine-regressor", "MODEL_VERSION": "1.0.0"}),
    # v1.1.0 : RandomForestRegressor — production + ab_test
    (
        "wine/upload_wine_RandomForest_abtest.py",
        {"MODEL_NAME": "wine-regressor", "MODEL_VERSION": "1.1.0"},
    ),
    # v1.2.0 : ExtraTreesRegressor — shadow
    (
        "wine/upload_wine_ExtraTrees_shadow.py",
        {"MODEL_NAME": "wine-regressor", "MODEL_VERSION": "1.2.0"},
    ),
    # v1.3.0 : Ridge — uploaded only
    (
        "wine/upload_wine_Ridge_uploaded.py",
        {"MODEL_NAME": "wine-regressor", "MODEL_VERSION": "1.3.0"},
    ),
    # ── cancer-classifier ────────────────────────────────────────────────────────
    # v1.0.0 : RandomForest — production + ab_test
    (
        "cancer/upload_cancer_model.py",
        {"MODEL_NAME": "cancer-classifier", "MODEL_VERSION": "1.0.0"},
    ),
    # v1.1.0 : GradientBoosting — production + ab_test
    (
        "cancer/upload_cancer_GradientBoosting_abtest.py",
        {"MODEL_NAME": "cancer-classifier", "MODEL_VERSION": "1.1.0"},
    ),
    # v1.2.0 : ExtraTrees — shadow
    (
        "cancer/upload_cancer_ExtraTrees_shadow.py",
        {"MODEL_NAME": "cancer-classifier", "MODEL_VERSION": "1.2.0"},
    ),
    # v1.3.0 : LogisticRegression — uploaded only
    (
        "cancer/upload_cancer_LogisticRegression_uploaded.py",
        {"MODEL_NAME": "cancer-classifier", "MODEL_VERSION": "1.3.0"},
    ),
    # ── titanic-survival ──────────────────────────────────────────────────────
    # v1.0.0 : GradientBoosting Pipeline (OneHotEncoder) — production + ab_test
    (
        "titanic/upload_titanic_model.py",
        {"MODEL_NAME": "titanic-survival", "MODEL_VERSION": "1.0.0"},
    ),
    # v1.1.0 : RandomForest Pipeline — production + ab_test
    (
        "titanic/upload_titanic_RandomForest_abtest.py",
        {"MODEL_NAME": "titanic-survival", "MODEL_VERSION": "1.1.0"},
    ),
    # v1.2.0 : ExtraTrees Pipeline — shadow
    (
        "titanic/upload_titanic_ExtraTrees_shadow.py",
        {"MODEL_NAME": "titanic-survival", "MODEL_VERSION": "1.2.0"},
    ),
    # v1.3.0 : LogisticRegression Pipeline — uploaded only
    (
        "titanic/upload_titanic_LogisticRegression_uploaded.py",
        {"MODEL_NAME": "titanic-survival", "MODEL_VERSION": "1.3.0"},
    ),
    # ── Predictions and ground truth ─────────────────────────────────────────
    ("iris/send_predictions_iris.py", {"MODEL_NAME": "iris-classifier", "SLEEP_BETWEEN": "1"}),
    ("iris/send_ground_truth_iris.py", {"MODEL_NAME": "iris-classifier"}),
    ("wine/send_predictions_wine.py", {"MODEL_NAME": "wine-regressor", "SLEEP_BETWEEN": "1"}),
    ("wine/send_ground_truth_wine.py", {"MODEL_NAME": "wine-regressor"}),
    (
        "cancer/send_predictions_cancer.py",
        {"MODEL_NAME": "cancer-classifier", "SLEEP_BETWEEN": "1"},
    ),
    ("cancer/send_ground_truth_cancer.py", {"MODEL_NAME": "cancer-classifier"}),
    (
        "titanic/send_predictions_titanic.py",
        {"MODEL_NAME": "titanic-survival", "SLEEP_BETWEEN": "1"},
    ),
    ("titanic/send_ground_truth_titanic.py", {"MODEL_NAME": "titanic-survival"}),
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


def predictions_exist(model_name: str) -> bool:
    try:
        r = requests.get(
            f"{API_URL}/predictions",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            params={"model_name": model_name, "limit": 1},
            timeout=5,
        )
        if r.status_code == 200:
            return len(r.json().get("predictions", [])) > 0
        return False
    except Exception:
        return False


def run_script(script_path_rel: str, extra_env: dict) -> bool:
    script_path = SCRIPTS_DIR / script_path_rel
    if not script_path.exists():
        print(f"⚠️   Script not found: {script_path} — skipped.")
        return True

    env = {
        **os.environ,
        "API_URL": API_URL,
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
        print(f"❌  {script_path_rel} failed (code {result.returncode})")
        return False
    return True


def main():
    print("=" * 62)
    print("  PredictML — Seeding sample data")
    print("=" * 62)

    if not api_healthy():
        print("❌  API unreachable — seed cancelled.")
        sys.exit(1)

    if model_exists("iris-classifier", "1.0.0") and predictions_exist("iris-classifier"):
        print("\n✅  Data already present (models + predictions) — seed skipped.")
        sys.exit(0)

    print("\n  Sample data absent — starting seed…\n")
    print("  Target schema:")
    print("    iris-classifier   v1.0.0 RandomForest        → production + ab_test")
    print("    iris-classifier   v1.1.0 GradientBoosting    → production + ab_test")
    print("    iris-classifier   v1.2.0 ExtraTrees          → shadow")
    print("    iris-classifier   v1.3.0 LogisticRegression  → uploaded")
    print("    wine-regressor    v1.0.0 GradientBoosting    → production + ab_test")
    print("    wine-regressor    v1.1.0 RandomForest        → production + ab_test")
    print("    wine-regressor    v1.2.0 ExtraTrees          → shadow")
    print("    wine-regressor    v1.3.0 Ridge               → uploaded")
    print("    cancer-classifier v1.0.0 RandomForest        → production + ab_test")
    print("    cancer-classifier v1.1.0 GradientBoosting    → production + ab_test")
    print("    cancer-classifier v1.2.0 ExtraTrees          → shadow")
    print("    cancer-classifier v1.3.0 LogisticRegression  → uploaded")
    print("    titanic-survival  v1.0.0 GradientBoosting+OneHotEncoder → production + ab_test")
    print("    titanic-survival  v1.1.0 RandomForest+OneHotEncoder     → production + ab_test")
    print("    titanic-survival  v1.2.0 ExtraTrees+OneHotEncoder       → shadow")
    print("    titanic-survival  v1.3.0 LogisticRegression+OneHotEncoder → uploaded")

    for script_path_rel, extra_env in SCRIPTS:
        ok = run_script(script_path_rel, extra_env)
        if not ok:
            print(f"\n❌  Seed stopped after failure of {script_path_rel}.")
            sys.exit(1)

    print("\n" + "=" * 62)
    print("  ✅  Seed completed successfully!")
    print("  → Dashboard: http://localhost:8501")
    print("=" * 62)


if __name__ == "__main__":
    main()
