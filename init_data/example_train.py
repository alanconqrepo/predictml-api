"""
Example train.py script for the PredictML retraining feature.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERFACE CONTRACT — REQUIRED environment variables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TRAIN_START_DATE  : start date in YYYY-MM-DD format
                      (e.g. "2025-01-01")
  TRAIN_END_DATE    : end date in YYYY-MM-DD format
                      (e.g. "2025-12-31")
  OUTPUT_MODEL_PATH : absolute path where the produced model (.joblib) must be saved
                      (e.g. "/tmp/abc123/output_model.joblib")

Optional environment variables injected by the API:
  MLFLOW_TRACKING_URI : MLflow server URI
  MODEL_NAME          : source model name
  TRAIN_DATA_PATH     : path to the CSV of production data exported by the API
                        (predictions + observed results).
                        Absent on the first training run (no production data yet).

IMPORTANT — MLflow is managed automatically by the API:
  The API creates the MLflow run itself after this script finishes.
  You do NOT need to call mlflow.start_run() here.
  To enrich MLflow logging, add the optional keys below
  to the stdout JSON output (section 6).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAIN_DATA_PATH CSV FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Columns: id_obs, input_features, prediction_result, observed_result,
           timestamp, model_version, response_time_ms

  - input_features   : JSON dict of features sent to /predict
  - observed_result  : actual observed value (JSON), empty if not provided
  - prediction_result: what the model predicted (JSON)

  For supervised training, filter rows where observed_result is non-empty:
    X = [json.loads(row["input_features"]) for row if row["observed_result"]]
    y = [json.loads(row["observed_result"]) for row if row["observed_result"]]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. The model MUST be saved to OUTPUT_MODEL_PATH via joblib.dump().
  2. Print to stdout a JSON as the LAST JSON line of the output:
       {
         "accuracy": 0.95,          # required for DB update
         "f1_score": 0.94,          # required for DB update
         "n_rows": 12450,           # optional — logged in MLflow
         "feature_stats": {         # optional — logged as MLflow metrics
           "sepal_length": {"mean": 5.8, "std": 0.83, "min": 4.3, "max": 7.9, "null_rate": 0.0}
         },
         "label_distribution": {    # optional — logged as MLflow metrics
           "setosa": 0.33, "versicolor": 0.33, "virginica": 0.34
         }
       }
  3. Progress logs may be printed to stderr freely.
  4. Exit with code 0 on success, non-zero on failure.
"""

import csv
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# 1. Read environment variables (REQUIRED)
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

# Optional variables
MODEL_NAME = os.environ.get("MODEL_NAME", "example_model")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH")  # absent on the first training run

print(
    f"[train.py] Retraining '{MODEL_NAME}' "
    f"from {TRAIN_START_DATE} to {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[train.py] Model output: {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load data
# ──────────────────────────────────────────────────────────────────────────────

if TRAIN_DATA_PATH:
    # ── Retrain: production data exported by the API ──────────────────────────
    print(f"[train.py] Loading production data: {TRAIN_DATA_PATH}", file=sys.stderr)

    features_list = []
    labels_list = []
    feature_names = None

    with open(TRAIN_DATA_PATH, newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            if not row["observed_result"]:
                continue  # skip predictions without an observed result
            features = json.loads(row["input_features"])
            label = json.loads(row["observed_result"])
            if feature_names is None:
                feature_names = sorted(features.keys())
            features_list.append([features[k] for k in feature_names])
            labels_list.append(label)

    if not features_list:
        print(
            json.dumps({"error": "No labelled data in the date window."}),
            flush=True,
        )
        sys.exit(1)

    X = np.array(features_list, dtype=float)
    y = np.array(labels_list)
    print(f"[train.py] {len(X)} labelled examples loaded.", file=sys.stderr)

else:
    # ── First training run: synthetic dataset (no production data available) ──
    print("[train.py] TRAIN_DATA_PATH absent — using the Iris dataset.", file=sys.stderr)
    from sklearn.datasets import load_iris

    iris = load_iris()
    X_full, y_full = iris.data, iris.target
    feature_names = list(iris.feature_names)

    # Simulate a temporal filter proportional to the date range
    start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
    end = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
    delta_days = max(1, (end - start).days)
    n_samples = min(len(X_full), max(30, delta_days * 2))
    rng = np.random.default_rng(seed=delta_days % 1000)
    indices = rng.choice(len(X_full), size=n_samples, replace=False)
    X, y = X_full[indices], y_full[indices]
    print(f"[train.py] {n_samples} examples selected.", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Training
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(
    f"[train.py] Training on {len(X_train)} examples, "
    f"evaluating on {len(X_test)}…",
    file=sys.stderr,
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

print(f"[train.py] Accuracy: {acc:.4f} | F1 Score: {f1:.4f}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Save model (REQUIRED — use joblib.dump)
# ──────────────────────────────────────────────────────────────────────────────
joblib.dump(model, OUTPUT_MODEL_PATH)

print(f"[train.py] Model saved: {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Capture library versions (read by the API for requirements.txt)
# ──────────────────────────────────────────────────────────────────────────────
import importlib.metadata as _imeta  # noqa: E402

_deps: dict = {}
for _pkg in ["scikit-learn", "numpy", "pandas", "mlflow", "python-dotenv"]:
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# 7. Metrics to stdout (last JSON line — read by the API for MLflow + DB)
# ──────────────────────────────────────────────────────────────────────────────
feature_stats = {}
if feature_names is not None:
    for i, fname in enumerate(feature_names):
        col = X_train[:, i] if X_train.ndim > 1 else X_train
        feature_stats[fname] = {
            "mean": round(float(np.mean(col)), 4),
            "std": round(float(np.std(col)), 4),
            "min": round(float(np.min(col)), 4),
            "max": round(float(np.max(col)), 4),
            "null_rate": 0.0,
        }

classes_arr = np.unique(y_train)
total_train = len(y_train)
label_distribution = {
    str(cls): round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in classes_arr
    if total_train > 0
}

print(
    json.dumps(
        {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "n_rows": len(X_train),
            "feature_stats": feature_stats,
            "label_distribution": label_distribution,
            "dependencies": _deps,
        }
    )
)
