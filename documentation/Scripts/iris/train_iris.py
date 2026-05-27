"""
train_iris.py — PredictML retraining script — Iris example
python .\train_iris.py
====================================================================

This script is designed to be uploaded with your model (field "Training script")
to enable automatic or scheduled retraining from the dashboard.

INTERFACE CONTRACT (environment variables automatically injected by the API)
-------------------------------------------------------------------------------------
  TRAIN_START_DATE   : start date  — format YYYY-MM-DD
  TRAIN_END_DATE     : end date    — format YYYY-MM-DD
  OUTPUT_MODEL_PATH  : absolute path where the produced .joblib should be saved

Optional variables:
  MLFLOW_TRACKING_URI      : MLflow server URI (e.g. http://localhost:5000)
  MLFLOW_TRACKING_USERNAME : MLflow username (if auth enabled)
  MLFLOW_TRACKING_PASSWORD : MLflow password (if auth enabled)
  MODEL_NAME               : source model name

EXPECTED OUTPUT
---------------
  - Model saved at OUTPUT_MODEL_PATH via joblib.dump()
  - Last JSON line on stdout with at minimum:
      {"accuracy": 0.95, "f1_score": 0.94}
  - Progress logs on stderr
  - Exit code 0 on success

MODULES ALLOWED by the PredictML sandbox
-------------------------------------------
  os, sys, json, joblib, datetime, dotenv, numpy, pandas, sklearn, mlflow
  (subprocess, requests, socket, urllib are blocked)

AUTOMATIC LIBRARY VERSION CAPTURE
-----------------------------------------------
  The API automatically generates a requirements.txt from the imports in this script.
  It is stored in MinIO  : {model_name}/v{version}_requirements.txt
  And logged as artifact : MLflow > environment/requirements.txt
  No action required in the script — everything is handled server-side at upload and retrain.
"""

import io
import json
import os
import joblib
import sys
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import logging
    import mlflow
    import mlflow.sklearn
    # pip version unresolved in sandbox → not actionable
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    # gRPC OTLP absent → tracing silently disabled
    logging.getLogger("mlflow.tracing.provider").setLevel(logging.ERROR)
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    import boto3
    from botocore.config import Config as BotocoreConfig
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False

DEBUG = True

# Force UTF-8 on Windows to avoid charmap crash on MLflow emojis (e.g. 🏃)
#if hasattr(sys.stdout, "buffer"):
#    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
#if hasattr(sys.stderr, "buffer"):
#    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_T0 = datetime.now()

def _ts(label: str) -> None:
    elapsed = (datetime.now() - _T0).total_seconds()
    print(f"[TIMING] {label} — +{elapsed:.2f}s", file=sys.stderr)

# ── 1. Environment variables (REQUIRED) ───────────────────────────────────────
if DEBUG:
    print(f"""
    -------------   BEFORE OVERRIDE  ------------
    [ENV] TRAIN_START_DATE           = {os.environ.get("TRAIN_START_DATE")}
    [ENV] TRAIN_END_DATE             = {os.environ.get("TRAIN_END_DATE")}
    [ENV] OUTPUT_MODEL_PATH          = {os.environ.get("OUTPUT_MODEL_PATH")}
    [ENV] MODEL_NAME                 = {os.environ.get("MODEL_NAME")}
    [ENV] MLFLOW_TRACKING_URI        = {os.environ.get("MLFLOW_TRACKING_URI")}
    [ENV] MLFLOW_TRACKING_USERNAME   = {os.environ.get("MLFLOW_TRACKING_USERNAME")}
    [ENV] MLFLOW_TRACKING_PASSWORD   = {os.environ.get("MLFLOW_TRACKING_PASSWORD")}
    [ENV] MLFLOW_HTTP_REQUEST_TIMEOUT= {os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT")}
    [ENV] MLFLOW_S3_ENDPOINT_URL     = {os.environ.get("MLFLOW_S3_ENDPOINT_URL")}
    [ENV] AWS_ACCESS_KEY_ID          = {os.environ.get("AWS_ACCESS_KEY_ID")}
    [ENV] AWS_SECRET_ACCESS_KEY      = {os.environ.get("AWS_SECRET_ACCESS_KEY")}
    [ENV] MINIO_ENDPOINT             = {os.environ.get("MINIO_ENDPOINT")}
    [ENV] MINIO_ROOT_USER            = {os.environ.get("MINIO_ROOT_USER")}
    [ENV] MINIO_ACCESS_KEY           = {os.environ.get("MINIO_ACCESS_KEY")}
    [ENV] MINIO_ROOT_PASSWORD        = {os.environ.get("MINIO_ROOT_PASSWORD")}
    [ENV] MINIO_SECRET_KEY           = {os.environ.get("MINIO_SECRET_KEY")}
    """)

TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE","2025-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE","2025-02-01")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH","default_model_path.joblib")

MODEL_NAME               = os.environ.get("MODEL_NAME", "iris-classifier")
MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")



# Short timeout for all MLflow HTTP calls (system default = 120 s → guaranteed timeout)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")

# MinIO credentials for log_model — boto3 reads os.environ directly.
# Without them, boto3 explores all credential providers (~15s) before giving up.
# Priority: AWS_* (standard boto3) > MINIO_ROOT_* (docker-compose)
_minio_endpoint   = (os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
                     or os.environ.get("MINIO_ENDPOINT", ""))
_minio_access_key = (os.environ.get("AWS_ACCESS_KEY_ID", "")
                     or os.environ.get("MINIO_ROOT_USER", ""))
_minio_secret_key = (os.environ.get("AWS_SECRET_ACCESS_KEY", "")
                     or os.environ.get("MINIO_ROOT_PASSWORD", ""))
if _minio_endpoint:
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = _minio_endpoint
if _minio_access_key:
    os.environ["AWS_ACCESS_KEY_ID"] = _minio_access_key
if _minio_secret_key:
    os.environ["AWS_SECRET_ACCESS_KEY"] = _minio_secret_key

# Outside Docker: substitute internal hostnames with localhost
# /.dockerenv is automatically created by Docker in every container
_in_docker = os.path.exists("/.dockerenv")
if not _in_docker:
    if "//mlflow:" in MLFLOW_TRACKING_URI:
        MLFLOW_TRACKING_URI = MLFLOW_TRACKING_URI.replace("//mlflow:", "//localhost:")
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    _s3_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
    if "//minio:" in _s3_url:
        _minio_internal_port = os.environ.get("MINIO_INTERNAL_PORT", "9000")
        _minio_host_port     = os.environ.get("MINIO_PORT", "9010")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = _s3_url.replace(
            f"//minio:{_minio_internal_port}", f"//localhost:{_minio_host_port}"
        )

if DEBUG:
    print(f"""
    -------------   AFTER OVERRIDE  ------------
    [ENV] TRAIN_START_DATE           = {TRAIN_START_DATE}
    [ENV] TRAIN_END_DATE             = {TRAIN_END_DATE}
    [ENV] OUTPUT_MODEL_PATH          = {OUTPUT_MODEL_PATH}
    [ENV] MODEL_NAME                 = {MODEL_NAME}
    [ENV] MLFLOW_TRACKING_URI        = {MLFLOW_TRACKING_URI}
    [ENV] MLFLOW_TRACKING_USERNAME   = {MLFLOW_TRACKING_USERNAME}
    [ENV] MLFLOW_TRACKING_PASSWORD   = {MLFLOW_TRACKING_PASSWORD}
    [ENV] MLFLOW_S3_ENDPOINT_URL     = {os.environ.get("MLFLOW_S3_ENDPOINT_URL")}
    [ENV] AWS_ACCESS_KEY_ID          = {os.environ.get("AWS_ACCESS_KEY_ID")}
    [ENV] AWS_SECRET_ACCESS_KEY      = {os.environ.get("AWS_SECRET_ACCESS_KEY")}
    """)



print(
    f"[{MODEL_NAME}] Retraining from {TRAIN_START_DATE} to {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[{MODEL_NAME}] Output: {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("startup")

# ── 2. Data loading ───────────────────────────────────────────────────────────
#
# REPLACE THIS BLOCK with your own data source:
#
#   import pandas as pd
#   df = pd.read_csv("data/training_data.csv", parse_dates=["date"])
#   df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]
#   if df.empty:
#       print(json.dumps({"error": "No data for this range"}))
#       sys.exit(1)
#   X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
#   y = df["species"]
#
# ─────────────────────────────────────────────────────────────────────────────

print(f"[{MODEL_NAME}] Loading Iris dataset (synthetic data)…", file=sys.stderr)
_ts("before data loading")

iris = load_iris()
X_full = pd.DataFrame(iris.data, columns=iris.feature_names)
y_full = iris.target

# Simulate a time filter: sample size proportional to the date range
start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end   = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
delta_days = max(1, (end - start).days)

n_samples = min(len(X_full), max(30, delta_days * 2))
rng     = np.random.default_rng(seed=delta_days % 1000)
indices = rng.choice(len(X_full), size=n_samples, replace=False)
X, y    = X_full.iloc[indices], y_full[indices]

print(f"[{MODEL_NAME}] {n_samples} samples retained out of {len(X_full)} available.", file=sys.stderr)
_ts("after data loading")

if n_samples < 20:
    print(json.dumps({"error": f"Not enough data ({n_samples} samples < 20 required)"}))
    sys.exit(1)

# ── 3. Training ───────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if n_samples >= 30 else None
)

print(
    f"[{MODEL_NAME}] Training on {len(X_train)} samples, "
    f"evaluating on {len(X_test)}…",
    file=sys.stderr,
)

HYPERPARAMS = {
    "n_estimators":      200,    # number of trees
    "max_depth":         10,     # max depth (None = unlimited)
    "min_samples_split": 4,      # min samples required to split a node
    "min_samples_leaf":  2,      # min samples required in a leaf
    "max_features":      "sqrt", # number of features considered at each split
    "class_weight":      "balanced",  # compensates for imbalanced classes
    "random_state":      42,
    "n_jobs":            -1,     # parallelism across all available cores
}
model = RandomForestClassifier(**HYPERPARAMS)
_ts("before fit")
model.fit(X_train, y_train)
_ts("after fit")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────

y_pred    = model.predict(X_test)
acc       = float(accuracy_score(y_test, y_pred))
f1        = float(f1_score(y_test,        y_pred, average="weighted", zero_division=0))
precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
recall    = float(recall_score(y_test,    y_pred, average="weighted", zero_division=0))

print(
    f"[{MODEL_NAME}] Accuracy: {acc:.4f} | F1: {f1:.4f}"
    f" | Precision: {precision:.4f} | Recall: {recall:.4f}",
    file=sys.stderr,
)
_ts("after evaluation")

# ── 5. Model saving (REQUIRED) ────────────────────────────────────────────────

joblib.dump(model, OUTPUT_MODEL_PATH)

print(f"[{MODEL_NAME}] Model saved → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("after model saving")

# ── 5b. Save training dataset → MinIO + MLflow artifact ──────────────────────

_dataset_minio_path = None

try:
    # Build the CSV (features + target)
    _df_train = X_train.copy()
    _df_train["target"] = y_train
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_train.csv"
    _csv_local = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    print(f"[{MODEL_NAME}] Dataset CSV created ({len(_df_train)} rows) → {_csv_local}", file=sys.stderr)

    # Upload to MinIO via boto3 (S3-compatible)
    _minio_endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
    _aws_key    = os.environ.get("AWS_ACCESS_KEY_ID", "")
    _aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    _bucket     = os.environ.get("MINIO_BUCKET", "models")

    if _BOTO3_AVAILABLE and _minio_endpoint_url and _aws_key:
        _s3 = boto3.client(
            "s3",
            endpoint_url=_minio_endpoint_url,
            aws_access_key_id=_aws_key,
            aws_secret_access_key=_aws_secret,
            config=BotocoreConfig(signature_version="s3v4"),
        )
        _object_key = f"{MODEL_NAME}/datasets/{_csv_filename}"
        _s3.upload_file(_csv_local, _bucket, _object_key)
        _dataset_minio_path = _object_key
        print(f"[{MODEL_NAME}] Dataset uploaded to MinIO → {_bucket}/{_object_key}", file=sys.stderr)
    else:
        print(f"[{MODEL_NAME}] Dataset not uploaded to MinIO (boto3={'ok' if _BOTO3_AVAILABLE else 'absent'}, endpoint='{_minio_endpoint_url}')", file=sys.stderr)

except Exception as _ds_exc:
    print(f"[{MODEL_NAME}] Dataset saving skipped: {_ds_exc}", file=sys.stderr)

_ts("after dataset saving")

# ── 6. Statistics for MLflow and drift detection ──────────────────────────────

feature_names = list(iris.feature_names)

feature_stats = {
    name: {
        "mean":      round(float(X_train[name].mean()), 4),
        "std":       round(float(X_train[name].std()),  4),
        "min":       round(float(X_train[name].min()),  4),
        "max":       round(float(X_train[name].max()),  4),
        "null_rate": 0.0,
    }
    for name in feature_names
}

total_train = len(y_train)
# Keys = integer index (str) — corresponds to prediction_result::text in the DB
label_distribution = {
    str(int(cls)): round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in np.unique(y_train)
}

# ── 7. MLflow logging (graceful degradation if MLflow unavailable) ────────────

mlflow_run_id = None

if _MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        _ts("MLflow — set_tracking_uri")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        _ts("MLflow — set_experiment (network call #1)")
        mlflow.set_experiment(f"predictml/{MODEL_NAME}")
        _ts("MLflow — set_experiment OK")

        run_name = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}"
        _ts("MLflow — start_run (network call #2)")
        with mlflow.start_run(run_name=run_name) as run:
            _ts("MLflow — start_run OK")

            # Params — single network call
            _ts("MLflow — log_params (network call #3)")
            mlflow.log_params({
                "algorithm":         "RandomForest",
                "train_start_date":  TRAIN_START_DATE,
                "train_end_date":    TRAIN_END_DATE,
                "n_samples_total":   n_samples,
                "test_size":         0.2,
                **HYPERPARAMS,
            })
            _ts("MLflow — log_params OK")

            # All metrics in a single call (instead of N individual calls)
            all_metrics: dict[str, float] = {
                "accuracy":     acc,
                "f1_score":     f1,
                "n_rows_train": float(len(X_train)),
                "n_rows_test":  float(len(X_test)),
            }
            for feat_name, stats in feature_stats.items():
                safe = feat_name.replace(" ", "_").replace("(", "").replace(")", "")[:40]
                for stat_key, val in stats.items():
                    all_metrics[f"feat_{safe}_{stat_key}"] = float(val)
            for label, ratio in label_distribution.items():
                all_metrics[f"label_{label}_ratio"] = float(ratio)

            _ts(f"MLflow — log_metrics x{len(all_metrics)} (network call #4)")
            mlflow.log_metrics(all_metrics)
            _ts("MLflow — log_metrics OK")

            # Tags — single network call
            _ts("MLflow — set_tags (network call #5)")
            mlflow.set_tags({
                "model_name":  MODEL_NAME,
                "algorithm":   "RandomForest",
                "trigger":     "script",
                "n_features":  str(len(feature_names)),
                "n_classes":   str(len(iris.target_names)),
            })
            _ts("MLflow — set_tags OK")

            # Artifact — sklearn model (graceful degradation if MinIO unreachable)
            # Explicit signature: float64 inputs + proba float outputs (avoids the
            # "integer column" warning caused by predict() returning integer classes)
            _X_example = X_test.astype("float64")
            _signature = mlflow.models.infer_signature(
                _X_example,
                model.predict_proba(_X_example),
            )
            _ts("MLflow — log_model start (network call #6 — MinIO)")
            try:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=_signature,
                )
                _ts("MLflow — log_model OK")
                print(f"[{MODEL_NAME}] Model artifact logged in MLflow.", file=sys.stderr)
            except Exception as art_exc:
                _ts("MLflow — log_model FAILED")
                print(f"[{MODEL_NAME}] Artifact skipped (MinIO unreachable): {art_exc}", file=sys.stderr)

            # Artifact — training dataset (graceful degradation)
            if _dataset_minio_path and os.path.exists(_csv_local):
                try:
                    _ts("MLflow — log_artifact dataset (network call #7 — MinIO)")
                    mlflow.log_artifact(_csv_local, artifact_path="dataset")
                    _ts("MLflow — log_artifact dataset OK")
                    print(f"[{MODEL_NAME}] Dataset logged as MLflow artifact.", file=sys.stderr)
                except Exception as _art_ds_exc:
                    print(f"[{MODEL_NAME}] Dataset artifact skipped: {_art_ds_exc}", file=sys.stderr)

            mlflow_run_id = run.info.run_id

        _ts("MLflow — run closed")
        print(f"[{MODEL_NAME}] MLflow run created: {mlflow_run_id}", file=sys.stderr)

    except Exception as exc:
        _ts("MLflow — EXCEPTION (entire block abandoned)")
        print(f"[{MODEL_NAME}] MLflow unavailable — retraining continues: {exc}", file=sys.stderr)
        mlflow_run_id = None

else:
    reason = "mlflow not installed" if not _MLFLOW_AVAILABLE else "MLFLOW_TRACKING_URI not defined"
    print(f"[{MODEL_NAME}] MLflow skipped ({reason}).", file=sys.stderr)

# ── 8. Library version capture (read by the API to generate requirements.txt) ─

import importlib.metadata as _imeta  # noqa: E402

_deps: dict = {}
for _pkg in ["scikit-learn", "numpy", "pandas", "mlflow", "python-dotenv", "boto3", "botocore"]:
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass

# ── 9. JSON stdout — LAST LINE (read by the API — do not add anything after) ──

output = {
    "accuracy":           round(acc, 4),
    "f1_score":           round(f1, 4),
    "precision":          round(precision, 4),
    "recall":             round(recall, 4),
    "n_rows":             len(X_train),
    "features_count":     len(iris.feature_names),
    "classes":            list(iris.target_names),
    "hyperparameters":    HYPERPARAMS,
    "training_dataset":   _dataset_minio_path or "scikit-learn Iris dataset (Fisher, 1936) - 150 observations, 3 classes",
    # confidence_threshold: if the max probability predicted by the model is below
    # this threshold, the API marks the prediction with low_confidence=True.
    # This allows callers to handle uncertain predictions differently
    # (e.g.: routing to a human, rejecting the prediction, monitoring alert).
    # 0.60 is a reasonable starting point for Iris: the model is very well calibrated
    # on this dataset, so cases below 60% confidence are genuine edge cases.
    "confidence_threshold": 0.60,
    "feature_stats":      feature_stats,
    "label_distribution": label_distribution,
    "dependencies":       _deps,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

_ts("script end")
print(json.dumps(output))
