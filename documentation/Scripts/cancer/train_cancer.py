"""
train_cancer.py — PredictML retraining script — Breast Cancer (binary)
python .\train_cancer.py
================================================================================

This script is designed to be uploaded with your model (field "Training script")
to enable automatic or scheduled retraining from the dashboard.

INTERFACE CONTRACT (environment variables automatically injected by the API)
-------------------------------------------------------------------------------------
  TRAIN_START_DATE   : start date  — format YYYY-MM-DD
  TRAIN_END_DATE     : end date    — format YYYY-MM-DD
  OUTPUT_MODEL_PATH  : absolute path where to save the produced .joblib

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

ALLOWED MODULES by the PredictML sandbox
-------------------------------------------
  os, sys, json, joblib, datetime, dotenv, numpy, pandas, sklearn, mlflow
  (subprocess, requests, socket, urllib are blocked)
"""

import json
import os
import joblib
import sys
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import logging
    import mlflow
    import mlflow.sklearn
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
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

_T0 = datetime.now()

def _ts(label: str) -> None:
    elapsed = (datetime.now() - _T0).total_seconds()
    print(f"[TIMING] {label} — +{elapsed:.2f}s", file=sys.stderr)

# ── 1. Environment variables ──────────────────────────────────────────────────

if DEBUG:
    print(f"""
    -------------   BEFORE OVERRIDE  ------------
    [ENV] TRAIN_START_DATE           = {os.environ.get("TRAIN_START_DATE")}
    [ENV] TRAIN_END_DATE             = {os.environ.get("TRAIN_END_DATE")}
    [ENV] OUTPUT_MODEL_PATH          = {os.environ.get("OUTPUT_MODEL_PATH")}
    [ENV] MODEL_NAME                 = {os.environ.get("MODEL_NAME")}
    [ENV] MLFLOW_TRACKING_URI        = {os.environ.get("MLFLOW_TRACKING_URI")}
    """)

TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE", "2025-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE",   "2025-02-01")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "default_model_path.joblib")

MODEL_NAME               = os.environ.get("MODEL_NAME", "cancer-classifier")
MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")

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

print(
    f"[{MODEL_NAME}] Retraining from {TRAIN_START_DATE} to {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[{MODEL_NAME}] Output: {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("startup")

# ── 2. Data loading ───────────────────────────────────────────────────────────

print(f"[{MODEL_NAME}] Loading Breast Cancer Wisconsin dataset...", file=sys.stderr)
_ts("before data loading")

cancer = load_breast_cancer()
X_full = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_full = cancer.target

# Simulate a time filter: size proportional to the date range
start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end   = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
delta_days = max(1, (end - start).days)

n_samples = min(len(X_full), max(40, delta_days * 3))
rng     = np.random.default_rng(seed=delta_days % 1000)
indices = rng.choice(len(X_full), size=n_samples, replace=False)
X, y    = X_full.iloc[indices], y_full[indices]

print(f"[{MODEL_NAME}] {n_samples} samples selected out of {len(X_full)} available.", file=sys.stderr)
_ts("after data loading")

if n_samples < 20:
    print(json.dumps({"error": f"Not enough data ({n_samples} samples < 20 required)"}))
    sys.exit(1)

# ── 3. Training ───────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if n_samples >= 40 else None
)

print(
    f"[{MODEL_NAME}] Training on {len(X_train)} samples, "
    f"evaluating on {len(X_test)}...",
    file=sys.stderr,
)

HYPERPARAMS = {
    "n_estimators":      200,
    "max_depth":         10,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}
model = RandomForestClassifier(**HYPERPARAMS)
_ts("before fit")
model.fit(X_train, y_train)
_ts("after fit")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────

y_pred     = model.predict(X_test)
y_proba    = model.predict_proba(X_test)[:, 1]
acc        = float(accuracy_score(y_test, y_pred))
f1         = float(f1_score(y_test,        y_pred, average="binary", zero_division=0))
precision  = float(precision_score(y_test, y_pred, average="binary", zero_division=0))
recall     = float(recall_score(y_test,    y_pred, average="binary", zero_division=0))
roc_auc    = float(roc_auc_score(y_test,   y_proba))

print(
    f"[{MODEL_NAME}] Accuracy: {acc:.4f} | F1: {f1:.4f}"
    f" | Precision: {precision:.4f} | Recall: {recall:.4f} | ROC-AUC: {roc_auc:.4f}",
    file=sys.stderr,
)
_ts("after evaluation")

# ── 5. Model saving ───────────────────────────────────────────────────────────

joblib.dump(model, OUTPUT_MODEL_PATH)

print(f"[{MODEL_NAME}] Model saved → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("after model saving")

# ── 5b. Save training dataset → MinIO ────────────────────────────────────────

_dataset_minio_path = None

try:
    _df_train = X_train.copy()
    _df_train["target"] = y_train
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_train.csv"
    _csv_local = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    print(f"[{MODEL_NAME}] CSV dataset created ({len(_df_train)} rows) → {_csv_local}", file=sys.stderr)

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
        print(f"[{MODEL_NAME}] Dataset not uploaded to MinIO (boto3={'ok' if _BOTO3_AVAILABLE else 'missing'}, endpoint='{_minio_endpoint_url}')", file=sys.stderr)

except Exception as _ds_exc:
    print(f"[{MODEL_NAME}] Dataset saving skipped: {_ds_exc}", file=sys.stderr)

_ts("after dataset saving")

# ── 6. Statistics ─────────────────────────────────────────────────────────────

feature_names = list(cancer.feature_names)

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


# -- Feature importances ------------------------------------------------------

feature_importances: dict = {}
try:
    import numpy as _np
    _est = model
    if hasattr(_est, "steps"):
        _est = _est.steps[-1][1]
    _raw_imp = None
    if hasattr(_est, "feature_importances_"):
        _raw_imp = _np.array(_est.feature_importances_)
    elif hasattr(_est, "coef_"):
        _c = _np.array(_est.coef_)
        _raw_imp = _np.mean(_np.abs(_c), axis=0) if _c.ndim > 1 else _np.abs(_c)
    if _raw_imp is not None:
        _total = _raw_imp.sum()
        if _total > 0:
            _raw_imp = _raw_imp / _total
        _feat_names = (
            list(model.feature_names_in_) if hasattr(model, "feature_names_in_")
            else list(X_train.columns) if hasattr(X_train, "columns")
            else [f"feature_{i}" for i in range(len(_raw_imp))]
        )
        feature_importances = dict(sorted(
            {n: round(float(v), 6) for n, v in zip(_feat_names, _raw_imp)}.items(),
            key=lambda kv: kv[1], reverse=True
        ))
        print(f"[{MODEL_NAME}] Feature importances extracted ({len(feature_importances)} features).", file=sys.stderr)
except Exception as _fi_exc:
    print(f"[{MODEL_NAME}] Feature importances skipped: {_fi_exc}", file=sys.stderr)

# ── 7. MLflow logging ─────────────────────────────────────────────────────────

mlflow_run_id = None

if _MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        _ts("MLflow — set_tracking_uri")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _ts("MLflow — set_experiment")
        mlflow.set_experiment(f"predictml/{MODEL_NAME}")

        run_name = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                "algorithm":        "RandomForest",
                "train_start_date": TRAIN_START_DATE,
                "train_end_date":   TRAIN_END_DATE,
                "n_samples_total":  n_samples,
                "test_size":        0.2,
                **HYPERPARAMS,
            })

            all_metrics: dict[str, float] = {
                "accuracy":     acc,
                "f1_score":     f1,
                "roc_auc":      roc_auc,
                "n_rows_train": float(len(X_train)),
                "n_rows_test":  float(len(X_test)),
            }
            for feat_name, stats in feature_stats.items():
                safe = feat_name.replace(" ", "_").replace("(", "").replace(")", "")[:40]
                for stat_key, val in stats.items():
                    all_metrics[f"feat_{safe}_{stat_key}"] = float(val)
            for label, ratio in label_distribution.items():
                all_metrics[f"label_{label}_ratio"] = float(ratio)
            for feat_name, importance in feature_importances.items():
                safe = feat_name.replace(" ", "_")[:50]
                all_metrics[f"fi_{safe}"] = float(importance)

            mlflow.log_metrics(all_metrics)
            mlflow.set_tags({
                "model_name": MODEL_NAME,
                "algorithm":  "RandomForest",
                "trigger":    "script",
                "n_features": str(len(feature_names)),
                "n_classes":  "2",
                "task_type":  "binary_classification",
            })

            _X_example = X_test.astype("float64")
            _signature = mlflow.models.infer_signature(
                _X_example,
                model.predict_proba(_X_example),
            )
            try:
                mlflow.sklearn.log_model(model, artifact_path="model", signature=_signature)
            except Exception as art_exc:
                print(f"[{MODEL_NAME}] Artifact skipped (MinIO unreachable): {art_exc}", file=sys.stderr)

            if _dataset_minio_path and os.path.exists(_csv_local):
                try:
                    mlflow.log_artifact(_csv_local, artifact_path="dataset")
                except Exception as _art_ds_exc:
                    print(f"[{MODEL_NAME}] Dataset artifact skipped: {_art_ds_exc}", file=sys.stderr)

            mlflow_run_id = run.info.run_id

        print(f"[{MODEL_NAME}] MLflow run created: {mlflow_run_id}", file=sys.stderr)

    except Exception as exc:
        print(f"[{MODEL_NAME}] MLflow unavailable — retraining continues: {exc}", file=sys.stderr)
        mlflow_run_id = None
else:
    reason = "mlflow not installed" if not _MLFLOW_AVAILABLE else "MLFLOW_TRACKING_URI not defined"
    print(f"[{MODEL_NAME}] MLflow skipped ({reason}).", file=sys.stderr)

# ── 8. JSON stdout — LAST LINE ────────────────────────────────────────────────

output = {
    "accuracy":           round(acc, 4),
    "f1_score":           round(f1, 4),
    "precision":          round(precision, 4),
    "recall":             round(recall, 4),
    "roc_auc":            round(roc_auc, 4),
    "n_rows":             len(X_train),
    "features_count":     len(feature_names),
    "classes":            list(cancer.target_names),
    "hyperparameters":    HYPERPARAMS,
    "training_dataset":   _dataset_minio_path or "scikit-learn Breast Cancer Wisconsin dataset — 569 observations, 2 classes",
    # confidence_threshold: 0.70 — higher threshold than iris because the decision (malignant/benign)
    # is critical. Below 70% confidence, the prediction is flagged low_confidence=True.
    "confidence_threshold": 0.70,
    "feature_importances": feature_importances,
    "feature_stats":      feature_stats,
    "label_distribution": label_distribution,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

import importlib.metadata as _imeta
_deps: dict = {}
for _pkg in ("scikit-learn", "numpy", "pandas", "joblib"):
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass
output["dependencies"] = _deps

_ts("end of script")
print(json.dumps(output))
