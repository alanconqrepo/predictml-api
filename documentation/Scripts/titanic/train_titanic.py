"""
train_titanic.py — PredictML retraining script — Titanic Survival (binary)
python .\train_titanic.py
=====================================================================================

This script is designed to be uploaded with your model (field "Training script")
to enable automatic or scheduled retraining from the dashboard.

Problem    : binary classification — predict passenger survival (0=dead, 1=survived)
Data       : synthetic dataset inspired by the Titanic with numerical AND categorical features
Model      : sklearn Pipeline — ColumnTransformer (StandardScaler + OneHotEncoder)
               + GradientBoostingClassifier
Metrics    : accuracy, f1_score, precision, recall, roc_auc

Input features (7 total):
  Numerical    : age (float), fare (float), parch (int), sibsp (int)
  Categorical  : pclass (str : "1st"/"2nd"/"3rd"),
                 sex (str : "male"/"female"),
                 embarked (str : "S"/"C"/"Q")

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
      {"accuracy": 0.82, "f1_score": 0.81}
  - Progress logs on stderr
  - Exit code 0 on success

ALLOWED MODULES by the PredictML sandbox
-------------------------------------------
  os, sys, json, joblib, datetime, dotenv, numpy, pandas, sklearn, mlflow
  (subprocess, requests, socket, urllib are blocked)

AUTOMATIC LIBRARY VERSION CAPTURE
-----------------------------------------------
  The API automatically generates a requirements.txt from the imports of this script.
  It is stored in MinIO  : {model_name}/v{version}_requirements.txt
  And logged as artifact  : MLflow > environment/requirements.txt
  No action required in the script — everything is handled server-side at upload and retrain.
"""

import json
import os
import sys
from datetime import datetime

import joblib
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

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


# ── 1. Environment variables (REQUIRED) ───────────────────────────────────────

if DEBUG:
    print(
        f"""
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
    [ENV] MINIO_ROOT_PASSWORD        = {os.environ.get("MINIO_ROOT_PASSWORD")}
    """,
        file=sys.stderr,
    )

TRAIN_START_DATE = os.environ.get("TRAIN_START_DATE", "2025-01-01")
TRAIN_END_DATE = os.environ.get("TRAIN_END_DATE", "2025-02-01")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "default_model_path.joblib")

MODEL_NAME = os.environ.get("MODEL_NAME", "titanic-survival")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

# Short timeout for all MLflow HTTP calls
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")

# MinIO credentials for log_model
_minio_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "") or os.environ.get(
    "MINIO_ENDPOINT", ""
)
_minio_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "") or os.environ.get("MINIO_ROOT_USER", "")
_minio_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "") or os.environ.get(
    "MINIO_ROOT_PASSWORD", ""
)
if _minio_endpoint:
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = _minio_endpoint
if _minio_access_key:
    os.environ["AWS_ACCESS_KEY_ID"] = _minio_access_key
if _minio_secret_key:
    os.environ["AWS_SECRET_ACCESS_KEY"] = _minio_secret_key

# Outside Docker: substitute internal hostnames with localhost
_in_docker = os.path.exists("/.dockerenv")
if not _in_docker:
    if "//mlflow:" in MLFLOW_TRACKING_URI:
        MLFLOW_TRACKING_URI = MLFLOW_TRACKING_URI.replace("//mlflow:", "//localhost:")
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    _s3_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
    if "//minio:" in _s3_url:
        _minio_internal_port = os.environ.get("MINIO_INTERNAL_PORT", "9000")
        _minio_host_port = os.environ.get("MINIO_PORT", "9010")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = _s3_url.replace(
            f"//minio:{_minio_internal_port}", f"//localhost:{_minio_host_port}"
        )

if DEBUG:
    print(
        f"""
    -------------   AFTER OVERRIDE  ------------
    [ENV] TRAIN_START_DATE           = {TRAIN_START_DATE}
    [ENV] TRAIN_END_DATE             = {TRAIN_END_DATE}
    [ENV] OUTPUT_MODEL_PATH          = {OUTPUT_MODEL_PATH}
    [ENV] MODEL_NAME                 = {MODEL_NAME}
    [ENV] MLFLOW_TRACKING_URI        = {MLFLOW_TRACKING_URI}
    [ENV] MLFLOW_TRACKING_USERNAME   = {MLFLOW_TRACKING_USERNAME}
    [ENV] MLFLOW_S3_ENDPOINT_URL     = {os.environ.get("MLFLOW_S3_ENDPOINT_URL")}
    [ENV] AWS_ACCESS_KEY_ID          = {os.environ.get("AWS_ACCESS_KEY_ID")}
    [ENV] AWS_SECRET_ACCESS_KEY      = {os.environ.get("AWS_SECRET_ACCESS_KEY")}
    """,
        file=sys.stderr,
    )

print(
    f"[{MODEL_NAME}] Retraining from {TRAIN_START_DATE} to {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[{MODEL_NAME}] Output: {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("startup")

# ── 2. Synthetic data generation ─────────────────────────────────────────────
#
# Synthetic data inspired by the Titanic dataset (891 real passengers).
# Distributions calibrated on historical statistics:
#   - 1st class : 24 %  — 2nd class : 21 %  — 3rd class : 55 %
#   - Sex       : male 65 %  —  female 35 %
#   - Embarked  : S=72 %  C=19 %  Q=9 %
#   - Survival  : ~38 % (influenced by class + sex + age)
#
# To retrain on your own production data, replace this block with:
#   df = pd.read_csv("data/training_data.csv", parse_dates=["date"])
#   df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]
#
# ─────────────────────────────────────────────────────────────────────────────

print(f"[{MODEL_NAME}] Generating synthetic Titanic dataset…", file=sys.stderr)
_ts("before data generation")

# Size proportional to the time range (like train_iris.py)
start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
delta_days = max(1, (end - start).days)

N_BASE = 891  # reference: real Titanic dataset size
n_samples = min(N_BASE * 3, max(100, delta_days * 5))

rng = np.random.default_rng(seed=delta_days % 1000)

# Ticket class sampling (historical distribution)
pclass_vals = rng.choice(["1st", "2nd", "3rd"], size=n_samples, p=[0.24, 0.21, 0.55])

# Sex sampling (historical distribution)
sex_vals = rng.choice(["male", "female"], size=n_samples, p=[0.65, 0.35])

# Port of embarkation (historical distribution)
embarked_vals = rng.choice(["S", "C", "Q"], size=n_samples, p=[0.72, 0.19, 0.09])

# Age: slightly dependent on class (1st class passengers older)
age_mu = np.where(pclass_vals == "1st", 39.0, np.where(pclass_vals == "2nd", 29.0, 25.0))
age_vals = rng.normal(age_mu, 14.0).clip(1.0, 80.0).round(1)

# Fare: strongly correlated with class
fare_mu = np.where(pclass_vals == "1st", 87.0, np.where(pclass_vals == "2nd", 21.0, 13.0))
fare_sigma = np.where(pclass_vals == "1st", 60.0, np.where(pclass_vals == "2nd", 13.0, 9.0))
fare_vals = np.exp(rng.normal(np.log(fare_mu), 0.6)).clip(5.0, 512.0).round(2)

# Siblings/spouses: empirical distribution (0 strongly dominant)
sibsp_probs = [0.68, 0.16, 0.10, 0.04, 0.02]
sibsp_vals = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=sibsp_probs)

# Parents/children
parch_probs = [0.76, 0.13, 0.07, 0.03, 0.01]
parch_vals = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=parch_probs)

# Survival: probabilistic model inspired by real data
#   - Women: survival rate ~74 %
#   - Men: survival rate ~19 %
#   - 1st class bonus: +15 %
#   - 3rd class penalty: -10 %
#   - Children (< 12 years): +15 %
base_survival = np.where(sex_vals == "female", 0.74, 0.19)
class_bonus = np.where(pclass_vals == "1st", 0.15, np.where(pclass_vals == "3rd", -0.10, 0.0))
age_bonus = np.where(age_vals < 12, 0.15, 0.0)
survival_prob = np.clip(base_survival + class_bonus + age_bonus, 0.05, 0.95)
survived_vals = (rng.uniform(size=n_samples) < survival_prob).astype(int)

# DataFrame assembly (order consistent with feature constants)
NUMERIC_FEATURES = ["age", "fare", "parch", "sibsp"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

df = pd.DataFrame(
    {
        "age": age_vals,
        "fare": fare_vals,
        "parch": parch_vals.astype(float),
        "sibsp": sibsp_vals.astype(float),
        "pclass": pclass_vals,
        "sex": sex_vals,
        "embarked": embarked_vals,
        "survived": survived_vals,
    }
)

X = df[ALL_FEATURES]
y = df["survived"]

print(
    f"[{MODEL_NAME}] {n_samples} passengers generated "
    f"(survival rate: {survived_vals.mean():.1%})",
    file=sys.stderr,
)
_ts("after data generation")

if n_samples < 50:
    print(json.dumps({"error": f"Not enough data ({n_samples} examples < 50 required)"}))
    sys.exit(1)

# ── 3. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"[{MODEL_NAME}] Training on {len(X_train)} examples, " f"evaluating on {len(X_test)}…",
    file=sys.stderr,
)

# ── 4. sklearn Pipeline — ColumnTransformer + GradientBoosting ────────────────
#
# The ColumnTransformer applies:
#   • StandardScaler        → numerical features  (age, fare, parch, sibsp)
#   • OneHotEncoder         → categorical features (pclass, sex, embarked)
#       handle_unknown="ignore"  → unknown categories in prod → zero vector
#       sparse_output=False      → dense array required by GBC
#
# The Pipeline is trained on a pandas DataFrame → pipeline.feature_names_in_
# contains the 7 original feature names (before encoding).
# The API uses this field for schema validation (/predict?strict_validation).
#
# ─────────────────────────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)

HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": 42,
}

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(**HYPERPARAMS)),
    ]
)

_ts("before fit")
pipeline.fit(X_train, y_train)
_ts("after fit")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc = float(accuracy_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
roc_auc = float(roc_auc_score(y_test, y_proba))

print(
    f"[{MODEL_NAME}] Accuracy: {acc:.4f} | F1: {f1:.4f}"
    f" | Precision: {precision:.4f} | Recall: {recall:.4f}"
    f" | AUC-ROC: {roc_auc:.4f}",
    file=sys.stderr,
)
_ts("after evaluation")

# ── 6. Model saving (REQUIRED) ─────────────────────────────────────────────────

joblib.dump(pipeline, OUTPUT_MODEL_PATH)
print(f"[{MODEL_NAME}] Pipeline saved → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("after model save")

# ── 6b. Training dataset save → MinIO + MLflow artifact ────────────────────────

_dataset_minio_path = None

try:
    _df_train = X_train.copy()
    _df_train["survived"] = y_train.values
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_train.csv"
    _csv_local = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    print(
        f"[{MODEL_NAME}] CSV dataset created ({len(_df_train)} rows) → {_csv_local}",
        file=sys.stderr,
    )

    _minio_endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
    _aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    _aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    _bucket = os.environ.get("MINIO_BUCKET", "models")

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
        print(
            f"[{MODEL_NAME}] Dataset uploaded to MinIO → {_bucket}/{_object_key}",
            file=sys.stderr,
        )
    else:
        print(
            f"[{MODEL_NAME}] Dataset not uploaded to MinIO "
            f"(boto3={'ok' if _BOTO3_AVAILABLE else 'missing'}, endpoint='{_minio_endpoint_url}')",
            file=sys.stderr,
        )

except Exception as _ds_exc:
    print(f"[{MODEL_NAME}] Dataset save skipped: {_ds_exc}", file=sys.stderr)

_ts("after dataset save")

# ── 7. Statistics for MLflow and drift detection ───────────────────────────────
#
# feature_stats contains ONLY numerical features (FeatureStats expects
# mean/std/min/max). Categorical features (pclass, sex, embarked) are
# listed in pipeline.feature_names_in_ and validated by name in the API, but
# they cannot have a numerical statistical baseline.
#
# ─────────────────────────────────────────────────────────────────────────────

feature_stats = {
    name: {
        "mean": round(float(X_train[name].mean()), 4),
        "std": round(float(X_train[name].std()), 4),
        "min": round(float(X_train[name].min()), 4),
        "max": round(float(X_train[name].max()), 4),
        "null_rate": 0.0,
    }
    for name in NUMERIC_FEATURES
}

total_train = len(y_train)
label_distribution = {
    str(int(cls)): round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in np.unique(y_train)
}

# Categorical feature distributions (informational — for MLflow)
cat_distributions = {}
for col in CATEGORICAL_FEATURES:
    vc = X_train[col].value_counts(normalize=True)
    cat_distributions[col] = {k: round(float(v), 4) for k, v in vc.items()}

# ── 8. MLflow logging (graceful degradation if MLflow unavailable) ────────────

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

            _ts("MLflow — log_params (network call #3)")
            mlflow.log_params(
                {
                    "algorithm": "GradientBoosting",
                    "pipeline": "ColumnTransformer(StandardScaler+OneHotEncoder)+GBC",
                    "numeric_features": ",".join(NUMERIC_FEATURES),
                    "categorical_features": ",".join(CATEGORICAL_FEATURES),
                    "train_start_date": TRAIN_START_DATE,
                    "train_end_date": TRAIN_END_DATE,
                    "n_samples_total": n_samples,
                    "test_size": 0.2,
                    **HYPERPARAMS,
                }
            )
            _ts("MLflow — log_params OK")

            all_metrics: dict = {
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "n_rows_train": float(len(X_train)),
                "n_rows_test": float(len(X_test)),
                "survival_rate_train": float(y_train.mean()),
            }
            for feat_name, stats in feature_stats.items():
                safe = feat_name.replace(" ", "_")[:40]
                for stat_key, val in stats.items():
                    all_metrics[f"feat_{safe}_{stat_key}"] = float(val)
            for label, ratio in label_distribution.items():
                all_metrics[f"label_{label}_ratio"] = float(ratio)

            _ts(f"MLflow — log_metrics x{len(all_metrics)} (network call #4)")
            mlflow.log_metrics(all_metrics)
            _ts("MLflow — log_metrics OK")

            _ts("MLflow — set_tags (network call #5)")
            mlflow.set_tags(
                {
                    "model_name": MODEL_NAME,
                    "algorithm": "GradientBoosting",
                    "trigger": "script",
                    "n_features_total": str(len(ALL_FEATURES)),
                    "n_features_numeric": str(len(NUMERIC_FEATURES)),
                    "n_features_categorical": str(len(CATEGORICAL_FEATURES)),
                    "n_classes": "2",
                    "problem_type": "binary_classification",
                    "preprocessing": "OneHotEncoding+StandardScaler",
                }
            )
            _ts("MLflow — set_tags OK")

            # Signature: inputs = 7 original features (4 float + 3 string)
            _ts("MLflow — log_model start (network call #6 — MinIO)")
            try:
                _signature = mlflow.models.infer_signature(
                    X_test,
                    pipeline.predict_proba(X_test),
                )
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    signature=_signature,
                )
                _ts("MLflow — log_model OK")
                print(
                    f"[{MODEL_NAME}] Model artifact logged in MLflow.",
                    file=sys.stderr,
                )
            except Exception as art_exc:
                _ts("MLflow — log_model FAILED")
                print(
                    f"[{MODEL_NAME}] Artifact skipped (MinIO unreachable): {art_exc}",
                    file=sys.stderr,
                )

            if _dataset_minio_path and os.path.exists(_csv_local):
                try:
                    _ts("MLflow — log_artifact dataset (network call #7 — MinIO)")
                    mlflow.log_artifact(_csv_local, artifact_path="dataset")
                    _ts("MLflow — log_artifact dataset OK")
                except Exception as _art_ds_exc:
                    print(
                        f"[{MODEL_NAME}] Dataset artifact skipped: {_art_ds_exc}",
                        file=sys.stderr,
                    )

            mlflow_run_id = run.info.run_id

        _ts("MLflow — run closed")
        print(f"[{MODEL_NAME}] MLflow run created: {mlflow_run_id}", file=sys.stderr)

    except Exception as exc:
        _ts("MLflow — EXCEPTION (entire block abandoned)")
        print(
            f"[{MODEL_NAME}] MLflow unavailable — retraining continues: {exc}",
            file=sys.stderr,
        )
        mlflow_run_id = None

else:
    reason = "mlflow not installed" if not _MLFLOW_AVAILABLE else "MLFLOW_TRACKING_URI not set"
    print(f"[{MODEL_NAME}] MLflow skipped ({reason}).", file=sys.stderr)

# ── 9. Library version capture ─────────────────────────────────────────────────

import importlib.metadata as _imeta  # noqa: E402

_deps: dict = {}
for _pkg in [
    "scikit-learn",
    "numpy",
    "pandas",
    "mlflow",
    "python-dotenv",
    "boto3",
    "botocore",
]:
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass

# ── 10. JSON stdout — LAST LINE (read by the API — do not add anything after) ──

output = {
    "accuracy": round(acc, 4),
    "f1_score": round(f1, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "roc_auc": round(roc_auc, 4),
    "n_rows": len(X_train),
    "features_count": len(ALL_FEATURES),
    "classes": ["0", "1"],
    "hyperparameters": {
        "pipeline": "ColumnTransformer(StandardScaler+OneHotEncoder)+GradientBoosting",
        **HYPERPARAMS,
    },
    "training_dataset": (
        _dataset_minio_path
        or f"Synthetic Titanic data — {n_samples} passengers, "
        "features: age, fare, parch, sibsp, pclass, sex, embarked"
    ),
    # confidence_threshold: probability threshold below which the prediction
    # is marked low_confidence=True in the API. 0.65 is appropriate for this
    # binary model: cases between 0.35 and 0.65 survival probability are ambiguous.
    "confidence_threshold": 0.65,
    "feature_stats": feature_stats,  # numerical features only
    "cat_distributions": cat_distributions,  # informational (non-standard API)
    "label_distribution": label_distribution,
    "dependencies": _deps,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

_ts("end of script")
print(json.dumps(output))
