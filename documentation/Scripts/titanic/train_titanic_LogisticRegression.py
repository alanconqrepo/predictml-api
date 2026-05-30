"""
train_titanic_LogisticRegression.py — PredictML retraining script — Titanic (LogisticRegression)
=================================================================================================

Model  : sklearn Pipeline — ColumnTransformer (StandardScaler + OneHotEncoder)
           + LogisticRegression (C=1.0, solver=lbfgs, max_iter=500)
Dataset: synthetic Titanic data — binary classification (survived: 0/1)
Differences vs v1.0.0 GradientBoosting:
  - LogisticRegression — linear model, maximum interpretability
  - L2 regularization (default C=1.0)
  - Same Pipeline/ColumnTransformer preprocessing

INTERFACE CONTRACT
-------------------------------------------------------------------------------------
  TRAIN_START_DATE   : start date  — format YYYY-MM-DD
  TRAIN_END_DATE     : end date    — format YYYY-MM-DD
  OUTPUT_MODEL_PATH  : absolute path where to save the produced .joblib
"""

import json
import os
import sys
from datetime import datetime

import joblib
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

_T0 = datetime.now()


def _ts(label: str) -> None:
    elapsed = (datetime.now() - _T0).total_seconds()
    print(f"[TIMING] {label} — +{elapsed:.2f}s", file=sys.stderr)


# ── 1. Environment variables ──────────────────────────────────────────────────

TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE", "2024-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE",   "2024-12-31")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "default_model_path.joblib")

MODEL_NAME               = os.environ.get("MODEL_NAME", "titanic-survival")
MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")

_minio_endpoint   = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "") or os.environ.get("MINIO_ENDPOINT", "")
_minio_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "") or os.environ.get("MINIO_ROOT_USER", "")
_minio_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "") or os.environ.get("MINIO_ROOT_PASSWORD", "")
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
        _minio_host_port = os.environ.get("MINIO_PORT", "9010")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = _s3_url.replace(
            f"//minio:{os.environ.get('MINIO_INTERNAL_PORT', '9000')}", f"//localhost:{_minio_host_port}"
        )

print(f"[{MODEL_NAME}] LogisticRegression — {TRAIN_START_DATE} → {TRAIN_END_DATE}", file=sys.stderr)
_ts("startup")

# ── 2. Synthetic data generation ─────────────────────────────────────────────

print(f"[{MODEL_NAME}] Generating synthetic Titanic dataset…", file=sys.stderr)
_ts("before data generation")

start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end   = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
delta_days = max(1, (end - start).days)

N_BASE = 891
n_samples = min(N_BASE * 3, max(100, delta_days * 5))

rng = np.random.default_rng(seed=(delta_days + 13) % 1000)

pclass_vals   = rng.choice(["1st", "2nd", "3rd"], size=n_samples, p=[0.24, 0.21, 0.55])
sex_vals      = rng.choice(["male", "female"], size=n_samples, p=[0.65, 0.35])
embarked_vals = rng.choice(["S", "C", "Q"], size=n_samples, p=[0.72, 0.19, 0.09])

age_mu   = np.where(pclass_vals == "1st", 39.0, np.where(pclass_vals == "2nd", 29.0, 25.0))
age_vals = rng.normal(age_mu, 14.0).clip(1.0, 80.0).round(1)

fare_mu   = np.where(pclass_vals == "1st", 87.0, np.where(pclass_vals == "2nd", 21.0, 13.0))
fare_vals = np.exp(rng.normal(np.log(fare_mu), 0.6)).clip(5.0, 512.0).round(2)

sibsp_vals = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.68, 0.16, 0.10, 0.04, 0.02])
parch_vals = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.76, 0.13, 0.07, 0.03, 0.01])

base_survival = np.where(sex_vals == "female", 0.74, 0.19)
class_bonus   = np.where(pclass_vals == "1st", 0.15, np.where(pclass_vals == "3rd", -0.10, 0.0))
age_bonus     = np.where(age_vals < 12, 0.15, 0.0)
survival_prob = np.clip(base_survival + class_bonus + age_bonus, 0.05, 0.95)
survived_vals = (rng.uniform(size=n_samples) < survival_prob).astype(int)

NUMERIC_FEATURES     = ["age", "fare", "parch", "sibsp"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

df = pd.DataFrame({
    "age": age_vals, "fare": fare_vals,
    "parch": parch_vals.astype(float), "sibsp": sibsp_vals.astype(float),
    "pclass": pclass_vals, "sex": sex_vals, "embarked": embarked_vals,
    "survived": survived_vals,
})

X = df[ALL_FEATURES]
y = df["survived"]

print(
    f"[{MODEL_NAME}] {n_samples} passengers generated "
    f"(survival rate: {survived_vals.mean():.1%})",
    file=sys.stderr,
)
_ts("after data generation")

if n_samples < 50:
    print(json.dumps({"error": f"Not enough data ({n_samples} < 50 required)"}))
    sys.exit(1)

# ── 3. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"[{MODEL_NAME}] Training on {len(X_train)} examples, "
    f"evaluating on {len(X_test)}…",
    file=sys.stderr,
)

# ── 4. Pipeline — ColumnTransformer + LogisticRegression ─────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

HYPERPARAMS = {
    "C":            1.0,
    "solver":       "lbfgs",
    "max_iter":     500,
    "class_weight": "balanced",
    "random_state": 42,
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(**HYPERPARAMS)),
])

_ts("before fit")
pipeline.fit(X_train, y_train)
_ts("after fit")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc       = float(accuracy_score(y_test, y_pred))
f1        = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
recall    = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
roc_auc   = float(roc_auc_score(y_test, y_proba))

print(
    f"[{MODEL_NAME}] Acc={acc:.4f} F1={f1:.4f} P={precision:.4f} R={recall:.4f} AUC={roc_auc:.4f}",
    file=sys.stderr,
)
_ts("evaluation")

# ── 6. Model saving ───────────────────────────────────────────────────────────

joblib.dump(pipeline, OUTPUT_MODEL_PATH)
print(f"[{MODEL_NAME}] Pipeline → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("model saved")

# ── 6b. Dataset CSV → MinIO ───────────────────────────────────────────────────

_dataset_minio_path = None
try:
    _df_train = X_train.copy()
    _df_train["survived"] = y_train.values
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_logreg_train.csv"
    _csv_local = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    _bucket = os.environ.get("MINIO_BUCKET", "models")
    if _BOTO3_AVAILABLE and os.environ.get("MLFLOW_S3_ENDPOINT_URL") and os.environ.get("AWS_ACCESS_KEY_ID"):
        _s3 = boto3.client(
            "s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            config=BotocoreConfig(signature_version="s3v4"),
        )
        _object_key = f"{MODEL_NAME}/datasets/{_csv_filename}"
        _s3.upload_file(_csv_local, _bucket, _object_key)
        _dataset_minio_path = _object_key
        print(f"[{MODEL_NAME}] Dataset → MinIO {_bucket}/{_object_key}", file=sys.stderr)
except Exception as _e:
    print(f"[{MODEL_NAME}] Dataset skipped: {_e}", file=sys.stderr)


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

# ── 7. Feature stats ──────────────────────────────────────────────────────────

feature_stats = {
    name: {
        "mean": round(float(X_train[name].mean()), 4),
        "std":  round(float(X_train[name].std()),  4),
        "min":  round(float(X_train[name].min()),  4),
        "max":  round(float(X_train[name].max()),  4),
        "null_rate": 0.0,
    }
    for name in NUMERIC_FEATURES
}
total_train = len(y_train)
label_distribution = {
    str(int(cls)): round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in np.unique(y_train)
}
cat_distributions = {}
for col in CATEGORICAL_FEATURES:
    vc = X_train[col].value_counts(normalize=True)
    cat_distributions[col] = {k: round(float(v), 4) for k, v in vc.items()}

# ── 8. MLflow (graceful degradation) ─────────────────────────────────────────

mlflow_run_id = None
if _MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"predictml/{MODEL_NAME}")
        with mlflow.start_run(run_name=f"{MODEL_NAME}_logreg_{TRAIN_START_DATE}") as run:
            mlflow.log_params({
                "algorithm": "LogisticRegression",
                "pipeline": "ColumnTransformer(StandardScaler+OneHotEncoder)+LR",
                "train_start_date": TRAIN_START_DATE, "train_end_date": TRAIN_END_DATE,
                **HYPERPARAMS,
            })
            mlflow.log_metrics({
                "accuracy": acc, "f1_score": f1, "roc_auc": roc_auc,
                "n_rows_train": float(len(X_train)), "n_rows_test": float(len(X_test)),
            })
            for _fi_name, _fi_val in feature_importances.items():
                try:
                    mlflow.log_metric(f"fi_{_fi_name.replace(' ','_')[:50]}", float(_fi_val))
                except Exception:
                    pass
            mlflow.set_tags({
                "model_name": MODEL_NAME, "algorithm": "LogisticRegression",
                "problem_type": "binary_classification",
            })
            try:
                _sig = mlflow.models.infer_signature(X_test, pipeline.predict_proba(X_test))
                mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=_sig)
            except Exception as _art:
                print(f"[{MODEL_NAME}] MLflow artifact skipped: {_art}", file=sys.stderr)
            mlflow_run_id = run.info.run_id
        print(f"[{MODEL_NAME}] MLflow run: {mlflow_run_id}", file=sys.stderr)
    except Exception as exc:
        print(f"[{MODEL_NAME}] MLflow unavailable: {exc}", file=sys.stderr)

# ── 9. JSON stdout ────────────────────────────────────────────────────────────

import importlib.metadata as _imeta
_deps: dict = {}
for _pkg in ["scikit-learn", "numpy", "pandas", "mlflow", "python-dotenv", "boto3", "botocore"]:
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass

output = {
    "accuracy":           round(acc, 4),
    "f1_score":           round(f1, 4),
    "precision":          round(precision, 4),
    "recall":             round(recall, 4),
    "roc_auc":            round(roc_auc, 4),
    "n_rows":             len(X_train),
    "features_count":     len(ALL_FEATURES),
    "classes":            ["0", "1"],
    "hyperparameters": {
        "pipeline": "ColumnTransformer(StandardScaler+OneHotEncoder)+LogisticRegression",
        **HYPERPARAMS,
    },
    "training_dataset": (
        _dataset_minio_path
        or f"Synthetic Titanic data — {n_samples} passengers, "
        "features: age, fare, parch, sibsp, pclass, sex, embarked"
    ),
    "confidence_threshold": 0.60,
    "feature_importances": feature_importances,
    "feature_stats":        feature_stats,
    "cat_distributions":    cat_distributions,
    "label_distribution":   label_distribution,
    "dependencies":         _deps,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

_ts("end of script")
print(json.dumps(output))
