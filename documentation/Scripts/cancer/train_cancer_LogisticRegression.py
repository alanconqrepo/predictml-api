"""
train_cancer_LogisticRegression.py — Breast Cancer — LogisticRegression (v1.3.0 uploaded)
===========================================================================================

Contrat d'interface identique à train_cancer.py.
Algorithme : Pipeline(StandardScaler → LogisticRegression)
La normalisation est indispensable pour la régression logistique sur ce dataset
dont les features ont des échelles très différentes (ex: area ~1000 vs symmetry ~0.18).
"""

import json
import os
import pickle
import sys
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# ── 1. Variables d'environnement ─────────────────────────────────────────────

TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE", "2025-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE",   "2025-02-01")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "default_model_path.pkl")

MODEL_NAME          = os.environ.get("MODEL_NAME", "cancer-classifier")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")

_minio_endpoint   = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "") or os.environ.get("MINIO_ENDPOINT", "")
_minio_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "") or os.environ.get("MINIO_ROOT_USER", "")
_minio_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "") or os.environ.get("MINIO_ROOT_PASSWORD", "")
if _minio_endpoint:   os.environ["MLFLOW_S3_ENDPOINT_URL"] = _minio_endpoint
if _minio_access_key: os.environ["AWS_ACCESS_KEY_ID"]      = _minio_access_key
if _minio_secret_key: os.environ["AWS_SECRET_ACCESS_KEY"]  = _minio_secret_key

_in_docker = os.path.exists("/.dockerenv")
if not _in_docker:
    if "//mlflow:" in MLFLOW_TRACKING_URI:
        MLFLOW_TRACKING_URI = MLFLOW_TRACKING_URI.replace("//mlflow:", "//localhost:")
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    _s3_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
    if "//minio:" in _s3_url:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = _s3_url.replace(
            f"//minio:{os.environ.get('MINIO_INTERNAL_PORT', '9000')}",
            f"//localhost:{os.environ.get('MINIO_PORT', '9010')}"
        )

print(f"[{MODEL_NAME}] LogisticRegression (Pipeline) — {TRAIN_START_DATE} → {TRAIN_END_DATE}", file=sys.stderr)
_ts("démarrage")

# ── 2. Données ────────────────────────────────────────────────────────────────

cancer = load_breast_cancer()
X_full = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_full = cancer.target

start      = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end        = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
delta_days = max(1, (end - start).days)
n_samples  = min(len(X_full), max(40, delta_days * 3))
rng        = np.random.default_rng(seed=(delta_days + 3) % 1000)
indices    = rng.choice(len(X_full), size=n_samples, replace=False)
X, y       = X_full.iloc[indices], y_full[indices]

if n_samples < 20:
    print(json.dumps({"error": f"Pas assez de données ({n_samples} < 20)"}))
    sys.exit(1)

# ── 3. Entraînement ───────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if n_samples >= 40 else None
)

HYPERPARAMS = {
    "C":            1.0,
    "max_iter":     1000,
    "solver":       "lbfgs",
    "class_weight": "balanced",
    "random_state": 42,
}
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(**HYPERPARAMS)),
])
_ts("avant fit")
model.fit(X_train, y_train)
_ts("après fit")

# ── 4. Évaluation ─────────────────────────────────────────────────────────────

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc       = float(accuracy_score(y_test, y_pred))
f1        = float(f1_score(y_test,        y_pred, average="binary", zero_division=0))
precision = float(precision_score(y_test, y_pred, average="binary", zero_division=0))
recall    = float(recall_score(y_test,    y_pred, average="binary", zero_division=0))
roc_auc   = float(roc_auc_score(y_test,   y_proba))

print(f"[{MODEL_NAME}] Acc={acc:.4f} F1={f1:.4f} ROC-AUC={roc_auc:.4f}", file=sys.stderr)

# ── 5. Sauvegarde ─────────────────────────────────────────────────────────────

with open(OUTPUT_MODEL_PATH, "wb") as fh:
    pickle.dump(model, fh)
print(f"[{MODEL_NAME}] Pipeline (Scaler+LR) sauvegardé → {OUTPUT_MODEL_PATH}", file=sys.stderr)

_dataset_minio_path = None
try:
    _df_train = X_train.copy()
    _df_train["target"] = y_train
    _csv_filename = f"{MODEL_NAME}_lr_{TRAIN_START_DATE}_{TRAIN_END_DATE}_train.csv"
    _csv_local    = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    _minio_ep = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
    _aws_key  = os.environ.get("AWS_ACCESS_KEY_ID", "")
    _aws_sec  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    _bucket   = os.environ.get("MINIO_BUCKET", "models")
    if _BOTO3_AVAILABLE and _minio_ep and _aws_key:
        _s3 = boto3.client("s3", endpoint_url=_minio_ep,
                           aws_access_key_id=_aws_key, aws_secret_access_key=_aws_sec,
                           config=BotocoreConfig(signature_version="s3v4"))
        _object_key = f"{MODEL_NAME}/datasets/{_csv_filename}"
        _s3.upload_file(_csv_local, _bucket, _object_key)
        _dataset_minio_path = _object_key
except Exception as _e:
    print(f"[{MODEL_NAME}] Sauvegarde dataset ignorée : {_e}", file=sys.stderr)

# ── 6. Statistiques ───────────────────────────────────────────────────────────

feature_names = list(cancer.feature_names)
feature_stats = {
    name: {
        "mean": round(float(X_train[name].mean()), 4),
        "std":  round(float(X_train[name].std()),  4),
        "min":  round(float(X_train[name].min()),  4),
        "max":  round(float(X_train[name].max()),  4),
        "null_rate": 0.0,
    }
    for name in feature_names
}
total_train = len(y_train)
label_distribution = {
    cancer.target_names[int(cls)]: round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in np.unique(y_train)
}

# ── 7. MLflow ─────────────────────────────────────────────────────────────────

mlflow_run_id = None
if _MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"predictml/{MODEL_NAME}")
        with mlflow.start_run(run_name=f"{MODEL_NAME}_lr_{TRAIN_START_DATE}_{TRAIN_END_DATE}") as run:
            mlflow.log_params({"algorithm": "LogisticRegression", "scaler": "StandardScaler", **HYPERPARAMS})
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc})
            mlflow.set_tags({"model_name": MODEL_NAME, "algorithm": "LogisticRegression", "task_type": "binary_classification"})
            try:
                _sig = mlflow.models.infer_signature(X_test.astype("float64"), model.predict_proba(X_test.astype("float64")))
                mlflow.sklearn.log_model(model, artifact_path="model", signature=_sig)
            except Exception as _ae:
                print(f"[{MODEL_NAME}] Artifact ignoré : {_ae}", file=sys.stderr)
            mlflow_run_id = run.info.run_id
        print(f"[{MODEL_NAME}] Run MLflow : {mlflow_run_id}", file=sys.stderr)
    except Exception as exc:
        print(f"[{MODEL_NAME}] MLflow indisponible : {exc}", file=sys.stderr)

# ── 8. JSON stdout ────────────────────────────────────────────────────────────

output = {
    "accuracy":             round(acc, 4),
    "f1_score":             round(f1, 4),
    "precision":            round(precision, 4),
    "recall":               round(recall, 4),
    "roc_auc":              round(roc_auc, 4),
    "n_rows":               len(X_train),
    "features_count":       len(feature_names),
    "classes":              list(cancer.target_names),
    "hyperparameters":      {**HYPERPARAMS, "scaler": "StandardScaler"},
    "training_dataset":     _dataset_minio_path or "scikit-learn Breast Cancer Wisconsin dataset — 569 observations, 2 classes",
    "confidence_threshold": 0.70,
    "feature_stats":        feature_stats,
    "label_distribution":   label_distribution,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

print(json.dumps(output))
