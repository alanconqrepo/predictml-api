"""
train_wine_ExtraTrees.py — Script de ré-entraînement PredictML — Wine (ExtraTrees)
=====================================================================================

Modèle   : ExtraTreesRegressor (n_estimators=100, max_depth=6, bootstrap=False)
Dataset  : scikit-learn Wine — régression (cible : teneur en alcool)
Différences vs v1.0.0 GradientBoosting :
  - ExtraTrees : splits complètement aléatoires, pas de gradient boosting
  - Plus rapide à entraîner, légèrement moins précis sur régression
  - n_estimators=100, max_depth=6 (plus shallow)

CONTRAT D'INTERFACE
-------------------------------------------------------------------------------------
  TRAIN_START_DATE   : date de début  — format YYYY-MM-DD
  TRAIN_END_DATE     : date de fin    — format YYYY-MM-DD
  OUTPUT_MODEL_PATH  : chemin absolu où sauvegarder le .pkl produit
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
from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

_T0 = datetime.now()

def _ts(label: str) -> None:
    elapsed = (datetime.now() - _T0).total_seconds()
    print(f"[TIMING] {label} — +{elapsed:.2f}s", file=sys.stderr)

# ── 1. Variables d'environnement ──────────────────────────────────────────────

TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE", "2024-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE",   "2024-12-31")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "default_model_path.joblib")

MODEL_NAME               = os.environ.get("MODEL_NAME", "wine-regressor")
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

print(f"[{MODEL_NAME}] ExtraTreesRegressor — {TRAIN_START_DATE} → {TRAIN_END_DATE}", file=sys.stderr)
_ts("démarrage")

# ── 2. Données ────────────────────────────────────────────────────────────────

wine = load_wine()
all_feature_names = list(wine.feature_names)
TARGET_FEATURE  = "alcohol"
FEATURE_NAMES   = [f for f in all_feature_names if f != TARGET_FEATURE]

df_full = pd.DataFrame(wine.data, columns=all_feature_names)
X_full  = df_full[FEATURE_NAMES]
y_full  = df_full[TARGET_FEATURE].values

start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end   = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
delta_days = max(1, (end - start).days)
n_samples = min(len(X_full), max(30, delta_days * 2))
rng = np.random.default_rng(seed=(delta_days + 11) % 1000)
indices = rng.choice(len(X_full), size=n_samples, replace=False)
X, y = X_full.iloc[indices], y_full[indices]

print(f"[{MODEL_NAME}] {n_samples} exemples.", file=sys.stderr)
_ts("données chargées")

if n_samples < 20:
    print(json.dumps({"error": f"Pas assez de données ({n_samples} < 20)"}))
    sys.exit(1)

# ── 3. Entraînement ───────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

HYPERPARAMS = {
    "n_estimators":      100,
    "max_depth":         6,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "bootstrap":         False,  # ExtraTrees : pas de bootstrap
    "random_state":      42,
    "n_jobs":            -1,
}
model = ExtraTreesRegressor(**HYPERPARAMS)
_ts("avant fit")
model.fit(X_train, y_train)
_ts("après fit")

# ── 4. Évaluation ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
mae    = float(mean_absolute_error(y_test, y_pred))
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2     = float(r2_score(y_test, y_pred))

print(f"[{MODEL_NAME}] MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}", file=sys.stderr)
_ts("évaluation")

# ── 5. Sauvegarde modèle ──────────────────────────────────────────────────────

joblib.dump(model, OUTPUT_MODEL_PATH)
print(f"[{MODEL_NAME}] Modèle → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("modèle sauvegardé")

# ── 5b. Dataset CSV → MinIO ───────────────────────────────────────────────────

_dataset_minio_path = None
_csv_local = None
try:
    _df_train = X_train.copy()
    _df_train[TARGET_FEATURE] = y_train
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_et_train.csv"
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
    print(f"[{MODEL_NAME}] Dataset ignoré : {_e}", file=sys.stderr)

# ── 6. Feature stats ──────────────────────────────────────────────────────────

feature_stats = {
    name: {
        "mean": round(float(X_train[name].mean()), 4),
        "std":  round(float(X_train[name].std()),  4),
        "min":  round(float(X_train[name].min()),  4),
        "max":  round(float(X_train[name].max()),  4),
        "null_rate": 0.0,
    }
    for name in FEATURE_NAMES
}
target_stats = {
    "mean": round(float(y_train.mean()), 4), "std":  round(float(y_train.std()),  4),
    "min":  round(float(y_train.min()),  4), "max":  round(float(y_train.max()),  4),
}

# ── 7. MLflow (dégradation gracieuse) ────────────────────────────────────────

mlflow_run_id = None
if _MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"predictml/{MODEL_NAME}")
        with mlflow.start_run(run_name=f"{MODEL_NAME}_et_{TRAIN_START_DATE}") as run:
            mlflow.log_params({"algorithm": "ExtraTreesRegressor", **HYPERPARAMS,
                               "target": TARGET_FEATURE,
                               "train_start_date": TRAIN_START_DATE, "train_end_date": TRAIN_END_DATE})
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2,
                                "n_rows_train": float(len(X_train)), "n_rows_test": float(len(X_test))})
            mlflow.set_tags({"model_name": MODEL_NAME, "algorithm": "ExtraTreesRegressor",
                             "task": "regression", "target": TARGET_FEATURE})
            try:
                _sig = mlflow.models.infer_signature(X_test.astype("float64"), model.predict(X_test.astype("float64")))
                mlflow.sklearn.log_model(model, artifact_path="model", signature=_sig)
            except Exception as _art:
                print(f"[{MODEL_NAME}] MLflow artifact ignoré : {_art}", file=sys.stderr)
            mlflow_run_id = run.info.run_id
        print(f"[{MODEL_NAME}] MLflow run : {mlflow_run_id}", file=sys.stderr)
    except Exception as exc:
        print(f"[{MODEL_NAME}] MLflow indisponible : {exc}", file=sys.stderr)

# ── 8. JSON stdout ────────────────────────────────────────────────────────────

output = {
    "accuracy":         round(r2, 4),
    "mae":              round(mae, 4),
    "rmse":             round(rmse, 4),
    "r2":               round(r2, 4),
    "n_rows":           len(X_train),
    "features_count":   len(FEATURE_NAMES),
    "training_dataset": _dataset_minio_path or "scikit-learn Wine dataset — ExtraTrees v1.2.0",
    "feature_stats":    feature_stats,
    "target_stats":     target_stats,
    "hyperparameters":  HYPERPARAMS,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

_ts("fin script")
print(json.dumps(output))
