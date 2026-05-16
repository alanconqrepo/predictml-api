"""
train_wine.py — Script de ré-entraînement PredictML — Exemple Wine (Régression)
python .\train_wine.py
=================================================================================

Ce script est conçu pour être uploadé avec votre modèle (champ "Script d'entraînement")
afin de permettre le ré-entraînement automatique ou planifié depuis le dashboard.

Problème : régression — prédire la teneur en alcool d'un vin (variable continue)
à partir des 12 autres mesures chimiques du dataset Wine de scikit-learn.
Modèle   : GradientBoostingRegressor
Métriques: MAE, RMSE, R²

CONTRAT D'INTERFACE (variables d'environnement injectées automatiquement par l'API)
-------------------------------------------------------------------------------------
  TRAIN_START_DATE   : date de début  — format YYYY-MM-DD
  TRAIN_END_DATE     : date de fin    — format YYYY-MM-DD
  OUTPUT_MODEL_PATH  : chemin absolu où sauvegarder le .pkl produit

Variables optionnelles :
  MLFLOW_TRACKING_URI      : URI du serveur MLflow (ex: http://localhost:5000)
  MLFLOW_TRACKING_USERNAME : identifiant MLflow (si auth activée)
  MLFLOW_TRACKING_PASSWORD : mot de passe MLflow (si auth activée)
  MODEL_NAME               : nom du modèle source

SORTIE ATTENDUE
---------------
  - Modèle sauvegardé à OUTPUT_MODEL_PATH via joblib.dump()
  - Dernière ligne JSON sur stdout avec au minimum :
      {"accuracy": 0.91, "mae": 0.12, "rmse": 0.18, "r2": 0.91}
    (accuracy = R² pour permettre l'affichage dans le dashboard)
  - Logs de progression sur stderr
  - Code de sortie 0 si succès

MODULES AUTORISÉS par le sandbox PredictML
-------------------------------------------
  os, sys, json, joblib, datetime, dotenv, numpy, pandas, sklearn, mlflow, boto3
  (subprocess, requests, socket, urllib sont bloqués)

CAPTURE AUTOMATIQUE DES VERSIONS DE LIBRAIRIES
-----------------------------------------------
  L'API génère automatiquement un requirements.txt à partir des imports de ce script.
  Il est stocké dans MinIO  : {model_name}/v{version}_requirements.txt
  Et loggué comme artefact  : MLflow > environment/requirements.txt
  Aucune action requise dans le script — tout se fait côté serveur à l'upload et au retrain.
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
from sklearn.ensemble import GradientBoostingRegressor
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

DEBUG = True

_T0 = datetime.now()

def _ts(label: str) -> None:
    elapsed = (datetime.now() - _T0).total_seconds()
    print(f"[TIMING] {label} — +{elapsed:.2f}s", file=sys.stderr)

# ── 1. Variables d'environnement (OBLIGATOIRES) ───────────────────────────────

if DEBUG:
    print(f"""
    -------------   AVANT SURCHARGE  ------------
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

TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE", "2025-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE",   "2025-02-01")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "default_model_path.joblib")

MODEL_NAME               = os.environ.get("MODEL_NAME", "wine-regressor")
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

if DEBUG:
    print(f"""
    -------------   APRES SURCHARGE  ------------
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
    f"[{MODEL_NAME}] Ré-entraînement du {TRAIN_START_DATE} au {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[{MODEL_NAME}] Sortie : {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("démarrage")

# ── 2. Chargement des données ─────────────────────────────────────────────────
#
# Régression : prédire la teneur en alcool (target = feature "alcohol")
# à partir des 12 autres mesures chimiques du dataset Wine.
#
# REMPLACEZ CE BLOC par votre propre source de données :
#
#   df = pd.read_csv("data/wine_data.csv", parse_dates=["date"])
#   df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]
#   X = df[FEATURE_NAMES]
#   y = df["alcohol"]
#
# ─────────────────────────────────────────────────────────────────────────────

print(f"[{MODEL_NAME}] Chargement du dataset Wine (données synthétiques)…", file=sys.stderr)
_ts("avant chargement données")

wine = load_wine()
all_feature_names = list(wine.feature_names)

# Target = alcohol (index 0) — variable continue [11.0, 14.8]
# Features = les 12 autres mesures chimiques
TARGET_FEATURE   = "alcohol"
FEATURE_NAMES    = [f for f in all_feature_names if f != TARGET_FEATURE]
TARGET_IDX       = all_feature_names.index(TARGET_FEATURE)
FEATURE_INDICES  = [i for i, f in enumerate(all_feature_names) if f != TARGET_FEATURE]

df_full = pd.DataFrame(wine.data, columns=all_feature_names)
X_full  = df_full[FEATURE_NAMES]
y_full  = df_full[TARGET_FEATURE].values

# Simulation d'un filtre temporel : taille de l'échantillon proportionnelle à la plage
start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end   = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
delta_days = max(1, (end - start).days)

n_samples = min(len(X_full), max(30, delta_days * 2))
rng     = np.random.default_rng(seed=delta_days % 1000)
indices = rng.choice(len(X_full), size=n_samples, replace=False)
X, y    = X_full.iloc[indices], y_full[indices]

print(f"[{MODEL_NAME}] {n_samples} exemples retenus sur {len(X_full)} disponibles.", file=sys.stderr)
_ts("après chargement données")

if n_samples < 20:
    print(json.dumps({"error": f"Pas assez de données ({n_samples} exemples < 20 requis)"}))
    sys.exit(1)

# ── 3. Entraînement ───────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(
    f"[{MODEL_NAME}] Entraînement GradientBoostingRegressor sur {len(X_train)} exemples, "
    f"évaluation sur {len(X_test)}…",
    file=sys.stderr,
)

HYPERPARAMS = {
    "n_estimators":   200,
    "learning_rate":  0.05,
    "max_depth":      4,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "subsample":      0.8,
    "max_features":   "sqrt",
    "random_state":   42,
}
model = GradientBoostingRegressor(**HYPERPARAMS)
_ts("avant fit")
model.fit(X_train, y_train)
_ts("après fit")

# ── 4. Évaluation ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
mae    = float(mean_absolute_error(y_test, y_pred))
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2     = float(r2_score(y_test, y_pred))

print(
    f"[{MODEL_NAME}] MAE : {mae:.4f} | RMSE : {rmse:.4f} | R² : {r2:.4f}",
    file=sys.stderr,
)
_ts("après évaluation")

# ── 5. Sauvegarde du modèle (OBLIGATOIRE) ─────────────────────────────────────

joblib.dump(model, OUTPUT_MODEL_PATH)

print(f"[{MODEL_NAME}] Modèle sauvegardé → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("après sauvegarde modèle")

# ── 5b. Sauvegarde du dataset d'entraînement → MinIO + artifact MLflow ────────

_dataset_minio_path = None
_csv_local = None

try:
    _df_train = X_train.copy()
    _df_train[TARGET_FEATURE] = y_train
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_train.csv"
    _csv_local = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    print(f"[{MODEL_NAME}] Dataset CSV créé ({len(_df_train)} lignes) → {_csv_local}", file=sys.stderr)

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
        print(f"[{MODEL_NAME}] Dataset uploadé dans MinIO → {_bucket}/{_object_key}", file=sys.stderr)
    else:
        print(f"[{MODEL_NAME}] Dataset non uploadé dans MinIO (boto3={'ok' if _BOTO3_AVAILABLE else 'absent'}, endpoint='{_minio_endpoint_url}')", file=sys.stderr)

except Exception as _ds_exc:
    print(f"[{MODEL_NAME}] Sauvegarde dataset ignorée : {_ds_exc}", file=sys.stderr)

_ts("après sauvegarde dataset")

# ── 6. Statistiques pour MLflow et détection de drift ─────────────────────────

feature_stats = {
    name: {
        "mean":      round(float(X_train[name].mean()), 4),
        "std":       round(float(X_train[name].std()),  4),
        "min":       round(float(X_train[name].min()),  4),
        "max":       round(float(X_train[name].max()),  4),
        "null_rate": 0.0,
    }
    for name in FEATURE_NAMES
}

target_stats = {
    "mean": round(float(y_train.mean()), 4),
    "std":  round(float(y_train.std()),  4),
    "min":  round(float(y_train.min()),  4),
    "max":  round(float(y_train.max()),  4),
}

# ── 7. Logging MLflow (dégradation gracieuse si MLflow indisponible) ──────────

mlflow_run_id = None

if _MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
    try:
        _ts("MLflow — set_tracking_uri")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        _ts("MLflow — set_experiment (appel réseau #1)")
        mlflow.set_experiment(f"predictml/{MODEL_NAME}")
        _ts("MLflow — set_experiment OK")

        run_name = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}"
        _ts("MLflow — start_run (appel réseau #2)")
        with mlflow.start_run(run_name=run_name) as run:
            _ts("MLflow — start_run OK")

            _ts("MLflow — log_params (appel réseau #3)")
            mlflow.log_params({
                "algorithm":        "GradientBoostingRegressor",
                **HYPERPARAMS,
                "target":           TARGET_FEATURE,
                "train_start_date": TRAIN_START_DATE,
                "train_end_date":   TRAIN_END_DATE,
                "n_samples_total":  n_samples,
                "test_size":        0.2,
            })
            _ts("MLflow — log_params OK")

            all_metrics: dict[str, float] = {
                "mae":          mae,
                "rmse":         rmse,
                "r2":           r2,
                "n_rows_train": float(len(X_train)),
                "n_rows_test":  float(len(X_test)),
                "target_mean":  target_stats["mean"],
                "target_std":   target_stats["std"],
            }
            for feat_name, stats in feature_stats.items():
                safe = feat_name.replace(" ", "_").replace("/", "_")[:40]
                for stat_key, val in stats.items():
                    all_metrics[f"feat_{safe}_{stat_key}"] = float(val)

            _ts(f"MLflow — log_metrics x{len(all_metrics)} (appel réseau #4)")
            mlflow.log_metrics(all_metrics)
            _ts("MLflow — log_metrics OK")

            _ts("MLflow — set_tags (appel réseau #5)")
            mlflow.set_tags({
                "model_name":   MODEL_NAME,
                "algorithm":    "GradientBoostingRegressor",
                "task":         "regression",
                "target":       TARGET_FEATURE,
                "trigger":      "script",
                "n_features":   str(len(FEATURE_NAMES)),
            })
            _ts("MLflow — set_tags OK")

            _X_example = X_test.astype("float64")
            _signature = mlflow.models.infer_signature(
                _X_example,
                model.predict(_X_example),
            )
            _ts("MLflow — log_model début (appel réseau #6 — MinIO)")
            try:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=_signature,
                )
                _ts("MLflow — log_model OK")
                print(f"[{MODEL_NAME}] Artifact modèle loggué dans MLflow.", file=sys.stderr)
            except Exception as art_exc:
                _ts("MLflow — log_model ÉCHEC")
                print(f"[{MODEL_NAME}] Artifact ignoré (MinIO inaccessible) : {art_exc}", file=sys.stderr)

            if _dataset_minio_path and _csv_local and os.path.exists(_csv_local):
                try:
                    _ts("MLflow — log_artifact dataset (appel réseau #7 — MinIO)")
                    mlflow.log_artifact(_csv_local, artifact_path="dataset")
                    _ts("MLflow — log_artifact dataset OK")
                    print(f"[{MODEL_NAME}] Dataset loggué comme artifact MLflow.", file=sys.stderr)
                except Exception as _art_ds_exc:
                    print(f"[{MODEL_NAME}] Artifact dataset ignoré : {_art_ds_exc}", file=sys.stderr)

            mlflow_run_id = run.info.run_id

        _ts("MLflow — run fermé")
        print(f"[{MODEL_NAME}] Run MLflow créé : {mlflow_run_id}", file=sys.stderr)

    except Exception as exc:
        _ts("MLflow — EXCEPTION (bloc entier abandonné)")
        print(f"[{MODEL_NAME}] MLflow indisponible — ré-entraînement continue : {exc}", file=sys.stderr)
        mlflow_run_id = None

else:
    reason = "mlflow non installé" if not _MLFLOW_AVAILABLE else "MLFLOW_TRACKING_URI non défini"
    print(f"[{MODEL_NAME}] MLflow ignoré ({reason}).", file=sys.stderr)

# ── 8. Capture des versions de librairies (lues par l'API pour générer requirements.txt) ──

import importlib.metadata as _imeta  # noqa: E402

_deps: dict = {}
for _pkg in ["scikit-learn", "numpy", "pandas", "mlflow", "python-dotenv", "boto3", "botocore"]:
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass

# ── 9. JSON stdout — DERNIÈRE LIGNE (lue par l'API — ne rien ajouter après) ───
#
# Pour un modèle de régression :
#   "accuracy" = R² (seul champ numérique affiché par défaut dans le dashboard)
#   "mae", "rmse", "r2" sont stockés dans training_metrics
#   Pas de "classes" ni de "confidence_threshold" (non applicables à la régression)

output = {
    "accuracy":          round(r2, 4),   # R² comme proxy de l'accuracy pour l'affichage
    "mae":               round(mae, 4),
    "rmse":              round(rmse, 4),
    "r2":                round(r2, 4),
    "n_rows":            len(X_train),
    "features_count":    len(FEATURE_NAMES),
    "training_dataset":  _dataset_minio_path or "scikit-learn Wine dataset (UCI) - 178 observations, cible : teneur en alcool",
    "feature_stats":     feature_stats,
    "target_stats":      target_stats,
    "hyperparameters":   HYPERPARAMS,
    "dependencies":      _deps,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

_ts("fin script")
print(json.dumps(output))
