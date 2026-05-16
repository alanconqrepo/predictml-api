"""
train_iris.py — Script de ré-entraînement PredictML — Exemple Iris
python .\train_iris.py
====================================================================

Ce script est conçu pour être uploadé avec votre modèle (champ "Script d'entraînement")
afin de permettre le ré-entraînement automatique ou planifié depuis le dashboard.

CONTRAT D'INTERFACE (variables d'environnement injectées automatiquement par l'API)
-------------------------------------------------------------------------------------
  TRAIN_START_DATE   : date de début  — format YYYY-MM-DD
  TRAIN_END_DATE     : date de fin    — format YYYY-MM-DD
  OUTPUT_MODEL_PATH  : chemin absolu où sauvegarder le .joblib produit

Variables optionnelles :
  MLFLOW_TRACKING_URI      : URI du serveur MLflow (ex: http://localhost:5000)
  MLFLOW_TRACKING_USERNAME : identifiant MLflow (si auth activée)
  MLFLOW_TRACKING_PASSWORD : mot de passe MLflow (si auth activée)
  MODEL_NAME               : nom du modèle source

SORTIE ATTENDUE
---------------
  - Modèle sauvegardé à OUTPUT_MODEL_PATH via joblib.dump()
  - Dernière ligne JSON sur stdout avec au minimum :
      {"accuracy": 0.95, "f1_score": 0.94}
  - Logs de progression sur stderr
  - Code de sortie 0 si succès

MODULES AUTORISÉS par le sandbox PredictML
-------------------------------------------
  os, sys, json, joblib, datetime, dotenv, numpy, pandas, sklearn, mlflow
  (subprocess, requests, socket, urllib sont bloqués)

CAPTURE AUTOMATIQUE DES VERSIONS DE LIBRAIRIES
-----------------------------------------------
  L'API génère automatiquement un requirements.txt à partir des imports de ce script.
  Il est stocké dans MinIO  : {model_name}/v{version}_requirements.txt
  Et loggué comme artefact  : MLflow > environment/requirements.txt
  Aucune action requise dans le script — tout se fait côté serveur à l'upload et au retrain.
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
    # pip version non résolue dans le sandbox → pas actionnable
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    # gRPC OTLP absent → tracing désactivé silencieusement
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

# Forcer UTF-8 sur Windows pour éviter le crash charmap sur les emojis MLflow (ex: 🏃)
#if hasattr(sys.stdout, "buffer"):
#    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
#if hasattr(sys.stderr, "buffer"):
#    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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

TRAIN_START_DATE  = os.environ.get("TRAIN_START_DATE","2025-01-01")
TRAIN_END_DATE    = os.environ.get("TRAIN_END_DATE","2025-02-01")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH","default_model_path.joblib")

MODEL_NAME               = os.environ.get("MODEL_NAME", "iris-classifier")
MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")



# Timeout court pour tous les appels HTTP MLflow (défaut système = 120 s → timeout garanti)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")

# Credentials MinIO pour log_model — boto3 lit os.environ directement.
# Sans eux, boto3 explore tous les credential providers (~15s) avant d'abandonner.
# Priorité : AWS_* (standard boto3) > MINIO_ROOT_* (docker-compose)
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

# Hors Docker : substituer les hostnames internes par localhost
# /.dockerenv est créé automatiquement par Docker dans tout container
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
# REMPLACEZ CE BLOC par votre propre source de données :
#
#   import pandas as pd
#   df = pd.read_csv("data/training_data.csv", parse_dates=["date"])
#   df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]
#   if df.empty:
#       print(json.dumps({"error": "Aucune donnée pour cette plage"}))
#       sys.exit(1)
#   X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
#   y = df["species"]
#
# ─────────────────────────────────────────────────────────────────────────────

print(f"[{MODEL_NAME}] Chargement du dataset Iris (données synthétiques)…", file=sys.stderr)
_ts("avant chargement données")

iris = load_iris()
X_full = pd.DataFrame(iris.data, columns=iris.feature_names)
y_full = iris.target

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
    X, y, test_size=0.2, random_state=42, stratify=y if n_samples >= 30 else None
)

print(
    f"[{MODEL_NAME}] Entraînement sur {len(X_train)} exemples, "
    f"évaluation sur {len(X_test)}…",
    file=sys.stderr,
)

HYPERPARAMS = {
    "n_estimators":      200,    # nombre d'arbres
    "max_depth":         10,     # profondeur max (None = illimitée)
    "min_samples_split": 4,      # nb min d'exemples pour splitter un nœud
    "min_samples_leaf":  2,      # nb min d'exemples dans une feuille
    "max_features":      "sqrt", # nb de features considérées à chaque split
    "class_weight":      "balanced",  # compense les classes déséquilibrées
    "random_state":      42,
    "n_jobs":            -1,     # parallélisme sur tous les cœurs disponibles
}
model = RandomForestClassifier(**HYPERPARAMS)
_ts("avant fit")
model.fit(X_train, y_train)
_ts("après fit")

# ── 4. Évaluation ─────────────────────────────────────────────────────────────

y_pred    = model.predict(X_test)
acc       = float(accuracy_score(y_test, y_pred))
f1        = float(f1_score(y_test,        y_pred, average="weighted", zero_division=0))
precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
recall    = float(recall_score(y_test,    y_pred, average="weighted", zero_division=0))

print(
    f"[{MODEL_NAME}] Accuracy : {acc:.4f} | F1 : {f1:.4f}"
    f" | Precision : {precision:.4f} | Recall : {recall:.4f}",
    file=sys.stderr,
)
_ts("après évaluation")

# ── 5. Sauvegarde du modèle (OBLIGATOIRE) ─────────────────────────────────────

joblib.dump(model, OUTPUT_MODEL_PATH)

print(f"[{MODEL_NAME}] Modèle sauvegardé → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("après sauvegarde modèle")

# ── 5b. Sauvegarde du dataset d'entraînement → MinIO + artifact MLflow ────────

_dataset_minio_path = None

try:
    # Construire le CSV (features + target)
    _df_train = X_train.copy()
    _df_train["target"] = y_train
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_train.csv"
    _csv_local = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    print(f"[{MODEL_NAME}] Dataset CSV créé ({len(_df_train)} lignes) → {_csv_local}", file=sys.stderr)

    # Upload vers MinIO via boto3 (S3-compatible)
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
label_distribution = {
    iris.target_names[int(cls)]: round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in np.unique(y_train)
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

            # Params — un seul appel réseau
            _ts("MLflow — log_params (appel réseau #3)")
            mlflow.log_params({
                "algorithm":         "RandomForest",
                "train_start_date":  TRAIN_START_DATE,
                "train_end_date":    TRAIN_END_DATE,
                "n_samples_total":   n_samples,
                "test_size":         0.2,
                **HYPERPARAMS,
            })
            _ts("MLflow — log_params OK")

            # Toutes les métriques en un seul appel (au lieu de N appels individuels)
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

            _ts(f"MLflow — log_metrics x{len(all_metrics)} (appel réseau #4)")
            mlflow.log_metrics(all_metrics)
            _ts("MLflow — log_metrics OK")

            # Tags — un seul appel réseau
            _ts("MLflow — set_tags (appel réseau #5)")
            mlflow.set_tags({
                "model_name":  MODEL_NAME,
                "algorithm":   "RandomForest",
                "trigger":     "script",
                "n_features":  str(len(feature_names)),
                "n_classes":   str(len(iris.target_names)),
            })
            _ts("MLflow — set_tags OK")

            # Artifact — modèle sklearn (dégradation gracieuse si MinIO inaccessible)
            # Signature explicite : inputs float64 + outputs proba float (évite le warning
            # "integer column" causé par predict() qui retourne des classes entières)
            _X_example = X_test.astype("float64")
            _signature = mlflow.models.infer_signature(
                _X_example,
                model.predict_proba(_X_example),
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

            # Artifact — dataset d'entraînement (dégradation gracieuse)
            if _dataset_minio_path and os.path.exists(_csv_local):
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
    # confidence_threshold : si la probabilité max prédite par le modèle est inférieure
    # à ce seuil, l'API marque la prédiction avec low_confidence=True.
    # Cela permet aux appelants de traiter différemment les prédictions incertaines
    # (ex : routing vers un humain, refus de la prédiction, alerte monitoring).
    # 0.60 est un point de départ raisonnable pour Iris : le modèle est très calibré
    # sur ce dataset, donc les cas sous 60 % de confiance sont de vrais cas limites.
    "confidence_threshold": 0.60,
    "feature_stats":      feature_stats,
    "label_distribution": label_distribution,
    "dependencies":       _deps,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

_ts("fin script")
print(json.dumps(output))
