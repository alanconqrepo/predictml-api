"""
train_titanic.py — Script de ré-entraînement PredictML — Titanic Survival (binaire)
python .\train_titanic.py
=====================================================================================

Ce script est conçu pour être uploadé avec votre modèle (champ "Script d'entraînement")
afin de permettre le ré-entraînement automatique ou planifié depuis le dashboard.

Problème   : classification binaire — prédire la survie d'un passager (0=mort, 1=survivant)
Données    : dataset synthétique inspiré du Titanic avec features numériques ET catégorielles
Modèle     : Pipeline sklearn — ColumnTransformer (StandardScaler + OneHotEncoder)
               + GradientBoostingClassifier
Métriques  : accuracy, f1_score, precision, recall, roc_auc

Features d'entrée (7 au total) :
  Numériques   : age (float), fare (float), parch (int), sibsp (int)
  Catégorielles: pclass (str : "1st"/"2nd"/"3rd"),
                 sex (str : "male"/"female"),
                 embarked (str : "S"/"C"/"Q")

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
      {"accuracy": 0.82, "f1_score": 0.81}
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


# ── 1. Variables d'environnement (OBLIGATOIRES) ───────────────────────────────

if DEBUG:
    print(
        f"""
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

# Timeout court pour tous les appels HTTP MLflow
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")

# Credentials MinIO pour log_model
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

# Hors Docker : substituer les hostnames internes par localhost
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
    -------------   APRES SURCHARGE  ------------
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
    f"[{MODEL_NAME}] Ré-entraînement du {TRAIN_START_DATE} au {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[{MODEL_NAME}] Sortie : {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("démarrage")

# ── 2. Génération des données synthétiques ────────────────────────────────────
#
# Données synthétiques inspirées du dataset Titanic (891 passagers réels).
# Distributions calibrées sur les statistiques historiques :
#   - 1st class : 24 %  — 2nd class : 21 %  — 3rd class : 55 %
#   - Sex       : male 65 %  —  female 35 %
#   - Embarked  : S=72 %  C=19 %  Q=9 %
#   - Survival  : ~38 % (influencé par class + sex + age)
#
# Pour ré-entraîner sur vos propres données de production, remplacez ce bloc par :
#   df = pd.read_csv("data/training_data.csv", parse_dates=["date"])
#   df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]
#
# ─────────────────────────────────────────────────────────────────────────────

print(f"[{MODEL_NAME}] Génération du dataset synthétique Titanic…", file=sys.stderr)
_ts("avant génération données")

# Taille proportionnelle à la plage temporelle (comme train_iris.py)
start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
delta_days = max(1, (end - start).days)

N_BASE = 891  # référence : taille réelle du dataset Titanic
n_samples = min(N_BASE * 3, max(100, delta_days * 5))

rng = np.random.default_rng(seed=delta_days % 1000)

# Tirage de la classe de billet (distribution historique)
pclass_vals = rng.choice(["1st", "2nd", "3rd"], size=n_samples, p=[0.24, 0.21, 0.55])

# Tirage du sexe (distribution historique)
sex_vals = rng.choice(["male", "female"], size=n_samples, p=[0.65, 0.35])

# Port d'embarquement (distribution historique)
embarked_vals = rng.choice(["S", "C", "Q"], size=n_samples, p=[0.72, 0.19, 0.09])

# Âge : dépend légèrement de la classe (1st class plus âgés)
age_mu = np.where(pclass_vals == "1st", 39.0, np.where(pclass_vals == "2nd", 29.0, 25.0))
age_vals = rng.normal(age_mu, 14.0).clip(1.0, 80.0).round(1)

# Tarif : fortement corrélé à la classe
fare_mu = np.where(pclass_vals == "1st", 87.0, np.where(pclass_vals == "2nd", 21.0, 13.0))
fare_sigma = np.where(pclass_vals == "1st", 60.0, np.where(pclass_vals == "2nd", 13.0, 9.0))
fare_vals = np.exp(rng.normal(np.log(fare_mu), 0.6)).clip(5.0, 512.0).round(2)

# Siblings/spouses : distribution empirique (0 très majoritaire)
sibsp_probs = [0.68, 0.16, 0.10, 0.04, 0.02]
sibsp_vals = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=sibsp_probs)

# Parents/children
parch_probs = [0.76, 0.13, 0.07, 0.03, 0.01]
parch_vals = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=parch_probs)

# Survie : modèle probabiliste inspiré des données réelles
#   - Femmes : taux de survie ~74 %
#   - Hommes : taux de survie ~19 %
#   - Bonus 1st class : +15 %
#   - Malus 3rd class : -10 %
#   - Enfants (< 12 ans) : +15 %
base_survival = np.where(sex_vals == "female", 0.74, 0.19)
class_bonus = np.where(pclass_vals == "1st", 0.15, np.where(pclass_vals == "3rd", -0.10, 0.0))
age_bonus = np.where(age_vals < 12, 0.15, 0.0)
survival_prob = np.clip(base_survival + class_bonus + age_bonus, 0.05, 0.95)
survived_vals = (rng.uniform(size=n_samples) < survival_prob).astype(int)

# Assemblage du DataFrame (ordre cohérent avec les constantes de features)
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
    f"[{MODEL_NAME}] {n_samples} passagers générés "
    f"(taux de survie : {survived_vals.mean():.1%})",
    file=sys.stderr,
)
_ts("après génération données")

if n_samples < 50:
    print(json.dumps({"error": f"Pas assez de données ({n_samples} exemples < 50 requis)"}))
    sys.exit(1)

# ── 3. Split train / test ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"[{MODEL_NAME}] Entraînement sur {len(X_train)} exemples, " f"évaluation sur {len(X_test)}…",
    file=sys.stderr,
)

# ── 4. Pipeline sklearn — ColumnTransformer + GradientBoosting ────────────────
#
# Le ColumnTransformer applique :
#   • StandardScaler        → features numériques  (age, fare, parch, sibsp)
#   • OneHotEncoder         → features catégorielles (pclass, sex, embarked)
#       handle_unknown="ignore"  → catégories inconnues en prod → vecteur zéro
#       sparse_output=False      → tableau dense requis par GBC
#
# Le Pipeline est entraîné sur un DataFrame pandas → pipeline.feature_names_in_
# contient les 7 noms de features originaux (avant encodage).
# L'API exploite ce champ pour la validation de schéma (/predict?strict_validation).
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

_ts("avant fit")
pipeline.fit(X_train, y_train)
_ts("après fit")

# ── 5. Évaluation ─────────────────────────────────────────────────────────────

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc = float(accuracy_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
roc_auc = float(roc_auc_score(y_test, y_proba))

print(
    f"[{MODEL_NAME}] Accuracy : {acc:.4f} | F1 : {f1:.4f}"
    f" | Precision : {precision:.4f} | Recall : {recall:.4f}"
    f" | AUC-ROC : {roc_auc:.4f}",
    file=sys.stderr,
)
_ts("après évaluation")

# ── 6. Sauvegarde du modèle (OBLIGATOIRE) ─────────────────────────────────────

joblib.dump(pipeline, OUTPUT_MODEL_PATH)
print(f"[{MODEL_NAME}] Pipeline sauvegardé → {OUTPUT_MODEL_PATH}", file=sys.stderr)
_ts("après sauvegarde modèle")

# ── 6b. Sauvegarde du dataset d'entraînement → MinIO + artifact MLflow ────────

_dataset_minio_path = None

try:
    _df_train = X_train.copy()
    _df_train["survived"] = y_train.values
    _csv_filename = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_train.csv"
    _csv_local = os.path.join(os.path.dirname(OUTPUT_MODEL_PATH), _csv_filename)
    _df_train.to_csv(_csv_local, index=False)
    print(
        f"[{MODEL_NAME}] Dataset CSV créé ({len(_df_train)} lignes) → {_csv_local}",
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
            f"[{MODEL_NAME}] Dataset uploadé dans MinIO → {_bucket}/{_object_key}",
            file=sys.stderr,
        )
    else:
        print(
            f"[{MODEL_NAME}] Dataset non uploadé dans MinIO "
            f"(boto3={'ok' if _BOTO3_AVAILABLE else 'absent'}, endpoint='{_minio_endpoint_url}')",
            file=sys.stderr,
        )

except Exception as _ds_exc:
    print(f"[{MODEL_NAME}] Sauvegarde dataset ignorée : {_ds_exc}", file=sys.stderr)

_ts("après sauvegarde dataset")

# ── 7. Statistiques pour MLflow et détection de drift ─────────────────────────
#
# feature_stats ne contient QUE les features numériques (FeatureStats attend
# mean/std/min/max). Les features catégorielles (pclass, sex, embarked) sont
# listées dans pipeline.feature_names_in_ et validées par nom dans l'API, mais
# elles ne peuvent pas avoir de baseline statistique numérique.
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

# Distribution des features catégorielles (informationnel — pour MLflow)
cat_distributions = {}
for col in CATEGORICAL_FEATURES:
    vc = X_train[col].value_counts(normalize=True)
    cat_distributions[col] = {k: round(float(v), 4) for k, v in vc.items()}

# ── 8. Logging MLflow (dégradation gracieuse si MLflow indisponible) ──────────

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

            _ts(f"MLflow — log_metrics x{len(all_metrics)} (appel réseau #4)")
            mlflow.log_metrics(all_metrics)
            _ts("MLflow — log_metrics OK")

            _ts("MLflow — set_tags (appel réseau #5)")
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

            # Signature : inputs = 7 features originales (4 float + 3 string)
            _ts("MLflow — log_model début (appel réseau #6 — MinIO)")
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
                    f"[{MODEL_NAME}] Artifact modèle loggué dans MLflow.",
                    file=sys.stderr,
                )
            except Exception as art_exc:
                _ts("MLflow — log_model ÉCHEC")
                print(
                    f"[{MODEL_NAME}] Artifact ignoré (MinIO inaccessible) : {art_exc}",
                    file=sys.stderr,
                )

            if _dataset_minio_path and os.path.exists(_csv_local):
                try:
                    _ts("MLflow — log_artifact dataset (appel réseau #7 — MinIO)")
                    mlflow.log_artifact(_csv_local, artifact_path="dataset")
                    _ts("MLflow — log_artifact dataset OK")
                except Exception as _art_ds_exc:
                    print(
                        f"[{MODEL_NAME}] Artifact dataset ignoré : {_art_ds_exc}",
                        file=sys.stderr,
                    )

            mlflow_run_id = run.info.run_id

        _ts("MLflow — run fermé")
        print(f"[{MODEL_NAME}] Run MLflow créé : {mlflow_run_id}", file=sys.stderr)

    except Exception as exc:
        _ts("MLflow — EXCEPTION (bloc entier abandonné)")
        print(
            f"[{MODEL_NAME}] MLflow indisponible — ré-entraînement continue : {exc}",
            file=sys.stderr,
        )
        mlflow_run_id = None

else:
    reason = "mlflow non installé" if not _MLFLOW_AVAILABLE else "MLFLOW_TRACKING_URI non défini"
    print(f"[{MODEL_NAME}] MLflow ignoré ({reason}).", file=sys.stderr)

# ── 9. Capture des versions de librairies ─────────────────────────────────────

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

# ── 10. JSON stdout — DERNIÈRE LIGNE (lue par l'API — ne rien ajouter après) ──

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
        or f"Données synthétiques Titanic — {n_samples} passagers, "
        "features : age, fare, parch, sibsp, pclass, sex, embarked"
    ),
    # confidence_threshold : seuil de probabilité en dessous duquel la prédiction
    # est marquée low_confidence=True dans l'API. 0.65 est adapté à ce modèle
    # binaire : les cas entre 0.35 et 0.65 de probabilité de survie sont ambigus.
    "confidence_threshold": 0.65,
    "feature_stats": feature_stats,  # uniquement features numériques
    "cat_distributions": cat_distributions,  # informationnel (non-standard API)
    "label_distribution": label_distribution,
    "dependencies": _deps,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

_ts("fin script")
print(json.dumps(output))
