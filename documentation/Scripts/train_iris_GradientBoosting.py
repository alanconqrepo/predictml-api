"""
train_iris_GradientBoosting.py — Script de ré-entraînement PredictML — Exemple Iris (GradientBoosting)
=============================================================================================

Identique à train_iris.py mais utilise un GradientBoostingClassifier au lieu du RandomForest.

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
  - Modèle sauvegardé à OUTPUT_MODEL_PATH via pickle.dump()
  - Dernière ligne JSON sur stdout avec au minimum :
      {"accuracy": 0.95, "f1_score": 0.94}
  - Logs de progression sur stderr
  - Code de sortie 0 si succès

MODULES AUTORISÉS par le sandbox PredictML
-------------------------------------------
  os, sys, json, pickle, datetime, numpy, pandas, sklearn, mlflow
  (subprocess, requests, socket, urllib sont bloqués)
"""

import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

# ── 1. Variables d'environnement (OBLIGATOIRES) ───────────────────────────────

TRAIN_START_DATE  = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE    = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

MODEL_NAME               = os.environ.get("MODEL_NAME", "iris-classifier")
MLFLOW_TRACKING_URI      = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

print(
    f"[{MODEL_NAME}] Ré-entraînement du {TRAIN_START_DATE} au {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[{MODEL_NAME}] Sortie : {OUTPUT_MODEL_PATH}", file=sys.stderr)

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

if n_samples < 20:
    print(json.dumps({"error": f"Pas assez de données ({n_samples} exemples < 20 requis)"}))
    sys.exit(1)

# ── 3. Entraînement ───────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if n_samples >= 30 else None
)

print(
    f"[{MODEL_NAME}] Entraînement GradientBoosting sur {len(X_train)} exemples, "
    f"évaluation sur {len(X_test)}…",
    file=sys.stderr,
)

HYPERPARAMS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
model = GradientBoostingClassifier(**HYPERPARAMS)
model.fit(X_train, y_train)

# ── 4. Évaluation ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

print(f"[{MODEL_NAME}] Accuracy : {acc:.4f} | F1 : {f1:.4f}", file=sys.stderr)

# ── 5. Sauvegarde du modèle (OBLIGATOIRE) ─────────────────────────────────────

with open(OUTPUT_MODEL_PATH, "wb") as fh:
    pickle.dump(model, fh)

print(f"[{MODEL_NAME}] Modèle sauvegardé → {OUTPUT_MODEL_PATH}", file=sys.stderr)

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
        if MLFLOW_TRACKING_USERNAME:
            os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"predictml/{MODEL_NAME}")

        run_name = f"{MODEL_NAME}_{TRAIN_START_DATE}_{TRAIN_END_DATE}_gb"
        with mlflow.start_run(run_name=run_name) as run:
            # Params — hyperparamètres + contexte temporel
            mlflow.log_params({
                "algorithm":        "GradientBoosting",
                "n_estimators":     HYPERPARAMS["n_estimators"],
                "learning_rate":    HYPERPARAMS["learning_rate"],
                "max_depth":        HYPERPARAMS["max_depth"],
                "random_state":     HYPERPARAMS["random_state"],
                "train_start_date": TRAIN_START_DATE,
                "train_end_date":   TRAIN_END_DATE,
                "n_samples_total":  n_samples,
                "test_size":        0.2,
            })

            # Métriques scalaires
            mlflow.log_metric("accuracy",     acc)
            mlflow.log_metric("f1_score",     f1)
            mlflow.log_metric("n_rows_train", float(len(X_train)))
            mlflow.log_metric("n_rows_test",  float(len(X_test)))

            # Métriques par feature (mean, std, min, max, null_rate)
            for feat_name, stats in feature_stats.items():
                safe = feat_name.replace(" ", "_").replace("(", "").replace(")", "")[:40]
                for stat_key, val in stats.items():
                    mlflow.log_metric(f"feat_{safe}_{stat_key}", float(val))

            # Distribution des labels
            for label, ratio in label_distribution.items():
                mlflow.log_metric(f"label_{label}_ratio", float(ratio))

            # Tags
            mlflow.set_tags({
                "model_name":  MODEL_NAME,
                "algorithm":   "GradientBoosting",
                "trigger":     "script",
                "n_features":  str(len(feature_names)),
                "n_classes":   str(len(iris.target_names)),
            })

            # Artifact — modèle sklearn (dégradation gracieuse si MinIO inaccessible)
            try:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    input_example=X_test.iloc[:3],
                )
                print(f"[{MODEL_NAME}] Artifact modèle loggué dans MLflow.", file=sys.stderr)
            except Exception as art_exc:
                print(f"[{MODEL_NAME}] Artifact ignoré (MinIO inaccessible) : {art_exc}", file=sys.stderr)

            mlflow_run_id = run.info.run_id

        print(f"[{MODEL_NAME}] Run MLflow créé : {mlflow_run_id}", file=sys.stderr)

    except Exception as exc:
        print(f"[{MODEL_NAME}] MLflow indisponible — ré-entraînement continue : {exc}", file=sys.stderr)
        mlflow_run_id = None

else:
    reason = "mlflow non installé" if not _MLFLOW_AVAILABLE else "MLFLOW_TRACKING_URI non défini"
    print(f"[{MODEL_NAME}] MLflow ignoré ({reason}).", file=sys.stderr)

# ── 8. JSON stdout — DERNIÈRE LIGNE (lue par l'API — ne rien ajouter après) ───

output = {
    "accuracy":           round(acc, 4),
    "f1_score":           round(f1, 4),
    "n_rows":             len(X_train),
    "feature_stats":      feature_stats,
    "label_distribution": label_distribution,
}
if mlflow_run_id:
    output["mlflow_run_id"] = mlflow_run_id

print(json.dumps(output))
