"""
train_iris.py — Script de ré-entraînement PredictML — Exemple Iris
====================================================================

Ce script est conçu pour être uploadé avec votre modèle (champ "Script d'entraînement")
afin de permettre le ré-entraînement automatique ou planifié depuis le dashboard.

CONTRAT D'INTERFACE (variables d'environnement injectées automatiquement par l'API)
-------------------------------------------------------------------------------------
  TRAIN_START_DATE   : date de début  — format YYYY-MM-DD
  TRAIN_END_DATE     : date de fin    — format YYYY-MM-DD
  OUTPUT_MODEL_PATH  : chemin absolu où sauvegarder le .pkl produit

Variables optionnelles :
  MLFLOW_TRACKING_URI : URI du serveur MLflow
  MODEL_NAME          : nom du modèle source

SORTIE ATTENDUE
---------------
  - Modèle sauvegardé à OUTPUT_MODEL_PATH via pickle.dump()
  - Dernière ligne JSON sur stdout avec au minimum :
      {"accuracy": 0.95, "f1_score": 0.94}
  - Logs de progression sur stderr
  - Code de sortie 0 si succès

MODULES AUTORISÉS par le sandbox PredictML
-------------------------------------------
  os, sys, json, pickle, datetime, numpy, sklearn
  (subprocess, requests, socket, urllib sont bloqués)
"""

import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ── 1. Variables d'environnement (OBLIGATOIRES) ───────────────────────────────

TRAIN_START_DATE  = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE    = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

MODEL_NAME          = os.environ.get("MODEL_NAME", "iris-classifier")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

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
X_full, y_full = iris.data, iris.target

# Simulation d'un filtre temporel : taille de l'échantillon proportionnelle à la plage
start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end   = datetime.strptime(TRAIN_END_DATE,   "%Y-%m-%d")
delta_days = max(1, (end - start).days)

n_samples = min(len(X_full), max(30, delta_days * 2))
rng     = np.random.default_rng(seed=delta_days % 1000)
indices = rng.choice(len(X_full), size=n_samples, replace=False)
X, y    = X_full[indices], y_full[indices]

print(f"[{MODEL_NAME}] {n_samples} exemples retenus sur {len(X_full)} disponibles.", file=sys.stderr)

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

model = RandomForestClassifier(n_estimators=100, random_state=42)
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

feature_names = list(iris.feature_names)  # ["sepal length (cm)", ...]

feature_stats = {
    name: {
        "mean":      round(float(np.mean(X_train[:, i])), 4),
        "std":       round(float(np.std(X_train[:, i])),  4),
        "min":       round(float(np.min(X_train[:, i])),  4),
        "max":       round(float(np.max(X_train[:, i])),  4),
        "null_rate": 0.0,
    }
    for i, name in enumerate(feature_names)
}

total_train = len(y_train)
label_distribution = {
    iris.target_names[int(cls)]: round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in np.unique(y_train)
}

# ── 7. JSON stdout — DERNIÈRE LIGNE (lue par l'API — ne rien ajouter après) ───

print(json.dumps({
    "accuracy":           round(acc, 4),
    "f1_score":           round(f1, 4),
    "n_rows":             len(X_train),
    "feature_stats":      feature_stats,
    "label_distribution": label_distribution,
}))
