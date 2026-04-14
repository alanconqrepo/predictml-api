"""
Exemple de script train.py pour la fonctionnalité de ré-entraînement PredictML.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTRAT D'INTERFACE — variables d'environnement OBLIGATOIRES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TRAIN_START_DATE  : date de début au format YYYY-MM-DD
                      (ex: "2025-01-01")
  TRAIN_END_DATE    : date de fin au format YYYY-MM-DD
                      (ex: "2025-12-31")
  OUTPUT_MODEL_PATH : chemin absolu où sauvegarder le modèle produit (.pkl)
                      (ex: "/tmp/abc123/output_model.pkl")

Variables d'environnement optionnelles injectées par l'API :
  MLFLOW_TRACKING_URI : URI du serveur MLflow pour logguer les métriques
  MODEL_NAME          : nom du modèle source (pour le contexte MLflow)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SORTIE ATTENDUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Le modèle DOIT être sauvegardé à OUTPUT_MODEL_PATH via pickle.dump().
  2. Pour remonter les métriques à l'API, imprimer sur stdout un JSON sur la
     DERNIÈRE ligne JSON de la sortie standard :
       {"accuracy": 0.95, "f1_score": 0.94}
  3. Les logs de progression peuvent être imprimés sur stderr à volonté.
  4. Quitter avec code 0 si succès, code non-nul si échec.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLE COMPLET — RandomForestClassifier sur Iris avec filtre de dates synthétique
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Remplacez la section « Chargement des données » par votre propre logique
(lecture CSV/Parquet, requête BDD, appel API, etc.) en filtrant sur la plage
de dates fournie.
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

# ──────────────────────────────────────────────────────────────────────────────
# 1. Lecture des variables d'environnement (OBLIGATOIRES)
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

# Variables optionnelles
MODEL_NAME = os.environ.get("MODEL_NAME", "example_model")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

print(
    f"[train.py] Ré-entraînement de '{MODEL_NAME}' " f"du {TRAIN_START_DATE} au {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[train.py] Sortie modèle : {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Chargement et filtrage des données
# ──────────────────────────────────────────────────────────────────────────────
# ⬇ REMPLACEZ CE BLOC par votre propre chargement de données.
#
# Exemple avec un DataFrame réel :
#
#   import pandas as pd
#   df = pd.read_csv("data/training_data.csv", parse_dates=["date"])
#   df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]
#   if df.empty:
#       print(json.dumps({"error": "Aucune donnée pour cette plage de dates"}))
#       sys.exit(1)
#   X = df.drop(columns=["target", "date"])
#   y = df["target"]
#
# ──────────────────────────────────────────────────────────────────────────────

print("[train.py] Chargement du dataset Iris (exemple synthétique)…", file=sys.stderr)
iris = load_iris()
X_full, y_full = iris.data, iris.target

# Simulation d'un filtre temporel : la taille de l'échantillon dépend de la plage de dates.
start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
end = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
delta_days = max(1, (end - start).days)

# Sélection d'un sous-ensemble proportionnel à la durée de la plage
n_samples = min(len(X_full), max(30, delta_days * 2))
rng = np.random.default_rng(seed=delta_days % 1000)
indices = rng.choice(len(X_full), size=n_samples, replace=False)
X, y = X_full[indices], y_full[indices]

print(f"[train.py] {n_samples} exemples retenus.", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Entraînement
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(
    f"[train.py] Entraînement sur {len(X_train)} exemples, " f"évaluation sur {len(X_test)}…",
    file=sys.stderr,
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Évaluation
# ──────────────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

print(f"[train.py] Accuracy : {acc:.4f} | F1 Score : {f1:.4f}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Sauvegarde du modèle (OBLIGATOIRE — utiliser pickle.dump)
# ──────────────────────────────────────────────────────────────────────────────
with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"[train.py] Modèle sauvegardé : {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Métriques sur stdout (JSON — doit être la dernière ligne JSON de stdout)
# ──────────────────────────────────────────────────────────────────────────────
# L'API lit la dernière ligne JSON de stdout pour mettre à jour accuracy et f1_score.
# Ne mettez rien après ce print.
print(json.dumps({"accuracy": round(acc, 4), "f1_score": round(f1, 4)}))
