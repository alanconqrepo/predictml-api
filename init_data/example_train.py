"""
Exemple de script train.py pour la fonctionnalité de ré-entraînement PredictML.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTRAT D'INTERFACE — variables d'environnement OBLIGATOIRES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TRAIN_START_DATE  : date de début au format YYYY-MM-DD
                      (ex: "2025-01-01")
  TRAIN_END_DATE    : date de fin au format YYYY-MM-DD
                      (ex: "2025-12-31")
  OUTPUT_MODEL_PATH : chemin absolu où sauvegarder le modèle produit (.joblib)
                      (ex: "/tmp/abc123/output_model.joblib")

Variables d'environnement optionnelles injectées par l'API :
  MLFLOW_TRACKING_URI : URI du serveur MLflow
  MODEL_NAME          : nom du modèle source
  TRAIN_DATA_PATH     : chemin vers le CSV des données de production exportées
                        par l'API (prédictions + résultats observés).
                        Absent lors du 1er entraînement (aucune donnée en prod).

IMPORTANT — MLflow est géré automatiquement par l'API :
  L'API crée elle-même le run MLflow après l'exécution de ce script.
  Vous n'avez PAS besoin d'appeler mlflow.start_run() ici.
  Pour enrichir le logging MLflow, ajoutez les clés optionnelles ci-dessous
  dans la sortie JSON stdout (section 6).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DU CSV TRAIN_DATA_PATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colonnes : id_obs, input_features, prediction_result, observed_result,
             timestamp, model_version, response_time_ms

  - input_features   : dict JSON des features envoyées à /predict
  - observed_result  : valeur réelle observée (JSON), vide si non renseignée
  - prediction_result: ce que le modèle avait prédit (JSON)

  Pour un train supervisé, filtrez les lignes où observed_result est non-vide :
    X = [json.loads(row["input_features"]) for row if row["observed_result"]]
    y = [json.loads(row["observed_result"]) for row if row["observed_result"]]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SORTIE ATTENDUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Le modèle DOIT être sauvegardé à OUTPUT_MODEL_PATH via joblib.dump().
  2. Imprimer sur stdout un JSON comme DERNIÈRE ligne JSON de la sortie :
       {
         "accuracy": 0.95,          # obligatoire pour mise à jour en DB
         "f1_score": 0.94,          # obligatoire pour mise à jour en DB
         "n_rows": 12450,           # optionnel — loggué dans MLflow
         "feature_stats": {         # optionnel — loggué comme metrics MLflow
           "sepal_length": {"mean": 5.8, "std": 0.83, "min": 4.3, "max": 7.9, "null_rate": 0.0}
         },
         "label_distribution": {    # optionnel — loggué comme metrics MLflow
           "setosa": 0.33, "versicolor": 0.33, "virginica": 0.34
         }
       }
  3. Les logs de progression peuvent être imprimés sur stderr à volonté.
  4. Quitter avec code 0 si succès, code non-nul si échec.
"""

import csv
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
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
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH")  # absent lors du 1er train

print(
    f"[train.py] Ré-entraînement de '{MODEL_NAME}' "
    f"du {TRAIN_START_DATE} au {TRAIN_END_DATE}",
    file=sys.stderr,
)
print(f"[train.py] Sortie modèle : {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Chargement des données
# ──────────────────────────────────────────────────────────────────────────────

if TRAIN_DATA_PATH:
    # ── Retrain : données de production exportées par l'API ──────────────────
    print(f"[train.py] Chargement des données de production : {TRAIN_DATA_PATH}", file=sys.stderr)

    features_list = []
    labels_list = []
    feature_names = None

    with open(TRAIN_DATA_PATH, newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            if not row["observed_result"]:
                continue  # ignorer les prédictions sans résultat observé
            features = json.loads(row["input_features"])
            label = json.loads(row["observed_result"])
            if feature_names is None:
                feature_names = sorted(features.keys())
            features_list.append([features[k] for k in feature_names])
            labels_list.append(label)

    if not features_list:
        print(
            json.dumps({"error": "Aucune donnée labellisée dans la fenêtre de dates."}),
            flush=True,
        )
        sys.exit(1)

    X = np.array(features_list, dtype=float)
    y = np.array(labels_list)
    print(f"[train.py] {len(X)} exemples labellisés chargés.", file=sys.stderr)

else:
    # ── Premier train : dataset synthétique (aucune donnée prod disponible) ──
    print("[train.py] TRAIN_DATA_PATH absent — utilisation du dataset Iris.", file=sys.stderr)
    from sklearn.datasets import load_iris

    iris = load_iris()
    X_full, y_full = iris.data, iris.target
    feature_names = list(iris.feature_names)

    # Simulation d'un filtre temporel proportionnel à la plage de dates
    start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
    end = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
    delta_days = max(1, (end - start).days)
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
    f"[train.py] Entraînement sur {len(X_train)} exemples, "
    f"évaluation sur {len(X_test)}…",
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
# 5. Sauvegarde du modèle (OBLIGATOIRE — utiliser joblib.dump)
# ──────────────────────────────────────────────────────────────────────────────
joblib.dump(model, OUTPUT_MODEL_PATH)

print(f"[train.py] Modèle sauvegardé : {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Capture des versions de librairies (lues par l'API pour requirements.txt)
# ──────────────────────────────────────────────────────────────────────────────
import importlib.metadata as _imeta  # noqa: E402

_deps: dict = {}
for _pkg in ["scikit-learn", "numpy", "pandas", "mlflow", "python-dotenv"]:
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# 7. Métriques sur stdout (dernière ligne JSON — lue par l'API pour MLflow + DB)
# ──────────────────────────────────────────────────────────────────────────────
feature_stats = {}
if feature_names is not None:
    for i, fname in enumerate(feature_names):
        col = X_train[:, i] if X_train.ndim > 1 else X_train
        feature_stats[fname] = {
            "mean": round(float(np.mean(col)), 4),
            "std": round(float(np.std(col)), 4),
            "min": round(float(np.min(col)), 4),
            "max": round(float(np.max(col)), 4),
            "null_rate": 0.0,
        }

classes_arr = np.unique(y_train)
total_train = len(y_train)
label_distribution = {
    str(cls): round(float(np.sum(y_train == cls)) / total_train, 4)
    for cls in classes_arr
    if total_train > 0
}

print(
    json.dumps(
        {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "n_rows": len(X_train),
            "feature_stats": feature_stats,
            "label_distribution": label_distribution,
            "dependencies": _deps,
        }
    )
)
