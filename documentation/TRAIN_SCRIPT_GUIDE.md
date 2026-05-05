# Guide — Écrire un script train.py compatible PredictML

Ce guide explique comment écrire un script `train.py` compatible avec le système de ré-entraînement automatique de PredictML.

---

## Pourquoi un train.py ?

Si vous fournissez un `train.py` lors de l'upload d'un modèle, vous pouvez ensuite :
- Déclencher un ré-entraînement manuel depuis l'API ou le dashboard
- Planifier un ré-entraînement automatique via une expression cron
- Déclencher un ré-entraînement réactif quand le drift dépasse un seuil

---

## Contrat obligatoire

Votre script **doit** :

1. **Lire trois variables d'environnement** (injectées automatiquement par l'API) :
   - `TRAIN_START_DATE` — date de début au format `YYYY-MM-DD`
   - `TRAIN_END_DATE` — date de fin au format `YYYY-MM-DD`
   - `OUTPUT_MODEL_PATH` — chemin absolu où sauvegarder le `.pkl`

2. **Sauvegarder le modèle** à `OUTPUT_MODEL_PATH` via `pickle.dump()`

3. **Afficher un JSON sur stdout** (dernière ligne JSON) avec au minimum :
   ```json
   {"accuracy": 0.95, "f1_score": 0.94}
   ```

4. **Quitter avec code 0** si succès, code non-nul si échec

---

## Variables d'environnement disponibles

| Variable | Obligatoire | Description |
|---|---|---|
| `TRAIN_START_DATE` | Oui | Date de début de la fenêtre d'entraînement (YYYY-MM-DD) |
| `TRAIN_END_DATE` | Oui | Date de fin de la fenêtre d'entraînement (YYYY-MM-DD) |
| `OUTPUT_MODEL_PATH` | Oui | Chemin absolu du fichier `.pkl` à produire |
| `MLFLOW_TRACKING_URI` | Non | URI du serveur MLflow (ex: `http://mlflow:5000`) |
| `MODEL_NAME` | Non | Nom du modèle source |

---

## Sortie JSON attendue (stdout)

La **dernière ligne JSON** de stdout est parsée par l'API pour mettre à jour la DB et le run MLflow.

### Champs obligatoires
```json
{"accuracy": 0.95, "f1_score": 0.94}
```

### Champs optionnels (enrichissent MLflow)
```json
{
    "accuracy": 0.95,
    "f1_score": 0.94,
    "n_rows": 12450,
    "feature_stats": {
        "sepal_length": {"mean": 5.8, "std": 0.83, "min": 4.3, "max": 7.9, "null_rate": 0.0}
    },
    "label_distribution": {
        "setosa": 0.33, "versicolor": 0.34, "virginica": 0.33
    }
}
```

> **Important** : Ne mettez rien après ce `print()`. La progression doit aller sur `stderr`.

---

## Template minimal

```python
"""train.py — Template minimal compatible PredictML"""
import json
import os
import pickle
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ── Variables d'environnement (OBLIGATOIRES) ──────────────────────────────────
TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE   = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

print(f"[train.py] Entraînement du {TRAIN_START_DATE} au {TRAIN_END_DATE}", file=sys.stderr)

# ── Chargement des données ────────────────────────────────────────────────────
# REMPLACEZ ce bloc par votre propre chargement (CSV, BDD, API, etc.)
# Filtrez obligatoirement sur [TRAIN_START_DATE, TRAIN_END_DATE]
import pandas as pd

df = pd.read_csv("data/training_data.csv", parse_dates=["date"])
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

if df.empty:
    print(json.dumps({"error": "Aucune donnée pour cette plage de dates"}))
    sys.exit(1)

feature_cols = ["feature_1", "feature_2", "feature_3"]
X = df[feature_cols]
y = df["target"]

# ── Entraînement ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Évaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
print(f"[train.py] Accuracy={acc:.4f}, F1={f1:.4f}", file=sys.stderr)

# ── Sauvegarde (OBLIGATOIRE) ──────────────────────────────────────────────────
with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"[train.py] Modèle sauvegardé : {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ── JSON stdout (DERNIÈRE LIGNE — lue par l'API) ──────────────────────────────
print(json.dumps({"accuracy": round(acc, 4), "f1_score": round(f1, 4)}))
```

---

## Template avec MLflow et feature stats

```python
"""train.py — Template avec MLflow + feature_stats pour drift detection"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Variables d'environnement ─────────────────────────────────────────────────
TRAIN_START_DATE    = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE      = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH   = os.environ["OUTPUT_MODEL_PATH"]
MODEL_NAME          = os.environ.get("MODEL_NAME", "my_model")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

print(f"[{MODEL_NAME}] Training {TRAIN_START_DATE} → {TRAIN_END_DATE}", file=sys.stderr)

# ── Données ───────────────────────────────────────────────────────────────────
df = pd.read_parquet("s3://my-bucket/data/training.parquet")  # Adaptez
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

FEATURES = ["age", "income", "score", "tenure_days"]
TARGET   = "churned"

if len(df) < 100:
    print(json.dumps({"error": f"Seulement {len(df)} lignes — entraînement annulé"}))
    sys.exit(1)

print(f"[{MODEL_NAME}] {len(df)} lignes chargées", file=sys.stderr)

X = df[FEATURES].fillna(0)
y = df[TARGET]

# ── Entraînement ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"[{MODEL_NAME}] Entraînement terminé sur {len(X_train)} exemples", file=sys.stderr)

# ── Évaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
print(f"[{MODEL_NAME}] Accuracy={acc:.4f} | F1={f1:.4f}", file=sys.stderr)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"[{MODEL_NAME}] Modèle → {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ── Stats pour drift detection et MLflow ─────────────────────────────────────
feature_stats = {
    col: {
        "mean":      round(float(X_train[col].mean()), 4),
        "std":       round(float(X_train[col].std()), 4),
        "min":       round(float(X_train[col].min()), 4),
        "max":       round(float(X_train[col].max()), 4),
        "null_rate": round(float(X_train[col].isna().mean()), 4),
    }
    for col in FEATURES
}

label_counts    = y_train.value_counts()
label_total     = len(y_train)
label_distribution = {
    str(k): round(float(v) / label_total, 4)
    for k, v in label_counts.items()
}

# ── JSON stdout (DERNIÈRE LIGNE — obligatoire) ─────────────────────────────
print(json.dumps({
    "accuracy":           round(acc, 4),
    "f1_score":           round(f1, 4),
    "n_rows":             len(X_train),
    "feature_stats":      feature_stats,
    "label_distribution": label_distribution,
}))
```

---

## Template pour modèle de régression

```python
"""train.py — Régression avec scikit-learn"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Variables d'environnement ─────────────────────────────────────────────────
TRAIN_START_DATE  = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE    = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

# ── Données ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/prices.csv", parse_dates=["date"])
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

FEATURES = ["surface_m2", "rooms", "floor", "distance_center_km"]
TARGET   = "price_eur"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Entraînement ──────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Évaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))

# Pour les modèles de régression, accuracy = R² (entre 0 et 1)
# f1_score = 1 - MAE/mean(y) (proportion d'erreur relative)
acc = max(0.0, r2)
f1  = max(0.0, 1.0 - mae / float(y_test.mean()))

print(f"[train.py] MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}", file=sys.stderr)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# ── JSON stdout ───────────────────────────────────────────────────────────────
print(json.dumps({
    "accuracy": round(acc, 4),
    "f1_score": round(f1, 4),
    "n_rows":   len(X_train),
    "feature_stats": {
        col: {"mean": round(float(X_train[col].mean()), 4), "std": round(float(X_train[col].std()), 4)}
        for col in FEATURES
    }
}))
```

---

## Uploader le modèle avec train.py

```bash
# curl
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@mon_modele.pkl;type=application/octet-stream" \
  -F "train_file=@train.py" \
  -F "name=mon_modele" \
  -F "version=1.0.0" \
  -F "description=Mon modèle de classification" \
  -F "accuracy=0.95" \
  -F "f1_score=0.94"
```

```python
# Python
import requests

with open("mon_modele.pkl", "rb") as pkl, open("train.py", "rb") as train:
    r = requests.post(
        "http://localhost:8000/models",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={
            "file":       ("mon_modele.pkl", pkl, "application/octet-stream"),
            "train_file": ("train.py", train, "text/plain"),
        },
        data={
            "name": "mon_modele", "version": "1.0.0",
            "accuracy": "0.95", "f1_score": "0.94",
        },
    )
```

---

## Déclencher un ré-entraînement

### Via l'API

```python
response = requests.post(
    "http://localhost:8000/models/mon_modele/1.0.0/retrain",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={
        "start_date":     "2025-01-01",
        "end_date":       "2025-12-31",
        "new_version":    "1.1.0",
        "set_production": False,
    }
)
data = response.json()
if data["success"]:
    print(f"Nouvelle version : {data['new_version']}")
    print(f"Accuracy : {data['accuracy']}")
    print(data["stdout"][-500:])  # derniers logs
else:
    print(f"Erreur : {data['error']}")
```

### Via le dashboard (page 8 — Retrain)
1. Sélectionnez le modèle et la version source
2. Renseignez les dates de début et fin
3. Saisissez le numéro de la nouvelle version
4. Cliquez **Lancer le ré-entraînement**
5. Les logs s'affichent en temps réel

---

## Planifier un ré-entraînement automatique

```python
# Chaque lundi à 3h UTC, fenêtre de 30 jours
requests.patch(
    "http://localhost:8000/models/mon_modele/1.0.0/schedule",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={
        "cron":         "0 3 * * 1",
        "lookback_days": 30,
        "auto_promote": True,
        "enabled":      True,
    }
)
```

### Exemples d'expressions cron
| Expression | Déclenchement |
|---|---|
| `0 3 * * 1` | Chaque lundi à 3h UTC |
| `0 0 1 * *` | Le 1er de chaque mois à minuit |
| `0 2 * * *` | Tous les jours à 2h UTC |
| `0 */6 * * *` | Toutes les 6 heures |

---

## Vérification de la syntaxe (validation à l'upload)

L'API vérifie automatiquement que votre `train.py` :
- A une syntaxe Python valide (`ast.parse()`)
- Référence `TRAIN_START_DATE`
- Référence `TRAIN_END_DATE`
- Référence `OUTPUT_MODEL_PATH`
- Contient un appel `pickle.dump`, `joblib.dump` ou `save_model`

Si la validation échoue, l'upload est rejeté avec un message d'erreur détaillé.

---

## Ré-entraînement déclenché par drift

Si vous configurez `trigger_on_drift` dans le schedule, un ré-entraînement est automatiquement déclenché quand le drift atteint le seuil :

```python
requests.patch(
    "http://localhost:8000/models/mon_modele/1.0.0/schedule",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={
        "cron":                       "0 3 * * 1",
        "lookback_days":               30,
        "trigger_on_drift":            "critical",  # ou "warning"
        "drift_retrain_cooldown_hours": 24,          # évite les boucles
        "enabled":                    True,
    }
)
```
