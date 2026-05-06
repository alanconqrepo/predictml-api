# PredictML API — Guide pour agent IA

Ce document explique comment interagir avec **PredictML API** en tant qu'utilisateur ou agent IA. Il couvre l'objectif du projet, la structure du code, toutes les routes disponibles, le cycle de vie des modèles, la détection de drift et le ré-entraînement.

---

## Objectif du projet

PredictML API est une plateforme ML-as-a-Service qui permet de :

- **Déployer** des modèles scikit-learn (`.pkl`) via une API REST
- **Servir des prédictions** en temps réel (unitaires ou batch)
- **Versionner** les modèles avec gestion de trafic A/B et déploiement shadow
- **Monitorer** les performances et détecter le drift des features
- **Ré-entraîner** automatiquement un modèle à partir d'un script `train.py`
- **Auditer** toutes les modifications via un historique complet avec rollback

Les modèles sont stockés dans **MinIO**, les expériences tracées dans **MLflow**, les métadonnées dans **PostgreSQL**, et les modèles chargés sont mis en cache dans **Redis**.

---

## Structure du projet

```
src/
├── api/
│   ├── models.py          # CRUD modèles, retrain, drift, A/B, historique
│   ├── predict.py         # Prédictions unitaires, batch, SHAP
│   ├── users.py           # Gestion utilisateurs (admin)
│   ├── observed_results.py# Résultats observés (ground truth)
│   └── monitoring.py      # Dashboard monitoring global et par modèle
├── core/
│   ├── config.py          # Variables d'environnement et settings
│   └── security.py        # Vérification token, rate limiting
├── db/
│   ├── models.py          # Modèles ORM SQLAlchemy
│   └── database.py        # Session async PostgreSQL
├── services/
│   ├── model_service.py   # Chargement, cache Redis, routing A/B
│   ├── db_service.py      # Accès données (prédictions, utilisateurs, modèles)
│   └── minio_service.py   # Upload/download artefacts MinIO
├── schemas/               # Schémas Pydantic (requêtes / réponses)
└── main.py                # App FastAPI, montage des routers

streamlit_app/             # Dashboard admin Streamlit (port 8501)
tests/                     # Tests pytest (sans Docker)
smoke-tests/               # Tests manuels contre l'API live
init_data/                 # Scripts d'initialisation et example_train.py
alembic/                   # Migrations de base de données
```

---

## Authentification

Toutes les routes protégées nécessitent un **Bearer token** dans le header :

```
Authorization: Bearer <api_token>
```

Les tokens sont générés à la création d'un utilisateur (et lors d'un renouvellement). Ils sont validés en base de données à chaque requête.

### Rôles utilisateur

| Rôle | Accès |
|------|-------|
| `admin` | Accès complet (gestion utilisateurs, modèles, retrain) |
| `user` | Prédictions, lecture des modèles et historique |
| `readonly` | Lecture seule |

### Rate limiting

Les routes `/predict` et `/predict-batch` sont soumises à un quota journalier (`rate_limit_per_day`, défaut 1000). Une requête dépassant le quota retourne `429 Too Many Requests`.

---

## Cycle de vie d'un modèle

### 1. Créer un modèle (première version)

```bash
POST /models
Content-Type: multipart/form-data
Authorization: Bearer <token>

Champs obligatoires :
  name        (string)   : Nom unique du modèle
  version     (string)   : Version (ex: "1.0.0")
  file        (fichier)  : Fichier .pkl (OU mlflow_run_id)

Champs optionnels :
  description, algorithm, accuracy, f1_score
  features_count, classes (JSON array)
  training_params (JSON object)
  training_dataset (string)
  feature_baseline (JSON)   : profil baseline pour le drift
  tags (JSON array)         : ex. ["production", "finance"]
  webhook_url               : URL POST appelée après chaque prédiction
  train_file (fichier)      : script train.py pour le ré-entraînement
```

**Exemple curl :**
```bash
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer <token>" \
  -F "name=churn_model" \
  -F "version=1.0.0" \
  -F "file=@models/churn_v1.pkl" \
  -F "algorithm=RandomForest" \
  -F "accuracy=0.92" \
  -F "classes=[0,1]" \
  -F "train_file=@init_data/train_churn.py"
```

### 2. Mettre en production une version

```bash
PATCH /models/{name}/{version}
Content-Type: application/json
Authorization: Bearer <token>

{
  "is_production": true
}
```

> Positionner `is_production: true` sur une version **démote automatiquement** toutes les autres versions du même modèle.

### 3. Créer une nouvelle version manuellement

Même appel que la création, avec un nouveau `version` (ex: `"1.1.0"`). La nouvelle version n'est pas en production par défaut.

```bash
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer <token>" \
  -F "name=churn_model" \
  -F "version=1.1.0" \
  -F "file=@models/churn_v2.pkl" \
  -F "accuracy=0.94"
```

### 4. Mettre à jour les métadonnées

```bash
PATCH /models/{name}/{version}
Content-Type: application/json
Authorization: Bearer <token>

Champs disponibles (tous optionnels) :
{
  "description": "string",
  "is_production": true/false,
  "accuracy": 0.94,
  "features_count": 12,
  "classes": [0, 1],
  "confidence_threshold": 0.75,
  "feature_baseline": {"feature_name": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0}},
  "tags": ["v2", "retrained"],
  "webhook_url": "https://...",
  "traffic_weight": 0.3,
  "deployment_mode": "production|ab_test|shadow"
}
```

### 5. Supprimer une version

```bash
DELETE /models/{name}/{version}
Authorization: Bearer <token>
```

### 6. Supprimer toutes les versions d'un modèle

```bash
DELETE /models/{name}
Authorization: Bearer <token>
```

---

## Prédictions

### Prédiction unitaire

```bash
POST /predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_name": "churn_model",
  "model_version": "1.0.0",   # optionnel — utilise la version en prod si omis
  "id_obs": "client_42",       # optionnel — identifiant d'observation
  "features": {
    "age": 35,
    "tenure_months": 24,
    "monthly_charges": 79.5
  }
}
```

**Réponse :**
```json
{
  "model_name": "churn_model",
  "model_version": "1.0.0",
  "id_obs": "client_42",
  "prediction": 1,
  "probability": [0.12, 0.88],
  "low_confidence": false,
  "selected_version": null
}
```

> Si `model_version` est omis, le routing A/B/Shadow s'applique automatiquement. `selected_version` indique la version choisie.

### Prédiction batch

```bash
POST /predict-batch
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_name": "churn_model",
  "model_version": null,
  "inputs": [
    {"id_obs": "c1", "features": {"age": 35, "tenure_months": 24, "monthly_charges": 79.5}},
    {"id_obs": "c2", "features": {"age": 58, "tenure_months": 6,  "monthly_charges": 120.0}}
  ]
}
```

> Le batch consomme autant d'appels que d'éléments dans `inputs` sur le quota journalier.

### Historique des prédictions

```bash
GET /predictions?name=churn_model&start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
  &version=1.0.0    # optionnel
  &user=alice       # optionnel
  &id_obs=client_42 # optionnel
  &limit=100        # 1-1000, défaut 100
  &cursor=450       # pagination curseur (id de la dernière prédiction vue)
Authorization: Bearer <token>
```

### Explainabilité SHAP

```bash
POST /explain
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_name": "churn_model",
  "model_version": "1.0.0",
  "features": {"age": 35, "tenure_months": 24, "monthly_charges": 79.5}
}
```

**Réponse :** `shap_values` par feature + `base_value` (E[f(X)]). Fonctionne avec les modèles arbres (RandomForest, GradientBoosting…) et linéaires (LogisticRegression, Ridge…).

---

## Résultats observés (ground truth)

Envoyer les vraies étiquettes pour calculer les performances réelles :

```bash
POST /observed-results
Content-Type: application/json
Authorization: Bearer <token>

{
  "data": [
    {
      "id_obs": "client_42",
      "model_name": "churn_model",
      "date_time": "2025-03-15T10:00:00",
      "observed_result": 1
    }
  ]
}
```

> L'appel est **upsert** : si `id_obs` + `model_name` existe déjà, la valeur est mise à jour.

**Récupérer les résultats :**
```bash
GET /observed-results?model_name=churn_model&start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
  &id_obs=client_42  # optionnel
  &limit=100&offset=0
```

---

## Drift et monitoring

### Prérequis : fournir un `feature_baseline`

Pour que la détection de drift fonctionne, le modèle doit avoir un profil baseline. Il peut être fourni à la création ou via PATCH :

```json
"feature_baseline": {
  "age":              {"mean": 42.3, "std": 12.1, "min": 18.0, "max": 90.0},
  "tenure_months":    {"mean": 18.5, "std": 9.4,  "min": 0.0,  "max": 72.0},
  "monthly_charges":  {"mean": 65.2, "std": 20.8, "min": 20.0, "max": 200.0}
}
```

### Rapport de drift par modèle

```bash
GET /models/{name}/drift?version=1.0.0&days=7&min_predictions=30
```

**Réponse par feature :**
- `z_score` : |moyenne_prod - moyenne_baseline| / std_baseline
- `psi` : Population Stability Index (binning sur les données de prod vs baseline)
- `drift_status` : `ok` | `warning` | `critical` | `insufficient_data` | `no_baseline`

**Seuils indicatifs :**
- Z-score > 2 → warning ; > 3 → critical
- PSI > 0.1 → warning ; > 0.2 → critical

### Performance réelle (après observed_results)

```bash
GET /models/{name}/performance?start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
  &version=1.0.0    # optionnel
  &granularity=day  # optionnel : day | week | month
```

Retourne accuracy/F1 (classification) ou MAE/RMSE/R² (régression) calculés sur les paires prédiction/résultat observé.

### Monitoring global

```bash
GET /monitoring/overview?start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
```

Retourne un tableau de bord global : total prédictions, taux d'erreur, latence p95, statut de drift et de santé par modèle.

### Monitoring détaillé par modèle

```bash
GET /monitoring/model/{name}?start=2025-01-01T00:00:00&end=2025-12-31T23:59:59
```

Retourne : stats par version, timeseries journalière, performance par jour, drift des features, comparaison A/B, dernières erreurs.

---

## Ré-entraînement (retrain en cas de drift)

### Flux complet

1. Détecter le drift via `GET /models/{name}/drift` ou observer une dégradation de performance via `GET /models/{name}/performance`
2. Déclencher le ré-entraînement via `POST /models/{name}/{version}/retrain`
3. L'API télécharge le `train.py` stocké dans MinIO, l'exécute dans un sous-processus isolé (timeout 600 s) et crée une nouvelle version
4. Optionnellement, promouvoir la nouvelle version en production

### Déclencher un ré-entraînement

```bash
POST /models/{name}/{version}/retrain
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "start_date": "2025-01-01",
  "end_date":   "2025-12-31",
  "new_version": "1.2.0",      # optionnel — auto-généré si null
  "set_production": false       # promouvoir automatiquement en production
}
```

**Réponse :**
```json
{
  "model_name": "churn_model",
  "source_version": "1.0.0",
  "new_version": "1.2.0",
  "success": true,
  "stdout": "...",
  "stderr": "",
  "error": null,
  "new_model_metadata": { ... }
}
```

### Contrat du script `train.py`

Le script fourni à l'upload doit respecter les contraintes suivantes (vérifiées statiquement au moment de l'upload) :

| Contrainte | Détail |
|---|---|
| Syntaxe Python valide | Vérifié via `ast.parse()` |
| Lire `TRAIN_START_DATE` | `os.environ["TRAIN_START_DATE"]` (format YYYY-MM-DD) |
| Lire `TRAIN_END_DATE` | `os.environ["TRAIN_END_DATE"]` |
| Lire `OUTPUT_MODEL_PATH` | Chemin où écrire le `.pkl` |
| Sauvegarder le modèle | `pickle.dump`, `joblib.dump` ou `save_model` |
| Retourner les métriques | Dernière ligne JSON sur stdout : `{"accuracy": 0.94, "f1_score": 0.93}` |

**Variables d'environnement injectées automatiquement :**

| Variable | Description |
|---|---|
| `TRAIN_START_DATE` | Date début (YYYY-MM-DD) |
| `TRAIN_END_DATE` | Date fin (YYYY-MM-DD) |
| `OUTPUT_MODEL_PATH` | Chemin absolu du `.pkl` à produire |
| `MLFLOW_TRACKING_URI` | URI MLflow (optionnel) |
| `MODEL_NAME` | Nom du modèle source |

**Squelette minimal de `train.py` :**
```python
import os, pickle, json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_START_DATE  = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE    = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

# Charger les données filtrées par date
df = pd.read_csv("data.csv")
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

accuracy = model.score(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# Retourner les métriques — doit être la dernière ligne JSON sur stdout
print(json.dumps({"accuracy": accuracy, "f1_score": accuracy}))
```

---

## Déploiement A/B et Shadow

### A/B testing

Configurer deux versions avec `deployment_mode: "ab_test"` et un `traffic_weight` (0.0–1.0, somme ≤ 1.0) :

```bash
PATCH /models/churn_model/1.0.0
{ "deployment_mode": "ab_test", "traffic_weight": 0.7 }

PATCH /models/churn_model/1.1.0
{ "deployment_mode": "ab_test", "traffic_weight": 0.3 }
```

Lors d'un appel `/predict` sans `model_version`, le routing est aléatoire pondéré. `selected_version` dans la réponse indique la version choisie.

### Shadow deployment

Tester une nouvelle version en parallèle sans l'exposer :

```bash
PATCH /models/churn_model/1.1.0
{ "deployment_mode": "shadow" }
```

Les prédictions shadow sont enregistrées (`is_shadow: true`) mais ne sont pas retournées au client. La comparaison se fait via `GET /models/{name}/ab-compare`.

### Comparaison A/B

```bash
GET /models/{name}/ab-compare?days=30
```

Retourne pour chaque version : nombre de prédictions, taux d'erreur, latences p50/p95, distribution des prédictions, taux d'accord shadow/production.

---

## Historique et rollback

### Consulter l'historique d'un modèle

```bash
GET /models/{name}/history?limit=50&offset=0
GET /models/{name}/{version}/history?limit=50&offset=0
```

Chaque entrée contient : `action` (CREATED, UPDATED, SET_PRODUCTION, ROLLBACK, DELETED), `snapshot` (état complet des métadonnées), `changed_fields`, `timestamp`, `changed_by_username`.

### Rollback vers un état précédent

```bash
POST /models/{name}/{version}/rollback/{history_id}
Authorization: Bearer <admin_token>
```

Restaure les métadonnées du modèle à l'état capturé dans `history_id`. Un nouvel enregistrement ROLLBACK est créé dans l'historique.

---

## Gestion des utilisateurs (admin)

### Créer un utilisateur

```bash
POST /users
Content-Type: application/json
Authorization: Bearer <admin_token>

{
  "username": "alice",
  "email": "alice@example.com",
  "role": "user",        # user | admin | readonly
  "rate_limit": 500      # quota journalier, 1-100000
}
```

Le `api_token` est retourné **uniquement** à la création.

### Lister, consulter, modifier, supprimer

```bash
GET    /users                   # tous les utilisateurs (admin)
GET    /users/{id}              # un utilisateur (admin ou soi-même)
PATCH  /users/{id}              # modifier rôle, statut, quota, renouveler token
DELETE /users/{id}              # supprimer (impossible sur son propre compte)
```

### Renouveler un token

```bash
PATCH /users/{id}
{ "regenerate_token": true }
```

Le nouveau token est retourné dans la réponse.

---

## Routes de service

```bash
GET /         # Informations API et modèles disponibles (pas d'auth)
GET /health   # Statut DB et cache Redis (pas d'auth)
GET /models/cached  # Modèles actuellement en cache Redis
```

---

## Statistiques de prédictions

```bash
GET /predictions/stats?model_name=churn_model&days=30
```

Retourne par modèle : total prédictions, taux d'erreur, latences moyennes p50/p95.

---

## Conseils pour un agent IA

1. **Toujours fournir un `feature_baseline`** à la création du modèle pour activer la détection de drift.
2. **Fournir un `train_file`** à la création si le ré-entraînement automatique est prévu.
3. **Utiliser `id_obs`** dans chaque prédiction pour pouvoir relier les résultats observés aux prédictions.
4. **Envoyer les `observed_results`** régulièrement pour que les métriques de performance réelle soient calculées.
5. **Surveiller `drift_status`** via `GET /models/{name}/drift` avant de décider un retrain.
6. **Ne pas mettre `set_production: true`** sur un retrain sans avoir vérifié les métriques retournées dans la réponse.
7. **Utiliser `model_version: null`** dans `/predict` pour bénéficier du routing automatique en A/B ou shadow.
8. **L'admin token** pour ce projet est `<ADMIN_TOKEN>` (à ne jamais exposer publiquement).
