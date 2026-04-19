# Référence API — PredictML API v2.0

Documentation complète de tous les endpoints, schémas de données et exemples de code Python.

---

## Authentification

Toutes les routes protégées utilisent un token Bearer dans l'en-tête HTTP.

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "votre-token-ici"

headers = {"Authorization": f"Bearer {TOKEN}"}
```

**Codes d'erreur d'authentification**

| Code | Raison |
|---|---|
| 401 | Token absent ou invalide |
| 403 | Compte inactif ou rôle insuffisant |
| 429 | Quota journalier de prédictions dépassé |

**Rôles disponibles**

| Rôle | Accès |
|---|---|
| `admin` | Accès complet — gestion utilisateurs, modèles, prédictions |
| `user` | Prédictions, consultation de son profil |
| `readonly` | Consultation uniquement, pas de prédiction |

---

## Endpoints publics

### `GET /`

Statut de l'API et liste des modèles actifs.

```python
response = requests.get(f"{BASE_URL}/")
print(response.json())
```

```json
{
  "message": "PredictML API - Multi Models",
  "status": "ok",
  "models_available": ["iris_model", "fraud_detector"],
  "models_count": 2,
  "models_cached": ["iris_model:1.0"]
}
```

---

### `GET /health`

Vérifie la connectivité à la base de données et l'état du cache Redis.

```python
response = requests.get(f"{BASE_URL}/health")
```

```json
{
  "status": "ok",
  "models_available": 2,
  "models_cached": 1
}
```

---

## Modèles

### `GET /models`

Liste tous les modèles actifs. Filtrable par tag.

```python
# Tous les modèles
response = requests.get(f"{BASE_URL}/models")

# Filtrés par tag
response = requests.get(f"{BASE_URL}/models", params={"tag": "production"})
models = response.json()
```

**Paramètre de requête**

| Paramètre | Type | Description |
|---|---|---|
| `tag` | str | Filtre par tag (optionnel) |

---

### `GET /models/cached`

Liste les modèles actuellement chargés en mémoire (cache Redis).

```python
response = requests.get(f"{BASE_URL}/models/cached")
```

```json
{
  "cached_models": ["iris_model:1.0", "fraud_detector:2.1"],
  "count": 2
}
```

---

### `GET /models/{name}/{version}`

Détail complet d'un modèle, incluant les métriques et les informations de chargement.

```python
response = requests.get(f"{BASE_URL}/models/iris_model/1.0.0")
model = response.json()
print(model["model_type"])        # "RandomForestClassifier"
print(model["feature_names"])     # ["sepal length (cm)", ...]
print(model["deployment_mode"])   # "production" | "ab_test" | "shadow"
print(model["traffic_weight"])    # 0.8
```

**Schéma de réponse `ModelGetResponse`**

```json
{
  "id": 1,
  "name": "iris_model",
  "version": "1.0.0",
  "description": "Classifieur Iris — 3 espèces",
  "algorithm": "RandomForestClassifier",
  "mlflow_run_id": "abc123def456",
  "minio_bucket": "models",
  "minio_object_key": "iris_model/v1.0.0_model.pkl",
  "file_size_bytes": 24576,
  "file_hash": "sha256:...",
  "accuracy": 0.97,
  "precision": 0.97,
  "recall": 0.96,
  "f1_score": 0.97,
  "confidence_threshold": 0.7,
  "features_count": 4,
  "classes": ["setosa", "versicolor", "virginica"],
  "training_params": {"n_estimators": 100, "max_depth": 5},
  "training_dataset": "iris_train_2024.csv",
  "trained_by": "alice",
  "training_date": "2024-01-15T10:30:00",
  "tags": ["iris", "classification"],
  "webhook_url": "https://hooks.example.com/predictml",
  "feature_baseline": {
    "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}
  },
  "is_active": true,
  "is_production": true,
  "deployment_mode": "production",
  "traffic_weight": 1.0,
  "train_script_object_key": "iris_model/v1.0.0_train.py",
  "created_at": "2024-01-15T10:35:00",
  "updated_at": "2024-01-20T08:00:00",
  "deprecated_at": null,
  "creator_username": "alice",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "feature_names": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
}
```

---

### `POST /models` — Uploader un modèle

Upload d'un fichier `.pkl` avec ses métadonnées. Utilise `multipart/form-data`.

**Auth requise** (rôle `user` ou supérieur)

```python
import requests

with open("iris_model.pkl", "rb") as f:
    files = {"file": ("iris_model.pkl", f, "application/octet-stream")}
    data = {
        "name": "iris_model",
        "version": "1.0.0",
        "description": "Classifieur Iris — 3 espèces",
        "algorithm": "RandomForestClassifier",
        "accuracy": "0.97",
        "f1_score": "0.97",
        "features_count": "4",
        "classes": '["setosa", "versicolor", "virginica"]',
        "training_params": '{"n_estimators": 100, "max_depth": 5}',
        "training_dataset": "iris_train_2024.csv",
        "tags": '["iris", "classification"]',
        "webhook_url": "https://hooks.example.com/predictml",
        "confidence_threshold": "0.7",
        "feature_baseline": '{"sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}}',
    }
    response = requests.post(
        f"{BASE_URL}/models",
        headers=headers,
        files=files,
        data=data,
    )
print(response.status_code)  # 201
```

**Avec un run MLflow (sans fichier `.pkl`)**

```python
data = {
    "name": "iris_model",
    "version": "2.0.0",
    "mlflow_run_id": "abc123def456789",
}
response = requests.post(f"{BASE_URL}/models", headers=headers, data=data)
```

**Avec un script de ré-entraînement**

```python
with open("iris_model.pkl", "rb") as f_model, open("train.py", "rb") as f_train:
    response = requests.post(
        f"{BASE_URL}/models",
        headers=headers,
        files={
            "file": ("iris_model.pkl", f_model, "application/octet-stream"),
            "train_file": ("train.py", f_train, "text/x-python"),
        },
        data={"name": "iris_model", "version": "1.0.0"},
    )
```

**Champs du formulaire**

| Champ | Type | Requis | Description |
|---|---|---|---|
| `name` | str | Oui | Nom unique du modèle |
| `version` | str | Oui | Version (ex: "1.0.0") |
| `file` | fichier `.pkl` | Si pas de MLflow | Fichier sérialisé |
| `train_file` | fichier `.py` | Non | Script de ré-entraînement |
| `description` | str | Non | Description lisible |
| `algorithm` | str | Non | Nom de l'algo |
| `mlflow_run_id` | str | Non | ID de run MLflow |
| `accuracy` | float | Non | Score de précision |
| `f1_score` | float | Non | Score F1 |
| `features_count` | int | Non | Nombre de features |
| `classes` | JSON str | Non | Labels des classes `["A","B"]` |
| `training_params` | JSON str | Non | Hyperparamètres `{"n": 100}` |
| `training_dataset` | str | Non | Nom/URI du dataset |
| `tags` | JSON str | Non | Tags `["tag1","tag2"]` |
| `webhook_url` | str | Non | URL de webhook post-prédiction |
| `confidence_threshold` | float | Non | Seuil de confiance (0.0–1.0) |
| `feature_baseline` | JSON str | Non | Stats baseline par feature |

---

### `PATCH /models/{name}/{version}` — Mettre à jour un modèle

Met à jour les métadonnées, le mode de déploiement ou les paramètres de production.

**Auth requise**

```python
# Passer en production
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=headers,
               json={"is_production": True})

# Configurer l'A/B testing
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=headers,
               json={"deployment_mode": "ab_test", "traffic_weight": 0.2})

# Mode shadow (test silencieux sans impacter les réponses)
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=headers,
               json={"deployment_mode": "shadow"})

# Configurer un seuil de confiance
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers,
               json={"confidence_threshold": 0.8})

# Ajouter une baseline pour la dérive
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers,
               json={
                   "feature_baseline": {
                       "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9}
                   }
               })

# Ajouter des tags et un webhook
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers,
               json={"tags": ["iris", "v1"], "webhook_url": "https://hooks.example.com/ml"})
```

**Schéma `ModelUpdateInput`**

| Champ | Type | Description |
|---|---|---|
| `description` | str | Nouvelle description |
| `is_production` | bool | Si `true`, les autres versions passent à `false` |
| `accuracy` | float | Score mis à jour |
| `features_count` | int | Nombre de features |
| `classes` | list | Labels des classes |
| `confidence_threshold` | float (0–1) | Seuil de confiance min pour `low_confidence` |
| `feature_baseline` | dict | Stats par feature `{nom: {mean, std, min, max}}` |
| `tags` | list[str] | Tags libres |
| `webhook_url` | str | URL appelée après chaque prédiction |
| `deployment_mode` | str | `"production"`, `"ab_test"` ou `"shadow"` |
| `traffic_weight` | float (0–1) | Part du trafic routée vers cette version |

---

### `DELETE /models/{name}/{version}` — Supprimer une version

Supprime le modèle de la base, MinIO et MLflow. Retourne 204.

```python
response = requests.delete(f"{BASE_URL}/models/iris_model/1.0.0", headers=headers)
assert response.status_code == 204
```

---

### `DELETE /models/{name}` — Supprimer toutes les versions

```python
response = requests.delete(f"{BASE_URL}/models/iris_model", headers=headers)
print(response.json())
# {"name": "iris_model", "deleted_versions": ["1.0.0", "2.0.0"],
#  "mlflow_runs_deleted": ["abc123"], "minio_objects_deleted": [...]}
```

---

### `GET /models/{name}/performance` — Performance réelle

Calcule les métriques réelles en joignant les prédictions aux résultats observés.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/performance",
    headers=headers,
    params={
        "start": "2025-01-01T00:00:00",
        "end": "2025-12-31T23:59:59",
        "version": "1.0.0",   # optionnel
        "period": "week",     # optionnel : "day", "week", "month"
    }
)
data = response.json()
```

**Schéma `ModelPerformanceResponse`** (classification)

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "period_start": "2025-01-01T00:00:00",
  "period_end": "2025-12-31T23:59:59",
  "total_predictions": 500,
  "matched_predictions": 120,
  "model_type": "classification",
  "accuracy": 0.95,
  "precision_weighted": 0.95,
  "recall_weighted": 0.94,
  "f1_weighted": 0.94,
  "confusion_matrix": [[40, 1, 0], [2, 38, 1], [0, 1, 37]],
  "classes": ["setosa", "versicolor", "virginica"],
  "per_class_metrics": {
    "setosa":     {"precision": 0.95, "recall": 0.98, "f1_score": 0.97, "support": 41},
    "versicolor": {"precision": 0.95, "recall": 0.93, "f1_score": 0.94, "support": 41},
    "virginica":  {"precision": 0.97, "recall": 0.97, "f1_score": 0.97, "support": 38}
  },
  "by_period": [
    {"period": "2025-W01", "matched_count": 30, "accuracy": 0.97, "f1_weighted": 0.96}
  ]
}
```

**Schéma `ModelPerformanceResponse`** (régression)

```json
{
  "model_type": "regression",
  "mae": 0.42,
  "mse": 0.31,
  "rmse": 0.56,
  "r2": 0.87
}
```

---

### `GET /models/{name}/drift` — Dérive des données

Rapport de dérive pour chaque feature (Z-score + PSI) par rapport à la baseline.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/drift",
    headers=headers,
    params={"days": 30, "version": "1.0.0"}
)
data = response.json()
print(data["drift_summary"])  # "ok" | "warning" | "critical" | "no_baseline"
```

**Schéma `DriftReportResponse`**

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "period_days": 30,
  "predictions_analyzed": 500,
  "baseline_available": true,
  "drift_summary": "warning",
  "features": {
    "sepal length (cm)": {
      "baseline_mean": 5.84, "baseline_std": 0.83,
      "production_mean": 6.12, "production_std": 0.91,
      "production_count": 500,
      "z_score": 1.9,
      "psi": 0.08,
      "drift_status": "warning"
    },
    "petal length (cm)": {
      "drift_status": "ok",
      "z_score": 0.3,
      "psi": 0.01
    }
  }
}
```

**Statuts de dérive**

| Statut | Signification |
|---|---|
| `ok` | Pas de dérive détectée |
| `warning` | Dérive modérée (Z-score > 1.5 ou PSI > 0.1) |
| `critical` | Dérive forte (Z-score > 2 ou PSI > 0.2) |
| `no_baseline` | Aucune baseline configurée pour ce modèle |
| `insufficient_data` | Pas assez de prédictions récentes |

---

### `GET /models/{name}/feature-importance` — Importance globale des features (SHAP agrégé)

Calcule la moyenne de `|SHAP|` par feature sur un échantillon de prédictions récentes.
Permet d'identifier les features les plus influentes du modèle en production et de détecter des dérives comportementales avant même que les métriques de performance ne bougent.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `version` | str | production | Version cible ; sinon `is_production=True`, sinon la plus récente |
| `last_n` | int | 100 | Nb de prédictions à échantillonner (max 500) |
| `days` | int | 7 | Fenêtre temporelle en jours (max 90) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/feature-importance",
    headers=headers,
    params={"version": "1.0.0", "last_n": 200, "days": 14}
)
data = response.json()

print(f"Analysé sur {data['sample_size']} prédictions")
for feat, info in sorted(
    data["feature_importance"].items(), key=lambda x: x[1]["rank"]
):
    print(f"  #{info['rank']} {feat}: mean |SHAP| = {info['mean_abs_shap']:.4f}")
```

**Schéma `FeatureImportanceResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "sample_size": 98,
  "feature_importance": {
    "petal length (cm)": { "mean_abs_shap": 0.42, "rank": 1 },
    "petal width (cm)":  { "mean_abs_shap": 0.31, "rank": 2 },
    "sepal length (cm)": { "mean_abs_shap": 0.18, "rank": 3 },
    "sepal width (cm)":  { "mean_abs_shap": 0.09, "rank": 4 }
  }
}
```

**Cas particuliers**

| Situation | Comportement |
|---|---|
| Aucune prédiction dans la fenêtre | `sample_size: 0`, `feature_importance: {}` |
| Modèle sans `feature_names_in_` | 422 — doit être entraîné avec un DataFrame pandas |
| Type de modèle non supporté par SHAP | 422 — voir `POST /explain` pour la liste des types |

> **Utilisation typique :** surveiller chaque semaine que les features les plus importantes restent stables. Un changement de classement indique souvent une dérive du comportement du modèle avant que l'accuracy ne baisse.

---

### `GET /models/{name}/history` — Historique complet

Journal de tous les changements d'état pour toutes les versions d'un modèle.

**Auth requise**

```python
response = requests.get(f"{BASE_URL}/models/iris_model/history", headers=headers)
data = response.json()
for entry in data["entries"]:
    print(f"[{entry['timestamp']}] {entry['action']} par {entry['changed_by_username']}")
```

**Schéma `ModelHistoryResponse`**

```json
{
  "model_name": "iris_model",
  "version": null,
  "total": 5,
  "entries": [
    {
      "id": 12,
      "model_name": "iris_model",
      "model_version": "1.0.0",
      "changed_by_username": "alice",
      "action": "set_production",
      "changed_fields": ["is_production"],
      "snapshot": {"is_production": true, "accuracy": 0.97},
      "timestamp": "2025-01-20T08:00:00"
    }
  ]
}
```

---

### `GET /models/{name}/{version}/history` — Historique d'une version

```python
response = requests.get(f"{BASE_URL}/models/iris_model/1.0.0/history", headers=headers)
```

Même schéma que `/models/{name}/history` mais filtré sur la version spécifiée.

---

### `POST /models/{name}/{version}/rollback/{history_id}` — Rollback

Restaure les métadonnées d'un modèle à un état précédent enregistré dans l'historique.

**Auth requise : admin**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/2.0.0/rollback/12",
    headers=headers
)
data = response.json()
print(f"Restauré vers history_id={data['rolled_back_to_history_id']}")
print(f"Champs restaurés : {data['restored_fields']}")
```

**Schéma `RollbackResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "rolled_back_to_history_id": 12,
  "new_history_id": 18,
  "restored_fields": ["is_production", "confidence_threshold"],
  "snapshot": {"is_production": false, "confidence_threshold": 0.7}
}
```

---

### `POST /models/{name}/{version}/retrain` — Ré-entraîner

Déclenche le ré-entraînement d'un modèle via son script `train.py` stocké dans MinIO.

**Auth requise : admin**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/retrain",
    headers=headers,
    json={
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "new_version": "1.1.0",      # optionnel, auto-généré si absent
        "set_production": false
    }
)
data = response.json()
print(f"Succès : {data['success']}")
print(f"Nouvelle version : {data['new_version']}")
print(data["stdout"])   # logs d'entraînement
```

**Schéma `RetrainRequest`**

| Champ | Type | Requis | Description |
|---|---|---|---|
| `start_date` | str (YYYY-MM-DD) | Oui | Date début des données |
| `end_date` | str (YYYY-MM-DD) | Oui | Date fin des données |
| `new_version` | str | Non | Version de la nouvelle sortie (auto si absent) |
| `set_production` | bool | Non | Passer en production après (défaut: false) |

**Schéma `RetrainResponse`**

```json
{
  "model_name": "iris_model",
  "source_version": "1.0.0",
  "new_version": "1.1.0",
  "success": true,
  "stdout": "Epoch 1/10 ... \n{\"accuracy\": 0.96, \"f1_score\": 0.95}",
  "stderr": "",
  "error": null,
  "new_model_metadata": { "id": 5, "name": "iris_model", "version": "1.1.0" }
}
```

> Le script `train.py` doit référencer les variables d'env `TRAIN_START_DATE`, `TRAIN_END_DATE`, `OUTPUT_MODEL_PATH`. Voir [CLAUDE.md](../CLAUDE.md) pour le contrat complet.

---

### `GET /models/{name}/ab-compare` — Rapport A/B

Comparaison côte à côte des versions en A/B test ou shadow sur une période.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/ab-compare",
    headers=headers,
    params={"days": 7}
)
data = response.json()
for v in data["versions"]:
    print(f"{v['version']} ({v['deployment_mode']}): "
          f"{v['total_predictions']} preds, error_rate={v['error_rate']:.2%}, "
          f"agreement={v['agreement_rate']}")
```

**Schéma `ABCompareResponse`**

```json
{
  "model_name": "iris_model",
  "period_days": 7,
  "versions": [
    {
      "version": "1.0.0",
      "deployment_mode": "production",
      "traffic_weight": 0.8,
      "total_predictions": 800,
      "shadow_predictions": 0,
      "error_rate": 0.01,
      "avg_response_time_ms": 12.5,
      "p95_response_time_ms": 45.0,
      "prediction_distribution": {"0": 450, "1": 200, "2": 150},
      "agreement_rate": null
    },
    {
      "version": "2.0.0",
      "deployment_mode": "shadow",
      "traffic_weight": 0.0,
      "total_predictions": 0,
      "shadow_predictions": 800,
      "error_rate": 0.005,
      "avg_response_time_ms": 9.8,
      "p95_response_time_ms": 38.0,
      "prediction_distribution": {"0": 460, "1": 195, "2": 145},
      "agreement_rate": 0.96
    }
  ]
}
```

---

## Prédictions

### `POST /predict`

Effectue une prédiction avec routage intelligent (A/B test, shadow).

**Auth requise** — contribue au quota journalier.

**Sélection de version** (ordre de priorité) :
1. `model_version` si fourni
2. Routage A/B si des versions `ab_test` sont configurées
3. Version avec `is_production=true`
4. Dernière version créée

```python
response = requests.post(
    f"{BASE_URL}/predict",
    headers=headers,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-2025-001",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
result = response.json()
```

**Schéma `PredictionInput`**

| Champ | Type | Requis | Description |
|---|---|---|---|
| `model_name` | str | Oui | Nom du modèle |
| `model_version` | str | Non | Version précise ; sinon auto-sélection |
| `id_obs` | str | Non | Identifiant d'observation |
| `features` | dict | Oui | `{"feature_name": valeur}` |

**Schéma `PredictionOutput`**

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "id_obs": "obs-2025-001",
  "prediction": 0,
  "probability": [0.97, 0.02, 0.01],
  "low_confidence": false,
  "selected_version": "1.0.0"
}
```

| Champ | Description |
|---|---|
| `prediction` | Résultat du modèle (classe ou valeur) |
| `probability` | Probabilités par classe (si `predict_proba` disponible) |
| `low_confidence` | `true` si prob max < `confidence_threshold` du modèle |
| `selected_version` | Version choisie par le routage A/B (si applicable) |

---

### `POST /predict-batch`

Prédictions en lot : le modèle est chargé une seule fois, toutes les prédictions sont persistées en une transaction.

**Auth requise**

```python
response = requests.post(
    f"{BASE_URL}/predict-batch",
    headers=headers,
    json={
        "model_name": "iris_model",
        "model_version": "1.0.0",   # optionnel
        "inputs": [
            {
                "id_obs": "obs-001",
                "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                             "petal length (cm)": 1.4, "petal width (cm)": 0.2}
            },
            {
                "id_obs": "obs-002",
                "features": {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0,
                             "petal length (cm)": 5.2, "petal width (cm)": 2.3}
            }
        ]
    }
)
data = response.json()
for item in data["predictions"]:
    print(f"{item['id_obs']} → {item['prediction']} (conf: {item['probability']})")
```

**Schéma `BatchPredictionInput`**

| Champ | Type | Requis | Description |
|---|---|---|---|
| `model_name` | str | Oui | Nom du modèle |
| `model_version` | str | Non | Version ; sinon auto-sélection |
| `inputs` | list | Oui | Liste d'items `{features, id_obs}` (min 1) |

**Schéma `BatchPredictionOutput`**

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "predictions": [
    {"id_obs": "obs-001", "prediction": 0, "probability": [0.97, 0.02, 0.01], "low_confidence": false},
    {"id_obs": "obs-002", "prediction": 2, "probability": [0.01, 0.05, 0.94], "low_confidence": false}
  ]
}
```

---

### `POST /explain`

Calcule les valeurs SHAP locales pour expliquer une prédiction.

**Auth requise**

```python
response = requests.post(
    f"{BASE_URL}/explain",
    headers=headers,
    json={
        "model_name": "iris_model",
        "model_version": "1.0.0",   # optionnel
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
data = response.json()
print(f"Prédiction : {data['prediction']}")
print(f"Base value : {data['base_value']}")
for feat, val in sorted(data["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑" if val > 0 else "↓"
    print(f"  {direction} {feat}: {val:+.4f}")
```

**Schéma `ExplainInput`**

| Champ | Type | Requis | Description |
|---|---|---|---|
| `model_name` | str | Oui | Nom du modèle |
| `model_version` | str | Non | Version ; sinon auto-sélection |
| `features` | dict | Oui | Features à expliquer |

**Schéma `ExplainOutput`**

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "prediction": 0,
  "shap_values": {
    "petal length (cm)": -1.32,
    "petal width (cm)": -0.87,
    "sepal length (cm)": -0.12,
    "sepal width (cm)": 0.05
  },
  "base_value": 1.0,
  "model_type": "tree"
}
```

> **Interprétation :** valeur SHAP positive = feature pousse vers la classe prédite, négative = pousse à l'opposé. La somme des SHAP values + `base_value` = prédiction brute.

> **Modèles supportés :** arborescents (`RandomForest`, `GradientBoosting`, `XGBoost`) via `TreeExplainer` ; linéaires (`LogisticRegression`, `LinearRegression`) via `LinearExplainer`.

---

### `GET /predictions`

Historique filtrable des prédictions avec pagination par curseur.

**Auth requise**

```python
from datetime import datetime, timedelta

params = {
    "name": "iris_model",
    "start": (datetime.now() - timedelta(days=7)).isoformat(),
    "end": datetime.now().isoformat(),
    "version": "1.0.0",        # optionnel
    "user": "alice",           # optionnel
    "limit": 50,
    "cursor": None,            # id de la dernière entrée pour la page suivante
}
response = requests.get(f"{BASE_URL}/predictions", headers=headers, params=params)
data = response.json()

# Page suivante
if data["next_cursor"]:
    params["cursor"] = data["next_cursor"]
    next_page = requests.get(f"{BASE_URL}/predictions", headers=headers, params=params)
```

**Paramètres de requête**

| Paramètre | Type | Requis | Description |
|---|---|---|---|
| `name` | str | Oui | Nom du modèle |
| `start` | datetime | Oui | Date de début (ISO 8601) |
| `end` | datetime | Oui | Date de fin (ISO 8601) |
| `version` | str | Non | Filtre sur la version |
| `user` | str | Non | Filtre sur le username |
| `limit` | int | Non | Max résultats (1-1000, défaut 100) |
| `cursor` | int | Non | ID de la dernière prédiction vue (pagination curseur) |

**Schéma `PredictionsListResponse`**

```json
{
  "total": 142,
  "limit": 50,
  "next_cursor": 1089,
  "predictions": [
    {
      "id": 1040,
      "model_name": "iris_model",
      "model_version": "1.0.0",
      "id_obs": "obs-2025-001",
      "input_features": {"sepal length (cm)": 5.1},
      "prediction_result": 0,
      "probabilities": [0.97, 0.02, 0.01],
      "response_time_ms": 12.5,
      "timestamp": "2025-01-15T14:32:00",
      "status": "success",
      "error_message": null,
      "username": "alice",
      "is_shadow": false
    }
  ]
}
```

> **Pagination curseur :** utiliser `next_cursor` (id de la dernière prédiction retournée) comme paramètre `cursor` de la requête suivante. Plus efficace que `offset` sur des volumes importants.

---

### `GET /predictions/stats`

Statistiques agrégées des prédictions par modèle.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/predictions/stats",
    headers=headers,
    params={"days": 7, "model_name": "iris_model"}  # model_name optionnel
)
data = response.json()
for stat in data["stats"]:
    print(f"{stat['model_name']}: {stat['total_predictions']} prédictions, "
          f"erreur={stat['error_rate']:.1%}, p95={stat['p95_response_time_ms']}ms")
```

**Schéma `PredictionStatsResponse`**

```json
{
  "days": 7,
  "model_name": "iris_model",
  "stats": [
    {
      "model_name": "iris_model",
      "total_predictions": 1250,
      "error_count": 5,
      "error_rate": 0.004,
      "avg_response_time_ms": 14.2,
      "p50_response_time_ms": 11.0,
      "p95_response_time_ms": 38.5
    }
  ]
}
```

---

## Résultats observés

Permet d'enregistrer les résultats réels après prédiction, pour évaluer les modèles en production.

### `POST /observed-results`

Enregistre ou met à jour des résultats observés. Idempotent sur `(id_obs, model_name)`.

**Auth requise**

```python
response = requests.post(
    f"{BASE_URL}/observed-results",
    headers=headers,
    json={
        "data": [
            {
                "id_obs": "obs-2025-001",
                "model_name": "iris_model",
                "date_time": "2025-01-16T08:00:00",
                "observed_result": 0
            },
            {
                "id_obs": "obs-2025-002",
                "model_name": "iris_model",
                "date_time": "2025-01-16T08:00:00",
                "observed_result": 1
            }
        ]
    }
)
print(response.json())  # {"upserted": 2}
```

**Schéma `ObservedResultInput`**

| Champ | Type | Description |
|---|---|---|
| `id_obs` | str | Identifiant de l'observation (lié à la prédiction) |
| `model_name` | str | Nom du modèle |
| `date_time` | datetime | Horodatage du résultat réel |
| `observed_result` | float/int/str | Résultat observé réel |

---

### `GET /observed-results`

Consulte les résultats observés avec filtres optionnels.

```python
params = {
    "model_name": "iris_model",  # optionnel
    "id_obs": "obs-2025-001",    # optionnel
    "start": "2025-01-01T00:00:00",
    "end": "2025-01-31T23:59:59",
    "limit": 100,
    "offset": 0,
}
response = requests.get(f"{BASE_URL}/observed-results", headers=headers, params=params)
data = response.json()
```

```json
{
  "total": 50,
  "limit": 100,
  "offset": 0,
  "results": [
    {
      "id": 1,
      "id_obs": "obs-2025-001",
      "model_name": "iris_model",
      "observed_result": 0,
      "date_time": "2025-01-16T08:00:00",
      "username": "alice"
    }
  ]
}
```

---

## Utilisateurs

### `POST /users` — Créer un utilisateur

**Auth requise : admin**

```python
response = requests.post(
    f"{BASE_URL}/users",
    headers=headers,
    json={
        "username": "alice",
        "email": "alice@example.com",
        "role": "user",
        "rate_limit": 5000
    }
)
user = response.json()
print(user["api_token"])  # Token à conserver — affiché une seule fois
```

**Schéma `UserCreateInput`**

| Champ | Type | Description |
|---|---|---|
| `username` | str (3-50 chars) | Nom unique |
| `email` | EmailStr | Email unique |
| `role` | str | `"admin"`, `"user"` ou `"readonly"` (défaut: `"user"`) |
| `rate_limit` | int (1-100000) | Prédictions par jour (défaut: 1000) |

**Schéma `UserResponse`**

```json
{
  "id": 3,
  "username": "alice",
  "email": "alice@example.com",
  "role": "user",
  "is_active": true,
  "rate_limit_per_day": 5000,
  "api_token": "eyJhbGciOiJIUzI1...",
  "created_at": "2025-01-15T10:00:00",
  "last_login": null
}
```

---

### `GET /users` — Lister les utilisateurs

**Auth requise : admin**

```python
response = requests.get(f"{BASE_URL}/users", headers=headers)
users = response.json()
```

---

### `GET /users/{user_id}` — Détail d'un utilisateur

Un utilisateur peut voir son propre profil. Un admin peut voir tous les profils.

```python
response = requests.get(f"{BASE_URL}/users/3", headers=headers)
```

---

### `PATCH /users/{user_id}` — Modifier un utilisateur

**Auth requise : admin**

```python
# Désactiver un compte
requests.patch(f"{BASE_URL}/users/3", headers=headers, json={"is_active": False})

# Changer le rôle et le quota
requests.patch(f"{BASE_URL}/users/3", headers=headers,
               json={"role": "readonly", "rate_limit": 100})

# Renouveler le token
response = requests.patch(
    f"{BASE_URL}/users/3",
    headers=headers,
    json={"regenerate_token": True}
)
new_token = response.json()["api_token"]
```

**Schéma `UserUpdateInput`**

| Champ | Type | Description |
|---|---|---|
| `is_active` | bool | Activer/désactiver le compte |
| `role` | str | Nouveau rôle |
| `rate_limit` | int | Nouveau quota journalier |
| `regenerate_token` | bool | Génère un nouveau token Bearer |

---

### `DELETE /users/{user_id}` — Supprimer un utilisateur

Supprime l'utilisateur et toutes ses prédictions en cascade. Retourne 204.

**Auth requise : admin**

```python
response = requests.delete(f"{BASE_URL}/users/3", headers=headers)
assert response.status_code == 204
```

---

## Monitoring

### `GET /monitoring/overview`

Tableau de bord global de santé pour tous les modèles sur une période.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/monitoring/overview",
    headers=headers,
    params={"days": 7}
)
data = response.json()
print(f"Erreurs globales : {data['global_stats']['error_rate']:.1%}")
for model in data["models"]:
    print(f"  {model['model_name']}: drift={model['drift_summary']}, "
          f"error_rate={model['error_rate']:.1%}")
```

**Schéma `GlobalDashboard`**

```json
{
  "period_days": 7,
  "global_stats": {
    "total_predictions": 5420,
    "error_count": 23,
    "error_rate": 0.004,
    "avg_response_time_ms": 13.2
  },
  "models": [
    {
      "model_name": "iris_model",
      "versions_count": 2,
      "total_predictions": 3200,
      "error_rate": 0.003,
      "drift_summary": "ok",
      "has_production_version": true
    }
  ]
}
```

---

### `GET /monitoring/model/{name}`

Détail complet du monitoring pour un modèle : timeseries, drift, A/B, erreurs récentes.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/monitoring/model/iris_model",
    headers=headers,
    params={"days": 30}
)
data = response.json()
# data["timeseries"] — points de prédiction par période
# data["drift"]      — rapport de dérive
# data["ab_compare"] — comparaison A/B si applicable
```

**Schéma `ModelDetailDashboard`**

```json
{
  "model_name": "iris_model",
  "period": {"days": 30, "start": "2025-01-01", "end": "2025-01-31"},
  "versions": [
    {
      "version": "1.0.0",
      "deployment_mode": "production",
      "total_predictions": 3200,
      "error_rate": 0.003,
      "avg_response_time_ms": 12.1
    }
  ],
  "timeseries": [
    {"timestamp": "2025-01-01T00:00:00", "count": 105, "error_count": 0, "avg_ms": 11.5}
  ],
  "drift_summary": "ok",
  "recent_errors": [
    {"timestamp": "2025-01-10T14:22:00", "error_message": "Feature manquante: petal length"}
  ]
}
```

---

## Client Python complet

```python
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List


class PredictMLClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def _get(self, path: str, params: dict = None):
        r = requests.get(f"{self.base_url}{path}", headers=self.headers, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict = None, **kwargs):
        r = requests.post(f"{self.base_url}{path}", headers=self.headers, json=json, **kwargs)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, json: dict):
        r = requests.patch(f"{self.base_url}{path}", headers=self.headers, json=json)
        r.raise_for_status()
        return r.json()

    # ── Prédictions ──────────────────────────────────────────────────────────

    def predict(self, model_name: str, features: Dict[str, Any],
                model_version: Optional[str] = None, id_obs: Optional[str] = None) -> dict:
        return self._post("/predict", json={
            "model_name": model_name, "model_version": model_version,
            "id_obs": id_obs, "features": features,
        })

    def predict_batch(self, model_name: str, inputs: List[dict],
                      model_version: Optional[str] = None) -> dict:
        return self._post("/predict-batch", json={
            "model_name": model_name, "model_version": model_version, "inputs": inputs,
        })

    def explain(self, model_name: str, features: Dict[str, Any],
                model_version: Optional[str] = None) -> dict:
        return self._post("/explain", json={
            "model_name": model_name, "model_version": model_version, "features": features,
        })

    def get_predictions(self, model_name: str, days: int = 7,
                        version: Optional[str] = None, limit: int = 100,
                        cursor: Optional[int] = None) -> dict:
        return self._get("/predictions", params={
            "name": model_name,
            "start": (datetime.now() - timedelta(days=days)).isoformat(),
            "end": datetime.now().isoformat(),
            "version": version, "limit": limit, "cursor": cursor,
        })

    def get_stats(self, days: int = 7, model_name: Optional[str] = None) -> dict:
        return self._get("/predictions/stats", params={"days": days, "model_name": model_name})

    # ── Modèles ───────────────────────────────────────────────────────────────

    def upload_model(self, pkl_path: str, name: str, version: str, **metadata) -> dict:
        with open(pkl_path, "rb") as f:
            files = {"file": (pkl_path, f, "application/octet-stream")}
            data = {"name": name, "version": version,
                    **{k: str(v) for k, v in metadata.items()}}
            r = requests.post(f"{self.base_url}/models", headers=self.headers,
                              files=files, data=data)
            r.raise_for_status()
            return r.json()

    def set_production(self, name: str, version: str) -> dict:
        return self._patch(f"/models/{name}/{version}", json={"is_production": True})

    def configure_ab_test(self, name: str, version: str, traffic_weight: float) -> dict:
        return self._patch(f"/models/{name}/{version}",
                           json={"deployment_mode": "ab_test", "traffic_weight": traffic_weight})

    def set_shadow(self, name: str, version: str) -> dict:
        return self._patch(f"/models/{name}/{version}", json={"deployment_mode": "shadow"})

    def get_performance(self, name: str, start: str, end: str,
                        version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/performance",
                         params={"start": start, "end": end, "version": version})

    def get_drift(self, name: str, days: int = 30, version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/drift", params={"days": days, "version": version})

    def get_ab_compare(self, name: str, days: int = 7) -> dict:
        return self._get(f"/models/{name}/ab-compare", params={"days": days})

    def retrain(self, name: str, version: str, start_date: str, end_date: str,
                new_version: Optional[str] = None, set_production: bool = False) -> dict:
        return self._post(f"/models/{name}/{version}/retrain", json={
            "start_date": start_date, "end_date": end_date,
            "new_version": new_version, "set_production": set_production,
        })

    # ── Résultats observés ────────────────────────────────────────────────────

    def submit_observed_results(self, records: List[dict]) -> dict:
        return self._post("/observed-results", json={"data": records})

    # ── Monitoring ────────────────────────────────────────────────────────────

    def get_overview(self, days: int = 7) -> dict:
        return self._get("/monitoring/overview", params={"days": days})

    def get_model_dashboard(self, name: str, days: int = 30) -> dict:
        return self._get(f"/monitoring/model/{name}", params={"days": days})

    # ── Divers ────────────────────────────────────────────────────────────────

    def get_models(self, tag: Optional[str] = None) -> list:
        return self._get("/models", params={"tag": tag} if tag else None)

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health").json()


# ── Exemple d'utilisation ─────────────────────────────────────────────────────

client = PredictMLClient("http://localhost:8000", "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA")

# Prédiction simple
result = client.predict(
    "iris_model",
    {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
     "petal length (cm)": 1.4, "petal width (cm)": 0.2},
    id_obs="obs-001"
)
print(f"Prédiction: {result['prediction']}, conf: {result['probability']}")

# Prédiction en lot
batch = client.predict_batch("iris_model", [
    {"id_obs": "obs-001", "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                                        "petal length (cm)": 1.4, "petal width (cm)": 0.2}},
    {"id_obs": "obs-002", "features": {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0,
                                        "petal length (cm)": 5.2, "petal width (cm)": 2.3}},
])

# Explication SHAP
explanation = client.explain(
    "iris_model",
    {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
     "petal length (cm)": 1.4, "petal width (cm)": 0.2}
)
print(f"Top feature: {max(explanation['shap_values'], key=lambda k: abs(explanation['shap_values'][k]))}")

# Upload + mise en production
client.upload_model("models/rf_v2.pkl", "iris_model", "2.0.0",
                    algorithm="RandomForestClassifier", accuracy=0.98)
client.set_production("iris_model", "2.0.0")

# Résultats observés
client.submit_observed_results([
    {"id_obs": "obs-001", "model_name": "iris_model",
     "date_time": "2025-01-16T08:00:00", "observed_result": 0},
])

# Performance réelle
perf = client.get_performance("iris_model", "2025-01-01T00:00:00", "2025-12-31T23:59:59")
print(f"Accuracy réelle: {perf.get('accuracy')}")

# Dérive
drift = client.get_drift("iris_model", days=30)
print(f"Drift: {drift['drift_summary']}")

# Monitoring global
overview = client.get_overview(days=7)
print(f"Taux d'erreur global: {overview['global_stats']['error_rate']:.1%}")
```

---

## Codes d'erreur courants

| Code | Situation |
|---|---|
| 400 | Corps de requête invalide (champ manquant, format incorrect) |
| 401 | Token Bearer absent ou invalide |
| 403 | Rôle insuffisant (ex: action admin avec rôle user) |
| 404 | Modèle, utilisateur ou history_id introuvable |
| 409 | Conflit (modèle `name+version` déjà existant) |
| 422 | Erreur de validation Pydantic |
| 429 | Quota journalier de prédictions dépassé |
| 500 | Erreur serveur interne |
| 503 | Service indisponible (Redis, MinIO, DB inaccessible) |
