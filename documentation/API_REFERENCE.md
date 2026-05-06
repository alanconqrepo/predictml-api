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
  "parent_version": null,
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
| `alert_thresholds` | dict | Seuils d'alerte spécifiques au modèle (ex: `{"error_rate": 0.05, "drift_zscore": 2.0}`) — surcharge les seuils globaux |

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

Le rapport couvre **4 dimensions de monitoring** :
1. **Dérive de distribution** (Z-score sur la moyenne, PSI sur la distribution)
2. **Dérive de performance** (accuracy/MAE vs baseline)
3. **Dérive de taux d'erreur** (HTTP 500 et erreurs de prédiction)
4. **Null rate** — taux de valeurs nulles par feature (`null_rate_current` vs `null_rate_baseline`)

Voir aussi `GET /models/{name}/output-drift` pour la dérive de distribution des sorties (label shift).

---

### `GET /models/{name}/output-drift` — Drift de distribution des sorties

Détecte le **label shift** : compare la distribution récente des prédictions à la distribution de référence (`training_stats.label_distribution`).

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `period_days` | int | 7 | Fenêtre d'analyse (max 90) |
| `model_version` | str | production | Version cible |
| `min_predictions` | int | 30 | Nb minimal de prédictions pour calculer (sinon `insufficient_data`) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/output-drift",
    headers=headers,
    params={"period_days": 7, "min_predictions": 50}
)
drift = response.json()

print(f"Statut : {drift['status']}")  # ok | warning | critical | insufficient_data | no_reference
print(f"PSI : {drift['psi']:.4f}")
for cls, data in drift["class_distribution"].items():
    print(f"  {cls}: référence={data['reference']:.2%}, récent={data['recent']:.2%}")
```

**Schéma `OutputDriftResponse`**

```json
{
  "model_name": "iris_model",
  "model_version": "2.0.0",
  "period_days": 7,
  "predictions_analyzed": 420,
  "status": "ok",
  "psi": 0.045,
  "class_distribution": {
    "setosa":     {"reference": 0.33, "recent": 0.35, "delta": 0.02},
    "versicolor": {"reference": 0.34, "recent": 0.32, "delta": -0.02},
    "virginica":  {"reference": 0.33, "recent": 0.33, "delta": 0.00}
  }
}
```

**Seuils PSI**

| PSI | Statut |
|---|---|
| < 0.1 | `ok` |
| 0.1 – 0.2 | `warning` |
| ≥ 0.2 | `critical` |

> Un `status: critical` déclenche une alerte webhook (`output_drift_critical`) et peut déclencher un retrain si `trigger_on_drift` est configuré dans le schedule.

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
  "auto_promoted": null,
  "auto_promote_reason": null,
  "new_model_metadata": {
    "id": 5,
    "name": "iris_model",
    "version": "1.1.0",
    "parent_version": "1.0.0",
    "training_stats": {
      "trained_at": "2026-04-25T03:00:00",
      "train_start_date": "2026-03-26",
      "train_end_date": "2026-04-25",
      "n_rows": 12450,
      "feature_stats": {"sepal_length": {"mean": 5.8, "std": 0.83}},
      "label_distribution": {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
    }
  }
}
```

> Le script `train.py` doit référencer les variables d'env `TRAIN_START_DATE`, `TRAIN_END_DATE`, `OUTPUT_MODEL_PATH`. Voir [CLAUDE.md](../CLAUDE.md) pour le contrat complet.

---

### `PATCH /models/{name}/{version}/schedule` — Planifier le ré-entraînement automatique

Configure un planning cron pour déclencher automatiquement le ré-entraînement d'une version.
Le scheduler APScheduler charge tous les plannings actifs au démarrage de l'API.

**Auth requise : admin**

```python
response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0/schedule",
    headers=headers,
    json={
        "cron": "0 3 * * 1",    # chaque lundi à 03h00 UTC
        "lookback_days": 30,     # TRAIN_START_DATE = today - 30j
        "auto_promote": False,   # évaluer la promotion_policy après retrain
        "enabled": True
    }
)
data = response.json()
print(f"Prochain déclenchement : {data['retrain_schedule']['next_run_at']}")
```

**Schéma `RetrainScheduleInput`**

| Champ | Type | Défaut | Description |
|---|---|---|---|
| `cron` | str | null | Expression cron 5 champs (ex : `"0 3 * * 1"`) |
| `lookback_days` | int ≥ 1 | 30 | Fenêtre d'historique passée au script (jours) |
| `auto_promote` | bool | false | Évaluer la `promotion_policy` après chaque retrain |
| `enabled` | bool | true | `false` = pause sans effacer le planning |
| `trigger_on_drift` | `"warning"` \| `"critical"` \| null | null | Niveau de drift déclenchant un retrain réactif (sans attendre le cron) |
| `drift_retrain_cooldown_hours` | int ≥ 1 | 24 | Cooldown minimal entre deux retrains drift-triggered (évite les boucles) |

**Schéma `ScheduleUpdateResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "retrain_schedule": {
    "cron": "0 3 * * 1",
    "lookback_days": 30,
    "auto_promote": false,
    "enabled": true,
    "last_run_at": null,
    "next_run_at": "2026-04-21T03:00:00"
  }
}
```

> Si `cron` est invalide, l'API retourne HTTP 422 avec le détail de l'erreur.  
> Si `enabled=True` sans `cron`, HTTP 422 est retourné.  
> Pour désactiver sans effacer le planning : `{"cron": "0 3 * * 1", "enabled": false}`.

---

### `GET /models/{name}/ab-compare` — Rapport A/B avec significativité statistique

Comparaison côte à côte des versions en A/B test ou shadow sur une période, enrichie d'un test
de significativité statistique automatique entre les deux versions les plus actives.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `days` | int | 30 | Fenêtre d'analyse en jours (1–90) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/ab-compare",
    headers=headers,
    params={"days": 7}
)
data = response.json()

# Métriques brutes par version
for v in data["versions"]:
    print(f"{v['version']} ({v['deployment_mode']}): "
          f"{v['total_predictions']} preds, error_rate={v['error_rate']:.2%}, "
          f"agreement={v['agreement_rate']}")

# Significativité statistique
sig = data.get("ab_significance")
if sig:
    status = "✅ significative" if sig["significant"] else "⚠️  non significative"
    print(f"\nDifférence {status} (p={sig['p_value']:.4f}, test={sig['test']})")
    if sig["winner"]:
        print(f"Meilleure version : {sig['winner']}")
    if sig["current_samples"][sig["winner"] or list(sig["current_samples"])[0]] < sig["min_samples_needed"]:
        print(f"⚠️  Données insuffisantes — {sig['min_samples_needed']} observations/version recommandées")
```

**Schéma `ABCompareResponse`**

```json
{
  "model_name": "iris_model",
  "period_days": 7,
  "versions": [
    {
      "version": "1.0.0",
      "deployment_mode": "ab_test",
      "traffic_weight": 0.8,
      "total_predictions": 800,
      "shadow_predictions": 0,
      "error_rate": 0.05,
      "avg_response_time_ms": 12.5,
      "p95_response_time_ms": 45.0,
      "prediction_distribution": {"0": 450, "1": 200, "2": 150},
      "agreement_rate": null
    },
    {
      "version": "2.0.0",
      "deployment_mode": "ab_test",
      "traffic_weight": 0.2,
      "total_predictions": 200,
      "shadow_predictions": 0,
      "error_rate": 0.01,
      "avg_response_time_ms": 9.8,
      "p95_response_time_ms": 38.0,
      "prediction_distribution": {"0": 115, "1": 50, "2": 35},
      "agreement_rate": null
    }
  ],
  "ab_significance": {
    "metric": "error_rate",
    "test": "chi2",
    "p_value": 0.008,
    "significant": true,
    "confidence_level": 0.95,
    "winner": "2.0.0",
    "min_samples_needed": 120,
    "current_samples": {"1.0.0": 800, "2.0.0": 200}
  }
}
```

**Champs `ab_significance`**

| Champ | Type | Description |
|---|---|---|
| `metric` | str | Métrique testée : `"error_rate"` ou `"response_time_ms"` |
| `test` | str | Test utilisé : `"chi2"` (erreur) ou `"mann_whitney_u"` (latence) |
| `p_value` | float | Valeur p du test statistique |
| `significant` | bool | `true` si `p_value < 1 - confidence_level` |
| `confidence_level` | float | Seuil de confiance (défaut `0.95`) |
| `winner` | str \| null | Version avec la meilleure métrique, `null` si égalité exacte |
| `min_samples_needed` | int | Observations/version recommandées pour détecter cet effet (puissance 80 %) |
| `current_samples` | dict | Nombre d'observations disponibles par version |

> **Logique de sélection du test :**  
> Chi-² si au moins une erreur est observée dans l'un des groupes (tableau de contingence succès/erreur).  
> Fallback Mann-Whitney U sur les temps de réponse si aucune erreur n'est présente.  
> `ab_significance: null` si moins de 2 versions actives ou données insuffisantes.

---

### `GET /models/{name}/shadow-compare` — Rapport shadow vs production

Comparaison enrichie entre le modèle shadow et le modèle de production : accuracy, latence, taux de désaccord et recommandation de promotion.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `days` | int | 7 | Fenêtre d'analyse |
| `shadow_version` | str | auto | Version shadow cible (sinon la première version `shadow` active) |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/shadow-compare",
    headers=headers,
    params={"days": 14}
)
cmp = response.json()

print(f"Production v{cmp['production_version']}: accuracy={cmp['production_accuracy']:.2%}")
print(f"Shadow v{cmp['shadow_version']}: accuracy={cmp['shadow_accuracy']:.2%}")
print(f"Delta accuracy: {cmp['accuracy_delta']:+.2%}")
print(f"Taux de désaccord: {cmp['disagreement_rate']:.2%}")
print(f"Recommandation: {cmp['recommendation']}")
```

**Schéma `ShadowCompareResponse`**

```json
{
  "model_name": "iris_model",
  "period_days": 14,
  "production_version": "1.0.0",
  "shadow_version": "2.0.0",
  "predictions_analyzed": 1240,
  "production_accuracy": 0.94,
  "shadow_accuracy": 0.97,
  "accuracy_delta": 0.03,
  "production_avg_latency_ms": 14.2,
  "shadow_avg_latency_ms": 12.8,
  "latency_delta_ms": -1.4,
  "confidence_delta": 0.02,
  "disagreement_rate": 0.08,
  "recommendation": "promote"
}
```

| `recommendation` | Signification |
|---|---|
| `promote` | Le shadow est meilleur sur toutes les métriques — promouvoir en production |
| `keep_shadow` | Métriques mitigées ou données insuffisantes — continuer l'observation |
| `no_shadow` | Aucun modèle shadow actif |

---

### `GET /models/{name}/{version}/card` — Model Card

Fiche récapitulative d'un modèle en un seul appel : métadonnées, performance réelle, drift, calibration, top features SHAP, info retrain et couverture ground truth.

**Auth requise**

```python
# JSON (défaut)
response = requests.get(
    f"{BASE_URL}/models/iris_model/2.0.0/card",
    headers=headers
)
card = response.json()
print(f"Modèle : {card['name']} v{card['version']}")
print(f"Accuracy réelle : {card['performance']['accuracy']:.2%}")
print(f"Drift : {card['drift']['summary']}")
print(f"Top feature : {card['feature_importance'][0]['feature']}")

# Markdown — prêt à partager ou insérer dans une PR
response_md = requests.get(
    f"{BASE_URL}/models/iris_model/2.0.0/card",
    headers={"Authorization": f"Bearer {TOKEN}", "Accept": "text/markdown"}
)
with open("model_card.md", "w") as f:
    f.write(response_md.text)
```

**Schéma `ModelCardResponse`**

```json
{
  "name": "iris_model",
  "version": "2.0.0",
  "description": "Classifieur Iris — 3 espèces",
  "algorithm": "RandomForestClassifier",
  "trained_by": "alice",
  "training_date": "2026-03-01T03:00:00",
  "parent_version": "1.0.0",
  "performance": {
    "accuracy": 0.97,
    "f1_score": 0.96,
    "matched_predictions": 920,
    "total_predictions": 1240
  },
  "drift": {
    "summary": "ok",
    "features_in_warning": [],
    "features_in_critical": []
  },
  "calibration": {
    "brier_score": 0.042,
    "overconfidence_gap": 0.031
  },
  "feature_importance": [
    {"feature": "petal length (cm)", "mean_abs_shap": 0.42, "rank": 1},
    {"feature": "petal width (cm)",  "mean_abs_shap": 0.31, "rank": 2}
  ],
  "retrain": {
    "last_retrain_at": "2026-04-01T03:00:00",
    "trained_by": "scheduler",
    "n_rows": 12450
  },
  "ground_truth_coverage": 0.74
}
```

---

### `GET /models/{name}/confidence-trend` — Tendance de confiance

Retourne l'évolution de la probabilité de confiance maximale moyenne sur une période, par fenêtre temporelle.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `version` | str | production | Version cible |
| `days` | int | 30 | Fenêtre d'analyse en jours |
| `granularity` | str | `"day"` | Granularité : `"hour"`, `"day"`, `"week"` |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/confidence-trend",
    headers=headers,
    params={"days": 14, "granularity": "day"}
)
data = response.json()

for point in data["trend"]:
    print(f"{point['period']}: conf_avg={point['avg_confidence']:.3f}, "
          f"low_conf_rate={point['low_confidence_rate']:.1%}")
```

**Schéma `ConfidenceTrendResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "days": 14,
  "granularity": "day",
  "trend": [
    {
      "period": "2026-04-11",
      "prediction_count": 142,
      "avg_confidence": 0.91,
      "low_confidence_rate": 0.04
    },
    {
      "period": "2026-04-12",
      "prediction_count": 158,
      "avg_confidence": 0.88,
      "low_confidence_rate": 0.07
    }
  ]
}
```

> Une baisse progressive de `avg_confidence` sans dégradation d'accuracy peut indiquer que le modèle rencontre des observations de plus en plus proches des frontières de décision — signe précoce de dérive.

---

### `POST /models/{name}/{version}/warmup` — Préchauffage du cache

Précharge le modèle dans le cache Redis sans attendre la première requête de prédiction. Réduit la latence à froid lors des déploiements.

**Auth requise**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/warmup",
    headers=headers
)
data = response.json()
print(f"Chargé en {data['load_time_ms']:.1f} ms — {data['status']}")
```

**Schéma `WarmupResponse`**

```json
{
  "model_name": "iris_model",
  "version": "1.0.0",
  "status": "loaded",
  "load_time_ms": 42.3,
  "cached": true
}
```

| `status` | Signification |
|---|---|
| `loaded` | Modèle chargé et mis en cache |
| `already_cached` | Modèle déjà présent dans le cache Redis |
| `error` | Échec du chargement (voir `detail`) |

---

## Prédictions

### `POST /models/{name}/{version}/validate-input` — Validation du schéma d'entrée

Valide les features d'entrée contre le schéma attendu d'une version de modèle, **sans effectuer de prédiction**.

**Auth requise**

Détecte :
- **features manquantes** — présentes dans le modèle, absentes dans la requête
- **features inattendues** — présentes dans la requête, absentes dans le modèle
- **coercitions de type** — valeurs `string` convertibles en `float` (avertissement non bloquant)

La source de vérité est, par priorité : `feature_names_in_` du modèle sklearn chargé, puis les clés de `feature_baseline` stockées en DB.

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/validate-input",
    headers=headers,
    json={
        "petal_length": 5.1,
        "petal_width": 1.8,
        "sepal_length": 6.3
        # sepal_width manquant
    }
)
print(response.json())
```

```json
{
  "valid": false,
  "errors": [
    { "type": "missing_feature",    "feature": "sepal_width" },
    { "type": "unexpected_feature", "feature": "petal_width_squared" }
  ],
  "warnings": [
    { "type": "type_coercion", "feature": "petal_length", "from_type": "string", "to_type": "float" }
  ],
  "expected_features": ["petal_length", "petal_width", "sepal_length", "sepal_width"]
}
```

| Champ | Description |
|---|---|
| `valid` | `true` seulement si `errors` est vide |
| `errors` | Liste d'erreurs bloquantes (`missing_feature`, `unexpected_feature`) |
| `warnings` | Avertissements non bloquants (`type_coercion`) |
| `expected_features` | Liste triée des features attendues ; `null` si aucun schéma disponible |

---

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

**Query parameter `explain`** (optionnel, défaut `false`) :

Ajouter `?explain=true` pour recevoir les valeurs SHAP directement dans la réponse, sans appel séparé à `POST /explain`.

```python
response = requests.post(
    f"{BASE_URL}/predict?explain=true",
    headers=headers,
    json={
        "model_name": "iris_model",
        "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                     "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    }
)
data = response.json()
# data["explanation"]["shap_values"] contient les valeurs SHAP locales
if data.get("explanation"):
    for feat, val in data["explanation"]["shap_values"].items():
        print(f"  {feat}: {val:+.4f}")
```

Le champ `explanation` suit le schéma `ExplainOutput` (voir `POST /explain`). Si le modèle ne supporte pas SHAP, `explanation` est `null`.

**Query parameter `strict_validation`** (optionnel, défaut `false`) :

Ajouter `?strict_validation=true` pour rejeter les requêtes avec des features **inattendues** (en plus des features manquantes déjà vérifiées par défaut). Retourne un `422` structuré si la validation échoue.

```python
# Rejette si features inattendues présentes
response = requests.post(
    f"{BASE_URL}/predict?strict_validation=true",
    headers=headers,
    json={
        "model_name": "iris_model",
        "features": {"sepal_length": 5.1, "extra_col": 99.0, ...}
    }
)
# → 422 avec detail.errors listant les features inattendues
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
| `id_obs` | str | Non | Filtre par identifiant d'observation |
| `min_confidence` | float (0–1) | Non | Confiance minimale (max des probabilités) |
| `max_confidence` | float (0–1) | Non | Confiance maximale (max des probabilités) |
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

### `GET /predictions/{prediction_id}` — Consulter une prédiction par ID

Retourne le détail complet d'une prédiction à partir de son identifiant interne.

**Auth requise**

```python
prediction_id = 1040

response = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}",
    headers=headers
)
data = response.json()
print(f"Modèle : {data['model_name']} v{data['model_version']}")
print(f"Résultat : {data['prediction_result']}")
print(f"Latence : {data['response_time_ms']} ms")
```

Retourne le même schéma qu'un élément de `GET /predictions` (voir ci-dessus). Retourne `404` si la prédiction n'existe pas ou appartient à un autre utilisateur (non admin).

---

### `GET /predictions/{prediction_id}/explain` — Explication SHAP post-hoc

Génère a posteriori l'explication SHAP d'une prédiction existante, en rechargeant les features depuis la base de données.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}/explain",
    headers=headers
)
data = response.json()
print(f"Prédiction : {data['prediction']}")
for feat, val in sorted(data["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {'↑' if val > 0 else '↓'} {feat}: {val:+.4f}")
```

Retourne le même schéma que `POST /explain`. Retourne `404` si la prédiction n'existe pas, `422` si le modèle ne supporte pas SHAP.

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

### `GET /predictions/anomalies` — Détection d'anomalies

Retourne les prédictions récentes dont au moins une feature présente un z-score anormal par rapport à la baseline du modèle.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `model_name` | str | Oui | Nom du modèle |
| `days` | int | 7 | Fenêtre d'analyse (max 90) |
| `z_threshold` | float | 3.0 | Seuil z-score au-dessus duquel une feature est considérée aberrante |
| `limit` | int | 200 | Nombre max de résultats (max 1000) |

```python
response = requests.get(
    f"{BASE_URL}/predictions/anomalies",
    headers=headers,
    params={"model_name": "iris_model", "days": 7, "z_threshold": 3.0}
)
data = response.json()

print(f"{data['anomaly_count']} prédictions anormales sur {data['total_analyzed']} analysées")
for pred in data["anomalies"]:
    print(f"  ID {pred['prediction_id']} ({pred['timestamp'][:10]}):")
    for feat in pred["anomalous_features"]:
        print(f"    {feat['feature']}: z={feat['z_score']:.1f} "
              f"(valeur={feat['value']}, baseline_mean={feat['baseline_mean']:.2f})")
```

**Schéma `AnomaliesResponse`**

```json
{
  "model_name": "iris_model",
  "period_days": 7,
  "z_threshold": 3.0,
  "total_analyzed": 840,
  "anomaly_count": 5,
  "anomalies": [
    {
      "prediction_id": 1082,
      "timestamp": "2026-04-25T14:32:00",
      "model_version": "2.0.0",
      "prediction_result": 1,
      "anomalous_features": [
        {
          "feature": "sepal length (cm)",
          "value": 12.4,
          "z_score": 7.8,
          "baseline_mean": 5.84,
          "baseline_std": 0.83
        }
      ]
    }
  ]
}
```

> Retourne `{"error": "no_baseline"}` si `feature_baseline` n'est pas configurée pour ce modèle.

---

### `DELETE /predictions/purge` — Purge RGPD

Supprime les prédictions antérieures à N jours. `dry_run=true` par défaut — aucune suppression sans confirmation explicite.

**Auth requise : admin**

**Paramètres de requête**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `older_than_days` | int | Oui | Supprimer les prédictions > N jours |
| `model_name` | str | Non | Limiter la purge à un seul modèle |
| `dry_run` | bool | `true` | Simuler sans supprimer |

```python
# Simuler (dry_run par défaut)
response = requests.delete(
    f"{BASE_URL}/predictions/purge",
    headers=headers,
    params={"older_than_days": 90}
)
data = response.json()
print(f"Seraient supprimées : {data['deleted_count']} prédictions")
print(f"Résultats observés liés : {data['linked_observed_results_count']}")

# Purger réellement
response = requests.delete(
    f"{BASE_URL}/predictions/purge",
    headers=headers,
    params={"older_than_days": 90, "model_name": "iris_model", "dry_run": "false"}
)
```

**Schéma `PurgeResponse`**

```json
{
  "dry_run": false,
  "deleted_count": 12450,
  "oldest_remaining": "2026-01-15T08:32:00",
  "models_affected": ["iris_model", "wine"],
  "linked_observed_results_count": 3
}
```

> `linked_observed_results_count > 0` indique que des prédictions liées à des `observed_results` seront supprimées — perte de données de performance historiques.

---

## Golden Tests

Tests de régression pour valider qu'un modèle produit toujours les sorties attendues sur des cas de référence. Particulièrement utile après un ré-entraînement.

### `GET /models/{name}/golden-tests` — Lister les cas de test

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/golden-tests",
    headers=headers
)
tests = response.json()

for t in tests:
    print(f"#{t['id']} [{t.get('description', '—')}] "
          f"→ attendu: {t['expected_output']}")
```

**Schéma `GoldenTestResponse`**

```json
[
  {
    "id": 1,
    "model_name": "iris_model",
    "description": "Iris setosa typique",
    "input_features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                       "petal length (cm)": 1.4, "petal width (cm)": 0.2},
    "expected_output": "setosa",
    "created_at": "2026-04-01T10:00:00",
    "created_by": "alice"
  }
]
```

---

### `POST /models/{name}/golden-tests` — Créer un cas de test

**Auth requise**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/golden-tests",
    headers=headers,
    json={
        "input_features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        },
        "expected_output": "setosa",
        "description": "Iris setosa typique — toutes features nominales"
    }
)
print(response.json()["id"])  # id du nouveau cas
```

| Champ | Type | Requis | Description |
|---|---|---|---|
| `input_features` | dict | Oui | Features d'entrée du cas |
| `expected_output` | str/int/float | Oui | Sortie attendue du modèle |
| `description` | str | Non | Description du cas de test |

---

### `DELETE /models/{name}/golden-tests/{test_id}` — Supprimer un cas

**Auth requise : admin**

```python
response = requests.delete(
    f"{BASE_URL}/models/iris_model/golden-tests/1",
    headers=headers
)
assert response.status_code == 204
```

---

### `POST /models/{name}/golden-tests/upload-csv` — Import en lot depuis CSV

**Auth requise : admin**

**Format du CSV**

```
description,input_features,expected_output
"Iris setosa typique","{""sepal length (cm)"": 5.1, ""sepal width (cm)"": 3.5, ""petal length (cm)"": 1.4, ""petal width (cm)"": 0.2}",setosa
"Iris virginica robuste","{""sepal length (cm)"": 6.7, ""sepal width (cm)"": 3.0, ""petal length (cm)"": 5.2, ""petal width (cm)"": 2.3}",virginica
```

```python
import io, csv

rows = [
    {"description": "Iris setosa",
     "input_features": '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}',
     "expected_output": "setosa"},
    {"description": "Iris virginica",
     "input_features": '{"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}',
     "expected_output": "virginica"},
]

buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=["description", "input_features", "expected_output"])
writer.writeheader()
writer.writerows(rows)
buf.seek(0)

response = requests.post(
    f"{BASE_URL}/models/iris_model/golden-tests/upload-csv",
    headers=headers,
    files={"file": ("tests.csv", buf, "text/csv")}
)
result = response.json()
print(f"{result['imported']} cas importés")
if result.get("errors"):
    for err in result["errors"]:
        print(f"  ❌ Ligne {err['row']}: {err['reason']}")
```

---

### `POST /models/{name}/{version}/run-golden-tests` — Exécuter les tests

Exécute tous les cas de test enregistrés pour un modèle sur une version donnée.

**Auth requise**

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/2.0.0/run-golden-tests",
    headers=headers
)
result = response.json()

print(f"✅ {result['passed']} / {result['total_tests']} tests passés "
      f"({result['pass_rate']:.1%})")

for detail in result["details"]:
    icon = "✅" if detail["passed"] else "❌"
    desc = detail.get("description", f"Test #{detail['test_id']}")
    print(f"  {icon} {desc}")
    if not detail["passed"]:
        print(f"      attendu: {detail['expected']!r} | reçu: {detail['actual']!r}")
```

**Schéma `GoldenTestRunResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "total_tests": 5,
  "passed": 4,
  "failed": 1,
  "pass_rate": 0.8,
  "details": [
    {
      "test_id": 1,
      "description": "Iris setosa typique",
      "input": {"sepal length (cm)": 5.1},
      "expected": "setosa",
      "actual": "setosa",
      "passed": true
    },
    {
      "test_id": 3,
      "description": "Cas limite versicolor",
      "input": {"sepal length (cm)": 6.0},
      "expected": "versicolor",
      "actual": "virginica",
      "passed": false
    }
  ]
}
```

> L'intégration avec `promotion_policy.min_golden_test_pass_rate` permet de bloquer l'auto-promotion si le taux de réussite est insuffisant.

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

### `GET /observed-results/export` — Export CSV/JSON

Exporte les résultats observés filtrés dans un fichier téléchargeable.

**Auth requise**

**Paramètres de requête**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `model_name` | str | Non | Filtre sur le modèle |
| `start` | datetime | Non | Date de début |
| `end` | datetime | Non | Date de fin |
| `format` | str | `"csv"` | `"csv"` ou `"json"` |

```python
response = requests.get(
    f"{BASE_URL}/observed-results/export",
    headers=headers,
    params={"model_name": "iris_model", "format": "csv"}
)

with open("observed_results.csv", "wb") as f:
    f.write(response.content)
# Colonnes : id_obs, model_name, observed_result, date_time, username
```

---

### `GET /observed-results/stats` — Statistiques de couverture

Retourne des statistiques sur la couverture ground truth : combien de prédictions ont un résultat observé associé.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/observed-results/stats",
    headers=headers,
    params={"model_name": "iris_model", "days": 30}
)
data = response.json()
print(f"Prédictions : {data['total_predictions']}")
print(f"Avec ground truth : {data['matched_count']}")
print(f"Couverture : {data['coverage_rate']:.1%}")
```

**Schéma `ObservedResultsStatsResponse`**

```json
{
  "model_name": "iris_model",
  "days": 30,
  "total_predictions": 1500,
  "matched_count": 420,
  "coverage_rate": 0.28,
  "unmatched_count": 1080
}
```

---

### `POST /observed-results/upload-csv` — Import CSV en lot

Importe un fichier CSV de résultats observés. Idempotent sur `(id_obs, model_name)`.

**Auth requise**

**Format du CSV**

```
id_obs,model_name,date_time,observed_result
obs-001,iris_model,2026-01-15T10:00:00,0
obs-002,iris_model,2026-01-15T10:01:00,2
```

```python
import io

csv_content = """id_obs,model_name,date_time,observed_result
obs-001,iris_model,2026-01-15T10:00:00,0
obs-002,iris_model,2026-01-15T10:01:00,2
obs-003,iris_model,2026-01-15T10:02:00,1
"""

response = requests.post(
    f"{BASE_URL}/observed-results/upload-csv",
    headers=headers,
    files={"file": ("results.csv", io.StringIO(csv_content), "text/csv")}
)
print(response.json())
```

**Schéma `CSVUploadResponse`**

```json
{
  "upserted": 3,
  "errors": []
}
```

Si certaines lignes sont invalides, elles sont listées dans `errors` et les lignes valides sont quand même insérées.

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

## Modèles — Endpoints complémentaires

### `GET /models/leaderboard` — Classement des modèles

Classe les modèles en production par métrique sur une fenêtre glissante. Résultat mis en cache (TTL configurable).

**Auth non requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `metric` | str | `accuracy` | Métrique de classement : `accuracy`, `f1_score`, `latency_p95_ms`, `predictions_count` |
| `days` | int | 30 | Fenêtre temporelle |
| `top_n` | int | 10 | Nombre de modèles retournés |

```python
response = requests.get(
    f"{BASE_URL}/models/leaderboard",
    params={"metric": "accuracy", "days": 30, "top_n": 5}
)
leaderboard = response.json()

for i, entry in enumerate(leaderboard["entries"], 1):
    print(f"#{i} {entry['model_name']} v{entry['version']} — "
          f"accuracy={entry.get('accuracy')}, p95={entry.get('latency_p95_ms')}ms")
```

**Schéma `LeaderboardResponse`**

```json
{
  "metric": "accuracy",
  "days": 30,
  "entries": [
    {
      "model_name": "iris_model",
      "version": "2.0.0",
      "is_production": true,
      "accuracy": 0.97,
      "f1_score": 0.96,
      "latency_p95_ms": 14.2,
      "predictions_count": 8450,
      "last_prediction_at": "2026-04-27T18:32:00"
    }
  ]
}
```

---

### `GET /models/{name}/performance-timeline` — Timeline de performance

Évolution des métriques de performance version par version, triée par date de déploiement.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `days` | int | 90 | Fenêtre temporelle |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/performance-timeline",
    headers=headers,
    params={"days": 90}
)
timeline = response.json()

for entry in timeline["versions"]:
    print(f"v{entry['version']} déployée le {entry['deployed_at']} — "
          f"accuracy={entry.get('accuracy')}, MAE={entry.get('mae')}")
```

**Schéma**

```json
{
  "model_name": "iris_model",
  "period_days": 90,
  "versions": [
    {
      "version": "1.0.0",
      "deployed_at": "2026-01-15T00:00:00",
      "accuracy": 0.94,
      "f1_score": 0.93,
      "mae": null,
      "predictions_count": 4200
    },
    {
      "version": "2.0.0",
      "deployed_at": "2026-03-01T00:00:00",
      "accuracy": 0.97,
      "f1_score": 0.96,
      "mae": null,
      "predictions_count": 8450
    }
  ]
}
```

---

### `GET /models/{name}/calibration` — Calibration des probabilités

Mesure la qualité de calibration des probabilités prédites : un modèle parfaitement calibré retourne 70% de confiance quand il a raison 70% du temps.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `days` | int | 30 | Fenêtre temporelle |
| `version` | str | production | Version cible |
| `bins` | int | 10 | Nombre de bins pour la courbe de reliability |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/calibration",
    headers=headers,
    params={"days": 30, "version": "2.0.0"}
)
cal = response.json()

print(f"Brier score: {cal['brier_score']:.4f}")  # 0 = parfait, 1 = pire
print(f"Gap de surconfiance: {cal['overconfidence_gap']:+.3f}")
# Positif = modèle trop confiant, négatif = pas assez confiant

for b in cal["reliability_diagram"]:
    print(f"  conf={b['bin_center']:.1f} → réalité={b['fraction_positive']:.2f} "
          f"(n={b['count']})")
```

**Schéma `CalibrationResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "period_days": 30,
  "sample_size": 920,
  "brier_score": 0.042,
  "overconfidence_gap": 0.031,
  "reliability_diagram": [
    {"bin_center": 0.05, "mean_predicted": 0.04, "fraction_positive": 0.02, "count": 48},
    {"bin_center": 0.15, "mean_predicted": 0.14, "fraction_positive": 0.11, "count": 72},
    {"bin_center": 0.85, "mean_predicted": 0.86, "fraction_positive": 0.91, "count": 235},
    {"bin_center": 0.95, "mean_predicted": 0.96, "fraction_positive": 0.94, "count": 187}
  ]
}
```

> **Interprétation :** Un `brier_score` < 0.1 est bon pour la classification. Un `overconfidence_gap` > 0.05 signale que le modèle surestime sa certitude — à surveiller avant un déploiement haute-criticité.

---

### `GET /models/{name}/confidence-distribution` — Distribution de confiance

Histogramme du niveau de confiance (`max(probabilities)`) sur les prédictions récentes. Permet d'identifier la proportion de prédictions incertaines.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `days` | int | 7 | Fenêtre temporelle |
| `version` | str | production | Version cible |
| `bins` | int | 10 | Résolution de l'histogramme |
| `high_threshold` | float | 0.9 | Seuil confiance haute |
| `uncertain_threshold` | float | 0.6 | Seuil incertitude |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/confidence-distribution",
    headers=headers,
    params={"days": 7, "uncertain_threshold": 0.7}
)
dist = response.json()

print(f"Prédictions très confiantes (>{dist['high_threshold']}) : "
      f"{dist['high_confidence_pct']:.1%}")
print(f"Prédictions incertaines (<{dist['uncertain_threshold']}) : "
      f"{dist['uncertain_pct']:.1%}")
```

**Schéma `ConfidenceDistributionResponse`**

```json
{
  "model_name": "iris_model",
  "version": "2.0.0",
  "period_days": 7,
  "total_predictions": 1240,
  "high_threshold": 0.9,
  "uncertain_threshold": 0.6,
  "high_confidence_pct": 0.82,
  "uncertain_pct": 0.04,
  "bins": [
    {"lower": 0.0, "upper": 0.1, "count": 3, "pct": 0.002},
    {"lower": 0.9, "upper": 1.0, "count": 1018, "pct": 0.821}
  ]
}
```

---

### `GET /models/{name}/performance-report` — Rapport consolidé

Agrège en un seul appel : performance, drift, feature importance, calibration, et comparaison A/B.
Idéal pour des scripts de monitoring automatique ou des alertes programmatiques.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `days` | int | 30 | Fenêtre temporelle commune à toutes les composantes |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/performance-report",
    headers=headers,
    params={"days": 30}
)
report = response.json()

# Vérification globale en une ligne
if (report["drift"]["drift_summary"] == "critical" or
        report["performance"].get("accuracy", 1.0) < 0.85):
    print("⚠️  Action requise sur iris_model")
```

**Schéma `PerformanceReportResponse`**

```json
{
  "model_name": "iris_model",
  "generated_at": "2026-04-28T10:00:00",
  "period_days": 30,
  "performance": {"accuracy": 0.97, "f1_score": 0.96, "matched_predictions": 920},
  "drift": {"drift_summary": "ok", "features": {}},
  "feature_importance": {"petal length (cm)": {"mean_abs_shap": 0.42, "rank": 1}},
  "calibration": {"brier_score": 0.042, "overconfidence_gap": 0.031},
  "ab_compare": null
}
```

---

### `GET /models/{name}/readiness` — Vérification de disponibilité

Vérifie qu'un modèle satisfait tous les prérequis avant d'être passé en production.

**Auth requise**

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/readiness",
    headers=headers
)
check = response.json()

if check["ready"]:
    print("✅ Modèle prêt pour la production")
else:
    for name, ok in check["checks"].items():
        if not ok:
            print(f"❌ {name} : non satisfait")
```

**Schéma `ReadinessResponse`**

```json
{
  "model_name": "iris_model",
  "ready": false,
  "checks": {
    "is_production": false,
    "file_accessible": true,
    "baseline_computed": false,
    "no_critical_drift": true
  }
}
```

| Check | Description |
|---|---|
| `is_production` | `is_production=True` sur au moins une version |
| `file_accessible` | Fichier `.pkl` accessible dans MinIO |
| `baseline_computed` | `feature_baseline` calculée (nécessaire pour le drift) |
| `no_critical_drift` | Aucun drift critique détecté dans la fenêtre récente |

---

### `GET /models/{name}/retrain-history` — Historique des ré-entraînements

Journal structuré de tous les événements de retrain pour un modèle : manuel ou planifié.

**Auth requise**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `limit` | int | 20 | Nombre d'entrées |
| `offset` | int | 0 | Pagination |

```python
response = requests.get(
    f"{BASE_URL}/models/iris_model/retrain-history",
    headers=headers,
    params={"limit": 10}
)
history = response.json()

print(f"Total retrains : {history['total']}")
for entry in history["history"]:
    promoted = "✅" if entry["auto_promoted"] else "⏭️"
    print(f"{promoted} {entry['timestamp'][:10]} "
          f"v{entry['source_version']} → v{entry['new_version']} "
          f"(accuracy={entry.get('accuracy')}, by={entry['trained_by']})")
```

**Schéma `RetrainHistoryResponse`**

```json
{
  "model_name": "iris_model",
  "total": 8,
  "history": [
    {
      "timestamp": "2026-04-01T03:00:00",
      "source_version": "1.0.0",
      "new_version": "1.1.0",
      "trained_by": "scheduler",
      "accuracy": 0.95,
      "f1_score": 0.94,
      "auto_promoted": true,
      "auto_promote_reason": "all criteria met",
      "n_rows": 12450,
      "train_start_date": "2026-03-01",
      "train_end_date": "2026-04-01"
    }
  ]
}
```

---

### `PATCH /models/{name}/{version}/deprecate` — Déprécier une version

Marque une version comme dépréciée. Les nouvelles prédictions sur cette version retournent **HTTP 410 Gone**.

**Auth requise : admin**

```python
response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0/deprecate",
    headers=headers
)
print(response.json())
# {"model_name": "iris_model", "version": "1.0.0",
#  "deprecated_at": "2026-04-28T10:00:00",
#  "message": "Version dépréciée. Les nouvelles prédictions sont bloquées."}
```

> **Note :** La dépréciation est irréversible via cet endpoint. Pour restaurer une version, utiliser `POST /models/{name}/{version}/rollback/{history_id}`.

---

### `PATCH /models/{name}/policy` — Politique d'auto-promotion post-retrain

Définit les critères que doit satisfaire un modèle retrained pour être promu automatiquement en production.

**Auth requise : admin**

**Corps de la requête**

```json
{
  "min_accuracy": 0.90,
  "max_latency_p95_ms": 200.0,
  "min_sample_validation": 50,
  "auto_promote": true
}
```

| Champ | Type | Défaut | Description |
|---|---|---|---|
| `min_accuracy` | float [0–1] | null | Précision minimale (sur les observed_results récents) |
| `max_latency_p95_ms` | float > 0 | null | Latence P95 maximale en ms |
| `min_sample_validation` | int ≥ 1 | 10 | Nb minimal de paires (prédiction, résultat) pour évaluer |
| `auto_promote` | bool | false | Activer l'auto-promotion post-retrain |
| `min_golden_test_pass_rate` | float [0–1] | null | Taux de réussite minimal des golden tests avant promotion |
| `auto_demote` | bool | false | Activer l'auto-demotion (circuit breaker) |
| `demote_on_drift` | `"warning"` \| `"critical"` \| null | null | Niveau de drift déclenchant la demotion automatique |
| `demote_on_accuracy_below` | float [0–1] | null | Accuracy minimale sous laquelle le modèle est démis |
| `demote_cooldown_hours` | int ≥ 1 | 24 | Délai minimal entre deux demotions automatiques |

```python
# Auto-promotion post-retrain
response = requests.patch(
    f"{BASE_URL}/models/iris_model/policy",
    headers=headers,
    json={
        "min_accuracy": 0.90,
        "max_latency_p95_ms": 200,
        "min_sample_validation": 50,
        "min_golden_test_pass_rate": 0.95,
        "auto_promote": True
    }
)

# Circuit breaker — auto-demotion si drift critique ou accuracy trop basse
response = requests.patch(
    f"{BASE_URL}/models/iris_model/policy",
    headers=headers,
    json={
        "auto_demote": True,
        "demote_on_drift": "critical",
        "demote_on_accuracy_below": 0.80,
        "demote_cooldown_hours": 48
    }
)
print(f"Politique activée : {response.json()['auto_promote']}")
```

> La politique est évaluée automatiquement à la fin de chaque ré-entraînement (auto-promotion) et lors de chaque cycle de supervision toutes les 6h (auto-demotion). Le résultat est retourné dans la réponse de `POST /models/{name}/{version}/retrain` via les champs `auto_promoted` et `auto_promote_reason`.

---

### `GET /models/{name}/{version}/download` — Télécharger le fichier .pkl

Télécharge le fichier modèle sérialisé depuis MinIO.

**Auth requise**

```python
import pathlib

response = requests.get(
    f"{BASE_URL}/models/iris_model/2.0.0/download",
    headers=headers,
    stream=True
)
response.raise_for_status()

output_path = pathlib.Path("iris_model_v2.0.0.pkl")
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Modèle téléchargé : {output_path} ({output_path.stat().st_size / 1024:.1f} Ko)")
```

La réponse est un flux binaire (`Content-Type: application/octet-stream`) avec en-tête `Content-Disposition: attachment; filename=iris_model_2.0.0.pkl`.

---

## Prédictions — Endpoints complémentaires

### `GET /predictions/export` — Export streaming

Exporte l'historique des prédictions en CSV, JSONL, ou Parquet. Utilise une pagination curseur en interne pour gérer les grands volumes sans surcharger la mémoire.

**Auth requise : admin**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `format` | str | `csv` | Format : `csv`, `jsonl`, `parquet` |
| `model_name` | str | — | Filtre par modèle (optionnel) |
| `start` | datetime | — | Début de la plage (ISO 8601) |
| `end` | datetime | — | Fin de la plage (ISO 8601) |
| `version` | str | — | Filtre par version (optionnel) |

```python
from datetime import datetime, timedelta

response = requests.get(
    f"{BASE_URL}/predictions/export",
    headers=headers,
    params={
        "format": "csv",
        "model_name": "iris_model",
        "start": (datetime.now() - timedelta(days=30)).isoformat(),
        "end": datetime.now().isoformat()
    },
    stream=True
)
response.raise_for_status()

with open("predictions_export.csv", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Pour Parquet (binaire, compatible pandas/polars)
response_parquet = requests.get(
    f"{BASE_URL}/predictions/export",
    headers=headers,
    params={"format": "parquet", "model_name": "iris_model"},
    stream=True
)
with open("predictions.parquet", "wb") as f:
    for chunk in response_parquet.iter_content(chunk_size=8192):
        f.write(chunk)

import pandas as pd
df = pd.read_parquet("predictions.parquet")
print(df.head())
```

---

## Utilisateurs — Endpoints complémentaires

### `GET /users/me` — Profil de l'utilisateur courant

Retourne le profil de l'utilisateur propriétaire du token Bearer utilisé.

**Auth requise**

```python
response = requests.get(f"{BASE_URL}/users/me", headers=headers)
me = response.json()
print(f"Connecté en tant que {me['username']} (rôle : {me['role']})")
```

**Schéma**

```json
{
  "id": 3,
  "username": "alice",
  "email": "alice@example.com",
  "role": "user",
  "is_active": true,
  "rate_limit_per_day": 5000,
  "created_at": "2026-01-10T09:00:00",
  "last_login": "2026-04-28T08:12:00"
}
```

---

### `GET /users/me/quota` — Quota journalier

Retourne la consommation de quota de l'utilisateur courant pour la journée en cours.

**Auth requise**

```python
response = requests.get(f"{BASE_URL}/users/me/quota", headers=headers)
quota = response.json()

print(f"Quota : {quota['used_today']} / {quota['rate_limit_per_day']} prédictions")
print(f"Restant : {quota['remaining']} — Réinitialisation à {quota['reset_at']}")
```

**Schéma `QuotaResponse`**

```json
{
  "rate_limit_per_day": 5000,
  "used_today": 342,
  "remaining": 4658,
  "reset_at": "2026-04-29T00:00:00"
}
```

---

### `GET /users/{user_id}/usage` — Statistiques d'utilisation

Statistiques de consommation d'un utilisateur : volume par modèle et par jour. Accessible par l'utilisateur lui-même ou par un admin.

**Auth requise (self ou admin)**

**Paramètres**

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `days` | int | 30 | Fenêtre temporelle |

```python
response = requests.get(
    f"{BASE_URL}/users/3/usage",
    headers=headers,
    params={"days": 30}
)
usage = response.json()

print(f"Total 30j : {usage['total_predictions']} prédictions")
for model in usage["by_model"]:
    print(f"  {model['model_name']}: {model['predictions']} pred, "
          f"{model['errors']} erreurs")
```

**Schéma `UserUsageResponse`**

```json
{
  "user_id": 3,
  "username": "alice",
  "period_days": 30,
  "total_predictions": 8420,
  "by_model": [
    {"model_name": "iris_model", "predictions": 6200, "errors": 12},
    {"model_name": "fraud_detector", "predictions": 2220, "errors": 3}
  ],
  "by_day": [
    {"date": "2026-04-28", "predictions": 342},
    {"date": "2026-04-27", "predictions": 489}
  ]
}
```

---

## Infrastructure

### `GET /health/dependencies` — Santé détaillée des dépendances

Vérifie la connectivité et la latence de chaque service dépendant. Utile pour le diagnostic en production et les health checks orchestrateurs (K8s readiness probe).

**Auth non requise**

```python
response = requests.get(f"{BASE_URL}/health/dependencies")
health = response.json()

print(f"Statut global : {health['status']}")
for service, info in health["dependencies"].items():
    status_icon = "✅" if info["status"] == "ok" else "❌"
    latency = f" ({info.get('latency_ms', '?')}ms)" if "latency_ms" in info else ""
    print(f"  {status_icon} {service}{latency}")
```

**Schéma `DependencyHealthResponse`**

```json
{
  "status": "ok",
  "dependencies": {
    "database": {"status": "ok", "latency_ms": 2.1},
    "redis":    {"status": "ok", "latency_ms": 0.4},
    "minio":    {"status": "ok", "latency_ms": 5.8},
    "mlflow":   {"status": "degraded", "latency_ms": null, "error": "Connection timeout"}
  }
}
```

| Statut | Signification |
|---|---|
| `ok` | Service joignable et fonctionnel |
| `degraded` | Service joignable mais lent ou erreur partielle |
| `unavailable` | Service inaccessible |

---

### `GET /metrics` — Métriques Prometheus

Expose les métriques de l'API au format texte Prometheus. Scraped automatiquement par Grafana LGTM via le dashboard `http://localhost:3000`.

**Auth optionnelle** — si `METRICS_TOKEN` est défini dans les variables d'environnement, le token Bearer est requis.

```python
# Sans token (si METRICS_TOKEN non configuré)
response = requests.get(f"{BASE_URL}/metrics")
print(response.text[:500])  # format texte Prometheus

# Avec token
response = requests.get(
    f"{BASE_URL}/metrics",
    headers={"Authorization": "Bearer <METRICS_TOKEN>"}
)
```

**Métriques exposées (exemples)**

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/predict",status="200"} 18420

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.05",endpoint="/predict"} 17832

# HELP predictions_total Total predictions by model
# TYPE predictions_total counter
predictions_total{model="iris_model",version="2.0.0"} 12540
```

> Pour configurer le scraping Grafana, ajouter l'endpoint dans `prometheus.yml` :
> ```yaml
> - job_name: predictml
>   static_configs:
>     - targets: ['api:8000']
>   metrics_path: /metrics
> ```

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

    def get_shadow_compare(self, name: str, days: int = 7,
                           shadow_version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/shadow-compare",
                         params={"days": days, "shadow_version": shadow_version})

    def get_output_drift(self, name: str, period_days: int = 7,
                         model_version: Optional[str] = None) -> dict:
        return self._get(f"/models/{name}/output-drift",
                         params={"period_days": period_days, "model_version": model_version})

    def get_model_card(self, name: str, version: str) -> dict:
        return self._get(f"/models/{name}/{version}/card")

    def get_anomalies(self, model_name: str, days: int = 7,
                      z_threshold: float = 3.0, limit: int = 200) -> dict:
        return self._get("/predictions/anomalies",
                         params={"model_name": model_name, "days": days,
                                 "z_threshold": z_threshold, "limit": limit})

    def retrain(self, name: str, version: str, start_date: str, end_date: str,
                new_version: Optional[str] = None, set_production: bool = False) -> dict:
        return self._post(f"/models/{name}/{version}/retrain", json={
            "start_date": start_date, "end_date": end_date,
            "new_version": new_version, "set_production": set_production,
        })

    # ── Golden Tests ──────────────────────────────────────────────────────────

    def list_golden_tests(self, model_name: str) -> list:
        return self._get(f"/models/{model_name}/golden-tests")

    def create_golden_test(self, model_name: str, input_features: Dict[str, Any],
                           expected_output, description: Optional[str] = None) -> dict:
        return self._post(f"/models/{model_name}/golden-tests", json={
            "input_features": input_features,
            "expected_output": expected_output,
            "description": description,
        })

    def run_golden_tests(self, model_name: str, version: str) -> dict:
        return self._post(f"/models/{model_name}/{version}/run-golden-tests")

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

client = PredictMLClient("http://localhost:8000", "<ADMIN_TOKEN>")

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
