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

Vérifie la connectivité à la base de données et l'état du cache modèles.

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

Liste tous les modèles actifs avec leurs métadonnées.

```python
response = requests.get(f"{BASE_URL}/models")
models = response.json()
```

**Réponse** : liste d'objets modèle avec `name`, `version`, `algorithm`, `accuracy`, `is_production`, etc.

---

### `GET /models/cached`

Liste les modèles actuellement chargés en mémoire.

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
response = requests.get(f"{BASE_URL}/models/iris_model/1.0")
model = response.json()
print(model["model_type"])       # "RandomForestClassifier"
print(model["feature_names"])    # ["sepal length (cm)", ...]
print(model["model_loaded"])     # True
```

**Schéma de réponse `ModelGetResponse`**

```python
{
  "id": 1,
  "name": "iris_model",
  "version": "1.0",
  "description": "Classifieur Iris — 3 espèces",
  "algorithm": "RandomForestClassifier",
  "mlflow_run_id": "abc123def456",       # optionnel
  "minio_bucket": "models",
  "minio_object_key": "iris_model/1.0/model.pkl",
  "file_size_bytes": 24576,
  "file_hash": "sha256:...",
  "accuracy": 0.97,
  "precision": 0.97,
  "recall": 0.96,
  "f1_score": 0.97,
  "features_count": 4,
  "classes": [0, 1, 2],
  "training_params": {"n_estimators": 100, "max_depth": 5},
  "training_dataset": "iris_train_2024.csv",
  "trained_by": "alice",
  "training_date": "2024-01-15T10:30:00",
  "is_active": True,
  "is_production": True,
  "created_at": "2024-01-15T10:35:00",
  "updated_at": "2024-01-20T08:00:00",
  "deprecated_at": None,
  "creator_username": "alice",
  "model_loaded": True,
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

files = {"file": ("iris_model.pkl", open("iris_model.pkl", "rb"), "application/octet-stream")}
data = {
    "name": "iris_model",
    "version": "1.0",
    "description": "Classifieur Iris — 3 espèces",
    "algorithm": "RandomForestClassifier",
    "accuracy": "0.97",
    "f1_score": "0.97",
    "features_count": "4",
    "classes": '["setosa", "versicolor", "virginica"]',       # JSON string
    "training_params": '{"n_estimators": 100, "max_depth": 5}', # JSON string
    "training_dataset": "iris_train_2024.csv",
}

response = requests.post(
    f"{BASE_URL}/models",
    headers=headers,
    files=files,
    data=data,
)
print(response.status_code)  # 201
print(response.json()["id"])
```

**Avec un run MLflow (sans fichier `.pkl`)**

```python
data = {
    "name": "iris_model",
    "version": "2.0",
    "mlflow_run_id": "abc123def456789",
}
response = requests.post(f"{BASE_URL}/models", headers=headers, data=data)
```

**Champs du formulaire**

| Champ | Type | Requis | Description |
|---|---|---|---|
| `name` | str | Oui | Nom unique du modèle |
| `version` | str | Oui | Version (ex: "1.0", "2024-01") |
| `file` | fichier `.pkl` | Si pas de MLflow | Fichier sérialisé |
| `description` | str | Non | Description lisible |
| `algorithm` | str | Non | Nom de l'algo (ex: "RandomForestClassifier") |
| `mlflow_run_id` | str | Non | ID de run MLflow |
| `accuracy` | float | Non | Score de précision |
| `f1_score` | float | Non | Score F1 |
| `features_count` | int | Non | Nombre de features |
| `classes` | JSON str | Non | Labels des classes `["A","B"]` |
| `training_params` | JSON str | Non | Hyperparamètres `{"n": 100}` |
| `training_dataset` | str | Non | Nom du dataset d'entraînement |

---

### `PATCH /models/{name}/{version}` — Mettre à jour un modèle

Met à jour les métadonnées ou passe un modèle en production.

**Auth requise**

```python
# Passer un modèle en production
response = requests.patch(
    f"{BASE_URL}/models/iris_model/2.0",
    headers=headers,
    json={"is_production": True}
)

# Mettre à jour les métriques
response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0",
    headers=headers,
    json={
        "description": "Version améliorée",
        "accuracy": 0.98,
        "features_count": 4,
        "classes": [0, 1, 2]
    }
)
```

**Schéma `ModelUpdateInput`**

| Champ | Type | Description |
|---|---|---|
| `description` | str | Nouvelle description |
| `is_production` | bool | Si `true`, les autres versions passent à `false` |
| `accuracy` | float | Score mis à jour |
| `features_count` | int | Nombre de features |
| `classes` | list | Labels des classes |

---

### `DELETE /models/{name}/{version}` — Supprimer une version

Supprime le modèle de la base, MinIO et MLflow. Retourne 204.

```python
response = requests.delete(
    f"{BASE_URL}/models/iris_model/1.0",
    headers=headers
)
assert response.status_code == 204
```

---

### `DELETE /models/{name}` — Supprimer toutes les versions

```python
response = requests.delete(
    f"{BASE_URL}/models/iris_model",
    headers=headers
)
print(response.json())
```

```json
{
  "name": "iris_model",
  "deleted_versions": ["1.0", "2.0"],
  "mlflow_runs_deleted": ["abc123"],
  "minio_objects_deleted": ["iris_model/1.0/model.pkl", "iris_model/2.0/model.pkl"]
}
```

---

## Prédictions

### `POST /predict`

Effectue une prédiction avec le modèle spécifié.

**Auth requise** — contribue au quota journalier.

**Sélection de version** (ordre de priorité) :
1. `model_version` si fourni
2. Version avec `is_production=true`
3. Dernière version créée

```python
response = requests.post(
    f"{BASE_URL}/predict",
    headers=headers,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-2024-001",        # optionnel, pour lier au résultat observé
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
  "model_version": "1.0",
  "id_obs": "obs-2024-001",
  "prediction": 0,
  "probability": [0.97, 0.02, 0.01]
}
```

> `probability` est présent uniquement si le modèle supporte `predict_proba`.

---

### `GET /predictions`

Historique filtrable des prédictions.

**Auth requise**

```python
from datetime import datetime, timedelta

params = {
    "name": "iris_model",
    "start": (datetime.now() - timedelta(days=7)).isoformat(),
    "end": datetime.now().isoformat(),
    "version": "1.0",        # optionnel
    "user": "alice",         # optionnel
    "limit": 50,
    "offset": 0,
}
response = requests.get(f"{BASE_URL}/predictions", headers=headers, params=params)
data = response.json()
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
| `offset` | int | Non | Pagination (défaut 0) |

**Schéma `PredictionsListResponse`**

```json
{
  "total": 142,
  "limit": 50,
  "offset": 0,
  "predictions": [
    {
      "id": 1,
      "model_name": "iris_model",
      "model_version": "1.0",
      "id_obs": "obs-2024-001",
      "input_features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
      "prediction_result": 0,
      "probabilities": [0.97, 0.02, 0.01],
      "response_time_ms": 12.5,
      "timestamp": "2024-01-15T14:32:00",
      "status": "success",
      "error_message": null,
      "username": "alice"
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
                "id_obs": "obs-2024-001",
                "model_name": "iris_model",
                "date_time": "2024-01-16T08:00:00",
                "observed_result": 0
            },
            {
                "id_obs": "obs-2024-002",
                "model_name": "iris_model",
                "date_time": "2024-01-16T08:00:00",
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
    "model_name": "iris_model",
    "start": "2024-01-01T00:00:00",
    "end": "2024-01-31T23:59:59",
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
      "id_obs": "obs-2024-001",
      "model_name": "iris_model",
      "observed_result": 0,
      "date_time": "2024-01-16T08:00:00",
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
  "created_at": "2024-01-15T10:00:00",
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
requests.patch(f"{BASE_URL}/users/3", headers=headers, json={"role": "readonly", "rate_limit": 100})

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

## Client Python complet

Exemple d'un client réutilisable pour PredictML API.

```python
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List


class PredictMLClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def _get(self, path: str, params: dict = None):
        response = requests.get(f"{self.base_url}{path}", headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, json: dict = None, **kwargs):
        response = requests.post(f"{self.base_url}{path}", headers=self.headers, json=json, **kwargs)
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        model_name: str,
        features: Dict[str, Any],
        model_version: Optional[str] = None,
        id_obs: Optional[str] = None,
    ) -> dict:
        return self._post("/predict", json={
            "model_name": model_name,
            "model_version": model_version,
            "id_obs": id_obs,
            "features": features,
        })

    def get_predictions(
        self,
        model_name: str,
        days: int = 7,
        version: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        return self._get("/predictions", params={
            "name": model_name,
            "start": (datetime.now() - timedelta(days=days)).isoformat(),
            "end": datetime.now().isoformat(),
            "version": version,
            "limit": limit,
        })

    def upload_model(self, pkl_path: str, name: str, version: str, **metadata) -> dict:
        with open(pkl_path, "rb") as f:
            files = {"file": (pkl_path, f, "application/octet-stream")}
            data = {"name": name, "version": version, **{k: str(v) for k, v in metadata.items()}}
            response = requests.post(
                f"{self.base_url}/models",
                headers=self.headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()

    def set_production(self, name: str, version: str) -> dict:
        response = requests.patch(
            f"{self.base_url}/models/{name}/{version}",
            headers=self.headers,
            json={"is_production": True},
        )
        response.raise_for_status()
        return response.json()

    def submit_observed_results(self, records: List[dict]) -> dict:
        return self._post("/observed-results", json={"data": records})

    def get_models(self) -> list:
        return self._get("/models")

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health").json()


# Usage
client = PredictMLClient("http://localhost:8000", "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA")

# Prédiction
result = client.predict(
    model_name="iris_model",
    features={"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
    id_obs="obs-001"
)
print(f"Prédiction: {result['prediction']}, Probabilités: {result['probability']}")

# Upload d'un modèle
client.upload_model(
    pkl_path="models/random_forest_v2.pkl",
    name="iris_model",
    version="2.0",
    algorithm="RandomForestClassifier",
    accuracy=0.98,
    f1_score=0.98,
    features_count=4,
)

# Passer en production
client.set_production("iris_model", "2.0")

# Enregistrer les résultats réels
client.submit_observed_results([
    {"id_obs": "obs-001", "model_name": "iris_model", "date_time": "2024-01-16T08:00:00", "observed_result": 0},
])
```

---

## Codes d'erreur courants

| Code | Situation |
|---|---|
| 400 | Corps de requête invalide (champ manquant, format incorrect) |
| 401 | Token Bearer absent ou invalide |
| 403 | Rôle insuffisant (ex: action admin avec rôle user) |
| 404 | Modèle ou utilisateur introuvable |
| 409 | Conflit (modèle `name+version` déjà existant) |
| 422 | Erreur de validation Pydantic |
| 429 | Quota journalier de prédictions dépassé |
| 500 | Erreur serveur interne |
