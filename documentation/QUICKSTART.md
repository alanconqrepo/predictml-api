# Quickstart — PredictML API

Guide de démarrage complet : de l'installation à votre premier workflow ML en production.

---

## Prérequis

- **Python 3.10+** (pour les scripts locaux)
- **Docker Desktop** avec Docker Compose v2
- **Git**

```bash
python --version   # 3.10+
docker --version   # Docker version 24+
docker compose version  # Docker Compose version v2+
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/alanconqrepo/predictml-api.git
cd predictml-api

# 2. Lancer tous les services (API, DB, MinIO, MLflow, Redis, Grafana, Streamlit)
docker-compose up -d --build

# 3. Initialiser la base de données (premier déploiement uniquement)
docker exec predictml-api python init_data/init_db.py
```

---

## Vérifier que tout fonctionne

```bash
# Health check de l'API
curl http://localhost:8000/health
# {"status": "ok", "models_available": 2, "models_cached": 1}

# Lister les modèles disponibles
curl http://localhost:8000/models
```

**Services disponibles**

| Service | URL | Identifiants |
|---|---|---|
| API | http://localhost:8000 | — |
| Swagger UI | http://localhost:8000/docs | — |
| Dashboard admin | http://localhost:8501 | token admin (voir ci-dessous) |
| MLflow | http://localhost:5000 | — |
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |
| Grafana | http://localhost:3000 | admin / admin |

**Token admin par défaut :** `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA`

---

## Workflow complet — De l'entraînement à la prédiction

### Étape 1 : Entraîner un modèle localement

```python
# train_iris.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Charger les données
iris = load_iris()
X, y = iris.data, iris.target

# Entraîner
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

# Sauvegarder
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Modèle sauvegardé : iris_model.pkl")
```

```bash
pip install scikit-learn
python train_iris.py
# Accuracy: 1.0000, F1: 1.0000
# Modèle sauvegardé : iris_model.pkl
```

### Étape 2 : Uploader le modèle via l'API

```python
# upload_model.py
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

with open("iris_model.pkl", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/models",
        headers=HEADERS,
        files={"file": ("iris_model.pkl", f, "application/octet-stream")},
        data={
            "name": "iris_model",
            "version": "1.0.0",
            "description": "Classifieur Iris — RandomForest",
            "algorithm": "RandomForestClassifier",
            "accuracy": "1.0",
            "f1_score": "1.0",
            "features_count": "4",
            "classes": '["setosa", "versicolor", "virginica"]',
        },
    )

print(response.status_code)  # 201
print(response.json()["id"])
```

### Étape 3 : Passer le modèle en production

```python
response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0",
    headers=HEADERS,
    json={"is_production": True}
)
print(response.json()["is_production"])  # True
```

### Étape 4 : Faire des prédictions

```python
# predict.py
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Prédiction unitaire
response = requests.post(
    f"{BASE_URL}/predict",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-001",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
result = response.json()
print(f"Prédiction : {result['prediction']}")        # 0 (setosa)
print(f"Probabilités : {result['probability']}")     # [0.97, 0.02, 0.01]
print(f"Faible confiance : {result['low_confidence']}")  # None (pas de seuil configuré)

# Prédictions en lot
batch_response = requests.post(
    f"{BASE_URL}/predict-batch",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "inputs": [
            {"id_obs": "obs-001", "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}},
            {"id_obs": "obs-002", "features": {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}},
        ]
    }
)
for item in batch_response.json()["predictions"]:
    print(f"  {item['id_obs']} → {item['prediction']}")
```

### Étape 5 : Enregistrer les résultats observés

Après que vous connaissez le vrai résultat, envoyez-le pour évaluer la performance réelle du modèle :

```python
response = requests.post(
    f"{BASE_URL}/observed-results",
    headers=HEADERS,
    json={
        "data": [
            {"id_obs": "obs-001", "model_name": "iris_model",
             "date_time": "2025-01-16T08:00:00", "observed_result": 0},
            {"id_obs": "obs-002", "model_name": "iris_model",
             "date_time": "2025-01-16T08:00:00", "observed_result": 2},
        ]
    }
)
print(response.json())  # {"upserted": 2}
```

### Étape 6 : Consulter les performances réelles

```python
perf = requests.get(
    f"{BASE_URL}/models/iris_model/performance",
    headers=HEADERS,
    params={"start": "2025-01-01T00:00:00", "end": "2025-12-31T23:59:59"}
)
data = perf.json()
print(f"Accuracy réelle : {data['accuracy']}")
print(f"F1 réel : {data['f1_weighted']}")
```

### Étape 7 : Consulter le dashboard Streamlit

Ouvrez **http://localhost:8501** et connectez-vous avec le token admin.

Le dashboard permet de :
- **Utilisateurs** — créer, désactiver, renouveler les tokens
- **Modèles** — voir les détails, passer en production, lien MLflow
- **Prédictions** — historique filtrable avec les features et résultats
- **Stats** — graphiques de volume, temps de réponse, taux d'erreur
- **Code Example** — exemples générés pour MLflow + API
- **A/B Testing** — shadow mode, comparaison statistique, décision de promotion
- **Supervision** — monitoring global, drift, alertes, tendances de performance
- **Retrain** — planifier et déclencher les ré-entraînements, consulter les logs

---

## Utiliser l'API directement (curl)

```bash
export TOKEN="ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"

# Statut
curl http://localhost:8000/

# Lister les modèles
curl http://localhost:8000/models

# Prédiction unitaire
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris_model",
    "id_obs": "obs-42",
    "features": {
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }
  }'

# Explicabilité SHAP
curl -X POST http://localhost:8000/explain \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris_model",
    "features": {
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }
  }'

# Dérive des données
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/models/iris_model/drift"

# Performance réelle
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/models/iris_model/performance?start=2025-01-01T00:00:00&end=2025-12-31T23:59:59"

# Monitoring global
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/monitoring/overview"

# Historique des prédictions (pagination curseur)
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/predictions?name=iris_model&start=2025-01-01T00:00:00&end=2025-12-31T23:59:59"

# Créer un utilisateur (admin)
curl -X POST http://localhost:8000/users \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "email": "alice@example.com", "role": "user", "rate_limit": 500}'

# Passer un modèle en production
curl -X PATCH http://localhost:8000/models/iris_model/1.0.0 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_production": true}'

# Renouveler un token (admin)
curl -X PATCH http://localhost:8000/users/2 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"regenerate_token": true}'
```

---

## Lancer les tests

```bash
# Tests automatisés (pas besoin de Docker)
pytest tests/ -v

# Smoke tests (Docker doit être démarré)
python smoke-tests/test_multimodel_api.py
```

---

## Commandes quotidiennes

```bash
# Démarrage
docker-compose up -d

# Voir les logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Rebuild après modification du code
docker-compose up -d --build api
docker-compose up -d --build streamlit

# Accès services
# Dashboard admin  : http://localhost:8501
# API Swagger      : http://localhost:8000/docs
# MLflow           : http://localhost:5000
# MinIO console    : http://localhost:9001  (minioadmin / minioadmin)
# Grafana          : http://localhost:3000  (admin / admin)
```

---

## Pour aller plus loin

- [Guide débutant complet](BEGINNER_GUIDE.md) — tutoriel détaillé avec A/B testing, SHAP, ré-entraînement
- [Référence API](API_REFERENCE.md) — tous les endpoints, schémas et exemples Python
- [Architecture](ARCHITECTURE.md) — structure du projet et flux de données
