# Quickstart

## Premier déploiement

```bash
# 1. Démarrer tous les services
docker-compose up -d --build

# 2. Générer les modèles .pkl (si Models/ est vide)
python init_data/create_multiple_models.py

# 3. Initialiser la DB et uploader les modèles vers MinIO
docker exec predictml-api python init_data/init_db.py
```

Le token admin est affiché une seule fois à l'étape 3 — sauvegardez-le.

## Accéder au dashboard

Ouvrir **http://localhost:8501** et saisir le token admin.

Le dashboard permet de :
- Gérer les utilisateurs (créer, désactiver, renouveler les tokens)
- Administrer les modèles (voir les détails, passer en production, lien MLflow)
- Consulter l'historique des prédictions avec filtres
- Visualiser les statistiques (temps de réponse, distribution, erreurs)
- Copier des exemples de code MLflow + API

## Utiliser l'API directement

```bash
export TOKEN="votre-token-admin"

# Statut
curl http://localhost:8000/

# Lister les modèles
curl http://localhost:8000/models

# Prédiction
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

# Historique des prédictions
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/predictions?name=iris_model&start=2024-01-01T00:00:00&end=2026-12-31T23:59:59"

# Créer un utilisateur (admin)
curl -X POST http://localhost:8000/users \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "email": "alice@example.com", "role": "user", "rate_limit": 500}'

# Renouveler un token (admin)
curl -X PATCH http://localhost:8000/users/2 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"regenerate_token": true}'

# Passer un modèle en production
curl -X PATCH http://localhost:8000/models/iris_model/1.0.0 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_production": true}'
```

## Lancer les tests

```bash
# Tests automatisés (pas besoin de Docker)
pytest tests/ -v

# Smoke tests (Docker doit être démarré)
python smoke-tests/test_multimodel_api.py
```

## Usages courants

```bash
# Démarrage quotidien
docker-compose up -d

# Voir les logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Accès services
# Dashboard admin : http://localhost:8501
# API Swagger     : http://localhost:8000/docs
# MLflow          : http://localhost:5000
# MinIO console   : http://localhost:9001  (minioadmin / minioadmin)
```
