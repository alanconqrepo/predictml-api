# Quickstart

## Premier déploiement

```bash
# 1. Démarrer les services
docker-compose up -d --build

# 2. Générer les modèles .pkl (si Models/ est vide)
python init_data/create_multiple_models.py

# 3. Initialiser la DB et uploader les modèles vers MinIO
docker exec predictml-api python init_data/init_db.py
```

Le token admin est affiché une seule fois à l'étape 3 — sauvegardez-le.

## Tester l'API

```bash
export TOKEN="votre-token-admin"

# Statut
curl http://localhost:8000/

# Lister les modèles
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/models

# Prédiction — format liste (features ordonnées)
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "iris_model", "features": [5.1, 3.5, 1.4, 0.2]}'

# Prédiction — format dict avec id_obs (modèle entraîné sur DataFrame pandas)
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris_model",
    "id_obs": "patient-42",
    "features": {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
  }'
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

# Accès MinIO console : http://localhost:9003  (minioadmin / minio_secure_password_123)
# Accès MLflow       : http://localhost:5000
```
