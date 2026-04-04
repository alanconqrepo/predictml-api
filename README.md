# predictml-api

API de prédiction Machine Learning construite avec FastAPI et scikit-learn.

**Version 2.0** — Multi-utilisateurs, stockage distribué, experiment tracking.

## Stack

- **API** : FastAPI (async) — port 8000
- **Base de données** : PostgreSQL 16 — port 5434
- **Stockage modèles** : MinIO (S3) — port 9002 / console 9003
- **Experiment tracking** : MLflow — port 5000

## Démarrage rapide

```bash
docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py
```

Voir [documentation/QUICKSTART.md](documentation/QUICKSTART.md) pour le guide complet.

## Endpoints

| Méthode | Route | Auth | Description |
|---|---|---|---|
| GET | `/` | Non | Statut et modèles disponibles |
| GET | `/health` | Non | Health check |
| GET | `/models` | Non | Liste des modèles |
| GET | `/models/{name}` | Non | Détail d'un modèle |
| POST | `/predict` | Oui | Faire une prédiction |

```bash
# Format liste (features ordonnées)
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "iris_model", "features": [5.1, 3.5, 1.4, 0.2]}'

# Format dict avec id_obs (features nommées — modèle entraîné sur DataFrame)
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris_model",
    "id_obs": "patient-42",
    "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
  }'
```

## Tests

```bash
# Tests automatisés (sans Docker)
pytest tests/ -v

# Smoke tests (Docker démarré)
python smoke-tests/test_multimodel_api.py
```

## Documentation

- [QUICKSTART.md](documentation/QUICKSTART.md) — Premier déploiement et usage courant
- [ARCHITECTURE.md](documentation/ARCHITECTURE.md) — Structure et flux de données
- [DOCKER.md](documentation/DOCKER.md) — Commandes Docker et dépannage
- [init_data/README.md](init_data/README.md) — Scripts d'initialisation
- [smoke-tests/README.md](smoke-tests/README.md) — Tests manuels live
