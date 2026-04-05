# predictml-api

API de prédiction Machine Learning construite avec FastAPI et scikit-learn.

**Version 2.0** — Multi-utilisateurs, stockage distribué, experiment tracking, dashboard admin.

[![Tests](https://github.com/alanconqrepo/predictml-api/actions/workflows/tests.yml/badge.svg)](https://github.com/alanconqrepo/predictml-api/actions/workflows/tests.yml)

## Stack

- **API** : FastAPI (async) — port 8000
- **Dashboard** : Streamlit Admin — port 8501
- **Base de données** : PostgreSQL 16 — port 5433
- **Stockage modèles** : MinIO (S3) — port 9000 / console 9001
- **Experiment tracking** : MLflow — port 5000

## Démarrage rapide

```bash
docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py
```

Le dashboard admin est accessible sur **http://localhost:8501**.

Voir [documentation/QUICKSTART.md](documentation/QUICKSTART.md) pour le guide complet.

## Endpoints API

| Méthode | Route | Auth | Description |
|---|---|---|---|
| GET | `/` | Non | Statut et modèles disponibles |
| GET | `/health` | Non | Health check |
| GET | `/models` | Non | Liste des modèles actifs |
| GET | `/models/{name}/{version}` | Non | Détail d'un modèle |
| POST | `/models` | Oui | Uploader un nouveau modèle |
| PATCH | `/models/{name}/{version}` | Oui | Mettre à jour (ex. passer en production) |
| DELETE | `/models/{name}/{version}` | Oui | Supprimer une version |
| DELETE | `/models/{name}` | Oui | Supprimer toutes les versions |
| POST | `/predict` | Oui | Faire une prédiction |
| GET | `/predictions` | Oui | Historique des prédictions |
| POST | `/users` | Admin | Créer un utilisateur |
| GET | `/users` | Admin | Lister les utilisateurs |
| GET | `/users/{id}` | Oui | Détail d'un utilisateur |
| PATCH | `/users/{id}` | Admin | Modifier (rôle, statut, renouveler token) |
| DELETE | `/users/{id}` | Admin | Supprimer un utilisateur |
| POST | `/observed-results` | Oui | Enregistrer des résultats observés |
| GET | `/observed-results` | Oui | Consulter les résultats observés |

```bash
export TOKEN="votre-token"

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris_model",
    "id_obs": "obs-42",
    "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}
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
