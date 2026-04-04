# predictml-api

API FastAPI de prédiction ML avec PostgreSQL, MinIO et MLflow.

## Stack

- **API** : FastAPI async — port 8000
- **DB** : PostgreSQL 16 — port 5434
- **Stockage modèles** : MinIO — port 9002 / console 9003
- **Experiment tracking** : MLflow — port 5000

## Structure

```
src/
├── api/            # Endpoints (models.py, predict.py)
├── core/           # Config & auth (config.py, security.py)
├── db/             # ORM SQLAlchemy + service DB
├── services/       # Logique métier (model_service, minio_service)
├── schemas/        # Pydantic
└── main.py

tests/              # Pytest — tests automatisés
smoke-tests/        # Tests manuels contre Docker live
init_data/          # Scripts one-shot (create_multiple_models, init_db)
Models/             # Fichiers .pkl locaux
notebooks/          # Jupyter
alembic/            # Migrations DB
```

## Commandes clés

```bash
# Démarrer
docker-compose up -d

# Initialiser (premier déploiement uniquement)
docker exec predictml-api python init_data/init_db.py

# Tests automatisés
pytest tests/ -v

# Smoke tests
python smoke-tests/test_multimodel_api.py

# Logs
docker-compose logs -f api

# PostgreSQL
docker exec -it predictml-postgres psql -U postgres -d sklearn_api
```

## Credentials

| Service | Valeur |
|---|---|
| Admin token | `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA` |
| DB | `postgres / postgres_secure_password_123` |
| MinIO | `minioadmin / minio_secure_password_123` |

## Tests

Les tests dans `tests/` utilisent `TestClient` de FastAPI — aucun Docker requis.
Ils couvrent : auth, endpoints publics, logique du `ModelService`.

```bash
pytest tests/ -v           # tous les tests
pytest tests/test_api.py   # endpoints uniquement
```

Les smoke tests dans `smoke-tests/` nécessitent Docker et frappent l'API live.
