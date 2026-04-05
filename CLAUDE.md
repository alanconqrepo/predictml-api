# predictml-api

API FastAPI de prédiction ML avec PostgreSQL, MinIO et MLflow. Dashboard admin Streamlit.

## Stack

- **API** : FastAPI async — port 8000
- **Dashboard** : Streamlit Admin — port 8501
- **DB** : PostgreSQL 16 — port 5433
- **Stockage modèles** : MinIO — port 9000 / console 9001
- **Experiment tracking** : MLflow — port 5000

## Structure

```
src/
├── api/            # Endpoints (models.py, predict.py, users.py, observed_results.py)
├── core/           # Config & auth (config.py, security.py)
├── db/             # ORM SQLAlchemy + service DB
├── services/       # Logique métier (db_service, model_service, minio_service)
├── schemas/        # Pydantic
└── main.py

streamlit_app/      # Dashboard admin multipage
├── app.py          # Login + accueil
├── utils/          # api_client.py, auth.py
└── pages/          # 1_Users, 2_Models, 3_Predictions, 4_Stats, 5_Code_Example

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
docker-compose logs -f streamlit

# PostgreSQL
docker exec -it predictml-postgres psql -U postgres -d sklearn_api
```

## Credentials

| Service | Valeur |
|---|---|
| Admin token | `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA` |
| DB | `postgres / postgres` |
| MinIO | `minioadmin / minioadmin` |

## Dépendances

Toute nouvelle dépendance doit être ajoutée dans **les deux fichiers** :
- `requirements.txt` — utilisé par le Dockerfile de l'API
- `pyproject.toml` — utilisé par la CI GitHub Actions

Pour le dashboard Streamlit, les dépendances sont dans `streamlit_app/requirements.txt`.

## Tests

Les tests dans `tests/` utilisent `TestClient` de FastAPI — aucun Docker requis.
Ils couvrent : auth, endpoints publics, logique du `ModelService`.

```bash
pytest tests/ -v           # tous les tests
pytest tests/test_api.py   # endpoints uniquement
```

Les smoke tests dans `smoke-tests/` nécessitent Docker et frappent l'API live.

## Endpoints principaux

- `POST /predict` — Prédiction (Bearer auth)
- `GET /predictions` — Historique (Bearer auth)
- `GET/POST/PATCH/DELETE /models` — Gestion modèles
- `GET/POST/PATCH/DELETE /users` — Gestion utilisateurs (admin)
- `POST/GET /observed-results` — Résultats observés
- `PATCH /users/{id}` avec `{"regenerate_token": true}` — Renouveler un token (admin)
