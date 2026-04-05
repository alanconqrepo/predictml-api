# Docker

## Commandes quotidiennes

```bash
# Démarrer tous les services
docker-compose up -d

# Arrêter
docker-compose down

# Logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Rebuild après modification du code API
docker-compose up -d --build api

# Rebuild après modification du dashboard Streamlit
docker-compose up -d --build streamlit
```

## Services et ports

| Service | URL | Description |
|---|---|---|
| API | http://localhost:8000 | API FastAPI |
| API docs | http://localhost:8000/docs | Swagger UI |
| Dashboard | http://localhost:8501 | Streamlit Admin Dashboard |
| MLflow | http://localhost:5000 | Experiment tracking |
| MinIO console | http://localhost:9001 | Gestion du stockage modèles |
| Grafana | http://localhost:3000 | Observabilité (admin/admin) |
| PostgreSQL | localhost:5433 | Base de données |

## Accès PostgreSQL

```bash
docker exec -it predictml-postgres psql -U postgres -d sklearn_api
```

## Réinitialisation complète

```bash
# Supprime tous les volumes (perte de données)
docker-compose down -v
docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py
```

## Dépannage

```bash
# Vérifier l'état des services
docker-compose ps

# Vérifier les ports utilisés
netstat -ano | grep -E "8000|8501|5433|9000|5000"

# Redémarrer un service
docker-compose restart api
docker-compose restart streamlit

# Inspecter les logs d'un service
docker-compose logs --tail=50 api
docker-compose logs --tail=50 streamlit
```

## Variables d'environnement (`.env`)

| Variable | Défaut | Description |
|---|---|---|
| `API_PORT` | `8000` | Port de l'API |
| `STREAMLIT_PORT` | `8501` | Port du dashboard |
| `POSTGRES_PORT` | `5433` | Port PostgreSQL externe |
| `MINIO_PORT` | `9000` | Port MinIO API |
| `MINIO_CONSOLE_PORT` | `9001` | Port console MinIO |
| `MLFLOW_URL` | `http://localhost:5000` | URL MLflow visible depuis le navigateur |
| `ENABLE_OTEL` | `false` | Activer OpenTelemetry vers Grafana |
