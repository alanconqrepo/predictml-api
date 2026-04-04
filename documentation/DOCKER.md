# Docker

## Commandes quotidiennes

```bash
# Démarrer
docker-compose up -d

# Arrêter
docker-compose down

# Logs
docker-compose logs -f api

# Rebuild après modification du code
docker-compose up -d --build api
```

## Services et ports

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MinIO console | http://localhost:9003 |
| MLflow | http://localhost:5000 |
| PostgreSQL | localhost:5434 |

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

# Ports déjà utilisés
netstat -ano | findstr "8000 5434 9002"

# Redémarrer un service
docker-compose restart api
```
