# Docker — PredictML API

## Commandes quotidiennes

```bash
# Démarrer tous les services
docker-compose up -d

# Arrêter
docker-compose down

# Logs
docker-compose logs -f api
docker-compose logs -f streamlit
docker-compose logs -f redis

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
| MinIO console | http://localhost:9001 | Gestion du stockage modèles (minioadmin/minioadmin) |
| Redis | localhost:6379 | Cache distribué des modèles |
| Grafana | http://localhost:3000 | Observabilité (admin/admin) |
| PostgreSQL | localhost:5433 | Base de données |

## Accès PostgreSQL

```bash
docker exec -it predictml-postgres psql -U postgres -d sklearn_api
```

## Cache Redis

```bash
# Se connecter à Redis
docker exec -it predictml-redis redis-cli

# Lister les clés en cache
docker exec -it predictml-redis redis-cli KEYS "*"

# Vider le cache des modèles (force le rechargement depuis MinIO)
docker exec -it predictml-redis redis-cli FLUSHDB

# Logs Redis
docker-compose logs -f redis
```

## Réinitialisation complète

```bash
# Supprime tous les volumes (perte de données)
docker-compose down -v
docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py
```

## Métriques Prometheus

L'API expose un endpoint de scrape standard sur `GET /metrics` (format `text/plain 0.0.4`).

### Métriques exposées

| Métrique | Type | Labels |
|---|---|---|
| `http_requests_total` | Counter | `method`, `handler`, `status_code` |
| `http_request_duration_seconds` | Histogram | `method`, `handler`, `status_code` |
| Métriques process Python | Gauge | — |

### Vérifier l'endpoint

```bash
curl http://localhost:8000/metrics
```

### Sécuriser l'endpoint (optionnel)

Par défaut `/metrics` est public, accessible depuis le réseau interne Docker. Pour le protéger :

```bash
# Dans .env
METRICS_TOKEN=mon-token-secret
```

Prometheus doit alors envoyer le token dans ses requêtes de scrape — décommenter la section `authorization` dans `monitoring/prometheus.yml` :

```yaml
- job_name: predictml-api
  static_configs:
    - targets: ['api:8000']
  authorization:
    credentials: "mon-token-secret"
```

### Scrape Prometheus (Grafana LGTM)

Le fichier `monitoring/prometheus.yml` est monté automatiquement dans le conteneur `grafana`. Prometheus scrape `api:8000/metrics` toutes les 15 secondes sans configuration supplémentaire.

Pour visualiser dans Grafana (http://localhost:3000) :
1. Aller dans **Explore → Prometheus**
2. Requêtes utiles :
   - `rate(http_requests_total[1m])` — débit par endpoint
   - `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` — latence p95

### Mode multi-workers

Si l'API est démarrée avec plusieurs workers (ex. `--workers 4`), `prometheus_client` agrège automatiquement les compteurs via `PROMETHEUS_MULTIPROC_DIR`. Le répertoire est créé au démarrage par `entrypoint.sh`.

## Dépannage

```bash
# Vérifier l'état des services
docker-compose ps

# Vérifier les ports utilisés
netstat -ano | grep -E "8000|8501|5433|9000|5000|6379|3000"

# Redémarrer un service
docker-compose restart api
docker-compose restart streamlit

# Inspecter les logs d'un service
docker-compose logs --tail=50 api
docker-compose logs --tail=50 streamlit

# Si l'API ne démarre pas : vérifier les migrations Alembic
docker-compose logs api | grep -i "alembic\|migration\|error"

# Si Redis est inaccessible : vérifier la connectivité
docker exec predictml-api python -c "import redis; r = redis.from_url('redis://redis:6379/0'); print(r.ping())"
```

## Variables d'environnement (`.env`)

> **SECRET_KEY est obligatoire.** L'API refuse de démarrer si cette variable est absente.
> Générez une valeur sécurisée avant le premier déploiement :
> ```bash
> python -c "import secrets; print(secrets.token_urlsafe(32))"
> ```

| Variable | Défaut | Description |
|---|---|---|
| `SECRET_KEY` | — | **Obligatoire.** Clé HMAC pour la signature du cache Redis. Aucune valeur par défaut — l'API lève une erreur au démarrage si absente. |
| `API_PORT` | `8000` | Port de l'API |
| `STREAMLIT_PORT` | `8501` | Port du dashboard |
| `POSTGRES_PORT` | `5433` | Port PostgreSQL externe |
| `MINIO_PORT` | `9000` | Port MinIO API |
| `MINIO_CONSOLE_PORT` | `9001` | Port console MinIO |
| `MLFLOW_URL` | `http://localhost:5000` | URL MLflow visible depuis le navigateur |
| `REDIS_URL` | `redis://redis:6379/0` | URL Redis (interne Docker) |
| `REDIS_CACHE_TTL` | `3600` | Durée de cache des modèles en secondes |
| `ENABLE_OTEL` | `false` | Activer OpenTelemetry vers Grafana |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://grafana:4317` | Endpoint OTLP Grafana |
| `METRICS_TOKEN` | `` | Token Bearer pour protéger `GET /metrics` (vide = public) |
| `PROMETHEUS_MULTIPROC_DIR` | `/tmp/prometheus_multiproc` | Répertoire de partage des métriques entre workers (multi-process) |
| `ENABLE_EMAIL_ALERTS` | `false` | Activer les alertes email |
| `SMTP_HOST` | `` | Serveur SMTP (ex: `smtp.gmail.com`) |
| `SMTP_PORT` | `587` | Port SMTP |
| `SMTP_USER` | `` | Utilisateur SMTP |
| `SMTP_PASSWORD` | `` | Mot de passe SMTP |
| `SMTP_FROM` | `` | Adresse expéditeur |
| `ALERT_EMAIL_TO` | `` | Destinataires alertes (séparés par virgules) |
| `WEEKLY_REPORT_ENABLED` | `false` | Activer le rapport hebdomadaire |
| `WEEKLY_REPORT_DAY` | `monday` | Jour du rapport (monday, tuesday…) |
| `WEEKLY_REPORT_HOUR` | `8` | Heure du rapport (0–23) |
| `PERFORMANCE_DRIFT_ALERT_THRESHOLD` | `0.10` | Seuil de chute d'accuracy déclenchant une alerte (ex: 0.10 = -10 pts) |
| `ERROR_RATE_ALERT_THRESHOLD` | `0.10` | Taux d'erreur déclenchant une alerte (ex: 0.10 = 10%) |
| `MAX_MODEL_SIZE_MB` | `500` | Taille max d'upload d'un fichier `.pkl` |
