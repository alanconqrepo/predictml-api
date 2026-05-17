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
docker-compose logs -f nginx
docker-compose logs -f redis-master
docker-compose logs -f prediction-writer

# Rebuild après modification du code API
docker-compose up -d --build api prediction-writer

# Rebuild après modification du dashboard Streamlit
docker-compose up -d --build streamlit
```

## Services et ports

L'API est exposée publiquement via **Nginx sur le port 80** (reverse proxy + load balancer).  
Le port 8000 de l'API n'est pas exposé directement à l'hôte.

| Service | URL externe | Description |
|---|---|---|
| **API** (via Nginx) | http://localhost | Reverse proxy Nginx → 3 réplicas FastAPI |
| API docs | http://localhost/docs | Swagger UI |
| Dashboard | http://localhost:8501 | Streamlit Admin Dashboard |
| MLflow | http://localhost:5000 | Experiment tracking |
| MinIO console | http://localhost:9001 | Gestion du stockage modèles |
| Redis (master) | localhost:6379 | Cache distribué (auth requise) |
| Grafana | http://localhost:3000 | Observabilité (Prometheus + Loki + Tempo) |
| PostgreSQL | localhost:5433 | Base de données principale (write) |

## Architecture des services Docker

```
                       ┌──────────────────────────────────────────────┐
                       │              réseau frontend                  │
                       │                                               │
Client ─── port 80 ──► │ nginx (reverse proxy, least_conn)            │
                       │   └─► api:8000 ×3 réplicas                  │
                       └──────────────────────────────────────────────┘
                                        │
                       ┌────────────────▼─────────────────────────────┐
                       │              réseau internal                  │
                       │                                               │
                       │ pgbouncer (pooler write)                      │
                       │   └─► postgres:5432 (primary)                │
                       │         └─► postgres-replica (streaming)      │
                       │               └─► pgbouncer-read (pooler)    │
                       │                                               │
                       │ redis-master ← redis-replica-1/2             │
                       │ redis-sentinel-1/2/3 (quorum : 2)            │
                       │                                               │
                       │ prediction-writer (Redis Stream consumer)     │
                       │ minio · mlflow · grafana · streamlit          │
                       └──────────────────────────────────────────────┘
```

### Services par rôle

| Service | Rôle |
|---|---|
| `nginx` | Reverse proxy, load balancer least_conn, entrée unique port 80 |
| `api` (×3 réplicas) | FastAPI — prédictions, modèles, utilisateurs |
| `migrate` | One-shot Alembic — s'exécute avant l'API, ne redémarre pas |
| `prediction-writer` | Worker Redis Streams — batch INSERT des prédictions en DB |
| `postgres` | PostgreSQL primary — toutes les écritures |
| `postgres-replica` | PostgreSQL replica en streaming — requêtes analytiques |
| `pgbouncer` | Connection pooler (transaction mode) pour le primary |
| `pgbouncer-read` | Connection pooler pour la replica |
| `redis-master` | Redis master — cache modèles (DB 0) + rate limiting (DB 1) |
| `redis-replica-1/2` | Redis réplicas — haute disponibilité |
| `redis-sentinel-1/2/3` | Sentinel — basculement automatique < 10 s (quorum : 2) |
| `minio` | Stockage objets S3-compatible (modèles .joblib, scripts train.py) |
| `mlflow` | Experiment tracking (métriques, artifacts) |
| `grafana` | Stack LGTM — Prometheus + Loki + Tempo + Grafana |
| `streamlit` | Dashboard admin multipage |

## Accès PostgreSQL

```bash
# Primary (container_name stable)
docker exec -it predictml-postgres psql -U postgres -d sklearn_api

# Ou via psql local
psql -h localhost -p 5433 -U postgres -d sklearn_api
```

## Cache Redis (Sentinel)

```bash
# Se connecter au master (avec auth)
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD"

# Vérifier le rôle du master
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" INFO replication | grep role

# Lister les clés en cache
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" KEYS "*"

# Vider le cache des modèles (force le rechargement depuis MinIO)
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" FLUSHDB

# Vérifier l'état du Sentinel
docker exec predictml-redis-sentinel-1 redis-cli -p 26379 SENTINEL masters
```

> **Note** : `REDIS_PASSWORD` est **obligatoire** dans `.env`. Le Sentinel utilise la même
> variable pour s'authentifier auprès du master et des réplicas.

## File d'attente des prédictions (Redis Streams)

Les prédictions POST /predict sont écrites de façon asynchrone via Redis Streams
pour découpler l'écriture DB du chemin critique de l'inférence.

```bash
# Longueur du stream en attente
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" XLEN predictions:new

# Dead Letter Queue (messages en échec après MAX_RETRIES)
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" XLEN predictions:dlq

# Logs du worker
docker-compose logs -f prediction-writer
```

En cas de Redis indisponible, le writer repasse automatiquement en mode synchrone.

## Migrations Alembic

Les migrations sont exécutées par le service `migrate` **avant** le démarrage de l'API.
En environnement multi-réplicas, cela évite les conflits de migration simultanés.

```bash
# Lancer manuellement les migrations (si nécessaire)
docker-compose run --rm migrate

# Voir l'historique des migrations
docker-compose run --rm migrate alembic history

# Créer une nouvelle migration
docker-compose run --rm migrate alembic revision --autogenerate -m "description"
```

## Initialisation (premier déploiement)

> L'API n'a plus de `container_name: predictml-api` (3 réplicas). Utiliser
> `docker-compose exec` à la place de `docker exec predictml-api`.

```bash
# Initialiser la base de données et l'utilisateur admin
docker-compose exec api python init_data/init_db.py
```

## Réinitialisation complète

```bash
# Supprime tous les volumes (perte de données)
docker-compose down -v
docker-compose up -d --build
docker-compose exec api python init_data/init_db.py
```

## Métriques Prometheus

L'API expose un endpoint de scrape standard sur `GET /metrics` (format `text/plain 0.0.4`).

> En mode multi-réplicas, Nginx distribue les requêtes Prometheus entre les réplicas.
> `PROMETHEUS_MULTIPROC_DIR` agrège les compteurs de tous les workers.

### Métriques HTTP (prometheus-fastapi-instrumentator)

| Métrique | Type | Labels |
|---|---|---|
| `http_requests_total` | Counter | `method`, `handler`, `status_code` |
| `http_request_duration_seconds` | Histogram | `method`, `handler`, `status_code` |
| Métriques process Python | Gauge | — |

### Métriques ML métier (src/core/ml_metrics.py)

| Métrique | Type | Labels | Description |
|---|---|---|---|
| `predictml_predictions_total` | Counter | `model_name`, `version`, `mode`, `status` | Prédictions par modèle et statut |
| `predictml_inference_duration_seconds` | Histogram | `model_name`, `version` | Durée d'inférence ML pure |
| `predictml_retrain_total` | Counter | `model_name`, `status` | Ré-entraînements déclenchés |
| `predictml_drift_detected_total` | Counter | `model_name`, `drift_type`, `severity` | Alertes de drift |

### Vérifier l'endpoint

```bash
# Via Nginx (port 80)
curl http://localhost/metrics

# Si METRICS_TOKEN est défini
curl -H "Authorization: Bearer $METRICS_TOKEN" http://localhost/metrics
```

### Sécuriser l'endpoint (recommandé en production)

```bash
# Dans .env
METRICS_TOKEN=mon-token-secret
```

Puis mettre à jour `monitoring/prometheus.yml` :

```yaml
- job_name: predictml-api
  static_configs:
    - targets: ['api:8000']
  authorization:
    credentials: "mon-token-secret"
```

## Dépannage

```bash
# Vérifier l'état des services
docker-compose ps

# Vérifier les ports utilisés
netstat -ano | grep -E "80|8501|5433|9000|5000|6379|3000"

# Redémarrer l'API (tous les réplicas)
docker-compose restart api

# Redémarrer Nginx
docker-compose restart nginx

# Redémarrer le Redis master
docker-compose restart redis-master

# Inspecter les logs d'un service
docker-compose logs --tail=50 api
docker-compose logs --tail=50 nginx
docker-compose logs --tail=50 prediction-writer

# Si l'API ne répond plus via Nginx
docker-compose logs --tail=50 nginx
docker-compose ps api  # vérifier le health check

# Si Redis est inaccessible
docker-compose ps redis-master redis-sentinel-1
docker-compose logs redis-sentinel-1

# Si les prédictions ne sont pas enregistrées en DB
docker-compose logs prediction-writer
docker exec -it predictml-redis-master redis-cli -a "$REDIS_PASSWORD" XLEN predictions:dlq
```

## Variables d'environnement (`.env`)

> **Variables obligatoires** — l'API ou docker-compose refuse de démarrer sans elles.

| Variable | Défaut | Obligatoire | Description |
|---|---|---|---|
| `SECRET_KEY` | — | **Oui** | Clé HMAC pour la signature des modèles. Générez avec `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `REDIS_PASSWORD` | — | **Oui** | Mot de passe Redis (master + réplicas + sentinels). |
| `MINIO_ROOT_USER` | — | **Oui** | Login du compte root MinIO. |
| `MINIO_ROOT_PASSWORD` | — | **Oui** | Mot de passe du compte root MinIO. |
| `GRAFANA_ADMIN_PASSWORD` | — | **Oui** | Mot de passe admin Grafana. |
| `TOKEN_LIFETIME_DAYS` | `90` | Non | Durée de validité des tokens Bearer en jours. |
| `API_PORT` | `8000` | Non | Port interne de l'API (Nginx écoute sur 80) |
| `STREAMLIT_PORT` | `8501` | Non | Port du dashboard |
| `POSTGRES_PORT` | `5433` | Non | Port PostgreSQL exposé à l'hôte |
| `MINIO_PORT` | `9000` | Non | Port MinIO API |
| `MINIO_CONSOLE_PORT` | `9001` | Non | Port console MinIO |
| `MLFLOW_URL` | `http://localhost:5000` | Non | URL MLflow visible depuis le navigateur |
| `REDIS_URL` | `redis://:$REDIS_PASSWORD@redis-master:6379/0` | Non | URL Redis master (interne Docker) |
| `REDIS_SENTINEL_HOSTS` | (auto dans compose) | Non | Adresses des sentinels `host:port,...` |
| `REDIS_CACHE_TTL` | `3600` | Non | Durée de cache des modèles en secondes |
| `ENABLE_OTEL` | `true` | Non | Activer OpenTelemetry vers Grafana (activé par défaut) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://grafana:4317` | Non | Endpoint OTLP Grafana |
| `METRICS_TOKEN` | `` | Non | Token Bearer pour protéger `GET /metrics` (vide = public) |
| `PROMETHEUS_MULTIPROC_DIR` | `/tmp/prometheus_multiproc` | Non | Partage des métriques entre workers |
| `PREDICTION_STREAM_ENABLED` | `true` | Non | Activer la queue asynchrone Redis Streams pour les writes de prédictions |
| `PREDICTION_STREAM_BATCH_SIZE` | `100` | Non | Lignes par commit du worker |
| `PREDICTION_STREAM_FLUSH_MS` | `500` | Non | Délai de flush max du worker en ms |
| `PREDICTION_STREAM_MAX_RETRIES` | `3` | Non | Tentatives avant envoi en DLQ |
| `MAX_ROWS_ANALYTICS` | `50000` | Non | Limite de lignes pour les requêtes analytiques |
| `ANALYTICS_MAX_DAYS` | `90` | Non | Fenêtre temporelle max pour les agrégations |
| `DATABASE_READ_REPLICA_URL` | (auto dans compose) | Non | URL PostgreSQL replica pour les lectures analytiques |
| `ADMIN_TOKEN` | `` | Non | Token admin personnalisé (généré auto si vide) |
| `ENABLE_EMAIL_ALERTS` | `false` | Non | Activer les alertes email |
| `SMTP_HOST` | `` | Non | Serveur SMTP |
| `SMTP_PORT` | `587` | Non | Port SMTP |
| `SMTP_USER` | `` | Non | Utilisateur SMTP |
| `SMTP_PASSWORD` | `` | Non | Mot de passe SMTP |
| `SMTP_FROM` | `` | Non | Adresse expéditeur |
| `ALERT_EMAIL_TO` | `` | Non | Destinataires alertes (séparés par virgules) |
| `WEEKLY_REPORT_ENABLED` | `false` | Non | Activer le rapport hebdomadaire |
| `PERFORMANCE_DRIFT_ALERT_THRESHOLD` | `0.10` | Non | Seuil de chute d'accuracy déclenchant une alerte |
| `ERROR_RATE_ALERT_THRESHOLD` | `0.10` | Non | Taux d'erreur déclenchant une alerte |
| `MAX_MODEL_SIZE_MB` | `500` | Non | Taille max d'upload d'un fichier `.joblib` |
