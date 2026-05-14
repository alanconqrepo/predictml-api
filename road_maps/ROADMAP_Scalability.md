# ROADMAP Scalabilité — predictml-api

> Stack actuelle : 1 container FastAPI · 1 PostgreSQL · 1 Redis · 1 MinIO · docker-compose
>
> Objectif : améliorer la scalabilité sans introduire Kubernetes, en partant des
> goulots d'étranglement les plus critiques et les moins coûteux à corriger.

---

## Seuils de saturation de la stack actuelle

| Dimension | Limite raisonnable aujourd'hui |
|---|---|
| Requêtes API concurrentes | ~10–20 (limité par pool_size=5 DB) |
| Prédictions/jour | < 500 000 (au-delà les endpoints stats deviennent lents) |
| Modèles en prod | < 50 (selon taille des .pkl en RAM Redis) |
| Utilisateurs actifs | < 200 |
| Taille d'équipe | Startup / PME < 50 personnes |

---

## Niveau 1 — Faible complexité (< 2h par item)

Ces corrections sont des **quick wins** : faible risque, pas de nouvelle infrastructure,
impact immédiat mesurable.

---

### 1. Pool de connexions DB explicite

**Pourquoi**

`src/db/database.py` appelle `create_async_engine()` sans paramètres de pool.
SQLAlchemy applique alors ses valeurs par défaut : `pool_size=5`, `max_overflow=10`.
Avec 3 workers Uvicorn et des requêtes async concurrentes, le pool est saturé dès
~15 connexions simultanées — PostgreSQL refuse les nouvelles et l'API retourne des
`TimeoutError` sans message clair.

**Comment**

Modifier `src/db/database.py` :

```python
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=20,          # connexions permanentes
    max_overflow=40,       # connexions temporaires sous pic
    pool_recycle=300,      # recycler les connexions > 5 min (évite les stale connections)
    pool_pre_ping=True,    # déjà présent, garder
    pool_timeout=30,       # timeout explicite plutôt qu'attente infinie
)
```

Ajouter dans `.env.example` :
```
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

**Impact attendu** : supporte 3× plus de requêtes concurrentes sans autre changement.

---

### 2. LIMIT de sécurité sur les requêtes d'agrégation non bornées

**Pourquoi**

Une douzaine de méthodes dans `src/services/db_service.py` exécutent des `SELECT`
sans clause `LIMIT` sur la table `predictions` :

- `get_global_monitoring_stats()` — aucun LIMIT ; à 1M de prédictions, charge 1M+
  lignes en RAM pour calculer des percentiles Python-side.
- `get_prediction_stats()`, `get_confidence_trend()`, `get_feature_production_stats()` —
  même pattern.

Un seul appel dashboard sur un déploiement actif (6 mois, 1 000 pred/jour = 180 000 lignes)
peut provoquer un timeout de 30–60 s et une consommation RAM de 500 MB+.

**Comment**

Ajouter un cap de sécurité sur toutes les requêtes d'agrégation non paginées.
Stratégie : limiter la fenêtre temporelle interrogée plutôt que le nombre de lignes brutes,
et ajouter un LIMIT absolu en dernier recours.

Dans chaque méthode d'agrégation, remplacer les fenêtres temporelles ouvertes par un
maximum configurable (ex. 90 jours) et ajouter `.limit(50_000)` sur les sous-requêtes
qui alimentent des calculs Python :

```python
# Avant
result = await db.execute(select(Prediction).where(Prediction.model_name == name))

# Après
MAX_ROWS_ANALYTICS = 50_000
result = await db.execute(
    select(Prediction)
    .where(Prediction.model_name == name)
    .where(Prediction.timestamp >= cutoff)
    .limit(MAX_ROWS_ANALYTICS)
)
```

Exposer `MAX_ROWS_ANALYTICS` comme variable d'environnement dans `config.py`.

**Impact attendu** : élimine les requêtes illimitées, borne la RAM consommée par les
endpoints analytiques à < 100 MB dans tous les cas.

---

### 3. Rate limiting sur `/predict-batch`

**Pourquoi**

`src/api/predict.py` applique `@limiter.limit("60/minute")` sur `/predict` mais
**aucun décorateur** sur `/predict-batch`. Un client peut envoyer des batches de
1 000 prédictions à volonté, contournant totalement le contrôle de débit. Sous haute
charge, cela sature le pool DB et bloque tous les autres utilisateurs.

**Comment**

Ajouter le décorateur sur l'endpoint `/predict-batch` dans `src/api/predict.py` :

```python
@router.post("/predict-batch")
@limiter.limit("10/minute")          # batches = charge plus lourde → limite plus basse
async def predict_batch(request: Request, ...):
    ...
```

Ajouter aussi une validation de taille de batch :

```python
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))
if len(payload.predictions) > MAX_BATCH_SIZE:
    raise HTTPException(status_code=422, detail=f"Batch trop grand (max {MAX_BATCH_SIZE})")
```

**Impact attendu** : protège l'API contre l'abus de l'endpoint le plus coûteux.

---

### 4. Backend Redis pour le rate limiter (slowapi)

**Pourquoi**

slowapi stocke ses compteurs en mémoire du processus. Dès qu'on lance 2 replicas de
l'API (item 6 ci-dessous), chaque instance tient ses propres compteurs — un client peut
envoyer 60 req/min par replica, soit 120+ req/min au total sans jamais déclencher la
limite. Le rate limiting devient inefficace.

**Comment**

slowapi supporte nativement un backend Redis. Modifier `src/main.py` :

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from redis.asyncio import from_url as redis_from_url

# Avant
limiter = Limiter(key_func=get_remote_address)

# Après
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.REDIS_URL,   # "redis://:password@redis:6379/1"
)
```

Utiliser une base Redis dédiée (ex. `/1`) pour ne pas mélanger cache et rate-limit.

**Impact attendu** : rate limiting correct dès qu'on passe à plusieurs replicas, sans
infrastructure supplémentaire (Redis est déjà dans le docker-compose).

---

## Niveau 2 — Complexité modérée (demi-journée par item)

Ces items nécessitent de nouveaux fichiers de configuration ou des changements
d'architecture docker-compose, mais ne touchent pas au code applicatif.

---

### 5. Nginx comme reverse proxy et load balancer

**Pourquoi**

L'API est exposée directement sur le port 8000. Il n'y a aucun point d'entrée
centralisé pour : router les requêtes entre replicas, terminer TLS, mettre en buffer
les requêtes lentes (slow clients saturent les workers Uvicorn), renvoyer des
`503` propres si l'API est down.

**Comment**

Ajouter un service `nginx` dans `docker-compose.yml` :

```yaml
nginx:
  image: nginx:1.27-alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf:ro
  depends_on:
    - api
  networks:
    - frontend
```

Créer `nginx.conf` à la racine :

```nginx
upstream api_backend {
    least_conn;                    # équilibrage par charge (pas round-robin)
    server api:8000;               # devient "api_1:8000, api_2:8000..." avec replicas
    keepalive 32;                  # connexions persistantes vers l'API
}

server {
    listen 80;
    client_max_body_size 512M;     # pour l'upload de modèles .pkl
    proxy_read_timeout 120s;       # pour les endpoints retrain lents

    location / {
        proxy_pass http://api_backend;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_http_version 1.1;
        proxy_set_header Connection "";   # keepalive HTTP/1.1
    }

    location /health {
        proxy_pass http://api_backend;
        access_log off;            # ne pas polluer les logs avec les healthchecks
    }
}
```

Supprimer le `ports: ["8000:8000"]` du service `api` (exposé uniquement via nginx).

**Impact attendu** : point d'entrée unique, buffering des clients lents, base pour le
load balancing entre replicas.

---

### 6. Replicas API dans docker-compose

**Pourquoi**

Un seul container API = zéro tolérance aux pannes et zéro scale-out. Un redémarrage
(déploiement, crash) coupe le service pour tous les utilisateurs.

**Comment**

Modifier le service `api` dans `docker-compose.yml` :

```yaml
api:
  build: .
  deploy:
    replicas: 3
    resources:
      limits:
        cpus: "1.0"
        memory: 1G
      reservations:
        cpus: "0.25"
        memory: 256M
  restart: unless-stopped
  # Supprimer ports: (nginx prend le relais)
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 15s
    timeout: 5s
    retries: 3
    start_period: 30s
```

> **Prérequis** : les items 4 (Redis rate limiter) et 7 (migrations séparées) doivent
> être faits avant, sinon les replicas créeront des races conditions sur les migrations
> et des rate limits inefficaces.

**Impact attendu** : ×3 throughput, tolérance à la panne d'un replica, rolling deploy
possible (arrêter/relancer un replica à la fois).

---

### 7. Séparation des migrations du démarrage de l'application

**Pourquoi**

`entrypoint.sh` lance `alembic upgrade head` **puis** démarre Uvicorn dans le même
processus. Avec 3 replicas démarrant simultanément, les 3 lancent les migrations en
parallèle — Alembic n'est pas conçu pour cela et peut corrompre la table
`alembic_version` ou provoquer des deadlocks sur le schéma PostgreSQL.

**Comment**

Créer un service `migrate` one-shot dans `docker-compose.yml` :

```yaml
migrate:
  build: .
  command: alembic upgrade head
  depends_on:
    postgres:
      condition: service_healthy
  networks:
    - internal
  restart: "no"        # s'exécute une seule fois, ne redémarre pas

api:
  depends_on:
    migrate:
      condition: service_completed_successfully
    redis:
      condition: service_healthy
    minio:
      condition: service_healthy
```

Modifier `entrypoint.sh` pour supprimer l'appel à alembic :

```bash
#!/bin/bash
exec uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Ajouter `--workers 2` à uvicorn (2 workers par container × 3 replicas = 6 workers
total, adapté à une machine 4–8 cœurs).

**Impact attendu** : démarrage plus rapide des replicas, zéro race condition sur les
migrations, migrations rejouables en isolation pour les déploiements.

---

### 8. PgBouncer pour le connection pooling PostgreSQL

**Pourquoi**

Même avec `pool_size=20` (item 1), chaque replica de l'API ouvre ses propres
connexions. Avec 3 replicas × 20 connexions = 60 connexions permanentes vers PostgreSQL.
PostgreSQL supporte ~100 connexions par défaut (`max_connections=100`) — il reste peu
de marge pour MLflow, Alembic, les outils d'administration. Au-delà, PostgreSQL refuse
avec `FATAL: sorry, too many clients`.

PgBouncer mutualise les connexions : les replicas pensent avoir 20 connexions chacun,
mais PgBouncer n'en maintient que 20–30 réelles vers PostgreSQL.

**Comment**

Ajouter dans `docker-compose.yml` :

```yaml
pgbouncer:
  image: pgbouncer/pgbouncer:1.23
  environment:
    DATABASES_HOST: postgres
    DATABASES_PORT: 5432
    DATABASES_USER: postgres
    DATABASES_PASSWORD: postgres
    DATABASES_DBNAME: sklearn_api
    POOL_MODE: transaction       # mode transaction = compatible asyncpg
    MAX_CLIENT_CONN: 200         # connexions clients (replicas API)
    DEFAULT_POOL_SIZE: 20        # connexions réelles vers PostgreSQL
    SERVER_RESET_QUERY: DISCARD ALL
  depends_on:
    postgres:
      condition: service_healthy
  networks:
    - internal
```

Modifier `DATABASE_URL` dans `.env` pour pointer vers PgBouncer :
```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@pgbouncer:5432/sklearn_api
```

> **Note** : le mode `transaction` est incompatible avec les transactions explicites
> longues (ex. `BEGIN ... COMMIT` manuels). Les sessions SQLAlchemy async classiques
> fonctionnent correctement.

**Impact attendu** : PostgreSQL ne voit jamais plus de 20–30 connexions réelles,
quelle que soit l'échelle des replicas.

---

## Niveau 3 — Complexité élevée (2–5 jours)

Ces items nécessitent des refactors significatifs du code applicatif ou l'introduction
de nouveaux composants.

---

### 9. Réécriture des agrégations Python → SQL (window functions)

**Pourquoi**

C'est le goulot d'étranglement **le plus critique pour les performances analytiques**.
Environ 12 méthodes dans `src/services/db_service.py` appliquent le même anti-pattern :

1. `SELECT *` sans LIMIT sur `predictions` (parfois + jointure `observed_results`)
2. Boucle Python pour grouper par date/version/modèle dans un `defaultdict`
3. Calcul de percentiles via `sorted()` + indexing manuel ou `np.percentile()`

Exemple concret de `get_prediction_stats()` : charge toutes les prédictions d'un modèle
en RAM, puis calcule P50/P95 latency en Python. À 100 000 prédictions × 8 champs JSON
≈ 80 MB de RAM + 2–5 secondes de traitement, pour une réponse qui pourrait tenir en
< 50 ms avec du SQL natif.

**Comment**

Réécrire ces méthodes en utilisant les fonctions analytiques PostgreSQL :

```sql
-- Avant (Python-side) :
-- fetch all rows, then sorted(times)[int(len*0.95)]

-- Après (SQL natif) :
SELECT
    DATE_TRUNC('day', timestamp)              AS day,
    COUNT(*)                                   AS count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms)  AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_ms,
    AVG(response_time_ms)                      AS avg_ms
FROM predictions
WHERE model_name = :name
  AND timestamp >= :start
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY day;
```

Méthodes prioritaires à réécrire (par ordre d'impact) :

| Méthode | Requête SQL cible | Fonction PostgreSQL |
|---|---|---|
| `get_global_monitoring_stats()` | GROUP BY model_name + percentiles | `PERCENTILE_CONT` |
| `get_prediction_stats()` | GROUP BY day + percentiles latence | `PERCENTILE_CONT`, `DATE_TRUNC` |
| `get_accuracy_drift()` | JOIN + GROUP BY day + AVG accuracy | `DATE_TRUNC`, `AVG`, `CASE` |
| `get_confidence_trend()` | GROUP BY day + percentiles confiance | `PERCENTILE_CONT` |
| `get_prediction_label_distribution()` | `GROUP BY prediction_result, COUNT(*)` | `COUNT`, `GROUP BY` |
| `get_feature_production_stats()` | Aggregate JSON keys (complexe) | `JSONB` operators + `AVG`/`STDDEV` |
| `get_ab_comparison_stats()` | GROUP BY version + percentiles | `PERCENTILE_CONT`, `FILTER` |

> **Note compatibilité** : `PERCENTILE_CONT` est PostgreSQL uniquement. Les tests
> utilisent SQLite en mémoire — il faudra mocker ces méthodes dans les tests ou
> ajouter une implémentation fallback pour SQLite détectée via `engine.dialect.name`.

**Impact attendu** : temps de réponse des endpoints analytiques de 5–30 s → < 200 ms,
consommation RAM divisée par 50–100×.

---

### 10. Cache warming des modèles au démarrage

**Pourquoi**

Après un redémarrage de Redis (mise à jour, crash), tous les modèles doivent être
rechargés depuis MinIO à la première prédiction. Si 10 utilisateurs font simultanément
une prédiction sur des modèles différents, l'API lance 10 téléchargements MinIO en
parallèle (potentiellement 500 MB × 10). Cela provoque des timeouts et une surcharge
de MinIO.

`src/services/model_service.py` n'implémente pas de warm-up : les modèles sont chargés
uniquement à la demande (`load_model()` déclenche un fetch MinIO sur cache miss).

**Comment**

Dans `src/main.py`, ajouter un warm-up dans le `lifespan` après l'initialisation de la DB :

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... init existant ...
    await _warmup_production_models()
    yield
    # ... shutdown existant ...

async def _warmup_production_models():
    """Précharge en Redis les modèles marqués is_production=True."""
    try:
        async with AsyncSessionLocal() as db:
            active_models = await db_service.get_production_models(db)
        tasks = [
            model_service.load_model(m.name, m.version)
            for m in active_models
        ]
        await asyncio.gather(*tasks, return_exceptions=True)  # echecs non bloquants
        logger.info(f"Warm-up: {len(active_models)} modèles préchargés")
    except Exception as e:
        logger.warning(f"Warm-up ignoré: {e}")  # ne pas bloquer le démarrage
```

Ajouter `get_production_models()` dans `src/services/db_service.py` si elle n'existe pas :

```python
async def get_production_models(self, db: AsyncSession) -> list[ModelMetadata]:
    result = await db.execute(
        select(ModelMetadata).where(ModelMetadata.is_production == True)
    )
    return result.scalars().all()
```

**Impact attendu** : premier appel après redémarrage aussi rapide qu'un appel normal,
pas de pic de charge MinIO au redémarrage.

---

### 11. Redis Sentinel pour la haute disponibilité

**Pourquoi**

Redis est un point de défaillance unique pour trois fonctions critiques :
- Cache des modèles (perte = rechargement lent depuis MinIO)
- Rate limiting distribué (perte = plus de contrôle de débit)
- Verrou Redis pour le retrain scheduler (perte = jobs dupliqués)

Un redémarrage Redis de 30 secondes suffit à provoquer des erreurs en cascade.

**Comment**

Remplacer le service Redis unique par une architecture Sentinel (1 master + 2 replicas +
3 sentinels) dans `docker-compose.yml` :

```yaml
redis-master:
  image: redis:7-alpine
  command: redis-server --requirepass ${REDIS_PASSWORD}
  networks: [internal]

redis-replica-1:
  image: redis:7-alpine
  command: >
    redis-server
    --replicaof redis-master 6379
    --requirepass ${REDIS_PASSWORD}
    --masterauth ${REDIS_PASSWORD}
  networks: [internal]

redis-replica-2:
  image: redis:7-alpine
  command: >
    redis-server
    --replicaof redis-master 6379
    --requirepass ${REDIS_PASSWORD}
    --masterauth ${REDIS_PASSWORD}
  networks: [internal]

redis-sentinel:
  image: redis:7-alpine
  command: >
    redis-sentinel /etc/redis/sentinel.conf
  volumes:
    - ./redis-sentinel.conf:/etc/redis/sentinel.conf
  deploy:
    replicas: 3
  networks: [internal]
```

Créer `redis-sentinel.conf` :

```
sentinel monitor mymaster redis-master 6379 2
sentinel auth-pass mymaster ${REDIS_PASSWORD}
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 10000
sentinel parallel-syncs mymaster 1
```

Modifier `src/core/config.py` pour supporter l'URL Sentinel :

```python
REDIS_URL: str = "redis://:password@redis-master:6379/0"
REDIS_SENTINEL_HOSTS: str = ""   # "sentinel1:26379,sentinel2:26379,sentinel3:26379"
```

**Impact attendu** : basculement automatique en < 10 s si le master Redis tombe,
sans intervention manuelle.

---

## Niveau 4 — Complexité très élevée (> 1 semaine)

Ces items nécessitent des changements architecturaux profonds et introduisent de
nouveaux composants de production.

---

### 12. Queue asynchrone pour les writes de prédictions (Redis Streams)

**Pourquoi**

Chaque appel à `POST /predict` effectue une écriture synchrone en DB
(`DBService.create_prediction()`) dans le chemin critique de la requête. À 100 req/s,
cela génère 100 INSERTs/s sur PostgreSQL — c'est la limite haute d'un PostgreSQL non
optimisé sur disque rotatif (SSD : ~1 000 INSERT/s mais avec latence accrue pour
les appels concurrents).

L'écriture synchrone lie la latence de la réponse API à la latence DB : un pic de
charge DB se traduit directement en timeout utilisateur.

**Comment**

Découpler l'écriture avec Redis Streams :

```
POST /predict
  → valider, prédire → répondre à l'utilisateur  (< 10 ms)
  → publier dans Redis Stream "predictions:new"   (< 1 ms)

Worker séparé (src/workers/prediction_writer.py) :
  → XREAD "predictions:new"
  → batch INSERT dans PostgreSQL (100 lignes / commit)
```

Fichiers à créer/modifier :
- `src/workers/prediction_writer.py` — consumer Redis Streams avec batch commits
- `src/api/predict.py` — remplacer `await db_service.create_prediction()` par publish Redis
- `docker-compose.yml` — ajouter service `prediction-writer`
- `src/schemas/prediction.py` — sérialisation JSON pour le stream

> **Risque** : les prédictions ne sont pas immédiatement disponibles dans
> `GET /predictions` (délai de quelques secondes). À documenter dans l'API.
> Implémenter un mécanisme de retry/DLQ pour les writes échoués.

**Impact attendu** : latence `/predict` décorrélée de la charge DB, capacité à absorber
des pics à 1 000+ req/s sans dégradation.

---

### 13. Read replica PostgreSQL — routage lecture/écriture

**Pourquoi**

Les endpoints analytiques (stats, drift, monitoring) et les endpoints de prédiction
partagent le même PostgreSQL. Un batch de calculs de drift lancé depuis le dashboard
peut saturer le CPU PostgreSQL et augmenter la latence des prédictions de 5×.

**Comment**

Ajouter un replica PostgreSQL en streaming replication et router les `SELECT` analytiques
vers le replica :

```yaml
# docker-compose.yml
postgres-replica:
  image: postgres:16-alpine
  environment:
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    PGUSER: postgres
  command: >
    postgres
    -c primary_conninfo='host=postgres port=5432 user=postgres password=postgres'
    -c recovery_target_timeline=latest
  volumes:
    - postgres_replica_data:/var/lib/postgresql/data
  networks: [internal]
```

Modifier `src/db/database.py` pour créer deux engines :

```python
write_engine = create_async_engine(settings.DATABASE_URL, ...)          # master
read_engine  = create_async_engine(settings.DATABASE_READ_REPLICA_URL, ...)  # replica

ReadSession  = async_sessionmaker(read_engine, ...)
WriteSession = async_sessionmaker(write_engine, ...)
```

Ajouter une dépendance FastAPI `get_read_db()` et la passer aux endpoints analytiques
(`/predictions/stats`, `/monitoring/*`, `/models/{name}/ab-compare`, etc.).

> **Note** : le replica a un délai de réplication de quelques millisecondes.
> Les données très récentes (dernière prédiction) peuvent ne pas être visibles
> immédiatement sur les endpoints analytiques — comportement acceptable.

**Impact attendu** : isolation complète entre charge analytique et charge de prédiction,
PostgreSQL master dédié aux writes, capacité analytique scalable indépendamment.

---

## Synthèse — Ordre de mise en œuvre recommandé

```
Semaine 1  → Items 1, 2, 3, 4   (quick wins, pas de risque)
Semaine 2  → Items 7, 5, 6      (infra docker-compose, dans cet ordre)
Semaine 3  → Item 8             (PgBouncer, après que les replicas tournent)
Semaine 4–6 → Item 9            (refactor agrégations SQL, le plus impactant)
Semaine 7  → Item 10            (cache warming, simple une fois le reste stable)
Semaine 8+ → Items 11, 12, 13   (HA Redis, queue async, read replica — si la charge
                                  justifie la complexité opérationnelle)
```

**Seuils de saturation après chaque niveau :**

| Après niveau | Req/s supportées | Utilisateurs | Modèles |
|---|---|---|---|
| Stack actuelle | ~15 | < 200 | < 50 |
| Niveau 1 | ~50 | < 500 | < 50 |
| Niveau 2 | ~150 | < 2 000 | < 100 |
| Niveau 3 | ~500 | < 10 000 | < 500 |
| Niveau 4 | ~2 000+ | < 50 000 | illimité |
