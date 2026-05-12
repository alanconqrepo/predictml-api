# Grafana & OpenTelemetry — Guide d'utilisation

## Architecture d'observabilité

PredictML utilise la stack **Grafana LGTM** tout-en-un (`grafana/otel-lgtm`) qui embarque :

| Composant | Rôle | Port interne |
|---|---|---|
| **Prometheus** | Scrape les métriques `/metrics` de l'API | 9090 |
| **Loki** | Stockage des logs structurés | 3100 |
| **Tempo** | Stockage des traces distribuées | 3200 |
| **Grafana** | Visualisation — UI accessible | **3000** |
| **OTel Collector** | Réception OTLP (traces + métriques + logs) | 4317 (gRPC), 4318 (HTTP) |

```
API FastAPI
  │
  ├─► /metrics (Prometheus scrape)  ──────────────► Prometheus ─► Grafana
  │
  └─► OTLP gRPC :4317 (si ENABLE_OTEL=true)
        ├─► Traces  ──► Tempo  ──► Grafana
        ├─► Métriques ► Prometheus ─► Grafana
        └─► Logs ────► Loki ───► Grafana
```

## Accéder à Grafana

```bash
# Démarrer la stack
docker-compose up -d

# Grafana est disponible sur
open http://localhost:3000

# Identifiants par défaut
# Login    : admin  (configurable via GRAFANA_ADMIN_USER dans .env)
# Password : valeur de GRAFANA_ADMIN_PASSWORD dans .env
```

## Dashboards pré-configurés

Les deux dashboards apparaissent automatiquement au démarrage, dans le dossier **General**.

### 1. PredictML — API Overview

**Fichier** : `monitoring/grafana/dashboards/api-overview.json`

| Panel | Métrique | Description |
|---|---|---|
| Débit (req/s) | `http_requests_total` | Nombre de requêtes par seconde (toutes routes) |
| Taux d'erreur 5xx | `http_requests_total{status_code=~"5.."}` | Part des erreurs serveur |
| Latence P95 | `http_request_duration_seconds_bucket` | 95e percentile de durée |
| Routes actives | `http_requests_total` | Nombre de routes ayant du trafic |
| Débit par route | par `handler` | Courbe temporelle par endpoint |
| Latence P50/P95/P99 | histogramme | Percentiles de latence globale |
| Erreurs HTTP | 4xx + 5xx par route | Évolution temporelle des erreurs |
| Répartition codes HTTP | donut | Part de chaque code de statut |
| Top endpoints | table | Routes les plus sollicitées + latences |
| Mémoire RSS | `process_resident_memory_bytes` | Consommation mémoire du worker |
| CPU | `process_cpu_seconds_total` | Consommation CPU du worker |

> Ce dashboard fonctionne **immédiatement** — il ne nécessite pas `ENABLE_OTEL=true`.

---

### 2. PredictML — Model Performance

**Fichier** : `monitoring/grafana/dashboards/model-performance.json`

| Panel | Source | Description |
|---|---|---|
| Volume /predict (req/s) | Prometheus | Taux de prédictions par seconde |
| Erreurs /predict (%) | Prometheus | Part des 4xx/5xx sur l'endpoint predict |
| Latence P95 /predict | Prometheus | Latence de l'inférence ML |
| Volume prédictions (graphe) | Prometheus | Succès vs erreurs client vs erreurs serveur |
| Latence /predict P50/P95/P99 | Prometheus | Percentiles de durée d'inférence |
| Erreurs /models | Prometheus | Erreurs sur les routes de gestion de modèles |
| Événements retrain | **Loki** | Logs des ré-entraînements (OTEL requis) |
| Alertes drift | **Loki** | Logs de détection de drift (OTEL requis) |
| Logs d'erreur récents | **Loki** | Lignes ERROR/WARNING (OTEL requis) |

> Les **panels Loki** (logs) restent vides sans `ENABLE_OTEL=true`.

---

## Activer OpenTelemetry (traces + logs → Loki/Tempo)

Ajouter dans le fichier `.env` :

```bash
ENABLE_OTEL=true
OTEL_SERVICE_NAME=predictml-api          # label service dans Loki/Tempo
OTEL_EXPORTER_OTLP_ENDPOINT=http://grafana:4317  # déjà défaut en Docker
```

Puis redémarrer l'API :

```bash
docker-compose restart api
```

Une fois activé :
- Les **logs Python** (structlog) sont bridgés vers Loki via l'OTLP Collector
- Les **traces** de chaque requête FastAPI et chaque requête SQL sont envoyées à Tempo
- Les **métriques OTEL** sont également exportées (en complément de Prometheus)

---

## Lire les logs dans Loki (Explore)

1. Aller dans **Grafana → Explore**
2. Sélectionner le datasource **Loki**
3. Exemples de requêtes LogQL :

```logql
# Tous les logs de l'API
{service_name="predictml-api"}

# Logs d'erreur uniquement
{service_name="predictml-api"} | json | level =~ "(?i)(error|critical)"

# Événements de ré-entraînement
{service_name="predictml-api"} |= "retrain" | json

# Prédictions sur un modèle spécifique
{service_name="predictml-api"} |= "predict" | json | model_name="iris"

# Détection de drift
{service_name="predictml-api"} |= "drift" | json

# Alertes de supervision
{service_name="predictml-api"} |= "alert" | json

# Requêtes lentes (latence > 1 s)
{service_name="predictml-api"} | json | response_time > 1000
```

---

## Lire les traces dans Tempo (Explore)

1. Aller dans **Grafana → Explore**
2. Sélectionner le datasource **Tempo**
3. Rechercher par :
   - **Service** : `predictml-api`
   - **Span name** : `POST /predict`, `GET /models`, etc.
   - **Tag** : `http.status_code=500` pour filtrer les erreurs

Les traces FastAPI sont instrumentées automatiquement par `FastAPIInstrumentor`.  
Chaque requête crée un span racine avec les tags `http.method`, `http.route`, `http.status_code`.  
Les requêtes SQL (SQLAlchemy) créent des spans enfants tracés par `SQLAlchemyInstrumentor`.

### Lien traces → logs

Dans Tempo, cliquer sur un span ouvre un lien **"Logs for this span"** qui filtre Loki  
sur le `trace_id` correspondant — corrélation automatique requête / logs.

---

## Métriques Prometheus disponibles

### Métriques HTTP (prometheus-fastapi-instrumentator)

| Métrique | Type | Labels | Description |
|---|---|---|---|
| `http_requests_total` | Counter | `handler`, `method`, `status_code` | Nombre total de requêtes |
| `http_request_duration_seconds` | Histogram | `handler`, `method`, `status_code` | Durée des requêtes |
| `http_request_duration_highr_seconds` | Histogram | `handler`, `method`, `status_code` | Durée (haute résolution) |
| `http_request_size_bytes` | Histogram | `handler`, `method` | Taille des corps de requêtes |
| `http_response_size_bytes` | Histogram | `handler`, `method` | Taille des corps de réponses |

### Métriques process Python

| Métrique | Description |
|---|---|
| `process_cpu_seconds_total` | Temps CPU consommé |
| `process_resident_memory_bytes` | Mémoire RSS |
| `process_virtual_memory_bytes` | Mémoire virtuelle |
| `process_open_fds` | Descripteurs de fichiers ouverts |
| `process_start_time_seconds` | Timestamp de démarrage |

### Exemples de requêtes PromQL utiles

```promql
# Débit global en req/s
sum(rate(http_requests_total[5m]))

# Taux d'erreur 5xx en %
100 * sum(rate(http_requests_total{status_code=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))

# P95 de latence globale (ms)
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
) * 1000

# P95 de latence /predict uniquement
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket{handler=~"/predict.*"}[5m])) by (le)
) * 1000

# Volume par endpoint
sum(rate(http_requests_total[5m])) by (handler)

# Erreurs par route
sum(rate(http_requests_total{status_code=~"[45].."}[5m])) by (handler, status_code)
```

---

## Authentification de l'endpoint /metrics

Par défaut en développement, `/metrics` est accessible sans token.  
En production, définir `METRICS_TOKEN` dans le `.env` :

```bash
METRICS_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Puis mettre à jour `monitoring/prometheus.yml` pour que Prometheus s'authentifie :

```yaml
scrape_configs:
  - job_name: predictml-api
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
    authorization:
      credentials: "<valeur de METRICS_TOKEN>"
```

---

## Modifier ou ajouter un dashboard

1. Éditer un dashboard dans l'interface Grafana
2. Exporter via **Dashboard → Share → Export → Save to file**
3. Remplacer le fichier `.json` correspondant dans `monitoring/grafana/dashboards/`
4. Grafana recharge automatiquement les fichiers toutes les 30 secondes (configurable via `updateIntervalSeconds` dans `dashboards.yaml`)

Pour créer un nouveau dashboard depuis zéro et le rendre persistant :

```bash
# 1. Créer dans Grafana UI, exporter en JSON
# 2. Placer dans monitoring/grafana/dashboards/mon-dashboard.json
# 3. Pas de restart nécessaire — le provider scrute le répertoire
```

---

## Troubleshooting

### Les dashboards n'apparaissent pas

```bash
# Vérifier que les fichiers sont bien montés dans le container
docker exec predictml-grafana ls /etc/grafana/provisioning/dashboards/

# Vérifier les logs Grafana
docker-compose logs grafana | grep -i "dashboard\|provision\|error"
```

### Loki est vide

Vérifier que `ENABLE_OTEL=true` est dans le `.env` et que l'API a redémarré :

```bash
docker-compose logs api | grep -i "otel\|telemetry"
# Doit afficher : "OpenTelemetry activé — endpoint: http://grafana:4317"
```

### Prometheus ne scrape pas les métriques

```bash
# Vérifier que l'API expose /metrics
curl http://localhost:8000/metrics

# Si METRICS_TOKEN est défini
curl -H "Authorization: Bearer <METRICS_TOKEN>" http://localhost:8000/metrics

# Vérifier dans Grafana → Explore → Prometheus → Metrics browser
# Chercher : http_requests_total
```

### Tempo ne reçoit pas de traces

```bash
# Vérifier la connectivité OTLP
docker-compose logs api | grep -i "otlp\|span\|trace"

# Tester l'envoi OTLP HTTP
curl -v http://localhost:4318/v1/traces
```
