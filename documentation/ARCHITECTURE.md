# Architecture — PredictML API

## Stack

| Service | Technologie | Port |
|---|---|---|
| API | FastAPI (async) | 8000 |
| Dashboard Admin | Streamlit | 8501 |
| Base de données | PostgreSQL 16 | 5433 |
| Stockage modèles | MinIO (S3-compatible) | 9000 / console 9001 |
| Experiment tracking | MLflow | 5000 |
| Cache distribué | Redis 7 | 6379 |
| Observabilité | Grafana LGTM (Loki + Tempo + Prometheus + OTLP) | 3000 |

---

## Structure du projet

```
predictml-api/
├── src/
│   ├── api/                        # Endpoints HTTP
│   │   ├── models.py               # CRUD + drift + history + retrain + A/B
│   │   ├── predict.py              # POST /predict, /predict-batch, /explain, GET /predictions, /predictions/{id}, /predictions/{id}/explain, /stats, DELETE /predictions/purge
│   │   ├── users.py                # CRUD /users
│   │   ├── observed_results.py     # /observed-results
│   │   └── monitoring.py           # /monitoring/overview, /monitoring/model/{name}
│   ├── core/
│   │   ├── config.py               # Settings (variables d'env via dotenv)
│   │   ├── security.py             # Auth Bearer token + rate limiting
│   │   └── telemetry.py            # OpenTelemetry → Grafana LGTM
│   ├── db/
│   │   ├── models/                 # ORM SQLAlchemy
│   │   │   ├── user.py             # Table users
│   │   │   ├── prediction.py       # Table predictions
│   │   │   ├── model_metadata.py   # Table model_metadata
│   │   │   ├── observed_result.py  # Table observed_results
│   │   │   └── model_history.py    # Table model_history
│   │   └── database.py             # Session async (asyncpg)
│   ├── services/
│   │   ├── db_service.py               # Toutes les requêtes DB
│   │   ├── model_service.py            # Chargement, cache Redis, routage A/B/shadow
│   │   ├── minio_service.py            # Upload/download MinIO
│   │   ├── drift_service.py            # Calcul dérive Z-score + PSI + null rate (4 dimensions)
│   │   ├── shap_service.py             # Explications SHAP locales (tree + linear)
│   │   ├── ab_significance_service.py  # Tests statistiques A/B (Chi-², Mann-Whitney U)
│   │   ├── auto_promotion_service.py   # Évaluation politique d'auto-promotion post-retrain
│   │   ├── input_validation_service.py # Validation schéma de features d'entrée
│   │   ├── supervision_reporter.py     # Rapports de supervision et seuils d'alerte par modèle
│   │   ├── email_service.py            # Alertes email & rapports hebdomadaires
│   │   └── webhook_service.py          # Webhooks HTTP post-prédiction
│   ├── schemas/                    # Schémas Pydantic (validation I/O)
│   │   ├── model.py
│   │   ├── prediction.py
│   │   ├── user.py
│   │   ├── observed_result.py
│   │   └── monitoring.py
│   └── main.py                     # App FastAPI + lifecycle hooks
├── streamlit_app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                      # Page login + accueil
│   ├── utils/
│   │   ├── api_client.py           # Client HTTP vers l'API
│   │   └── auth.py                 # Helpers session_state
│   └── pages/
│       ├── 1_Users.py
│       ├── 2_Models.py
│       ├── 3_Predictions.py
│       ├── 4_Stats.py
│       ├── 5_Code_Example.py
│       ├── 6_AB_Testing.py      # A/B testing, shadow mode, comparaison statistique
│       ├── 7_Supervision.py     # Monitoring global, drift, alertes, performance
│       └── 8_Retrain.py         # Gestion centralisée des retrains (manuel + planifié)
├── tests/                          # Tests pytest (automatisés, sans Docker)
├── smoke-tests/                    # Tests manuels (Docker live)
├── init_data/                      # Scripts one-shot (init_db, create_multiple_models)
│   └── example_train.py            # Exemple de script train.py compatible retrain
├── Models/                         # Fichiers .pkl locaux
├── notebooks/                      # Jupyter notebooks
├── alembic/                        # Migrations DB
├── docker-compose.yml
└── .env
```

---

## Flux de données — Prédiction

```
Client
  │  POST /predict + Bearer Token
  ▼
security.py
  → vérifie le token en DB
  → contrôle le rate limit (quota journalier)
  ▼
predict.py
  → valide la requête (Pydantic)
  ▼
model_service.py
  → select_routing_versions() : A/B test / shadow / production
  → charge le modèle (cache Redis → MinIO → MLflow)
  ▼
model.predict(X)  ← sklearn
  ▼
[si shadow] background_task → prédiction shadow loguée séparément
  ▼
db_service.py
  → log la prédiction en PostgreSQL (features, résultat, latence, user)
  ▼
[si webhook_url configuré] webhook_service.send_webhook()
  ▼
Client ← JSON { prediction, probability, low_confidence, selected_version }
```

### Routage A/B / Shadow

- `deployment_mode="production"` : version principale, reçoit tout le trafic
- `deployment_mode="ab_test"` : reçoit `traffic_weight` fraction du trafic (ex: 0.2 = 20%)
- `deployment_mode="shadow"` : exécutée en arrière-plan sur chaque requête production, résultat non retourné au client

Le service `model_service.select_routing_versions()` détermine quelle version répond à la requête et quelles versions shadow tourner en parallèle.

---

## Flux de données — Dashboard Streamlit

```
Navigateur
  │  HTTP + API Token (session_state)
  ▼
streamlit_app/utils/api_client.py
  │  requests HTTP
  ▼
API FastAPI (http://api:8000)
  ▼
PostgreSQL / MinIO / Redis / MLflow
```

Le dashboard Streamlit ne parle **jamais directement** à la DB ou à MinIO — l'API FastAPI est le seul backend.

---

## Base de données

| Table | Rôle |
|---|---|
| `users` | Auth, rôles (admin/user/readonly), rate limiting, token Bearer |
| `model_metadata` | Registre des modèles (versioning, localisation MinIO/MLflow, tags, A/B config) |
| `predictions` | Log complet de chaque appel API (features, résultat, temps de réponse, shadow flag) |
| `observed_results` | Résultats réels observés (pour comparer aux prédictions) |
| `model_history` | Journal des changements d'état des modèles (pour rollback) |

### Table `predictions` — colonnes notables

| Colonne | Type | Description |
|---|---|---|
| `id_obs` | VARCHAR(255), nullable | Identifiant métier de l'observation |
| `input_features` | JSON | Features envoyées |
| `prediction_result` | JSON | Résultat du modèle |
| `probabilities` | JSON, nullable | Probabilités par classe |
| `response_time_ms` | Float | Temps d'inférence en millisecondes |
| `status` | VARCHAR(20) | `success` ou `error` |
| `is_shadow` | Boolean | `true` si prédiction shadow (non retournée au client) |

### Table `model_history` — colonnes notables

| Colonne | Type | Description |
|---|---|---|
| `action` | str | Type de changement (`set_production`, `update_metadata`, `rollback`…) |
| `snapshot` | JSON | État complet des métadonnées au moment du changement |
| `changed_fields` | JSON | Liste des champs modifiés |
| `changed_by_user_id` | int | Auteur du changement |

### Table `model_metadata` — colonnes notables

| Colonne | Type | Description |
|---|---|---|
| `deployment_mode` | str | `production`, `ab_test`, `shadow` |
| `traffic_weight` | float | Part du trafic (0.0–1.0) |
| `confidence_threshold` | float | Seuil de confiance min |
| `feature_baseline` | JSON | Stats par feature pour drift detection |
| `tags` | JSON | Liste de tags libres |
| `webhook_url` | str | URL de callback post-prédiction |
| `train_script_object_key` | str | Clé MinIO du script train.py |
| `parent_version` | str | Version source du retrain (traçabilité de lignée) |
| `promotion_policy` | JSON | Politique d'auto-promotion post-retrain (`min_accuracy`, `max_latency_p95_ms`, etc.) |
| `retrain_schedule` | JSON | Planning cron de ré-entraînement automatique (`cron`, `lookback_days`, `enabled`, etc.) |
| `alert_thresholds` | JSON | Seuils d'alerte spécifiques au modèle (surcharge les seuils globaux) |
| `training_stats` | JSON | Snapshot des données d'entraînement du dernier retrain (`n_rows`, `feature_stats`, etc.) |

---

## Stockage des données d'entraînement

### Stratégie : MinIO (stockage) + MLflow (traçabilité)

Les données d'entraînement ne sont **pas** stockées comme artifacts MLflow (évite la duplication à chaque run). Elles vivent dans un bucket MinIO dédié, et MLflow en garde uniquement la **référence**.

```
MinIO bucket: datasets/
└── loan_model/
    └── v1/
        ├── train.parquet    ← données d'entraînement
        └── test.parquet

MLflow Run:
  params:  dataset_uri = "s3://datasets/loan_model/v1"
  inputs:  mlflow.log_input(dataset, context="training")  ← référence, pas de copie
```

**Avantages :**
- Pas de duplication : 10 runs → 1 seul exemplaire du dataset
- Fichiers volumineux gérés nativement par MinIO (multipart, streaming)
- Traçabilité complète via MLflow

### Workflow complet

```python
import os, mlflow, mlflow.sklearn, pandas as pd
from minio import Minio
from io import BytesIO

minio = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

if not minio.bucket_exists("datasets"):
    minio.make_bucket("datasets")

parquet_bytes = df_train.to_parquet(index=False)
minio.put_object("datasets", "loan_model/v1/train.parquet",
                 BytesIO(parquet_bytes), length=len(parquet_bytes))

DATASET_URI = "s3://datasets/loan_model/v1"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "minioadmin"
mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run() as run:
    dataset = mlflow.data.from_numpy(X_train.to_numpy(),
                                     source=f"{DATASET_URI}/train.parquet")
    mlflow.log_input(dataset, context="training")
    mlflow.log_param("dataset_uri", DATASET_URI)
    mlflow.sklearn.log_model(pipeline, "model")
    RUN_ID = run.info.run_id

requests.post("http://localhost:8000/models", headers=HEADERS, data={
    "name": "loan_model", "version": "1.0.0",
    "mlflow_run_id": RUN_ID, "training_dataset": DATASET_URI,
})
```

---

## Authentification

- **Mécanisme** : HTTP Bearer token (`Authorization: Bearer <token>`)
- **Token** : `secrets.token_urlsafe(32)` stocké dans `users.api_token`
- **Rôles** : `admin` (accès total), `user` (prédictions + lecture), `readonly`
- **Rate limiting** : quota journalier par utilisateur (`rate_limit_per_day`)
- **Renouvellement** : `PATCH /users/{id}` avec `{"regenerate_token": true}` (admin)

---

## Observabilité

### Logging structuré

L'API utilise `structlog` pour produire des logs JSON structurés. Chaque requête loguée contient : `model_name`, `model_version`, `response_time_ms`, `status`, `user_id`.

### Endpoint Prometheus `/metrics`

L'API expose `GET /metrics` au format Prometheus text 0.0.4 via `prometheus-fastapi-instrumentator`.

Métriques collectées automatiquement :
- `http_requests_total` (counter) — labels `method`, `handler`, `status_code`
- `http_request_duration_seconds` (histogramme) — latence par endpoint et code de retour
- Métriques process Python standard (mémoire, CPU, file descriptors)

Le fichier `monitoring/prometheus.yml` configure un job de scrape `predictml-api` ciblant `api:8000/metrics`. Il est monté dans le conteneur `grafana/otel-lgtm` au démarrage.

Sécurisation optionnelle via `METRICS_TOKEN` (Bearer token). Le mode multi-workers est supporté via `PROMETHEUS_MULTIPROC_DIR` — les compteurs sont agrégés par worker dans un répertoire partagé.

### OpenTelemetry → Grafana LGTM

Quand `ENABLE_OTEL=true`, les traces et métriques sont envoyées au collecteur OTLP (Grafana LGTM sur le port 4317).

Grafana LGTM regroupe :
- **Loki** — agrégation des logs
- **Tempo** — traces distribuées
- **Prometheus** — métriques (dont le scrape de `/metrics`)
- **Grafana** — visualisation unifiée (http://localhost:3000)

### Alertes email

Déclenchées automatiquement par `email_service.py` (scheduler APScheduler) quand :
- La dérive d'accuracy dépasse `PERFORMANCE_DRIFT_ALERT_THRESHOLD` (défaut: 10%)
- Le taux d'erreur dépasse `ERROR_RATE_ALERT_THRESHOLD` (défaut: 10%)
- Rapport hebdomadaire si `WEEKLY_REPORT_ENABLED=true`

### Scheduler de ré-entraînement planifié

`src/tasks/retrain_scheduler.py` — second scheduler APScheduler (AsyncIOScheduler) dédié aux retrains automatiques.

**Lifecycle :**
1. Au démarrage (`lifespan` dans `main.py`), charge tous les `ModelMetadata` actifs ayant `retrain_schedule.enabled=True`.
2. Crée un job cron par version (ID : `retrain_schedule:{name}:{version}`).
3. À chaque déclenchement, acquiert un **verrou Redis** `SET NX EX 700` pour éviter les exécutions concurrentes en multi-réplicas.
4. Exécute la logique retrain dans un sous-processus (timeout 600 s), crée une nouvelle version, met à jour `last_run_at` / `next_run_at`.

**Modification d'un planning en live :** l'endpoint `PATCH /models/{name}/{version}/schedule` met à jour la DB *et* le scheduler en cours d'exécution (`add_retrain_job` / `remove_retrain_job`).
