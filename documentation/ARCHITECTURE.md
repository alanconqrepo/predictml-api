# Architecture

## Stack

| Service | Technologie | Port |
|---|---|---|
| API | FastAPI (async) | 8000 |
| Dashboard Admin | Streamlit | 8501 |
| Base de données | PostgreSQL 16 | 5433 |
| Stockage modèles | MinIO (S3-compatible) | 9000 / console 9001 |
| Experiment tracking | MLflow | 5000 |
| Observabilité | Grafana LGTM (Loki + Tempo + Prometheus) | 3000 |

## Structure du projet

```
predictml-api/
├── src/
│   ├── api/                  # Endpoints HTTP
│   │   ├── models.py         # CRUD /models
│   │   ├── predict.py        # POST /predict, GET /predictions
│   │   ├── users.py          # CRUD /users
│   │   └── observed_results.py  # /observed-results
│   ├── core/
│   │   ├── config.py         # Settings (variables d'env)
│   │   └── security.py       # Auth Bearer token + rate limiting
│   ├── db/
│   │   ├── models/           # SQLAlchemy ORM (User, Prediction, ModelMetadata, ObservedResult)
│   │   └── database.py       # Session async
│   ├── services/
│   │   ├── db_service.py     # Toutes les requêtes DB
│   │   ├── model_service.py  # Chargement & cache des modèles
│   │   └── minio_service.py  # Upload/download MinIO
│   ├── schemas/              # Pydantic (validation I/O)
│   └── main.py
├── streamlit_app/            # Dashboard admin Streamlit
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                # Page login + accueil
│   ├── utils/
│   │   ├── api_client.py     # Client HTTP vers l'API
│   │   └── auth.py           # Helpers session_state
│   └── pages/
│       ├── 1_Users.py        # Gestion utilisateurs (admin)
│       ├── 2_Models.py       # Gestion modèles
│       ├── 3_Predictions.py  # Historique prédictions
│       ├── 4_Stats.py        # Statistiques & graphiques
│       └── 5_Code_Example.py # Exemple MLflow + API
├── tests/                    # Tests pytest (automatisés)
├── smoke-tests/              # Tests manuels (Docker live)
├── init_data/                # Scripts d'initialisation (one-shot)
├── Models/                   # Fichiers .pkl locaux
├── notebooks/                # Jupyter notebooks
├── alembic/                  # Migrations DB
├── docker-compose.yml
└── .env
```

## Flux de données — Prédiction

```
Client
  │  POST /predict + Bearer Token
  ▼
security.py          → vérifie le token en DB + rate limit
  ▼
predict.py           → valide la requête (Pydantic)
  ▼
model_service.py     → charge le modèle (cache mémoire ou MinIO/MLflow)
  ▼
model.predict(X)     → sklearn
  ▼
db_service.py        → log la prédiction en PostgreSQL
  ▼
Client ← JSON
```

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
PostgreSQL / MinIO / MLflow
```

Le dashboard Streamlit ne parle **jamais directement** à la DB ou à MinIO — l'API FastAPI est le seul backend.

## Base de données

| Table | Rôle |
|---|---|
| `users` | Auth, rôles (ADMIN/USER/READONLY), rate limiting |
| `model_metadata` | Registre des modèles (versioning, localisation MinIO/MLflow) |
| `predictions` | Log complet de chaque appel API (features, résultat, temps de réponse) |
| `observed_results` | Résultats réels observés (pour comparer aux prédictions) |

### Table `predictions` — colonnes notables

| Colonne | Type | Description |
|---|---|---|
| `id_obs` | VARCHAR(255), nullable | Identifiant métier de l'observation |
| `input_features` | JSON | Features envoyées |
| `prediction_result` | JSON | Résultat du modèle |
| `probabilities` | JSON, nullable | Probabilités par classe |
| `response_time_ms` | Float | Temps d'inférence en millisecondes |
| `status` | VARCHAR(20) | `success` ou `error` |

## Authentification

- **Mécanisme** : HTTP Bearer token (`Authorization: Bearer <token>`)
- **Token** : `secrets.token_urlsafe(32)` stocké en clair dans `users.api_token`
- **Rôles** : `admin` (accès total), `user` (prédictions + lecture), `readonly`
- **Rate limiting** : quota journalier par utilisateur (`rate_limit_per_day`)
- **Renouvellement token** : `PATCH /users/{id}` avec `{"regenerate_token": true}` (admin)

## Format de requête `/predict`

Seul le format dict (features nommées) est accepté depuis la v2 :

```json
{
  "model_name": "iris_model",
  "model_version": "1.0.0",
  "id_obs": "obs-42",
  "features": {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
  }
}
```

`model_version` est optionnel — sans lui, la version `is_production=True` est utilisée (ou la plus récente).
