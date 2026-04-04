# Architecture

## Stack

| Service | Technologie | Port |
|---|---|---|
| API | FastAPI (async) | 8000 |
| Base de données | PostgreSQL 16 | 5434 |
| Stockage modèles | MinIO (S3-compatible) | 9002 / console 9003 |
| Experiment tracking | MLflow | 5000 |

## Structure du projet

```
predictml-api/
├── src/
│   ├── api/                # Endpoints HTTP
│   │   ├── models.py       # GET /models
│   │   └── predict.py      # POST /predict
│   ├── core/
│   │   ├── config.py       # Settings (variables d'env)
│   │   └── security.py     # Auth Bearer token
│   ├── db/
│   │   ├── models/         # SQLAlchemy ORM
│   │   └── service.py      # Requêtes DB
│   ├── services/
│   │   ├── model_service.py  # Chargement & cache des modèles
│   │   └── minio_service.py  # Upload/download MinIO
│   ├── schemas/            # Pydantic (validation I/O)
│   └── main.py
├── tests/                  # Tests pytest (automatisés)
├── smoke-tests/            # Tests manuels (Docker live)
├── init_data/              # Scripts d'initialisation (one-shot)
├── Models/                 # Fichiers .pkl locaux
├── notebooks/              # Jupyter notebooks
├── alembic/                # Migrations DB
├── docker-compose.yml
└── .env
```

## Flux de données

```
Client
  │  POST /predict + Bearer Token
  ▼
security.py          → vérifie le token en DB
  ▼
predict.py           → valide la requête (Pydantic)
  ▼
model_service.py     → charge le modèle (cache mémoire ou MinIO)
  ▼
model.predict(X)     → sklearn
  ▼
db/service.py        → log la prédiction en PostgreSQL
  ▼
Client ← JSON
```

## Base de données

| Table | Rôle |
|---|---|
| `users` | Auth, rôles (ADMIN/USER/READONLY), rate limiting |
| `model_metadata` | Registre des modèles (versioning, localisation MinIO) |
| `predictions` | Log complet de chaque appel API |

### Table `predictions` — colonnes notables

| Colonne | Type | Description |
|---|---|---|
| `id_obs` | VARCHAR(255), nullable | Identifiant métier de l'observation (fourni par le client) |
| `input_features` | JSON | Features envoyées (liste ou dict selon le format utilisé) |
| `prediction_result` | JSON | Résultat retourné par le modèle |

## Format de requête `/predict`

Deux formats de `features` sont acceptés :

**Format liste** (backward compatible) — l'ordre doit correspondre à l'ordre d'entraînement :
```json
{"model_name": "iris_model", "features": [5.1, 3.5, 1.4, 0.2]}
```

**Format dict** — les features sont nommées. Le modèle **doit** exposer `feature_names_in_`
(entraîné avec un DataFrame pandas). L'ordre des clés n'importe pas.
`id_obs` est optionnel dans les deux formats :
```json
{
  "model_name": "iris_model",
  "id_obs": "patient-42",
  "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
}
```

> **Exiger `feature_names_in_`** : lors de l'entraînement, passer un `pd.DataFrame` (et non
> un array numpy) à `model.fit()` — sklearn sauvegarde alors automatiquement les noms de colonnes.
