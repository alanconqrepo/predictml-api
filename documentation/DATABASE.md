# Documentation Base de Données — PredictML API

Schéma SQL complet, requêtes utiles et exemples de connexion Python.

---

## Connexion

**Paramètres de connexion**

| Paramètre | Valeur |
|---|---|
| Hôte | `localhost` |
| Port | `5433` |
| Base | `sklearn_api` |
| Utilisateur | `postgres` |
| Mot de passe | `postgres` |

---

## Se connecter à la base de données

### Via psql (ligne de commande)

```bash
# Connexion directe via Docker
docker exec -it predictml-postgres psql -U postgres -d sklearn_api

# Ou avec les paramètres complets
psql -h localhost -p 5433 -U postgres -d sklearn_api
```

### Via Python — SQLAlchemy async (utilisé par l'API)

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5433/sklearn_api"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def example():
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM predictions"))
        count = result.scalar()
        print(f"Nombre de prédictions : {count}")
```

### Via Python — psycopg2 (connexion synchrone classique)

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="sklearn_api",
    user="postgres",
    password="postgres"
)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM predictions")
count = cursor.fetchone()[0]
print(f"Nombre de prédictions : {count}")

cursor.close()
conn.close()
```

### Via Python — pandas (pour l'analyse)

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

# Charger les prédictions dans un DataFrame
df = pd.read_sql("""
    SELECT p.timestamp, p.model_name, p.model_version, p.prediction_result,
           p.response_time_ms, u.username
    FROM predictions p
    LEFT JOIN users u ON p.user_id = u.id
    WHERE p.timestamp >= NOW() - INTERVAL '7 days'
    ORDER BY p.timestamp DESC
""", engine)

print(df.head())
```

---

## Schéma de la base de données

### Table `users`

Stocke les comptes utilisateurs et leurs tokens d'authentification.

```sql
CREATE TABLE users (
    id                 SERIAL PRIMARY KEY,
    username           VARCHAR(50)  NOT NULL UNIQUE,
    email              VARCHAR(100) NOT NULL UNIQUE,
    api_token          VARCHAR(255) NOT NULL UNIQUE,
    role               VARCHAR(20)  NOT NULL DEFAULT 'user',   -- 'admin', 'user', 'readonly'
    is_active          BOOLEAN      NOT NULL DEFAULT TRUE,
    rate_limit_per_day INTEGER      NOT NULL DEFAULT 1000,
    created_at         TIMESTAMP    NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMP    NOT NULL DEFAULT NOW(),
    last_login         TIMESTAMP
);

CREATE INDEX ix_users_id       ON users (id);
CREATE INDEX ix_users_username ON users (username);
CREATE INDEX ix_users_email    ON users (email);
CREATE INDEX ix_users_api_token ON users (api_token);
```

**Colonnes**

| Colonne | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Identifiant unique |
| `username` | VARCHAR(50) | Nom d'utilisateur unique |
| `email` | VARCHAR(100) | Email unique |
| `api_token` | VARCHAR(255) | Token Bearer pour l'authentification |
| `role` | VARCHAR(20) | `admin`, `user` ou `readonly` |
| `is_active` | BOOLEAN | Compte actif ou désactivé |
| `rate_limit_per_day` | INTEGER | Quota journalier de prédictions |
| `created_at` | TIMESTAMP | Date de création |
| `updated_at` | TIMESTAMP | Dernière modification |
| `last_login` | TIMESTAMP | Dernière connexion authentifiée |

---

### Table `model_metadata`

Stocke les métadonnées des modèles ML déployés.

```sql
CREATE TABLE model_metadata (
    id                SERIAL PRIMARY KEY,
    name              VARCHAR(100) NOT NULL,
    version           VARCHAR(50)  NOT NULL,
    minio_bucket      VARCHAR(100),
    minio_object_key  VARCHAR(255),               -- chemin dans MinIO
    file_size_bytes   INTEGER,
    file_hash         VARCHAR(64),                -- hash SHA256 du fichier
    description       TEXT,
    algorithm         VARCHAR(100),               -- ex: 'RandomForestClassifier'
    features_count    INTEGER,
    classes           JSONB,                      -- ex: [0, 1, 2] ou ["cat","dog"]
    accuracy          FLOAT,
    precision         FLOAT,
    recall            FLOAT,
    f1_score          FLOAT,
    training_metrics  JSONB,                      -- métriques additionnelles
    mlflow_run_id     VARCHAR(255),               -- ID de run MLflow
    user_id_creator   INTEGER REFERENCES users(id),
    trained_by        VARCHAR(100),
    training_date     TIMESTAMP,
    training_dataset  VARCHAR(255),
    training_params   JSONB,                      -- hyperparamètres
    is_active         BOOLEAN   NOT NULL DEFAULT TRUE,
    is_production     BOOLEAN   NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    deprecated_at     TIMESTAMP
);

CREATE INDEX ix_model_metadata_id           ON model_metadata (id);
CREATE INDEX ix_model_metadata_name         ON model_metadata (name);
CREATE INDEX ix_model_metadata_version      ON model_metadata (version);
CREATE INDEX ix_model_metadata_is_active    ON model_metadata (is_active);
CREATE INDEX ix_model_metadata_mlflow_run_id ON model_metadata (mlflow_run_id);
CREATE INDEX ix_model_metadata_user_id_creator ON model_metadata (user_id_creator);
```

**Colonnes**

| Colonne | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Identifiant unique |
| `name` | VARCHAR(100) | Nom du modèle |
| `version` | VARCHAR(50) | Version (ex: "1.0", "2024-01") |
| `minio_bucket` | VARCHAR(100) | Bucket MinIO contenant le fichier |
| `minio_object_key` | VARCHAR(255) | Chemin du fichier dans MinIO |
| `file_size_bytes` | INTEGER | Taille du fichier `.pkl` en octets |
| `file_hash` | VARCHAR(64) | Hash SHA256 du fichier |
| `description` | TEXT | Description lisible |
| `algorithm` | VARCHAR(100) | Classe de l'algorithme |
| `features_count` | INTEGER | Nombre de features attendues |
| `classes` | JSONB | Labels des classes de sortie |
| `accuracy` | FLOAT | Précision sur le jeu de test |
| `precision` | FLOAT | Précision (au sens ML) |
| `recall` | FLOAT | Rappel |
| `f1_score` | FLOAT | Score F1 |
| `training_metrics` | JSONB | Métriques supplémentaires |
| `mlflow_run_id` | VARCHAR(255) | Lien vers un run MLflow |
| `user_id_creator` | INTEGER FK | Créateur du modèle |
| `trained_by` | VARCHAR(100) | Nom du formateur |
| `training_date` | TIMESTAMP | Date d'entraînement |
| `training_dataset` | VARCHAR(255) | Jeu de données utilisé |
| `training_params` | JSONB | Hyperparamètres d'entraînement |
| `is_active` | BOOLEAN | Modèle disponible ou non |
| `is_production` | BOOLEAN | Version de production (une par `name`) |
| `deployment_mode` | VARCHAR | `production`, `ab_test`, `shadow` |
| `traffic_weight` | FLOAT | Fraction du trafic A/B (0.0–1.0) |
| `confidence_threshold` | FLOAT | Seuil de confiance min (`low_confidence`) |
| `feature_baseline` | JSONB | Stats par feature pour drift detection |
| `tags` | JSONB | Liste de tags libres |
| `webhook_url` | VARCHAR | URL de callback post-prédiction |
| `train_script_object_key` | VARCHAR | Clé MinIO du script `train.py` |
| `parent_version` | VARCHAR | Version source du retrain (lignée) |
| `promotion_policy` | JSONB | Politique d'auto-promotion (`min_accuracy`, `max_latency_p95_ms`, `min_sample_validation`, `auto_promote`) |
| `retrain_schedule` | JSONB | Planning cron (`cron`, `lookback_days`, `enabled`, `last_run_at`, `next_run_at`) |
| `alert_thresholds` | JSONB | Seuils d'alerte spécifiques au modèle (surcharge les variables d'env globales) |
| `training_stats` | JSONB | Snapshot du dernier retrain (`n_rows`, `feature_stats`, `label_distribution`, `trained_at`) |
| `created_at` | TIMESTAMP | Date de création |
| `updated_at` | TIMESTAMP | Dernière mise à jour |
| `deprecated_at` | TIMESTAMP | Date de dépréciation |

---

### Table `predictions`

Log de toutes les prédictions effectuées via l'API.

```sql
CREATE TABLE predictions (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id),
    model_name       VARCHAR(100) NOT NULL,
    model_version    VARCHAR(50),
    id_obs           VARCHAR(255),
    input_features   JSONB NOT NULL,                  -- features envoyées
    prediction_result JSONB NOT NULL,                 -- résultat retourné
    probabilities    JSONB,                           -- probabilités par classe
    response_time_ms FLOAT NOT NULL,
    timestamp        TIMESTAMP NOT NULL DEFAULT NOW(),
    client_ip        VARCHAR(45),                     -- IPv4/IPv6
    user_agent       TEXT,
    status           VARCHAR(20) NOT NULL DEFAULT 'success',  -- 'success' ou 'error'
    error_message    TEXT
);

CREATE INDEX ix_predictions_id          ON predictions (id);
CREATE INDEX ix_predictions_user_id     ON predictions (user_id);
CREATE INDEX ix_predictions_model_name  ON predictions (model_name);
CREATE INDEX ix_predictions_timestamp   ON predictions (timestamp);
CREATE INDEX ix_predictions_id_obs      ON predictions (id_obs);
```

**Colonnes**

| Colonne | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Identifiant unique |
| `user_id` | INTEGER FK | Utilisateur ayant fait la prédiction |
| `model_name` | VARCHAR(100) | Nom du modèle utilisé |
| `model_version` | VARCHAR(50) | Version utilisée |
| `id_obs` | VARCHAR(255) | Identifiant d'observation (lien avec `observed_results`) |
| `input_features` | JSONB | Dictionnaire des features |
| `prediction_result` | JSONB | Résultat de la prédiction |
| `probabilities` | JSONB | Liste de probabilités par classe |
| `response_time_ms` | FLOAT | Latence en millisecondes |
| `timestamp` | TIMESTAMP | Horodatage de la prédiction |
| `client_ip` | VARCHAR(45) | Adresse IP du client |
| `user_agent` | TEXT | User-Agent HTTP |
| `status` | VARCHAR(20) | `success` ou `error` |
| `error_message` | TEXT | Message d'erreur si échec |

---

### Table `observed_results`

Résultats réels observés, pour évaluer les modèles après prédiction.

```sql
CREATE TABLE observed_results (
    id              SERIAL PRIMARY KEY,
    id_obs          VARCHAR(255) NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    observed_result JSONB NOT NULL,
    date_time       TIMESTAMP NOT NULL DEFAULT NOW(),
    user_id         INTEGER NOT NULL REFERENCES users(id),

    CONSTRAINT uq_observed_result_obs_model UNIQUE (id_obs, model_name)
);

CREATE INDEX ix_observed_results_id         ON observed_results (id);
CREATE INDEX ix_observed_results_id_obs     ON observed_results (id_obs);
CREATE INDEX ix_observed_results_model_name ON observed_results (model_name);
CREATE INDEX ix_observed_results_date_time  ON observed_results (date_time);
```

**Colonnes**

| Colonne | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Identifiant unique |
| `id_obs` | VARCHAR(255) | Identifiant de l'observation |
| `model_name` | VARCHAR(100) | Modèle concerné |
| `observed_result` | JSONB | Résultat réel observé |
| `date_time` | TIMESTAMP | Date d'observation |
| `user_id` | INTEGER FK | Utilisateur ayant soumis le résultat |

**Contrainte unique** : `(id_obs, model_name)` — un upsert met à jour si la paire existe déjà.

---

### Table `golden_tests`

Cas de test de régression (golden set) pour valider qu'un modèle produit toujours les sorties attendues.

```sql
CREATE TABLE golden_tests (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    input_features  JSONB        NOT NULL,
    expected_output JSONB        NOT NULL,
    description     TEXT,
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER   REFERENCES users(id)
);

CREATE INDEX ix_golden_tests_id         ON golden_tests (id);
CREATE INDEX ix_golden_tests_model_name ON golden_tests (model_name);
```

**Colonnes**

| Colonne | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Identifiant unique |
| `model_name` | VARCHAR(100) | Modèle ciblé |
| `input_features` | JSONB | Features d'entrée du cas de test |
| `expected_output` | JSONB | Sortie attendue (classe, valeur, etc.) |
| `description` | TEXT | Description optionnelle du cas |
| `created_at` | TIMESTAMP | Date de création |
| `created_by_user_id` | INTEGER FK | Auteur du cas de test |

**Exemple — créer et lister des cas**

```sql
-- Cas de test pour le modèle iris
INSERT INTO golden_tests (model_name, input_features, expected_output, description)
VALUES (
    'iris_model',
    '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}',
    '"setosa"',
    'Iris setosa typique — toutes features nominales'
);

-- Lister tous les cas d'un modèle
SELECT id, description, expected_output, created_at
FROM golden_tests
WHERE model_name = 'iris_model'
ORDER BY created_at DESC;
```

---

## Diagramme des relations

```
users
  │
  ├──< predictions (user_id → users.id)
  ├──< model_metadata (user_id_creator → users.id)
  ├──< observed_results (user_id → users.id)
  └──< golden_tests (created_by_user_id → users.id)

predictions.id_obs ──── observed_results.id_obs  (jointure applicative)
```

---

## Requêtes SQL utiles

### Utilisateurs

```sql
-- Tous les utilisateurs actifs avec leur quota
SELECT username, email, role, rate_limit_per_day, last_login
FROM users
WHERE is_active = TRUE
ORDER BY created_at DESC;

-- Vérifier les utilisateurs n'ayant jamais prédit
SELECT u.username, u.email, u.created_at
FROM users u
LEFT JOIN predictions p ON p.user_id = u.id
WHERE p.id IS NULL;
```

### Modèles

```sql
-- Modèles en production
SELECT name, version, algorithm, accuracy, f1_score, created_at
FROM model_metadata
WHERE is_production = TRUE AND is_active = TRUE;

-- Toutes les versions d'un modèle
SELECT version, accuracy, is_production, created_at
FROM model_metadata
WHERE name = 'iris_model' AND is_active = TRUE
ORDER BY created_at DESC;

-- Modèles avec leur taille de fichier
SELECT name, version, minio_object_key,
       ROUND(file_size_bytes / 1024.0, 1) AS size_kb,
       algorithm, accuracy
FROM model_metadata
WHERE is_active = TRUE
ORDER BY file_size_bytes DESC;
```

### Prédictions

```sql
-- Nombre de prédictions par modèle aujourd'hui
SELECT model_name, model_version, COUNT(*) AS nb_predictions
FROM predictions
WHERE timestamp::date = CURRENT_DATE
GROUP BY model_name, model_version
ORDER BY nb_predictions DESC;

-- Latence moyenne par modèle sur 7 jours
SELECT model_name,
       AVG(response_time_ms)  AS avg_ms,
       MIN(response_time_ms)  AS min_ms,
       MAX(response_time_ms)  AS max_ms,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_ms
FROM predictions
WHERE timestamp >= NOW() - INTERVAL '7 days'
  AND status = 'success'
GROUP BY model_name;

-- Prédictions par utilisateur ce mois
SELECT u.username, COUNT(*) AS nb_predictions,
       u.rate_limit_per_day,
       ROUND(COUNT(*) * 100.0 / (30 * u.rate_limit_per_day), 1) AS pct_quota
FROM predictions p
JOIN users u ON p.user_id = u.id
WHERE p.timestamp >= DATE_TRUNC('month', NOW())
GROUP BY u.username, u.rate_limit_per_day
ORDER BY nb_predictions DESC;

-- Taux d'erreur par modèle
SELECT model_name,
       COUNT(*) AS total,
       SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
       ROUND(100.0 * SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) / COUNT(*), 2) AS error_rate_pct
FROM predictions
GROUP BY model_name
ORDER BY error_rate_pct DESC;

-- Toutes les prédictions pour une observation donnée
SELECT p.id_obs, p.model_name, p.model_version,
       p.prediction_result, p.timestamp,
       u.username
FROM predictions p
JOIN users u ON p.user_id = u.id
WHERE p.id_obs = 'obs-2024-001';
```

### Évaluation — comparaison prédiction vs résultat observé

```sql
-- Joindre prédictions et résultats observés
SELECT
    p.id_obs,
    p.model_name,
    p.prediction_result AS prediction,
    o.observed_result   AS actual,
    CASE WHEN p.prediction_result = o.observed_result THEN 1 ELSE 0 END AS correct,
    p.timestamp         AS predicted_at,
    o.date_time         AS observed_at
FROM predictions p
INNER JOIN observed_results o
    ON p.id_obs = o.id_obs AND p.model_name = o.model_name
WHERE p.model_name = 'iris_model'
ORDER BY p.timestamp DESC;

-- Précision réelle d'un modèle en production
SELECT
    p.model_name,
    COUNT(*) AS total,
    SUM(CASE WHEN p.prediction_result = o.observed_result THEN 1 ELSE 0 END) AS correct,
    ROUND(
        100.0 * SUM(CASE WHEN p.prediction_result = o.observed_result THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS accuracy_pct
FROM predictions p
INNER JOIN observed_results o ON p.id_obs = o.id_obs AND p.model_name = o.model_name
GROUP BY p.model_name;
```

### Monitoring et maintenance

```sql
-- Volume de prédictions par heure sur 24h
SELECT DATE_TRUNC('hour', timestamp) AS heure,
       COUNT(*) AS volume
FROM predictions
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY heure;

-- Quota consommé aujourd'hui par utilisateur actif
SELECT u.username, u.rate_limit_per_day,
       COUNT(p.id) AS predictions_today,
       u.rate_limit_per_day - COUNT(p.id) AS remaining
FROM users u
LEFT JOIN predictions p ON p.user_id = u.id
    AND p.timestamp::date = CURRENT_DATE
WHERE u.is_active = TRUE
GROUP BY u.id, u.username, u.rate_limit_per_day;

-- Taille de chaque table
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

---

## Exemples Python complets

### Analyse des prédictions avec pandas

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

# Charger les prédictions des 30 derniers jours
df_predictions = pd.read_sql("""
    SELECT
        p.id,
        p.model_name,
        p.model_version,
        p.id_obs,
        p.prediction_result,
        p.response_time_ms,
        p.timestamp,
        p.status,
        u.username
    FROM predictions p
    JOIN users u ON p.user_id = u.id
    WHERE p.timestamp >= NOW() - INTERVAL '30 days'
    ORDER BY p.timestamp DESC
""", engine)

print(f"Total prédictions : {len(df_predictions)}")
print(f"Modèles utilisés : {df_predictions['model_name'].unique()}")
print(f"Latence moyenne : {df_predictions['response_time_ms'].mean():.1f} ms")
print(df_predictions.groupby('model_name')['response_time_ms'].describe())
```

### Calculer la précision réelle d'un modèle

```python
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

def evaluate_model(model_name: str) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT
            p.id_obs,
            p.prediction_result::text AS predicted,
            o.observed_result::text   AS actual,
            p.timestamp
        FROM predictions p
        INNER JOIN observed_results o
            ON p.id_obs = o.id_obs AND p.model_name = o.model_name
        WHERE p.model_name = %(model_name)s
        ORDER BY p.timestamp DESC
    """, engine, params={"model_name": model_name})

    if df.empty:
        print(f"Aucune donnée pour le modèle '{model_name}'")
        return df

    accuracy = (df["predicted"] == df["actual"]).mean()
    print(f"\n=== {model_name} — {len(df)} observations ===")
    print(f"Accuracy réelle : {accuracy:.2%}")
    print(classification_report(df["actual"], df["predicted"]))
    return df

evaluate_model("iris_model")
```

### Monitorer la dérive de latence

```python
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

df = pd.read_sql("""
    SELECT
        DATE_TRUNC('hour', timestamp) AS hour,
        model_name,
        AVG(response_time_ms) AS avg_latency,
        COUNT(*) AS volume
    FROM predictions
    WHERE timestamp >= NOW() - INTERVAL '48 hours'
      AND status = 'success'
    GROUP BY DATE_TRUNC('hour', timestamp), model_name
    ORDER BY hour
""", engine)

for model, group in df.groupby("model_name"):
    print(f"\n{model}")
    print(group[["hour", "avg_latency", "volume"]].to_string(index=False))
```

### Exportation pour rapport

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

# Rapport mensuel complet
df = pd.read_sql("""
    SELECT
        DATE_TRUNC('day', p.timestamp)::date AS day,
        p.model_name,
        p.model_version,
        COUNT(*) AS total_predictions,
        SUM(CASE WHEN p.status = 'error' THEN 1 ELSE 0 END) AS errors,
        AVG(p.response_time_ms) AS avg_latency_ms,
        COUNT(DISTINCT p.user_id) AS unique_users
    FROM predictions p
    WHERE p.timestamp >= DATE_TRUNC('month', NOW())
    GROUP BY DATE_TRUNC('day', p.timestamp), p.model_name, p.model_version
    ORDER BY day, p.model_name
""", engine)

df.to_csv("rapport_mensuel.csv", index=False)
print("Rapport exporté dans rapport_mensuel.csv")
print(df)
```

### Connexion async (même pattern que l'API)

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func, text

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5433/sklearn_api"
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_stats():
    async with AsyncSessionLocal() as session:
        # Nombre de prédictions aujourd'hui
        result = await session.execute(
            text("SELECT COUNT(*) FROM predictions WHERE timestamp::date = CURRENT_DATE")
        )
        count = result.scalar()
        print(f"Prédictions aujourd'hui : {count}")

        # Modèles en production
        result = await session.execute(
            text("SELECT name, version, accuracy FROM model_metadata WHERE is_production = TRUE")
        )
        for row in result.fetchall():
            print(f"Production: {row.name} v{row.version} — accuracy={row.accuracy}")

asyncio.run(get_stats())
```

---

## Migrations avec Alembic

```bash
# Créer une nouvelle migration après modification des modèles ORM
alembic revision --autogenerate -m "description du changement"

# Appliquer toutes les migrations en attente
alembic upgrade head

# Revenir à la migration précédente
alembic downgrade -1

# Voir l'historique des migrations
alembic history

# Voir la migration actuelle
alembic current
```

---

## Sauvegarde et restauration

```bash
# Sauvegarde complète
docker exec predictml-postgres pg_dump -U postgres sklearn_api > backup_$(date +%Y%m%d).sql

# Restauration
docker exec -i predictml-postgres psql -U postgres sklearn_api < backup_20240115.sql

# Sauvegarde d'une table spécifique
docker exec predictml-postgres pg_dump -U postgres -t predictions sklearn_api > predictions_backup.sql
```
