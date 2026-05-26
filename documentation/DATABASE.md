# Database Documentation — PredictML API

Complete SQL schema, useful queries and Python connection examples.

---

## Connection

**Connection parameters**

| Parameter | Value |
|---|---|
| Host | `localhost` |
| Port | `5433` |
| Database | `sklearn_api` |
| User | `postgres` |
| Password | `postgres` |

---

## Connecting to the Database

### Via psql (command line)

```bash
# Direct connection via Docker
docker exec -it predictml-postgres psql -U postgres -d sklearn_api

# Or with full parameters
psql -h localhost -p 5433 -U postgres -d sklearn_api
```

### Via Python — SQLAlchemy async (used by the API)

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
        print(f"Number of predictions: {count}")
```

### Via Python — psycopg2 (classic synchronous connection)

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
print(f"Number of predictions: {count}")

cursor.close()
conn.close()
```

### Via Python — pandas (for analysis)

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

# Load predictions into a DataFrame
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

## Database Schema

### Table `users`

Stores user accounts and their authentication tokens.

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

**Columns**

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Unique identifier |
| `username` | VARCHAR(50) | Unique username |
| `email` | VARCHAR(100) | Unique email |
| `api_token` | VARCHAR(255) | Bearer token for authentication |
| `role` | VARCHAR(20) | `admin`, `user` or `readonly` |
| `is_active` | BOOLEAN | Account active or deactivated |
| `rate_limit_per_day` | INTEGER | Daily prediction quota |
| `created_at` | TIMESTAMP | Creation date |
| `updated_at` | TIMESTAMP | Last modification |
| `last_login` | TIMESTAMP | Last authenticated login |

---

### Table `model_metadata`

Stores metadata for deployed ML models.

```sql
CREATE TABLE model_metadata (
    id                SERIAL PRIMARY KEY,
    name              VARCHAR(100) NOT NULL,
    version           VARCHAR(50)  NOT NULL,
    minio_bucket      VARCHAR(100),
    minio_object_key  VARCHAR(255),               -- path in MinIO
    file_size_bytes   INTEGER,
    file_hash         VARCHAR(64),                -- SHA256 hash of the file
    description       TEXT,
    algorithm         VARCHAR(100),               -- e.g. 'RandomForestClassifier'
    features_count    INTEGER,
    classes           JSONB,                      -- e.g. [0, 1, 2] or ["cat","dog"]
    accuracy          FLOAT,
    precision         FLOAT,
    recall            FLOAT,
    f1_score          FLOAT,
    training_metrics  JSONB,                      -- additional metrics
    mlflow_run_id     VARCHAR(255),               -- MLflow run ID
    user_id_creator   INTEGER REFERENCES users(id),
    trained_by        VARCHAR(100),
    training_date     TIMESTAMP,
    training_dataset  VARCHAR(255),
    training_params   JSONB,                      -- hyperparameters
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

**Columns**

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Unique identifier |
| `name` | VARCHAR(100) | Model name |
| `version` | VARCHAR(50) | Version (e.g. "1.0", "2024-01") |
| `minio_bucket` | VARCHAR(100) | MinIO bucket containing the file |
| `minio_object_key` | VARCHAR(255) | File path in MinIO |
| `file_size_bytes` | INTEGER | Size of the `.joblib` file in bytes |
| `file_hash` | VARCHAR(64) | SHA256 hash of the file |
| `description` | TEXT | Human-readable description |
| `algorithm` | VARCHAR(100) | Algorithm class |
| `features_count` | INTEGER | Number of expected features |
| `classes` | JSONB | Output class labels |
| `accuracy` | FLOAT | Accuracy on the test set |
| `precision` | FLOAT | Precision (ML sense) |
| `recall` | FLOAT | Recall |
| `f1_score` | FLOAT | F1 score |
| `training_metrics` | JSONB | Additional metrics |
| `mlflow_run_id` | VARCHAR(255) | Link to an MLflow run |
| `user_id_creator` | INTEGER FK | Model creator |
| `trained_by` | VARCHAR(100) | Trainer name |
| `training_date` | TIMESTAMP | Training date |
| `training_dataset` | VARCHAR(255) | Dataset used |
| `training_params` | JSONB | Training hyperparameters |
| `is_active` | BOOLEAN | Model available or not |
| `is_production` | BOOLEAN | Production version (one per `name`) |
| `deployment_mode` | VARCHAR | `production`, `ab_test`, `shadow` |
| `traffic_weight` | FLOAT | A/B traffic fraction (0.0–1.0) |
| `confidence_threshold` | FLOAT | Min confidence threshold (`low_confidence`) |
| `feature_baseline` | JSONB | Per-feature stats for drift detection |
| `tags` | JSONB | List of free tags |
| `webhook_url` | VARCHAR | Post-prediction callback URL |
| `train_script_object_key` | VARCHAR | MinIO key of the `train.py` script |
| `parent_version` | VARCHAR | Source version of retrain (lineage) |
| `promotion_policy` | JSONB | Auto-promotion policy (`min_accuracy`, `max_latency_p95_ms`, `min_sample_validation`, `auto_promote`) |
| `retrain_schedule` | JSONB | Cron schedule (`cron`, `lookback_days`, `enabled`, `last_run_at`, `next_run_at`) |
| `alert_thresholds` | JSONB | Model-specific alert thresholds (overrides global env variables) |
| `training_stats` | JSONB | Snapshot from the last retrain (`n_rows`, `feature_stats`, `label_distribution`, `trained_at`) |
| `created_at` | TIMESTAMP | Creation date |
| `updated_at` | TIMESTAMP | Last update |
| `deprecated_at` | TIMESTAMP | Deprecation date |

---

### Table `predictions`

Log of all predictions made via the API.

```sql
CREATE TABLE predictions (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id),
    model_name       VARCHAR(100) NOT NULL,
    model_version    VARCHAR(50),
    id_obs           VARCHAR(255),
    input_features   JSONB NOT NULL,                  -- sent features
    prediction_result JSONB NOT NULL,                 -- returned result
    probabilities    JSONB,                           -- per-class probabilities
    response_time_ms FLOAT NOT NULL,
    timestamp        TIMESTAMP NOT NULL DEFAULT NOW(),
    client_ip        VARCHAR(45),                     -- IPv4/IPv6
    user_agent       TEXT,
    status           VARCHAR(20) NOT NULL DEFAULT 'success',  -- 'success' or 'error'
    error_message    TEXT
);

CREATE INDEX ix_predictions_id          ON predictions (id);
CREATE INDEX ix_predictions_user_id     ON predictions (user_id);
CREATE INDEX ix_predictions_model_name  ON predictions (model_name);
CREATE INDEX ix_predictions_timestamp   ON predictions (timestamp);
CREATE INDEX ix_predictions_id_obs      ON predictions (id_obs);
```

**Columns**

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Unique identifier |
| `user_id` | INTEGER FK | User who made the prediction |
| `model_name` | VARCHAR(100) | Name of the model used |
| `model_version` | VARCHAR(50) | Version used |
| `id_obs` | VARCHAR(255) | Observation identifier (link with `observed_results`) |
| `input_features` | JSONB | Features dictionary |
| `prediction_result` | JSONB | Prediction result |
| `probabilities` | JSONB | List of per-class probabilities |
| `response_time_ms` | FLOAT | Latency in milliseconds |
| `timestamp` | TIMESTAMP | Prediction timestamp |
| `client_ip` | VARCHAR(45) | Client IP address |
| `user_agent` | TEXT | HTTP User-Agent |
| `status` | VARCHAR(20) | `success` or `error` |
| `error_message` | TEXT | Error message if failure |

---

### Table `observed_results`

Actual observed results, to evaluate models after prediction.

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

**Columns**

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Unique identifier |
| `id_obs` | VARCHAR(255) | Observation identifier |
| `model_name` | VARCHAR(100) | Model concerned |
| `observed_result` | JSONB | Actual observed result |
| `date_time` | TIMESTAMP | Observation date |
| `user_id` | INTEGER FK | User who submitted the result |

**Unique constraint**: `(id_obs, model_name)` — an upsert updates if the pair already exists.

---

### Table `golden_tests`

Regression test cases (golden set) to validate that a model always produces the expected outputs.

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

**Columns**

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Unique identifier |
| `model_name` | VARCHAR(100) | Target model |
| `input_features` | JSONB | Input features for the test case |
| `expected_output` | JSONB | Expected output (class, value, etc.) |
| `description` | TEXT | Optional description of the case |
| `created_at` | TIMESTAMP | Creation date |
| `created_by_user_id` | INTEGER FK | Test case author |

**Example — create and list cases**

```sql
-- Test case for the iris model
INSERT INTO golden_tests (model_name, input_features, expected_output, description)
VALUES (
    'iris_model',
    '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}',
    '"setosa"',
    'Typical Iris setosa — all nominal features'
);

-- List all cases for a model
SELECT id, description, expected_output, created_at
FROM golden_tests
WHERE model_name = 'iris_model'
ORDER BY created_at DESC;
```

---

## Relationship Diagram

```
users
  │
  ├──< predictions (user_id → users.id)
  ├──< model_metadata (user_id_creator → users.id)
  ├──< observed_results (user_id → users.id)
  └──< golden_tests (created_by_user_id → users.id)

predictions.id_obs ──── observed_results.id_obs  (application-level join)
```

---

## Useful SQL Queries

### Users

```sql
-- All active users with their quota
SELECT username, email, role, rate_limit_per_day, last_login
FROM users
WHERE is_active = TRUE
ORDER BY created_at DESC;

-- Check users who have never made a prediction
SELECT u.username, u.email, u.created_at
FROM users u
LEFT JOIN predictions p ON p.user_id = u.id
WHERE p.id IS NULL;
```

### Models

```sql
-- Production models
SELECT name, version, algorithm, accuracy, f1_score, created_at
FROM model_metadata
WHERE is_production = TRUE AND is_active = TRUE;

-- All versions of a model
SELECT version, accuracy, is_production, created_at
FROM model_metadata
WHERE name = 'iris_model' AND is_active = TRUE
ORDER BY created_at DESC;

-- Models with their file size
SELECT name, version, minio_object_key,
       ROUND(file_size_bytes / 1024.0, 1) AS size_kb,
       algorithm, accuracy
FROM model_metadata
WHERE is_active = TRUE
ORDER BY file_size_bytes DESC;
```

### Predictions

```sql
-- Number of predictions per model today
SELECT model_name, model_version, COUNT(*) AS nb_predictions
FROM predictions
WHERE timestamp::date = CURRENT_DATE
GROUP BY model_name, model_version
ORDER BY nb_predictions DESC;

-- Average latency per model over 7 days
SELECT model_name,
       AVG(response_time_ms)  AS avg_ms,
       MIN(response_time_ms)  AS min_ms,
       MAX(response_time_ms)  AS max_ms,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_ms
FROM predictions
WHERE timestamp >= NOW() - INTERVAL '7 days'
  AND status = 'success'
GROUP BY model_name;

-- Predictions per user this month
SELECT u.username, COUNT(*) AS nb_predictions,
       u.rate_limit_per_day,
       ROUND(COUNT(*) * 100.0 / (30 * u.rate_limit_per_day), 1) AS pct_quota
FROM predictions p
JOIN users u ON p.user_id = u.id
WHERE p.timestamp >= DATE_TRUNC('month', NOW())
GROUP BY u.username, u.rate_limit_per_day
ORDER BY nb_predictions DESC;

-- Error rate per model
SELECT model_name,
       COUNT(*) AS total,
       SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
       ROUND(100.0 * SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) / COUNT(*), 2) AS error_rate_pct
FROM predictions
GROUP BY model_name
ORDER BY error_rate_pct DESC;

-- All predictions for a given observation
SELECT p.id_obs, p.model_name, p.model_version,
       p.prediction_result, p.timestamp,
       u.username
FROM predictions p
JOIN users u ON p.user_id = u.id
WHERE p.id_obs = 'obs-2024-001';
```

### Evaluation — comparing prediction vs observed result

```sql
-- Join predictions and observed results
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

-- Real accuracy of a production model
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

### Monitoring and Maintenance

```sql
-- Prediction volume by hour over 24h
SELECT DATE_TRUNC('hour', timestamp) AS hour,
       COUNT(*) AS volume
FROM predictions
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour;

-- Quota consumed today by active user
SELECT u.username, u.rate_limit_per_day,
       COUNT(p.id) AS predictions_today,
       u.rate_limit_per_day - COUNT(p.id) AS remaining
FROM users u
LEFT JOIN predictions p ON p.user_id = u.id
    AND p.timestamp::date = CURRENT_DATE
WHERE u.is_active = TRUE
GROUP BY u.id, u.username, u.rate_limit_per_day;

-- Size of each table
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

---

## Complete Python Examples

### Analysing Predictions with pandas

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

# Load predictions from the last 30 days
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

print(f"Total predictions: {len(df_predictions)}")
print(f"Models used: {df_predictions['model_name'].unique()}")
print(f"Average latency: {df_predictions['response_time_ms'].mean():.1f} ms")
print(df_predictions.groupby('model_name')['response_time_ms'].describe())
```

### Computing Real Accuracy of a Model

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
        print(f"No data for model '{model_name}'")
        return df

    accuracy = (df["predicted"] == df["actual"]).mean()
    print(f"\n=== {model_name} — {len(df)} observations ===")
    print(f"Real accuracy: {accuracy:.2%}")
    print(classification_report(df["actual"], df["predicted"]))
    return df

evaluate_model("iris_model")
```

### Monitoring Latency Drift

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

### Exporting for a Report

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5433/sklearn_api")

# Complete monthly report
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

df.to_csv("monthly_report.csv", index=False)
print("Report exported to monthly_report.csv")
print(df)
```

### Async Connection (same pattern as the API)

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func, text

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5433/sklearn_api"
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_stats():
    async with AsyncSessionLocal() as session:
        # Number of predictions today
        result = await session.execute(
            text("SELECT COUNT(*) FROM predictions WHERE timestamp::date = CURRENT_DATE")
        )
        count = result.scalar()
        print(f"Predictions today: {count}")

        # Production models
        result = await session.execute(
            text("SELECT name, version, accuracy FROM model_metadata WHERE is_production = TRUE")
        )
        for row in result.fetchall():
            print(f"Production: {row.name} v{row.version} — accuracy={row.accuracy}")

asyncio.run(get_stats())
```

---

## Migrations with Alembic

```bash
# Create a new migration after modifying ORM models
alembic revision --autogenerate -m "description of change"

# Apply all pending migrations
alembic upgrade head

# Revert to the previous migration
alembic downgrade -1

# View migration history
alembic history

# View current migration
alembic current
```

---

## Backup and Restore

```bash
# Full backup
docker exec predictml-postgres pg_dump -U postgres sklearn_api > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i predictml-postgres psql -U postgres sklearn_api < backup_20240115.sql

# Backup a specific table
docker exec predictml-postgres pg_dump -U postgres -t predictions sklearn_api > predictions_backup.sql
```
