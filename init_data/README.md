# init_data

Initialization scripts to run **once only** on first deployment.

## Execution order

### Step 1 — Create sklearn models locally

```bash
python init_data/create_multiple_models.py
```

**What it does:** trains 3 scikit-learn models (iris, wine, cancer) and saves them as `.joblib` files in `Models/`.

**When:** before `init_db.py`, if the `Models/` directory is empty or missing.

---

### Step 2 — Initialize the database and upload the models

```bash
# Inside the Docker container (recommended)
docker exec predictml-api python init_data/init_db.py

# Or locally if services are accessible
python init_data/init_db.py
```

**What it does:**
1. Creates the PostgreSQL tables (`users`, `predictions`, `model_metadata`)
2. Creates the admin user with a randomly generated token
3. Uploads all `.joblib` files from `Models/` to MinIO and registers them in the database

**When:** once, after the first `docker-compose up -d --build`.

> **Important:** the admin token is displayed only once in the terminal. Save it immediately.

---

## Prerequisites

Docker services must be started before step 2:

```bash
docker-compose up -d
```

## Re-initialization

If you need to start from scratch:

```bash
# Remove volumes and recreate
docker-compose down -v
docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py
```

## Summary

| Script | Prerequisites | Frequency |
|---|---|---|
| `create_multiple_models.py` | Python + scikit-learn | Once (or if `Models/` is empty) |
| `init_db.py` | Docker started | Once per deployment |
