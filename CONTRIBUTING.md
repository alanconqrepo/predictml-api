# Contributing to predictml-api

## Workflow

Each work session = one branch. Never push directly to `main`.

### 1. Start from an up-to-date main

```bash
git checkout main
git pull origin main
```

### 2. Create a branch

```bash
git checkout -b feature/feature-name
# or
git checkout -b fix/bug-name
```

### 3. Work, commit

```bash
git add file.py
git commit -m "feat: short description"
```

Commit prefixes: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`

### 4. Push and open a PR

```bash
git push -u origin feature/feature-name
```

Then open a Pull Request targeting `main` on GitHub.

### 5. CI and merge

CI automatically runs `pytest tests/ -v`.
Merging is blocked until tests pass.
Once CI is green, merge the PR and delete the branch.

### 6. Clean up locally

```bash
git checkout main
git pull origin main
git branch -d feature/feature-name
```

## Run tests locally

```bash
# All tests
pytest tests/ -v

# By feature
pytest tests/test_golden_tests.py -v          # golden tests
pytest tests/test_drift.py -v                 # drift input + output
pytest tests/test_ab_significance.py -v       # A/B significance
pytest tests/test_auto_promotion_policy.py -v # auto-promotion / demotion
pytest tests/test_scheduled_retraining.py -v  # scheduled retrain + drift-triggered
```

Tests use FastAPI's `TestClient` — no Docker required.

---

## Scope of a complete contribution

Any contribution that adds or modifies a feature must include:

### New API endpoint

- [ ] Route in `src/api/` (models.py, predict.py, users.py, etc.)
- [ ] Pydantic schema in `src/schemas/` (request + response)
- [ ] Service in `src/services/` if non-trivial business logic
- [ ] Tests in `tests/test_<feature>.py` (unit + edge cases)
- [ ] Entry in the endpoints table in `README.md`
- [ ] Section in `documentation/API_REFERENCE.md` (description + Python example + JSON schema)

### New Streamlit page

- [ ] File in `streamlit_app/pages/N_Name.py` (follow the pattern of existing pages)
- [ ] Required methods in `streamlit_app/utils/api_client.py`
- [ ] Update the page tree in `documentation/ARCHITECTURE.md`

### New DB table

- [ ] ORM model in `src/db/models/`
- [ ] Alembic migration in `alembic/versions/`
- [ ] Documentation in `documentation/DATABASE.md` (DDL + columns + SQL example)

---

## Recent features to know (V10–V14)

Before touching code related to the following areas, read the corresponding sections of `CLAUDE.md`:

### V10–V12 — Monitoring & ML

| Feature | Key files |
|---|---|
| **Output drift** (label shift) | `src/services/drift_service.py` · `GET /models/{name}/output-drift` |
| **Enriched shadow-compare** | `src/api/models.py` · `GET /models/{name}/shadow-compare` |
| **Auto-demotion / circuit breaker** | `src/services/auto_promotion_service.py` · `promotion_policy.auto_demote` |
| **Drift-triggered retrain** | `src/tasks/supervision_reporter.py` (lines 258–287) · `retrain_schedule.trigger_on_drift` |
| **Golden tests** | `src/services/golden_test_service.py` · `streamlit_app/pages/9_Golden_Tests.py` |
| **Anomaly detection** | `GET /predictions/anomalies` · `DBService.get_anomalies()` |
| **Model card** | `GET /models/{name}/{version}/card` · `ModelCardResponse` in `src/schemas/model.py` |
| **Confidence filters** | `GET /predictions?min_confidence=&max_confidence=` |
| **Retrain history** | `GET /models/{name}/retrain-history` · `ModelHistory` table |
| **Free-text search** | `GET /models?search=` · `ModelService.get_available_models()` |

### V13–V14 — Security & robustness (PRs #159–#177)

| Feature | Key files |
|---|---|
| **HMAC-SHA256 models** | Signed on upload + verified before `joblib.load()` · `SECRET_KEY` in `src/core/config.py` |
| **Sandbox retrain scripts** | Import whitelist + resource limits · `src/tasks/retrain_scheduler.py` · see `documentation/TRAIN_SCRIPT_GUIDE.md` |
| **Per-IP rate limiting** | Middleware `slowapi` · `src/core/rate_limit.py` · HTTP 429 if exceeded |
| **Token expiration** | `TOKEN_LIFETIME_DAYS` in `src/core/config.py` · verification in `src/core/security.py` |
| **Audit logging** | Admin operations logged as JSON · `src/core/audit.py` · called from `src/api/models.py`, `src/api/users.py` |
| **Pagination /users** | `skip` / `limit` in `GET /users` · `src/api/users.py` |
| **Auth on /health/dependencies** | Requires admin role · `src/api/monitoring.py` |
| **METRICS_TOKEN in production** | Bearer token for `GET /metrics` · `src/core/config.py` · doc in `documentation/DOCKER.md` |
| **Name/version validation** | Validated formats (path traversal prevention) · `src/api/models.py` |

### Associated test files

| File | Tested feature |
|---|---|
| `tests/test_rate_limit.py` | Per-IP rate limiting (HTTP 429) |
| `tests/test_security.py` | Bearer auth, token expiration, roles |
| `tests/test_monitoring_api.py` | Monitoring endpoints, auth on `/health/dependencies` |
