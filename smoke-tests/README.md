# Smoke Tests

Manual tests against a live environment. Require Docker services to be running.

## Prerequisites

```bash
docker-compose up -d
```

## Running the smoke tests

```bash
# With the default admin token
python smoke-tests/test_multimodel_api.py

# With a custom token or URL
API_TOKEN=mon-token API_BASE_URL=http://localhost:8000 python smoke-tests/test_multimodel_api.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | API base URL |
| `API_TOKEN` | admin token | Bearer authentication token |

## Difference with `tests/`

| | `smoke-tests/` | `tests/` |
|---|---|---|
| Server required | Yes (Docker) | No (mocked TestClient) |
| Launched by | Manually | `pytest` |
| Purpose | Verify the live environment | Validate business logic |
