# Coding Standards — predictml-api

> This file defines the project's coding rules. It must be followed in every Claude Code session or manual contribution.

---

## 1. Formatting and linting tools

| Tool | Role | Command |
|---|---|---|
| **black** | Automatic formatting (line-length=100) | `black src/` |
| **ruff** | Lint (E, F, I, N, W) | `ruff check src/` |

**Before each commit:**
```bash
ruff check src/ && black --check src/
```

**Auto-fix:**
```bash
ruff check src/ --fix && black src/
```

---

## 2. Code style

### Line length
- Maximum **100 characters** (managed by black)
- E501 ignored in ruff (black takes precedence)

### Imports
- Automatically sorted by ruff (rule I)
- Order: stdlib → third-party → local
- **Never an unused import** — ruff F401 detects it

### Naming
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private/internal utility methods: `_` prefix (e.g. `_utcnow`)

---

## 3. Type hints

- **Required** on all public functions (parameters + return)
- Use `Any` (capitalized, from `typing`) — **never `any`** (builtin)
- Correct examples:
  ```python
  from typing import Any
  def upload_model(self, model: Any, name: str) -> str: ...
  def _utcnow() -> datetime: ...
  ```

---

## 4. Structured logging (JSON)

- **Never use `print()` in production code**
- **Never use `import logging` / `logging.getLogger()`** — use `structlog` exclusively
- Declare a logger at the top of each module:
  ```python
  import structlog
  logger = structlog.get_logger(__name__)
  ```
- Calls with **structured kwargs** (never `%s` formatting):
  ```python
  # Correct — usable by ELK/Datadog/CloudWatch
  logger.info("Model loaded", model_name=name, version=str(v))
  logger.error("MinIO upload error", object_name=key, error=str(e))

  # Incorrect
  logger.info("Model '%s' v%s loaded", name, v)
  ```
- Levels to follow:
  - `logger.debug(...)` — debug information
  - `logger.info(...)` — normal flow information
  - `logger.warning(...)` — abnormal non-blocking situations
  - `logger.error(...)` — handled errors
- The global configuration is in **`src/core/logging.py`** (`setup_logging(debug: bool)`).
  It is called once in `src/main.py` at startup.
  - `DEBUG=False` (production) → **JSON** output on stdout (compatible with ELK/Datadog/CloudWatch)
  - `DEBUG=True` (development) → colored readable output in the terminal

---

## 5. Docstrings

- **Required** on all public functions and methods
- **Google-style** format, written in **English**
- Example:
  ```python
  def create_user(db: AsyncSession, username: str, token: str) -> User:
      """Creates a new user in the database.

      Args:
          db: Active async SQLAlchemy session.
          username: Unique username.
          token: User's Bearer token.

      Returns:
          The created and persisted User object.

      Raises:
          IntegrityError: If the username already exists.
      """
  ```
- Private functions (prefix `_`) can have a short one-line docstring

---

## 6. Shared utility functions

- Any utility function used in multiple modules must be centralized in **`src/core/utils.py`**
- Examples: `_utcnow()`, formatting helpers, etc.
- **Don't duplicate** — import from `src.core.utils`

---

## 7. Dead code management

- **Remove** all unused imports (ruff F401)
- **Remove** all assigned but unused variables (ruff F841)
- **Remove** any configuration specific to an unused technology (e.g., SQLite in a PostgreSQL project)

---

## 8. SQLAlchemy

- Idiomatic boolean comparison:
  ```python
  # Correct
  ModelMetadata.is_production.is_(True)
  # Incorrect
  ModelMetadata.is_production == True
  ```
- Always use async sessions (`AsyncSession`) in services

---

## 9. Tests

- Every new endpoint must have tests in `tests/`
- Cover: nominal case, missing auth (401), invalid token (401/403), missing resource (404), conflict (409)
- Use FastAPI's `TestClient` — no Docker required
- Naming: `test_<endpoint>_<scenario>.py` or group by resource

---

## 10. Commits

- Messages in **English**, in the **imperative present**
- Format: `<type>: <short description>`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Example: `refactor: centralize _utcnow() in src/core/utils.py`

---

## 11. Dependencies

Any new dependency must be added in **both files**:
- `requirements.txt` — API Dockerfile
- `pyproject.toml` — CI GitHub Actions

For the Streamlit dashboard: `streamlit_app/requirements.txt`
