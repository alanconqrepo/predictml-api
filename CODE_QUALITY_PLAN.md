# Plan — Code quality improvement

## Identified issues

### Dead / unused code
| File | Issue | Line |
|---|---|---|
| `src/core/config.py` | Unused `Path` import | ~5 |
| `src/core/security.py` | Unused `datetime` import | ~4 |
| `src/api/predict.py` | Unused variable `status_str = "error"` | ~185 |
| `src/db/models/model_metadata.py` | `__table_args__` SQLite (PostgreSQL project) | ~66-68 |

### Duplication
| Issue | Files concerned |
|---|---|
| `_utcnow()` defined 5 times identically | `src/services/db_service.py`, `src/db/models/user.py`, `src/db/models/prediction.py`, `src/db/models/model_metadata.py`, `src/db/models/observed_result.py` |

### Incorrect type hints (`any` instead of `Any`)
| File | Line |
|---|---|
| `src/services/minio_service.py` | ~44, ~115 |
| `src/services/db_service.py` | ~122 |

### `print()` in production (should be `logger`)
| File | Line | Suggested level |
|---|---|---|
| `src/api/predict.py` | ~204 | `logger.error` |
| `src/api/models.py` | ~318 | `logger.warning` |
| `src/api/models.py` | ~327 | `logger.warning` |

### Style violations
| File | Issue | Line |
|---|---|---|
| `src/api/models.py` | `== True` on SQLAlchemy column (E712) | ~286 |

### Missing docstrings
| File | Method |
|---|---|
| `src/db/models/user.py` | `__repr__` |
| `src/db/models/model_metadata.py` | `__repr__` |
| `src/db/models/observed_result.py` | `__repr__` |
| `src/services/db_service.py` | `update_user_last_login` |

---

## Implementation steps

### Step 1 — Create `src/core/utils.py` (centralize `_utcnow`)
```python
# src/core/utils.py
from datetime import datetime, timezone

def _utcnow() -> datetime:
    """Returns the current UTC datetime (without timezone info)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
```
Then in each concerned file: remove the local definition, add `from src.core.utils import _utcnow`.

### Step 2 — Remove dead code
- `src/core/config.py`: remove `from pathlib import Path`
- `src/core/security.py`: remove `from datetime import datetime`
- `src/api/predict.py`: remove `status_str = "error"`
- `src/db/models/model_metadata.py`: remove the `__table_args__` block

### Step 3 — Fix type hints
- `src/services/minio_service.py` + `src/services/db_service.py`: `any` → `Any`, add `from typing import Any`

### Step 4 — Replace `print()` with logging
- 3 occurrences in `predict.py` and `models.py` (loggers already present in these files)

### Step 5 — Fix E712
- `src/api/models.py` line ~286: `== True` → `.is_(True)`

### Step 6 — Add missing docstrings
- `__repr__` in the 3 ORM models
- `update_user_last_login` in db_service.py

### Step 7 — Update `CLAUDE.md`
- Add a section referencing `CODING_STANDARDS.md` and the lint command.

---

## Final verification

```bash
# Zero lint errors
ruff check src/

# Zero formatting diff
black --check src/

# All tests pass
pytest tests/ -v
```

---

## Files to create
- `src/core/utils.py` — shared utilities (new)
- `CODING_STANDARDS.md` — coding rules (created)

## Files to modify
- `src/core/config.py`
- `src/core/security.py`
- `src/api/predict.py`
- `src/api/models.py`
- `src/services/minio_service.py`
- `src/services/db_service.py`
- `src/db/models/model_metadata.py`
- `src/db/models/user.py`
- `src/db/models/prediction.py`
- `src/db/models/observed_result.py`
- `CLAUDE.md`
