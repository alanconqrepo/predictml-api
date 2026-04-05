# Plan — Amélioration qualité de code

## Problèmes identifiés

### Code mort / inutile
| Fichier | Problème | Ligne |
|---|---|---|
| `src/core/config.py` | Import `Path` inutilisé | ~5 |
| `src/core/security.py` | Import `datetime` inutilisé | ~4 |
| `src/api/predict.py` | Variable `status_str = "error"` inutilisée | ~185 |
| `src/db/models/model_metadata.py` | `__table_args__` SQLite (projet PostgreSQL) | ~66-68 |

### Duplication
| Problème | Fichiers concernés |
|---|---|
| `_utcnow()` définie 5 fois à l'identique | `src/services/db_service.py`, `src/db/models/user.py`, `src/db/models/prediction.py`, `src/db/models/model_metadata.py`, `src/db/models/observed_result.py` |

### Type hints incorrects (`any` au lieu de `Any`)
| Fichier | Ligne |
|---|---|
| `src/services/minio_service.py` | ~44, ~115 |
| `src/services/db_service.py` | ~122 |

### `print()` en production (doit être `logger`)
| Fichier | Ligne | Niveau suggéré |
|---|---|---|
| `src/api/predict.py` | ~204 | `logger.error` |
| `src/api/models.py` | ~318 | `logger.warning` |
| `src/api/models.py` | ~327 | `logger.warning` |

### Violations de style
| Fichier | Problème | Ligne |
|---|---|---|
| `src/api/models.py` | `== True` sur colonne SQLAlchemy (E712) | ~286 |

### Docstrings manquantes
| Fichier | Méthode |
|---|---|
| `src/db/models/user.py` | `__repr__` |
| `src/db/models/model_metadata.py` | `__repr__` |
| `src/db/models/observed_result.py` | `__repr__` |
| `src/services/db_service.py` | `update_user_last_login` |

---

## Étapes d'implémentation

### Étape 1 — Créer `src/core/utils.py` (centraliser `_utcnow`)
```python
# src/core/utils.py
from datetime import datetime, timezone

def _utcnow() -> datetime:
    """Retourne la date/heure UTC courante (sans info de timezone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
```
Puis dans chaque fichier concerné : supprimer la définition locale, ajouter `from src.core.utils import _utcnow`.

### Étape 2 — Supprimer le code mort
- `src/core/config.py` : supprimer `from pathlib import Path`
- `src/core/security.py` : supprimer `from datetime import datetime`
- `src/api/predict.py` : supprimer `status_str = "error"`
- `src/db/models/model_metadata.py` : supprimer le bloc `__table_args__`

### Étape 3 — Corriger les type hints
- `src/services/minio_service.py` + `src/services/db_service.py` : `any` → `Any`, ajouter `from typing import Any`

### Étape 4 — Remplacer `print()` par du logging
- 3 occurrences dans `predict.py` et `models.py` (loggers déjà présents dans ces fichiers)

### Étape 5 — Corriger E712
- `src/api/models.py` ligne ~286 : `== True` → `.is_(True)`

### Étape 6 — Ajouter les docstrings manquantes
- `__repr__` dans les 3 modèles ORM
- `update_user_last_login` dans db_service.py

### Étape 7 — Mettre à jour `CLAUDE.md`
Ajouter une section référençant `CODING_STANDARDS.md` et la commande lint.

---

## Vérification finale

```bash
# Zéro erreur de lint
ruff check src/

# Zéro diff de formatage
black --check src/

# Tous les tests passent
pytest tests/ -v
```

---

## Fichiers à créer
- `src/core/utils.py` — utilitaires partagés (nouveau)
- `CODING_STANDARDS.md` — règles de codage (créé)

## Fichiers à modifier
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
