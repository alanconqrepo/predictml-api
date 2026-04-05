# Coding Standards — predictml-api

> Ce fichier définit les règles de codage du projet. Il doit être respecté lors de toute session Claude Code ou contribution manuelle.

---

## 1. Outils de formatage et de lint

| Outil | Rôle | Commande |
|---|---|---|
| **black** | Formatage automatique (line-length=100) | `black src/` |
| **ruff** | Lint (E, F, I, N, W) | `ruff check src/` |

**Avant chaque commit :**
```bash
ruff check src/ && black --check src/
```

**Corriger automatiquement :**
```bash
ruff check src/ --fix && black src/
```

---

## 2. Style de code

### Longueur de ligne
- Maximum **100 caractères** (géré par black)
- E501 ignoré dans ruff (black fait autorité)

### Imports
- Triés automatiquement par ruff (règle I)
- Ordre : stdlib → third-party → local
- **Jamais d'import inutilisé** — ruff F401 le détecte

### Nommage
- Fonctions et variables : `snake_case`
- Classes : `PascalCase`
- Constantes : `UPPER_SNAKE_CASE`
- Méthodes privées/utilitaires internes : préfixe `_` (ex: `_utcnow`)

---

## 3. Type hints

- **Obligatoires** sur toutes les fonctions publiques (paramètres + retour)
- Utiliser `Any` (capital, depuis `typing`) — **jamais `any`** (builtin)
- Exemples corrects :
  ```python
  from typing import Any
  def upload_model(self, model: Any, name: str) -> str: ...
  def _utcnow() -> datetime: ...
  ```

---

## 4. Logging

- **Jamais de `print()` en code de production**
- Déclarer un logger en haut de chaque module :
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```
- Niveaux à respecter :
  - `logger.debug(...)` — informations de débogage
  - `logger.info(...)` — informations de flux normal
  - `logger.warning(...)` — situations anormales non-bloquantes
  - `logger.error(...)` — erreurs gérées

---

## 5. Docstrings

- **Obligatoires** sur toutes les fonctions et méthodes publiques
- Format **Google-style**, rédigé en **français**
- Exemple :
  ```python
  def create_user(db: AsyncSession, username: str, token: str) -> User:
      """Crée un nouvel utilisateur en base de données.

      Args:
          db: Session SQLAlchemy async active.
          username: Nom d'utilisateur unique.
          token: Token Bearer de l'utilisateur.

      Returns:
          L'objet User créé et persisté.

      Raises:
          IntegrityError: Si le username existe déjà.
      """
  ```
- Les fonctions privées (préfixe `_`) peuvent avoir une docstring courte d'une ligne

---

## 6. Fonctions utilitaires partagées

- Toute fonction utilitaire utilisée dans plusieurs modules doit être centralisée dans **`src/core/utils.py`**
- Exemples : `_utcnow()`, helpers de formatage, etc.
- **Ne pas dupliquer** — importer depuis `src.core.utils`

---

## 7. Gestion du code mort

- **Supprimer** tout import inutilisé (ruff F401)
- **Supprimer** toute variable assignée mais non utilisée (ruff F841)
- **Supprimer** toute configuration spécifique à une technologie non utilisée (ex: SQLite sur un projet PostgreSQL)

---

## 8. SQLAlchemy

- Comparaison booléenne idiomatique :
  ```python
  # Correct
  ModelMetadata.is_production.is_(True)
  # Incorrect
  ModelMetadata.is_production == True
  ```
- Toujours utiliser les sessions async (`AsyncSession`) dans les services

---

## 9. Tests

- Chaque nouvel endpoint doit avoir des tests dans `tests/`
- Couvrir : cas nominal, auth manquante (401), token invalide (401/403), ressource absente (404), conflit (409)
- Utiliser `TestClient` de FastAPI — pas de Docker requis
- Nommage : `test_<endpoint>_<scenario>.py` ou regrouper par ressource

---

## 10. Commits

- Messages en **français**, à l'**impératif présent**
- Format : `<type>: <description courte>`
- Types : `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Exemple : `refactor: centraliser _utcnow() dans src/core/utils.py`

---

## 11. Dépendances

Toute nouvelle dépendance doit être ajoutée dans **les deux fichiers** :
- `requirements.txt` — Dockerfile API
- `pyproject.toml` — CI GitHub Actions

Pour le dashboard Streamlit : `streamlit_app/requirements.txt`
