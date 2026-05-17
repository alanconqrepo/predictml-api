# Contribuer à predictml-api

## Workflow

Chaque session de travail = une branche. On ne pousse jamais directement sur `main`.

### 1. Partir de main à jour

```bash
git checkout main
git pull origin main
```

### 2. Créer une branche

```bash
git checkout -b feature/nom-feature
# ou
git checkout -b fix/nom-bug
```

### 3. Travailler, commiter

```bash
git add fichier.py
git commit -m "feat: description courte"
```

Préfixes de commit : `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`

### 4. Pousser et ouvrir une PR

```bash
git push -u origin feature/nom-feature
```

Puis ouvrir une Pull Request vers `main` sur GitHub.

### 5. CI et merge

Le CI lance `pytest tests/ -v` automatiquement.
Le merge est bloqué tant que les tests ne passent pas.
Une fois le CI vert, merger la PR et supprimer la branche.

### 6. Nettoyer en local

```bash
git checkout main
git pull origin main
git branch -d feature/nom-feature
```

## Lancer les tests en local

```bash
# Tous les tests
pytest tests/ -v

# Par fonctionnalité
pytest tests/test_golden_tests.py -v          # golden tests
pytest tests/test_drift.py -v                 # drift input + output
pytest tests/test_ab_significance.py -v       # significativité A/B
pytest tests/test_auto_promotion_policy.py -v # auto-promotion / demotion
pytest tests/test_scheduled_retraining.py -v  # retrain planifié + drift-triggered
```

Les tests utilisent `TestClient` FastAPI — aucun Docker requis.

---

## Périmètre d'une contribution complète

Toute contribution qui ajoute ou modifie une fonctionnalité doit inclure :

### Nouvel endpoint API

- [ ] Route dans `src/api/` (models.py, predict.py, users.py, etc.)
- [ ] Schéma Pydantic dans `src/schemas/` (request + response)
- [ ] Service dans `src/services/` si logique métier non triviale
- [ ] Tests dans `tests/test_<fonctionnalité>.py` (unitaire + edge cases)
- [ ] Entrée dans le tableau des endpoints de `README.md`
- [ ] Section dans `documentation/API_REFERENCE.md` (description + exemple Python + schéma JSON)

### Nouvelle page Streamlit

- [ ] Fichier dans `streamlit_app/pages/N_Nom.py` (suivre le pattern des pages existantes)
- [ ] Méthodes nécessaires dans `streamlit_app/utils/api_client.py`
- [ ] Mise à jour de l'arbre des pages dans `documentation/ARCHITECTURE.md`

### Nouvelle table DB

- [ ] Modèle ORM dans `src/db/models/`
- [ ] Migration Alembic dans `alembic/versions/`
- [ ] Documentation dans `documentation/DATABASE.md` (DDL + colonnes + exemple SQL)

---

## Fonctionnalités récentes à connaître (V10–V14)

Avant de toucher au code lié aux domaines suivants, lire les sections correspondantes du `CLAUDE.md` :

### V10–V12 — Monitoring & ML

| Fonctionnalité | Fichiers clés |
|---|---|
| **Drift output** (label shift) | `src/services/drift_service.py` · `GET /models/{name}/output-drift` |
| **Shadow-compare enrichi** | `src/api/models.py` · `GET /models/{name}/shadow-compare` |
| **Auto-demotion / circuit breaker** | `src/services/auto_promotion_service.py` · `promotion_policy.auto_demote` |
| **Drift-triggered retrain** | `src/tasks/supervision_reporter.py` (lignes 258–287) · `retrain_schedule.trigger_on_drift` |
| **Golden tests** | `src/services/golden_test_service.py` · `streamlit_app/pages/9_Golden_Tests.py` |
| **Anomaly detection** | `GET /predictions/anomalies` · `DBService.get_anomalies()` |
| **Model card** | `GET /models/{name}/{version}/card` · `ModelCardResponse` dans `src/schemas/model.py` |
| **Confidence filters** | `GET /predictions?min_confidence=&max_confidence=` |
| **Retrain history** | `GET /models/{name}/retrain-history` · `ModelHistory` table |
| **Free-text search** | `GET /models?search=` · `ModelService.get_available_models()` |

### V13–V14 — Sécurité & robustesse (PRs #159–#177)

| Fonctionnalité | Fichiers clés |
|---|---|
| **HMAC-SHA256 modèles** | Signature à l'upload + vérification avant `joblib.load()` · `SECRET_KEY` dans `src/core/config.py` |
| **Sandbox retrain scripts** | Whitelist d'imports + limites ressources · `src/tasks/retrain_scheduler.py` · voir `documentation/TRAIN_SCRIPT_GUIDE.md` |
| **Rate limiting per-IP** | Middleware `slowapi` · `src/core/rate_limit.py` · HTTP 429 si dépassé |
| **Token expiration** | `TOKEN_LIFETIME_DAYS` dans `src/core/config.py` · vérification dans `src/core/security.py` |
| **Audit logging** | Opérations admin loguées en JSON · `src/core/audit.py` · appelé depuis `src/api/models.py`, `src/api/users.py` |
| **Pagination /users** | `skip` / `limit` dans `GET /users` · `src/api/users.py` |
| **Auth sur /health/dependencies** | Requiert rôle admin · `src/api/monitoring.py` |
| **METRICS_TOKEN en production** | Token Bearer pour `GET /metrics` · `src/core/config.py` · doc dans `documentation/DOCKER.md` |
| **Validation name/version** | Formats validés (prévention path traversal) · `src/api/models.py` |

### Fichiers de tests associés

| Fichier | Fonctionnalité testée |
|---|---|
| `tests/test_rate_limit.py` | Rate limiting per-IP (HTTP 429) |
| `tests/test_security.py` | Auth Bearer, token expiration, roles |
| `tests/test_monitoring_api.py` | Endpoints monitoring, auth sur `/health/dependencies` |
