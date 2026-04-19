# predictml-api

API FastAPI de prédiction ML avec PostgreSQL, MinIO et MLflow. Dashboard admin Streamlit.

## Stack

- **API** : FastAPI async — port 8000
- **Dashboard** : Streamlit Admin — port 8501
- **DB** : PostgreSQL 16 — port 5433
- **Stockage modèles** : MinIO — port 9000 / console 9001
- **Experiment tracking** : MLflow — port 5000

## Structure

```
src/
├── api/            # Endpoints (models.py, predict.py, users.py, observed_results.py)
├── core/           # Config & auth (config.py, security.py)
├── db/             # ORM SQLAlchemy + service DB
├── services/       # Logique métier (db_service, model_service, minio_service)
├── schemas/        # Pydantic
└── main.py

streamlit_app/      # Dashboard admin multipage
├── app.py          # Login + accueil
├── utils/          # api_client.py, auth.py
└── pages/          # 1_Users, 2_Models, 3_Predictions, 4_Stats, 5_Code_Example

tests/              # Pytest — tests automatisés
smoke-tests/        # Tests manuels contre Docker live
init_data/          # Scripts one-shot (create_multiple_models, init_db)
Models/             # Fichiers .pkl locaux
notebooks/          # Jupyter
alembic/            # Migrations DB
```

## Qualité de code

Les règles de codage sont définies dans **[CODING_STANDARDS.md](./CODING_STANDARDS.md)**.

```bash
# Vérifier le lint
ruff check src/

# Vérifier le formatage
black --check src/

# Corriger automatiquement
ruff check src/ --fix && black src/
```

## Commandes clés

```bash
# Démarrer
docker-compose up -d

# Initialiser (premier déploiement uniquement)
docker exec predictml-api python init_data/init_db.py

# Tests automatisés
pytest tests/ -v

# Smoke tests
python smoke-tests/test_multimodel_api.py

# Logs
docker-compose logs -f api
docker-compose logs -f streamlit

# PostgreSQL
docker exec -it predictml-postgres psql -U postgres -d sklearn_api
```

## Credentials

| Service | Valeur |
|---|---|
| Admin token | `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA` |
| DB | `postgres / postgres` |
| MinIO | `minioadmin / minioadmin` |

## Dépendances

Toute nouvelle dépendance doit être ajoutée dans **les deux fichiers** :
- `requirements.txt` — utilisé par le Dockerfile de l'API
- `pyproject.toml` — utilisé par la CI GitHub Actions

Pour le dashboard Streamlit, les dépendances sont dans `streamlit_app/requirements.txt`.

## Tests

Les tests dans `tests/` utilisent `TestClient` de FastAPI — aucun Docker requis.
Ils couvrent : auth, endpoints publics, logique du `ModelService`.

```bash
pytest tests/ -v           # tous les tests
pytest tests/test_api.py   # endpoints uniquement
```

Les smoke tests dans `smoke-tests/` nécessitent Docker et frappent l'API live.

## Endpoints principaux

- `POST /predict` — Prédiction (Bearer auth)
- `GET /predictions` — Historique (Bearer auth)
- `GET/POST/PATCH/DELETE /models` — Gestion modèles
- `GET/POST/PATCH/DELETE /users` — Gestion utilisateurs (admin)
- `POST/GET /observed-results` — Résultats observés
- `PATCH /users/{id}` avec `{"regenerate_token": true}` — Renouveler un token (admin)
- `POST /models/{name}/{version}/retrain` — Ré-entraîner un modèle (admin)
- `PATCH /models/{name}/policy` — Définir la politique d'auto-promotion post-retrain (admin)
- `GET /models/{name}/feature-importance` — Importance globale des features (SHAP agrégé, Bearer auth)
- `GET /models/{name}/ab-compare` — Comparaison A/B avec test de significativité statistique (Bearer auth)
- `DELETE /predictions/purge` — Purge RGPD des prédictions anciennes (admin)

## Fonctionnalité A/B Significativité statistique

L'endpoint `GET /models/{name}/ab-compare` enrichit sa réponse avec un bloc `ab_significance` :

```json
{
  "ab_significance": {
    "metric": "error_rate",
    "test": "chi2",
    "p_value": 0.023,
    "significant": true,
    "confidence_level": 0.95,
    "winner": "v2.0.0",
    "min_samples_needed": 200,
    "current_samples": { "v1.0.0": 450, "v2.0.0": 380 }
  }
}
```

### Logique de sélection du test

| Condition | Test utilisé | Métrique |
|---|---|---|
| ≥ 1 erreur observée dans l'un des groupes | Chi-² sur tableau de contingence | `error_rate` |
| 0 erreur + temps de réponse disponibles | Mann-Whitney U | `response_time_ms` |
| Données insuffisantes (< 2 versions actives) | — | `ab_significance: null` |

### Calcul de `min_samples_needed`

- **Chi-²** : formule de puissance basée sur l'effet de taille de Cohen h (comparaison de proportions)
- **Mann-Whitney U** : formule basée sur Cohen d (distributions continues)
- Puissance cible : 80 % — seuil : `confidence_level` (défaut 95 %)

### Implémentation

- Service : `src/services/ab_significance_service.py`
- Tests unitaires : `tests/test_ab_significance.py` (20 tests)

## Fonctionnalité Retrain (ré-entraînement)

### Comment ça fonctionne

1. À l'upload d'un modèle (`POST /models`), fournir optionnellement un script `train_file`
   (fichier Python `train.py`).
2. Si fourni, le script est **validé statiquement** puis stocké dans MinIO
   (`{name}/v{version}_train.py`).
3. L'admin peut déclencher un ré-entraînement via `POST /models/{name}/{version}/retrain`
   en précisant une plage de dates.
4. Le script s'exécute dans un **sous-processus isolé** (timeout 600 s) avec les variables
   d'environnement injectées automatiquement.
5. Le `.pkl` produit est uploadé dans MinIO et enregistré comme **nouvelle version** du modèle.
6. Si `set_production: true`, la nouvelle version est automatiquement mise en production.
7. L'intégralité des logs `stdout`/`stderr` est retournée dans la réponse et affichée dans
   le dashboard Streamlit.

### Contraintes du script `train.py` (vérifiées à l'upload)

Le script doit impérativement :

| Contrainte | Détail |
|---|---|
| Syntaxe Python valide | Vérifié via `ast.parse()` |
| Référencer `TRAIN_START_DATE` | Lire `os.environ["TRAIN_START_DATE"]` |
| Référencer `TRAIN_END_DATE` | Lire `os.environ["TRAIN_END_DATE"]` |
| Référencer `OUTPUT_MODEL_PATH` | Chemin où sauvegarder le `.pkl` |
| Sauvegarder le modèle | Appel à `pickle.dump`, `joblib.dump` ou `save_model` |

### Variables d'environnement injectées par l'API

| Variable | Description |
|---|---|
| `TRAIN_START_DATE` | Date début (YYYY-MM-DD) |
| `TRAIN_END_DATE` | Date fin (YYYY-MM-DD) |
| `OUTPUT_MODEL_PATH` | Chemin absolu pour le `.pkl` produit |
| `MLFLOW_TRACKING_URI` | URI MLflow (optionnel) |
| `MODEL_NAME` | Nom du modèle source (optionnel) |

### Retour des métriques

Pour que l'API mette à jour `accuracy` et `f1_score` de la nouvelle version,
imprimer sur **stdout** un JSON sur la **dernière ligne JSON** de la sortie :

```json
{"accuracy": 0.95, "f1_score": 0.94}
```

### Upload avec train.py

```bash
curl -X POST http://localhost:8000/models \
  -H "Authorization: Bearer <token>" \
  -F "name=mon_modele" -F "version=1.0.0" \
  -F "file=@mon_modele.pkl" \
  -F "train_file=@init_data/example_train.py"
```

### Lancer un ré-entraînement

```bash
curl -X POST http://localhost:8000/models/mon_modele/1.0.0/retrain \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-01-01",
    "end_date":   "2025-12-31",
    "new_version": "1.1.0",
    "set_production": false
  }'
```

Un exemple complet de `train.py` respectant le contrat est disponible dans
`init_data/example_train.py`.

## Fonctionnalité Auto-promotion post-retrain

### Comment ça fonctionne

1. L'admin définit une politique via `PATCH /models/{name}/policy`.
2. La politique est stockée dans `ModelMetadata.promotion_policy` (champ JSON)
   et propagée à **toutes les versions actives** du modèle.
3. À la fin de chaque ré-entraînement (`POST /models/{name}/{version}/retrain`),
   si `auto_promote: true` :
   - Récupère les paires (prédiction, résultat observé) historiques du modèle.
   - Si `len(paires) < min_sample_validation` → non promu.
   - Si `min_accuracy` défini : vérifie l'accuracy sur les N dernières paires.
   - Si `max_latency_p95_ms` défini : vérifie le P95 de latence des prédictions.
   - Si tous les critères sont satisfaits → `is_production = true` automatiquement.
4. La réponse du retrain inclut `auto_promoted: true|false` et `auto_promote_reason`.

### Champs de la politique

| Champ | Type | Défaut | Description |
|---|---|---|---|
| `min_accuracy` | float [0–1] | null | Précision minimale requise |
| `max_latency_p95_ms` | float > 0 | null | Latence P95 maximale en ms |
| `min_sample_validation` | int ≥ 1 | 10 | Nombre minimal de paires de validation |
| `auto_promote` | bool | false | Activer l'auto-promotion |

### Sémantique de `auto_promoted` dans la réponse du retrain

| Valeur | Signification |
|---|---|
| `null` | Pas de policy configurée, ou `set_production=True` (promotion manuelle) |
| `false` | Policy évaluée : critères non satisfaits (voir `auto_promote_reason`) |
| `true` | Policy évaluée : critères satisfaits, version promue en production |

### Définir une politique

```bash
curl -X PATCH http://localhost:8000/models/mon_modele/policy \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "min_accuracy": 0.90,
    "max_latency_p95_ms": 200,
    "min_sample_validation": 50,
    "auto_promote": true
  }'
```

### Implémentation

- Endpoint : `PATCH /models/{name}/policy` dans `src/api/models.py`
- Service d'évaluation : `src/services/auto_promotion_service.py`
- Champ DB : `promotion_policy` (JSON) dans `src/db/models/model_metadata.py`
- Migration : `alembic/versions/20260419_5ab8c1f0_add_promotion_policy.py`
- Tests : `tests/test_auto_promotion_policy.py` (25 tests)

## Fonctionnalité Purge RGPD (rétention des données)

### Pourquoi

La table `predictions` grossit indéfiniment. Sur un déploiement actif (1 000 prédictions/jour),
elle atteint 365 000 lignes/an. Sans politique de rétention, les performances des requêtes
analytiques se dégradent et la conformité RGPD devient un problème.

### Signature

```
DELETE /predictions/purge
  ?older_than_days=90    # supprimer les prédictions > 90 jours
  &model_name=iris       # optionnel : purger un seul modèle
  &dry_run=true          # simuler sans supprimer (défaut : true)
```

### Réponse

```json
{
  "dry_run": false,
  "deleted_count": 12450,
  "oldest_remaining": "2026-01-15T08:32:00",
  "models_affected": ["iris", "wine"],
  "linked_observed_results_count": 3
}
```

- `linked_observed_results_count > 0` → avertissement : des prédictions supprimées sont liées à des
  `observed_results` (perte de données de performance historiques).

### Comportement

- `dry_run=true` par défaut — aucune suppression sans confirmation explicite (`dry_run=false`).
- Filtre SQL : `WHERE timestamp < now() - interval 'N days'` + filtre optionnel `model_name`.
- Admin uniquement.

### Implémentation

- Endpoint : `DELETE /predictions/purge` dans `src/api/predict.py`
- Service DB : `DBService.purge_predictions()` dans `src/services/db_service.py`
- Schéma réponse : `PurgeResponse` dans `src/schemas/prediction.py`
- Tests : `tests/test_predictions_purge.py` (16 tests)

### Exemple

```bash
# Simuler une purge (dry_run par défaut)
curl -X DELETE "http://localhost:8000/predictions/purge?older_than_days=90" \
  -H "Authorization: Bearer <admin_token>"

# Purger réellement les prédictions iris > 90 jours
curl -X DELETE "http://localhost:8000/predictions/purge?older_than_days=90&model_name=iris&dry_run=false" \
  -H "Authorization: Bearer <admin_token>"
```
