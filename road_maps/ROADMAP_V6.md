# ROADMAP V6 — predictml-api

> **Perspective** : Data Scientist / MLOps Engineer — Avril 2026

---

## État des lieux

Les ROADMAP V3, V4 et V5 sont entièrement en production. L'API couvre désormais :

- Prédiction simple + batch, SHAP local/global (`POST /explain`, `GET /models/{name}/feature-importance`)
- Validation stricte du schéma d'entrée, calibration, tendance de confiance
- A/B testing + shadow avec significativité statistique (Chi-² / Mann-Whitney)
- Drift (Z-score + PSI + null rate), compute-baseline, performance drift
- Retrain manuel + planifié (cron), auto-promotion, audit log + rollback
- Purge RGPD, export CSV/JSONL (prédictions + observed results), import CSV ground truth
- Couverture ground truth (`GET /observed-results/stats`), préchauffage cache (`POST /warmup`)
- Prometheus `/metrics`, dashboard Streamlit 8 pages, email + webhook d'alerte

**Ce roadmap ne propose que des améliorations à valeur réelle pour ≥ 80 % des utilisateurs.**

---

## CHUNK 1 — Priorité HAUTE · Difficulté FACILE

*Chaque feature est implémentable en moins d'une demi-journée.*

---

### 1.1 `GET /predictions/{id}` — Lookup direct d'une prédiction

#### Pourquoi

Quand une alerte de monitoring signale une prédiction problématique ou qu'un client remonte
un identifiant, il est impossible de l'aller chercher directement. Il faut appeler
`GET /predictions?id_obs=...` (liste filtrée), parcourir le résultat et identifier l'entrée
manuellement. Ce pattern "récupérer une liste pour trouver une ligne" est une friction
quotidienne pour les data scientists qui déboguent en production. Un lookup par ID est
l'opération CRUD la plus basique — son absence est incohérente avec le reste de l'API.

#### Endpoint

```
GET /predictions/{id}

Réponse : PredictionResponse (schéma déjà existant)
{
  "id": 4821,
  "model_name": "iris",
  "model_version": "2.0.0",
  "id_obs": "client_78",
  "input_features": {"petal_length": 5.1, "petal_width": 1.8},
  "prediction_result": "virginica",
  "probabilities": {"setosa": 0.02, "versicolor": 0.11, "virginica": 0.87},
  "response_time_ms": 23,
  "timestamp": "2026-04-20T14:32:00",
  "status": "success"
}
```

Contrôle d'accès : un utilisateur ne voit que ses propres prédictions ; un admin voit tout.

#### Implémentation

- **`src/services/db_service.py`** : `get_prediction_by_id(db, prediction_id)` —
  `SELECT * FROM predictions WHERE id = ?` avec `scalar_one_or_none()`.
- **`src/api/predict.py`** : endpoint `GET /predictions/{prediction_id}` (Bearer auth).
  Retourne 404 si inexistant, 403 si l'utilisateur ne possède pas la prédiction.
- Aucun nouveau schéma — `PredictionResponse` déjà défini dans `src/schemas/prediction.py`.

---

### 1.2 `GET /predictions/{id}/explain` — Explication d'une prédiction stockée

#### Pourquoi

`POST /explain` nécessite de re-soumettre les features. Or, `input_features` est déjà
stocké sur chaque prédiction en base. Pour analyser une prédiction passée (audit,
debugging, feedback client), il faut soit mémoriser les features envoyées à l'époque,
soit les reconstruire depuis un autre système — effort inutile quand la donnée existe.
Ce pattern "explication post-hoc sur historique" est l'un des plus courants en
explicabilité ML en production : l'analyste consulte les prédictions, identifie une
anomalie, et veut immédiatement comprendre pourquoi le modèle a tranché ainsi.

#### Endpoint

```
GET /predictions/{id}/explain

Réponse : ExplainOutput (schéma déjà existant)
{
  "prediction_id": 4821,
  "model_name": "iris",
  "shap_values": {
    "petal_length": 0.42,
    "petal_width":  0.31,
    "sepal_length": 0.08,
    "sepal_width":  -0.03
  },
  "base_value": 0.33,
  "model_type": "tree"
}
```

Garde : 422 si `status != "success"` ou si `input_features` est null.

#### Implémentation

- **`src/api/predict.py`** : endpoint `GET /predictions/{prediction_id}/explain`.
  1. Appel interne `DBService.get_prediction_by_id()` (feature 1.1).
  2. Charge le modèle via `model_service.load_model(db, model_name, model_version)`
     (déjà implémenté, avec cache Redis).
  3. Appel `shap_service.compute_shap_explanation()` avec les `input_features` stockées
     (déjà implémenté — supporte tree et linear models).
- `ExplainOutput` déjà défini dans `src/schemas/prediction.py` — aucun nouveau schéma.

---

### 1.3 `POST /predict?explain=true` — SHAP inline dans la réponse de prédiction

#### Pourquoi

Obtenir une prédiction ET son explication nécessite actuellement deux appels API distincts :
`POST /predict` puis `POST /explain`. Pour les applications temps-réel (dashboards métier,
systèmes de triage, interfaces de scoring) qui affichent "pourquoi ce score ?", ce double
aller-retour double la latence perçue et complexifie le code client. L'explication inline
est la fonctionnalité la plus demandée lors des démos MLOps — c'est souvent l'argument
décisif lors d'un audit réglementaire ("prouvez que votre modèle n'est pas une boîte noire").

#### Endpoint

```
POST /predict?explain=true
Body : { "model_name": "iris", "features": {"petal_length": 5.1, ...} }

Réponse enrichie :
{
  "prediction": "virginica",
  "probabilities": { "virginica": 0.87, ... },
  "response_time_ms": 34,
  ...
  "shap_values": {
    "petal_length": 0.42,
    "petal_width":  0.31,
    "sepal_length": 0.08
  },
  "shap_base_value": 0.33
}
```

Si le modèle ne supporte pas SHAP (pas d'arbre ni de modèle linéaire) : `shap_values: null`,
sans erreur — comportement silencieux pour ne pas casser les intégrations existantes.

#### Implémentation

- **`src/api/predict.py`** : ajouter `explain: bool = Query(False)` sur `POST /predict`.
  Après la prédiction, si `explain=True` : appel conditionnel à `shap_service.compute_shap_explanation()`
  dans un bloc `try/except` (silently skip si modèle non supporté).
- **`src/schemas/prediction.py`** : étendre `PredictionOutput` avec deux champs optionnels :
  ```python
  shap_values: dict[str, float] | None = None
  shap_base_value: float | None = None
  ```
- `?explain=true` est ignoré sur `POST /predict-batch` (déjà async) et sur les shadow calls.
- Overhead latence : SHAP sur modèle sklearn simple ≈ 10–50 ms — acceptable en temps réel.

---

## CHUNK 2 — Priorité HAUTE/MOYENNE · Difficulté MOYENNE

*Chaque feature est implémentable en 1–2 jours.*

---

### 2.1 Seuils d'alerte par modèle

#### Pourquoi

`supervision_reporter.py` utilise des variables d'environnement globales
(`ERROR_RATE_ALERT_THRESHOLD`, `PERFORMANCE_DRIFT_ALERT_THRESHOLD`) comme seuils
d'alerte pour tous les modèles. En production, les SLAs sont hétérogènes : un modèle
de recommandation tolère 15 % d'erreurs, un modèle de détection de fraude doit alerter
dès 2 %. Abaisser le seuil global pour protéger le modèle critique génère du bruit
intempestif sur tous les autres — les équipes finissent par ignorer les alertes
(**alert fatigue**). Des seuils par modèle permettent de calibrer la surveillance sur
les vrais niveaux de risque métier, sans modifier la configuration globale.

#### Endpoint

```
PATCH /models/{name}/{version}
Body : {
  "alert_thresholds": {
    "accuracy_min": 0.90,
    "error_rate_max": 0.05,
    "drift_auto_alert": true
  }
}

Réponse : ModelCreateResponse (schéma existant, enrichi)
```

`drift_auto_alert: true` déclenche un email/webhook dès qu'une feature passe en statut
`critical` lors du check toutes les 6 heures (si `false`, les alertes drift sont muettes
pour ce modèle même si le statut est critique).

#### Implémentation

- **`src/db/models/model_metadata.py`** : nouveau champ `alert_thresholds: Mapped[dict | None]`
  (`JSON`, nullable).
- **Migration Alembic** : `alembic/versions/20260421_add_alert_thresholds.py`.
- **`src/schemas/model.py`** : nouveau schéma Pydantic `AlertThresholds` avec validation
  (`accuracy_min ∈ [0,1]`, `error_rate_max ∈ [0,1]`). Ajout dans `ModelUpdateInput`.
- **`src/tasks/supervision_reporter.py`** : fonction utilitaire `_get_model_threshold()` :
  retourne le seuil du modèle si défini, sinon le seuil global de config. Appliquée avant
  chaque appel `send_error_spike_alert()`, `send_performance_alert()`, `send_drift_alert()`.

---

### 2.2 Snapshot des données d'entraînement au retrain

#### Pourquoi

Quand les performances d'un modèle changent après un retrain, la question est invariablement :
"a-t-on utilisé les bonnes données ?" Aujourd'hui, `POST /retrain` retourne les logs
stdout/stderr et les métriques finales, mais **rien sur les données utilisées**.
`TRAIN_START_DATE` et `TRAIN_END_DATE` sont injectées dans le subprocess mais ne sont pas
persistées sur la nouvelle version créée. Six mois plus tard, impossible de savoir sur
quelle fenêtre temporelle a été entraîné le modèle v3.2.0.
Ce snapshot est la base du **data card** (documentation ML standardisée) et la première
chose qu'un auditeur ou un régulateur demande. Il est capturé automatiquement sans
modifier le contrat des scripts `train.py` existants.

#### Données capturées

```json
{
  "train_start_date": "2026-01-01",
  "train_end_date":   "2026-03-31",
  "trained_at":       "2026-04-21T08:32:00",
  "n_rows":           null,
  "feature_stats":    null,
  "label_distribution": null
}
```

Les trois derniers champs (`n_rows`, `feature_stats`, `label_distribution`) sont populés
automatiquement si le script `train.py` les imprime sur stdout en JSON (même mécanisme
que `accuracy`/`f1_score` déjà supportés) :

```python
# Dans train.py (optionnel)
print(json.dumps({
    "accuracy": 0.95,
    "f1_score": 0.94,
    "n_rows": 12450,
    "label_distribution": {"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}
}))
```

#### Implémentation

- **`src/db/models/model_metadata.py`** : nouveau champ `training_stats: Mapped[dict | None]`
  (`JSON`, nullable).
- **Migration Alembic** : `alembic/versions/20260421_add_training_stats.py`.
- **`src/api/models.py`** (fonction `retrain_model()`) : après création de la nouvelle version,
  construire `training_stats` avec les dates (déjà disponibles dans `retrain_request`) +
  enrichissement optionnel depuis `_parse_json_from_stdout(logs.stdout)`.
- **`src/schemas/model.py`** : ajouter `training_stats: dict | None` dans `ModelGetResponse`
  et `RetrainResponse`.
- **`CLAUDE.md`** : documenter les clés optionnelles `n_rows`, `label_distribution`,
  `feature_stats` dans la section "Retour des métriques" du script `train.py`.

---

## Tableau récapitulatif

| # | Feature | Endpoint(s) / Composant | Priorité | Difficulté | Effort estimé |
|---|---------|------------------------|----------|------------|---------------|
| 1.1 | Lookup prédiction par ID | `GET /predictions/{id}` | **Haute** | Facile | ~1 h |
| 1.2 | Expliquer une prédiction stockée | `GET /predictions/{id}/explain` | **Haute** | Facile | ~2 h |
| 1.3 | SHAP inline sur /predict | `POST /predict?explain=true` | **Haute** | Facile | ~3 h |
| 2.1 | Seuils d'alerte par modèle | PATCH metadata + supervision_reporter | **Haute** | Moyenne | ~1 jour |
| 2.2 | Snapshot données d'entraînement | Champ `training_stats` + retrain | Moyenne | Moyenne | ~1.5 jours |

---

## Fichiers à modifier

| Fichier | Modifications |
|---------|--------------|
| `src/api/predict.py` | `GET /predictions/{id}`, `GET /predictions/{id}/explain`, `POST /predict?explain=true` |
| `src/api/models.py` | `PATCH`: `alert_thresholds` ; `POST /retrain`: capture `training_stats` |
| `src/services/db_service.py` | `get_prediction_by_id()` |
| `src/schemas/prediction.py` | `PredictionOutput` : `shap_values`, `shap_base_value` (optionnels) |
| `src/schemas/model.py` | Schéma `AlertThresholds` ; `training_stats` dans `ModelGetResponse` + `RetrainResponse` |
| `src/db/models/model_metadata.py` | `alert_thresholds` (JSON), `training_stats` (JSON) |
| `src/tasks/supervision_reporter.py` | `_get_model_threshold()` à 2 niveaux (model vs global) |
| `alembic/versions/` | 2 nouvelles migrations |
| `CLAUDE.md` | Section train.py : clés optionnelles `n_rows`, `label_distribution`, `feature_stats` |

---

## Ce qui n'a PAS été retenu

| Idée | Raison |
|------|--------|
| `GET /models/{name}/shadow-comparison` | `agreement_rate` + distribution par version déjà dans `ab-compare` |
| `POST /explain/batch` | `POST /predict-batch` + boucle `/explain` côté client suffit pour l'analyse offline |
| Rate limiting par modèle | Quota par utilisateur + monitoring par modèle couvre 95 % des besoins |
| Prediction replay (rejouer l'historique sur un nouveau modèle) | Complexité disproportionnée ; le shadow mode couvre ce besoin en pré-déploiement |
| Comparaison cross-modèles (`/models/compare?names=iris,wine`) | `monitoring/overview` + `performance` par modèle couvrent les cas d'usage |
| `GET /models/{name}/parent-chain` (arbre de lignée) | `parent_version` (V5 pending) suffit pour l'usage courant ; arbre = complexité DB non justifiée |
