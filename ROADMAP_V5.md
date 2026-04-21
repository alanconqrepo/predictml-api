# ROADMAP V5 — predictml-api

> **Perspective** : Data Scientist / MLOps Engineer — Avril 2026

---

## État des lieux

Les ROADMAP V3 et V4 sont entièrement en production. L'API couvre désormais :

- Prédiction simple + batch, SHAP local/global, validation stricte du schéma d'entrée
- A/B testing + shadow avec significativité statistique (Chi-² / Mann-Whitney)
- Drift (Z-score + PSI), calibration, tendance de confiance, compute-baseline
- Retrain manuel + planifié (cron), auto-promotion, audit log + rollback
- Purge RGPD, export CSV/JSONL, rate limiting, webhooks, email de supervision
- Prometheus `/metrics`, dashboard Streamlit 8 pages

**Ce roadmap ne propose que des améliorations à valeur réelle pour ≥ 80 % des utilisateurs.**

---

## CHUNK 1 — Priorité HAUTE · Difficulté FACILE

*Chaque feature est implémentable en moins d'une journée.*

---

### 1.1 `GET /observed-results/stats` — Taux de couverture du ground truth

#### Pourquoi

Les métriques de performance (`GET /models/{name}/performance`) ne sont fiables qu'à condition
d'avoir suffisamment de résultats observés. Aujourd'hui, il est impossible de savoir rapidement
**combien de prédictions ont été labellisées** sans faire soi-même le calcul. Un taux de
couverture de 8 % vs 85 % change radicalement la confiance qu'on peut accorder aux courbes de
performance et aux décisions d'auto-promotion.

#### Endpoint

```
GET /observed-results/stats?model_name=iris

Réponse :
{
  "model_name": "iris",               // null si global
  "total_predictions": 12540,
  "labeled_count": 3210,
  "coverage_rate": 0.256,             // labeled / total
  "oldest_label": "2026-01-03T14:22:00",
  "newest_label": "2026-04-19T09:11:00",
  "by_version": [
    { "version": "2.0.0", "predictions": 8200, "labeled": 2900, "coverage": 0.354 },
    { "version": "1.0.0", "predictions": 4340, "labeled": 310,  "coverage": 0.071 }
  ]
}
```

Si `model_name` est omis : retourne le global + `by_model` (même structure).

#### Implémentation

- **`src/services/db_service.py`** : méthode `get_observed_results_stats(db, model_name)` —
  2 requêtes COUNT : une sur `predictions`, une sur `observed_results` avec LEFT JOIN optionnel.
  Utilise le pattern de `get_prediction_stats()` déjà en place.
- **`src/api/observed_results.py`** : `GET /observed-results/stats` (auth Bearer, accessible
  à tous les rôles).
- **`src/schemas/observed_result.py`** : nouveau schéma `ObservedResultsStatsResponse`.
- **Streamlit `pages/3_Predictions.py`** : encart "Couverture du ground truth" en haut de page —
  `st.metric("Couverture", "25.6 %")` + barre de progression par modèle.

---

### 1.2 `GET /observed-results/export` — Export du ground truth

#### Pourquoi

`GET /predictions/export` (CSV/JSONL) existe et est utilisé quotidiennement par les data
scientists pour l'analyse offline. L'asymétrie est gênante : les résultats observés —
données d'entraînement du cycle suivant — ne sont pas exportables sans passer par l'API paginée.
C'est un copier-coller quasi-exact de l'endpoint d'export des prédictions.

#### Endpoint

```
GET /observed-results/export
  ?model_name=iris
  &start=2026-01-01
  &end=2026-04-01
  &format=csv|jsonl    # défaut : csv

Réponse : StreamingResponse (même pattern que /predictions/export)
Colonnes CSV : id_obs, model_name, observed_result, date_time
```

#### Implémentation

- **`src/services/db_service.py`** : `get_observed_results_for_export(db, model_name, start, end)`
  — curseur + yield par lot de 500, identique à `get_predictions_for_export()`.
- **`src/api/observed_results.py`** : `GET /observed-results/export` (admin uniquement,
  données sensibles).
- Aucun nouveau schéma — `StreamingResponse` direct comme l'endpoint symétrique.

---

### 1.3 `POST /models/{name}/{version}/warmup` — Préchauffage du cache

#### Pourquoi

Les modèles se chargent paresseusement : la première prédiction après un déploiement ou un
redémarrage subit une latence élevée (download MinIO + désérialisation pickle). Dans un
déploiement A/B, cela crée une fausse asymétrie de latence entre la version en place (chaude)
et la nouvelle version (froide) qui biaise les métriques de comparaison.

Le `ModelService.load_model()` est déjà entièrement fonctionnel — il suffit d'un endpoint
pour l'appeler proactivement.

#### Endpoint

```
POST /models/{name}/{version}/warmup

Réponse :
{
  "model_name": "iris",
  "version": "2.0.0",
  "already_cached": false,
  "load_time_ms": 312,
  "cache_key": "iris:2.0.0"
}
```

#### Implémentation

- **`src/api/models.py`** : endpoint admin, appelle
  `await model_service.load_model(name, version)` dans un bloc try/except + mesure le temps.
- Retourne `already_cached: true` si le modèle est déjà en mémoire (interroge
  `model_service.get_cached_models()`).
- **Streamlit `pages/2_Models.py`** : bouton "Préchauffer" à côté de chaque version
  dans la vue liste — visible uniquement si le modèle n'est pas dans le cache (badge
  "En cache" / "Non chargé").

---

## CHUNK 2 — Priorité HAUTE/MOYENNE · Difficulté MOYENNE

*Chaque feature est implémentable en 1–2 jours.*

---

### 2.1 Taux de valeurs nulles dans la détection de drift

#### Pourquoi

Le drift actuel (Z-score + PSI) analyse uniquement les valeurs non-nulles. Il est aveugle aux
**patterns de données manquantes**, qui sont pourtant le signal de dégradation silencieux le
plus fréquent en production : un pipeline amont change, une source de données tombe, une
feature est renommée — le taux de nulls explose sans que les métriques de distribution ne
bougent.

#### Modification

Étendre `FeatureDriftResult` et `DriftService.compute_feature_drift()` :

```python
# Nouvelles métriques dans FeatureDriftResult
null_rate_production: float   # % de nulls sur la fenêtre courante
null_rate_baseline: float     # % de nulls au moment du compute-baseline
null_rate_status: str         # ok | warning | critical

# Seuils
# ok       : null_rate_production < baseline + 0.05  (écart absolu < 5 pts)
# warning  : écart 5–15 pts
# critical : écart > 15 pts ou null_rate_production > 0.30 (30 % de nulls)
```

`compute-baseline` enrichit `feature_baseline` avec `null_rate` par feature (simple
`COUNT(WHERE feature IS NULL) / COUNT(*)`).

La méthode `get_feature_production_stats()` dans `db_service.py` est étendue pour compter
les nulls depuis `input_features` (JSON) — extraction via requête SQLAlchemy.

Le statut global `drift_summary` prend en compte le `null_rate_status` comme quatrième
dimension (en plus de Z-score et PSI).

---

### 2.2 Streamlit — Page `8_Retrain.py` : gestion centralisée des retrains

#### Pourquoi

Le ré-entraînement et les plannings cron sont uniquement accessibles via l'API. Un admin
non-technique ne peut pas :
- Voir en un coup d'œil tous les modèles ayant un schedule actif
- Savoir quand le dernier retrain a tourné et si ça s'est bien passé
- Déclencher manuellement un retrain sans écrire un `curl`
- Configurer la promotion policy sans mémoriser le schéma JSON

#### Page `streamlit_app/pages/8_Retrain.py`

**Section 1 — Vue d'ensemble des schedules (admin)**
- Tableau `st.dataframe` : modèle · version · cron · `last_run_at` · `next_run_at` · statut
  (activé/désactivé)
- Badge coloré : actif / désactivé / aucun schedule

**Section 2 — Déclencher un retrain manuel**
- Sélecteur modèle/version + date pickers `start_date` / `end_date`
- Champ `new_version` (optionnel) + toggle `set_production`
- Bouton "Lancer" → spinner pendant l'appel → affiche `stdout`/`stderr` dans
  `st.code(logs, language="text")`

**Section 3 — Configurer un planning cron**
- Champ cron expression + aide contextuelle (`"0 3 * * 1"` = lundi 3h UTC)
- Slider `lookback_days` + toggle `auto_promote` + toggle `enabled`
- Bouton "Sauvegarder"

**Section 4 — Politique d'auto-promotion**
- Formulaire : `min_accuracy` (slider 0–1), `max_latency_p95_ms` (number input),
  `min_sample_validation` (number input), `auto_promote` (toggle)
- Affiche la policy actuelle avant modification

**`api_client.py`** : méthodes `trigger_retrain()`, `set_schedule()`, `set_policy()` déjà
présentes dans l'API client — vérifier qu'elles couvrent tous les paramètres.

---

### 2.3 `parent_version` — Traçabilité de la lignée des modèles

#### Pourquoi

Quand un modèle est ré-entraîné (`POST /retrain`), la nouvelle version créée n'a aucun lien
explicite avec la version source. Il est impossible de répondre à "ce modèle v3.0.0 a-t-il
été entraîné à partir de v2.0.0 ou directement depuis v1.0.0 ?" — information cruciale lors
d'un incident ou d'un audit réglementaire.

#### Modification

- **DB** : ajouter colonne `parent_version VARCHAR` nullable sur `model_metadata` (+ migration
  Alembic).
- **Retrain** (`src/api/models.py`) : lors de la création de la nouvelle version après retrain,
  renseigner `parent_version = source_version`.
- **Upload** (`POST /models`) : paramètre optionnel `parent_version` dans le form data, pour
  tracer les uploads manuels de modèles dérivés.
- **Schémas** : ajouter `parent_version: str | None` dans `ModelCreateResponse` et
  `ModelGetResponse`.
- **Streamlit `pages/2_Models.py`** : afficher `parent_version` dans la vue détail d'une
  version si renseigné (`"Dérivé de v2.0.0"`).

---

## Tableau récapitulatif

| # | Feature | Endpoint(s) / Composant | Priorité | Difficulté | Effort estimé |
|---|---------|------------------------|----------|------------|---------------|
| 1.1 | Couverture ground truth | `GET /observed-results/stats` | **Haute** | Facile | ~3 h |
| 1.2 | Export ground truth | `GET /observed-results/export` | **Haute** | Facile | ~2 h |
| 1.3 | Warmup du cache modèle | `POST /models/{n}/{v}/warmup` | **Haute** | Facile | ~2 h |
| 2.1 | Null rate dans le drift | Extend `DriftService` + `compute-baseline` | **Haute** | Moyenne | ~1 jour |
| 2.2 | Page retrain Streamlit | `streamlit_app/pages/8_Retrain.py` | Moyenne | Moyenne | ~1.5 jours |
| 2.3 | Lignée des modèles (`parent_version`) | Champ DB + retrain + upload | Moyenne | Facile | ~3 h |

---

## Fichiers à modifier

| Fichier | Modifications |
|---------|--------------|
| `src/api/observed_results.py` | `GET /observed-results/stats`, `GET /observed-results/export` |
| `src/api/models.py` | `POST /models/{n}/{v}/warmup` |
| `src/services/db_service.py` | `get_observed_results_stats()`, `get_observed_results_for_export()`, extension `get_feature_production_stats()` pour null rate |
| `src/services/drift_service.py` | Ajout `null_rate_*` dans `compute_feature_drift()` et `summarize_drift()` |
| `src/schemas/observed_result.py` | `ObservedResultsStatsResponse` |
| `src/schemas/model.py` | Extension `FeatureDriftResult` (null_rate), `parent_version` dans réponses modèle |
| `src/db/models/model_metadata.py` | Colonne `parent_version` |
| `alembic/versions/` | Nouvelle migration pour `parent_version` + null rate baseline |
| `streamlit_app/utils/api_client.py` | `get_observed_results_stats()`, `export_observed_results()`, `warmup_model()` |
| `streamlit_app/pages/2_Models.py` | Bouton warmup + affichage parent_version |
| `streamlit_app/pages/3_Predictions.py` | Encart couverture ground truth |
| `streamlit_app/pages/8_Retrain.py` | Nouvelle page retrain management |

---

## Ce qui n'a PAS été retenu

| Idée | Raison |
|------|--------|
| `GET /models/compare` (comparaison arbitraire) | `GET /monitoring/overview` + `performance` par modèle suffisent ; l'A/B compare déjà les versions actives |
| OpenTelemetry distributed tracing | Prometheus `/metrics` déjà en place ; traçage distribué = infrastructure séparée hors scope |
| Multi-output models | Cas d'usage < 10 % des utilisateurs ; changements de schéma invasifs |
| WebSocket streaming predictions | Overkill ; `POST /predict-batch` couvre les cas haute fréquence |
| Alerting configurable sur le drift | Email de supervision hebdomadaire + webhooks déjà en place ; règles configurables = usine à gaz |
| Retrain history table | `model_history` (audit log) + `last_run_at` suffisent pour les cas d'usage courants |
| Per-model rate limiting | Quota par utilisateur + monitoring par modèle couvre 95 % des besoins |
| `GET /health/detailed` | `/monitoring/overview` fournit déjà un tableau de bord de santé complet |
