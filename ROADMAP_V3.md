# ROADMAP V3 — predictml-api

> Perspective : Data Scientist / MLOps Engineer  
> Date : Avril 2026  
> Objectif : fonctionnalités utiles à 80 % des utilisateurs, sans usine à gaz

---

## Ce qui est déjà en place (ne pas réimplémenter)

- Prédiction simple et batch (`POST /predict`, `POST /predict-batch`)
- Explication locale SHAP par feature (`POST /explain`)
- A/B testing + shadow deployment avec pondération du trafic
- Versioning complet + rollback via snapshots
- Détection de drift (Z-score + PSI) — `GET /models/{name}/drift`
- Métriques de performance vs résultats observés — `GET /models/{name}/performance`
- Alertes e-mail (drift, erreurs, rapport hebdomadaire)
- Webhooks post-prédiction (async)
- Ré-entraînement manuel — `POST /models/{name}/{version}/retrain`
- Monitoring complet — `GET /monitoring/overview` + `GET /monitoring/model/{name}`
- Dashboard Streamlit admin 7 pages (A/B, supervision, stats, retrain…)
- Rate limiting par utilisateur, cache Redis, intégration MLflow + MinIO

---

## CHUNK 1 — Priorité 1 : Quick Wins (< 1 jour chacun)

### 1.1 `GET /metrics` — Endpoint Prometheus

**Pourquoi :** Standard de facto pour l'observabilité en production. Permet l'intégration immédiate avec Grafana, Datadog ou tout stack de monitoring sans modifier le dashboard Streamlit. OpenTelemetry est déjà configuré pour le tracing, mais il n'existe pas de scrape endpoint Prometheus exposant les métriques HTTP.

**Métriques exposées automatiquement :**
- `http_requests_total` (labels : `method`, `handler`, `status`)
- `http_request_duration_seconds` (histogramme)
- Métriques custom optionnelles : `predictions_total`, `model_cache_hits_total`

**Implémentation :**
- Ajouter `prometheus-fastapi-instrumentator` dans `requirements.txt` + `pyproject.toml`
- 3 lignes dans `src/main.py` pour exposer `/metrics`
- Aucune modification des endpoints existants

**Effort :** ~2h  
**Impact :** ⭐⭐⭐⭐⭐ — intégration monitoring externe sans friction

---

### 1.2 `GET /predictions/export` — Export bulk CSV / JSONL

**Pourquoi :** Le dashboard Streamlit a un bouton CSV limité à 500 lignes. Les data scientists ont besoin d'exporter l'historique complet pour analyse offline (feature analysis, re-labelling, audit rétrospectif).

**Signature proposée :**
```
GET /predictions/export
  ?model_name=iris
  &start=2026-01-01
  &end=2026-04-01
  &format=csv|jsonl        # défaut : csv
  &include_features=true   # inclure input_features dans l'export
  &status=success|error    # filtre optionnel
```

**Réponse :** `StreamingResponse` avec `Content-Disposition: attachment` — streaming par curseur pour éviter de charger tout en mémoire.

**Implémentation :**
- Nouveau endpoint dans `src/api/predict.py` (ou `src/api/export.py`)
- Réutiliser `db_service.get_predictions()` avec itération par curseur (déjà en place — pagination keyset)
- `StreamingResponse` + générateur `yield` ligne par ligne

**Effort :** ~4h  
**Impact :** ⭐⭐⭐⭐ — indispensable pour les équipes DS

---

### 1.3 `GET /models/{name}/feature-importance` — SHAP agrégé global

**Pourquoi :** `POST /explain` retourne l'importance locale pour une prédiction donnée. Il manque une vue globale : *"quelles features influencent le plus ce modèle sur les 7 derniers jours ?"* — indispensable pour détecter des dérives comportementales avant même que les métriques de performance ne bougent.

**Signature proposée :**
```
GET /models/{name}/feature-importance
  ?version=1.0.0     # optionnel, défaut : version production
  &last_n=100        # nb de prédictions à échantillonner (défaut 100, max 500)
  &days=7            # fenêtre temporelle
```

**Réponse :**
```json
{
  "model_name": "iris",
  "version": "1.0.0",
  "sample_size": 98,
  "feature_importance": {
    "petal_length": { "mean_abs_shap": 0.42, "rank": 1 },
    "petal_width":  { "mean_abs_shap": 0.31, "rank": 2 },
    "sepal_length": { "mean_abs_shap": 0.18, "rank": 3 },
    "sepal_width":  { "mean_abs_shap": 0.09, "rank": 4 }
  }
}
```

**Implémentation :**
- Endpoint dans `src/api/models.py`
- Charger le modèle (cache Redis déjà en place via `model_service.load_model()`)
- Récupérer N prédictions récentes via `db_service.get_predictions()`
- Appeler `shap_service.compute_shap_explanation()` (déjà implémenté) sur chaque input
- Moyenner `|shap_values|` par feature → ranking

**Effort :** ~6h  
**Impact :** ⭐⭐⭐⭐⭐ — manque clairement dans le workflow MLOps actuel

---

## CHUNK 2 — Priorité 2 : Haute valeur, effort modéré (1–3 jours)

### 2.1 Significativité statistique A/B — enrichir `GET /models/{name}/ab-compare`

**Pourquoi :** L'endpoint `/ab-compare` compare déjà les métriques brutes (taux d'erreur, latence) entre versions. Mais sans test de significativité, impossible de savoir si une différence de 2 % d'accuracy entre v1 et v2 est réelle ou due au hasard — et on risque de promouvoir du bruit.

**Ajout à la réponse existante :**
```json
{
  "ab_significance": {
    "metric": "accuracy",
    "test": "mann_whitney_u",
    "p_value": 0.023,
    "significant": true,
    "confidence_level": 0.95,
    "winner": "v2.0.0",
    "min_samples_needed": 200,
    "current_samples": { "v1.0.0": 450, "v2.0.0": 380 }
  }
}
```

**Implémentation :**
- `scipy` est déjà présent en dépendance transitoire (scikit-learn)
- Mann-Whitney U pour distributions continues (latence, probabilité de prédiction)
- Chi-² pour métriques catégorielles (taux d'erreur, distribution des labels)
- Calcul `min_samples_needed` via formule de puissance statistique (d de Cohen)
- Modifier `src/api/models.py` — fonction `get_ab_comparison()`

**Effort :** ~1 jour  
**Impact :** ⭐⭐⭐⭐⭐ — évite les promotions basées sur du bruit statistique

---

### 2.2 `PATCH /models/{name}/policy` — Politique d'auto-promotion post-retrain

**Pourquoi :** Après un retrain, l'admin doit manuellement vérifier les métriques et décider de promouvoir. Avec une politique définie à l'avance, le workflow devient : retrain → vérification automatique → promotion si seuils atteints.

**Signature proposée :**
```
PATCH /models/{name}/policy
Body:
{
  "min_accuracy": 0.90,
  "max_latency_p95_ms": 200,
  "min_sample_validation": 50,
  "auto_promote": true
}
```

**Comportement :**
1. Stocker la policy dans `ModelMetadata` (nouveau champ JSON `promotion_policy`)
2. À la fin de `POST /models/{name}/{version}/retrain`, si `auto_promote: true` :
   - Évaluer le nouveau modèle sur les N derniers `observed_results`
   - Si `accuracy >= min_accuracy` ET `latency_p95 <= max_latency_p95_ms` → `is_production = true` automatiquement
3. La réponse du retrain inclut `"auto_promoted": true|false` + raison si non-promu

**Implémentation :**
- Nouveau champ `promotion_policy` (JSON) dans `src/db/models/model_metadata.py` + migration Alembic
- Modifier fin du handler retrain dans `src/api/models.py`
- Réutiliser `db_service.get_performance_pairs()` pour l'évaluation (déjà en place)

**Effort :** ~2 jours  
**Impact :** ⭐⭐⭐⭐ — élimine la promotion manuelle pour les workflows de retrain régulier

---

### 2.3 `DELETE /predictions/purge` — Rétention des données / RGPD

**Pourquoi :** La table `predictions` grossit indéfiniment. Sur un déploiement actif (1 000 prédictions/jour), elle atteint 365 000 lignes/an. Sans politique de rétention, les performances des requêtes analytiques se dégradent et la conformité RGPD devient un problème.

**Signature proposée :**
```
DELETE /predictions/purge
  ?older_than_days=90    # supprimer les prédictions > 90 jours
  &model_name=iris       # optionnel : purger un seul modèle
  &dry_run=true          # simuler sans supprimer (défaut : true)
```

**Réponse :**
```json
{
  "dry_run": false,
  "deleted_count": 12450,
  "oldest_remaining": "2026-01-15T08:32:00Z",
  "models_affected": ["iris", "wine"]
}
```

**Contraintes :**
- Admin seulement
- `dry_run=true` par défaut — suppression explicite uniquement avec `dry_run=false`
- Avertissement si des prédictions supprimées sont liées à des `observed_results` non encore traités

**Implémentation :**
- Endpoint dans `src/api/predict.py` ou nouveau `src/api/admin.py`
- Nouveau helper `db_service.purge_predictions(older_than_days, model_name, dry_run)`
- Query `WHERE timestamp < now() - interval '{N} days'`

**Effort :** ~4h  
**Impact :** ⭐⭐⭐⭐ — hygiène données et conformité sur le long terme

---

## CHUNK 3 — Priorité 3 : Moyen terme (3–7 jours)

### 3.1 Retraining planifié — champ `cron_schedule` sur les modèles

**Pourquoi :** Le retrain manuel convient aux équipes qui pilotent activement leurs modèles. Mais pour des modèles en production stable, déclencher un retrain automatique chaque semaine sans intervention humaine est la norme MLOps. Le système de retrain existant est complet — il manque uniquement le déclencheur temporel.

**Design proposé :**

Nouveau endpoint pour configurer le planning :
```
PATCH /models/{name}/{version}/schedule
Body:
{
  "cron": "0 3 * * 1",    // chaque lundi à 3h UTC
  "lookback_days": 30,     // TRAIN_START_DATE = today - 30j
  "auto_promote": false,   // utilise la policy définie (voir 2.2)
  "enabled": true
}
```

Nouveau champ `retrain_schedule` (JSON) sur `ModelMetadata` :
```json
{
  "cron": "0 3 * * 1",
  "lookback_days": 30,
  "auto_promote": false,
  "enabled": true,
  "last_run_at": "2026-04-14T03:00:00Z",
  "next_run_at": "2026-04-21T03:00:00Z"
}
```

**Implémentation :**
- Ajouter `APScheduler` avec `AsyncIOScheduler` dans `requirements.txt`
- Au démarrage de l'API (dans le `lifespan` de `src/main.py`), charger les schedules actifs depuis la DB
- Créer un job par modèle schedulé qui appelle la logique retrain existante
- Stocker `last_run_at` + `next_run_at` sur `ModelMetadata` après chaque exécution
- **Point de vigilance :** si l'API tourne en plusieurs replicas, utiliser un verrou Redis (`SET NX EX`) pour éviter les retrains simultanés

**Effort :** ~5 jours  
**Impact :** ⭐⭐⭐⭐ — indispensable pour les équipes avec des modèles à fraîcheur contrainte

---

### 3.2 `POST /models/{name}/{version}/validate-input` — Validation du schéma d'entrée

**Pourquoi :** Les pannes silencieuses les plus fréquentes en ML en production viennent de pipelines de données qui envoient des features manquantes, renommées, ou dans le mauvais type. Actuellement `/predict` accepte n'importe quel JSON — les incohérences causent des erreurs 500 non explicites ou, pire, des prédictions silencieusement fausses.

**Signature proposée :**
```
POST /models/{name}/{version}/validate-input
Body: { "petal_length": 5.1, "petal_width": 1.8, "sepal_length": 6.3 }
```

**Réponse :**
```json
{
  "valid": false,
  "errors": [
    { "type": "missing_feature",     "feature": "sepal_width" },
    { "type": "unexpected_feature",  "feature": "petal_width_squared" }
  ],
  "warnings": [
    { "type": "type_coercion", "feature": "petal_length", "from": "string", "to": "float" }
  ],
  "expected_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
}
```

**Mode strict optionnel sur `/predict` :**
Ajouter `?strict_validation=true` à `/predict` pour rejeter avec 422 si les features ne correspondent pas au schéma enregistré (au lieu du comportement silencieux actuel).

**Implémentation :**
- Endpoint dans `src/api/models.py`
- Lire `feature_baseline` et `features_count` depuis `ModelMetadata` (déjà stockés)
- Logique de validation : features manquantes, features inattendues, coercition de type
- Optionnel : brancher sur `/predict` avec query param `strict_validation`

**Effort :** ~2 jours  
**Impact :** ⭐⭐⭐ — surtout utile lors des migrations ou refactorisations de pipelines de données

---

## Tableau récapitulatif

| # | Feature | Endpoint | Effort | Impact | Difficulté |
|---|---------|----------|--------|--------|------------|
| 1 | Prometheus metrics | `GET /metrics` | 2h | ⭐⭐⭐⭐⭐ | Très facile |
| 2 | Export bulk prédictions | `GET /predictions/export` | 4h | ⭐⭐⭐⭐ | Facile |
| 3 | Feature importance globale (SHAP) | `GET /models/{name}/feature-importance` | 6h | ⭐⭐⭐⭐⭐ | Facile |
| 4 | Significativité statistique A/B | Enrichir `GET /models/{name}/ab-compare` | 1j | ⭐⭐⭐⭐⭐ | Moyen |
| 5 | Auto-promotion policy post-retrain | `PATCH /models/{name}/policy` | 2j | ⭐⭐⭐⭐ | Moyen |
| 6 | Purge / rétention RGPD | `DELETE /predictions/purge` | 4h | ⭐⭐⭐⭐ | Facile |
| 7 | Retraining planifié (cron) | `PATCH /models/{name}/{version}/schedule` | 5j | ⭐⭐⭐⭐ | Difficile |
| 8 | Validation schéma d'entrée | `POST /models/{name}/{version}/validate-input` | 2j | ⭐⭐⭐ | Moyen |

---

> **Hors scope volontaire :** feature store, ensemble orchestration, online learning, calibration curves, distributed tracing — utiles dans des contextes spécifiques mais hors cible pour 80 % des usages de ce projet.
