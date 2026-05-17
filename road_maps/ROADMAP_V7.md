# ROADMAP V7 — predictml-api

> **Perspective** : Data Scientist / MLOps Engineer — Avril 2026

---

## État des lieux

Les ROADMAP V3 à V6 sont entièrement en production. L'API couvre désormais :

- Prédiction simple + batch, SHAP local (`POST /explain`, inline via `?explain=true`) et global (`GET /models/{name}/feature-importance`)
- Validation stricte du schéma d'entrée, calibration, tendance de confiance, drift (Z-score + PSI + null rate)
- A/B testing + shadow avec significativité statistique (Chi-² / Mann-Whitney)
- Retrain manuel + planifié (cron), auto-promotion, snapshot `training_stats`, audit log + rollback
- Alertes par modèle (`alert_thresholds`) + supervision reporter toutes les 6h + rapport hebdo email
- Purge RGPD, export CSV/JSONL, import CSV ground truth, couverture ground truth
- Préchauffage cache, lookup `GET /predictions/{id}`, explication post-hoc `GET /predictions/{id}/explain`

**Ce roadmap ne propose que des améliorations à valeur réelle pour ≥ 80 % des utilisateurs.**
Chaque fonctionnalité est autonome — elle peut être implémentée indépendamment.

---

## CHUNK 1 — Priorité HAUTE · Difficulté FACILE

*Chaque feature est implémentable en moins d'une demi-journée.*

---

### 1.1 `GET /models/{name}/compare` — Comparaison multi-versions en un seul appel

#### Pourquoi

Décider quelle version promouvoir en production est l'une des décisions les plus fréquentes
en MLOps. Aujourd'hui, pour comparer trois versions d'un modèle, il faut enchaîner :
`GET /models/iris/1.0.0`, `GET /models/iris/2.0.0`, `GET /models/iris/3.0.0` pour les
métadonnées, puis `GET /models/iris/performance` trois fois avec un filtre de version,
plus `GET /models/iris/drift` par version — soit **9 appels minimum**. Le dashboard
Streamlit `2_Models.py` fait exactement ça en boucle, ce qui explique ses temps de
chargement élevés. Un endpoint de comparaison agrège tout en un seul appel et devient
le socle naturel d'un tableau de bord décisionnel.

#### Signature

```
GET /models/{name}/compare?versions=1.0.0,2.0.0,3.0.0

Réponse :
{
  "model_name": "iris",
  "compared_at": "2026-04-25T10:00:00",
  "versions": [
    {
      "version": "1.0.0",
      "is_production": false,
      "accuracy": 0.89,
      "f1_score": 0.88,
      "latency_p50_ms": 18,
      "latency_p95_ms": 42,
      "drift_status": "ok",
      "brier_score": 0.08,
      "trained_at": "2026-01-15T08:00:00",
      "n_rows_trained": 12450
    },
    {
      "version": "2.0.0",
      "is_production": true,
      "accuracy": 0.94,
      ...
    }
  ]
}
```

Si `?versions` est omis, retourne toutes les versions actives du modèle.

#### Comment implémenter

- **`src/api/models.py`** : nouvel endpoint `GET /models/{name}/compare`.
  Itère sur les versions demandées, appelle en parallèle (`asyncio.gather`) :
  1. `DBService.get_model_metadata(db, name, v)` — accuracy, f1, trained_at
  2. `DBService.get_prediction_stats(db, model_name=name, version=v, days=7)` — latence p50/p95
  3. Appel interne `DriftService.summarize_drift()` sur les stats de production récentes
  4. `training_stats` extrait du champ JSON existant (n_rows, trained_at)
- **`src/schemas/model.py`** : nouveau schéma `ModelCompareResponse` + `ModelVersionSummary`.
- Pas de nouvelle table DB ni migration — tout est calculé depuis l'existant.

---

### 1.2 `GET /health/dependencies` — Statut détaillé des dépendances

#### Pourquoi

Le `GET /health` actuel retourne uniquement `{ "status": "healthy", "database": "connected" }`.
Il ne vérifie ni Redis, ni MinIO, ni MLflow — les trois services dont une panne silencieuse
dégrade l'API sans la faire tomber (cache désactivé, modèles non chargeables, tracking perdu).
En environnement Kubernetes, les readiness probes branchées sur `/health` ne détectent pas
ces dégradations partielles. Les équipes ops découvrent les pannes via des 500 en production
plutôt que via le monitoring. Un endpoint détaillé avec latence mesurée par dépendance
permet de configurer des alertes ciblées et d'afficher un status page précis.

#### Signature

```
GET /health/dependencies

Réponse :
{
  "status": "degraded",
  "checked_at": "2026-04-25T10:00:00",
  "dependencies": {
    "database": { "status": "ok",      "latency_ms": 3  },
    "redis":    { "status": "ok",      "latency_ms": 1  },
    "minio":    { "status": "error",   "latency_ms": null, "detail": "Connection refused" },
    "mlflow":   { "status": "ok",      "latency_ms": 12 }
  }
}
```

`status` global = `ok` si tout est ok, `degraded` si ≥ 1 dépendance en erreur,
`critical` si la DB est en erreur (l'API ne peut plus fonctionner du tout).

#### Comment implémenter

- **`src/main.py`** ou **`src/api/`** : nouvel endpoint `GET /health/dependencies`.
  Quatre checks en parallèle (`asyncio.gather`) avec `time.monotonic()` pour la latence :
  1. **DB** : `SELECT 1` via `db.execute(text("SELECT 1"))`
  2. **Redis** : `redis_client.ping()`
  3. **MinIO** : `minio_service.list_models()` (bucket_name) ou `stat_object` sur un objet connu
  4. **MLflow** : `GET {MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list?max_results=1`
     via `httpx.AsyncClient` (déjà dépendance du projet)
- Chaque check est isolé dans un `try/except` — la panne d'une dépendance n'empêche pas
  les autres checks.
- **`src/schemas/`** : schéma `DependencyHealthResponse` (ou inline dans main.py si léger).
- Aucune migration DB.

---

### 1.3 `GET /models` — Filtres avancés

#### Pourquoi

`GET /models` ne supporte aujourd'hui qu'un seul filtre : `?tag`. Dans un déploiement
avec 20+ modèles (plusieurs algorithmes, plusieurs équipes, plusieurs états), retrouver
"tous les modèles RandomForest en production avec une accuracy > 0.90" nécessite de
récupérer toute la liste et de filtrer côté client. C'est une friction quotidienne pour
les Data Scientists qui gèrent un parc de modèles hétérogène. Les filtres proposés ne
nécessitent aucun nouveau champ en base — ils s'appliquent sur des colonnes déjà indexées.

#### Signature

```
GET /models?is_production=true&algorithm=RandomForest&min_accuracy=0.85&deployment_mode=ab_test

Paramètres ajoutés (tous optionnels, cumulables) :
  is_production: bool        — filtre sur model_metadata.is_production
  algorithm:     str         — filtre exact sur model_metadata.algorithm
  min_accuracy:  float       — filtre model_metadata.accuracy >= valeur
  deployment_mode: enum      — production | ab_test | shadow
```

Les filtres sont combinés en `AND`. Le filtre `?tag` existant reste inchangé.

#### Comment implémenter

- **`src/services/db_service.py`** : étendre `get_available_models()` avec 4 paramètres
  optionnels. Ajout de clauses `WHERE` conditionnelles sur la requête SQLAlchemy existante :
  ```python
  if is_production is not None:
      query = query.where(ModelMetadata.is_production == is_production)
  if algorithm:
      query = query.where(ModelMetadata.algorithm == algorithm)
  if min_accuracy is not None:
      query = query.where(ModelMetadata.accuracy >= min_accuracy)
  if deployment_mode:
      query = query.where(ModelMetadata.deployment_mode == deployment_mode)
  ```
- **`src/api/models.py`** : ajouter 4 `Query(None)` params sur `GET /models` et
  les passer à `get_available_models()`.
- Pas de nouveau schéma, pas de migration.

---

## CHUNK 2 — Priorité HAUTE/MOYENNE · Difficulté MOYENNE

*Chaque feature est implémentable en 1–2 jours.*

---

### 2.1 `GET /models/{name}/performance-timeline` — Évolution des métriques par version

#### Pourquoi

`POST /retrain` crée de nouvelles versions en continu. La question "est-ce que ce retrain
a amélioré le modèle ?" est la plus posée après chaque cycle d'entraînement. Aujourd'hui,
`GET /models/{name}/performance` donne la performance sur une période temporelle, mais
pas par version de modèle. Pour répondre à la question, il faut croiser manuellement
l'historique des versions (dates de déploiement) avec les observed results correspondants —
une jointure complexe que chaque Data Scientist refait à la main dans un notebook.
La timeline automatise ce calcul et rend visible la trajectoire du modèle sur sa durée
de vie : montée en performance, régression après un retrain sur données corrompues,
oscillation entre versions.

#### Signature

```
GET /models/{name}/performance-timeline

Réponse :
{
  "model_name": "iris",
  "timeline": [
    {
      "version": "1.0.0",
      "deployed_at": "2026-01-15T08:00:00",
      "accuracy": 0.89,
      "mae": null,
      "f1_score": 0.88,
      "sample_count": 1240,
      "trained_at": "2026-01-14T20:00:00",
      "n_rows_trained": 12450
    },
    {
      "version": "2.0.0",
      "deployed_at": "2026-02-10T09:30:00",
      "accuracy": 0.94,
      "mae": null,
      "f1_score": 0.93,
      "sample_count": 890,
      "trained_at": "2026-02-09T22:15:00",
      "n_rows_trained": 15200
    }
  ]
}
```

`sample_count` = nombre de paires (prédiction, observed_result) utilisées pour le calcul.
`accuracy` / `mae` sont `null` si aucun observed_result n'est disponible pour la version.

#### Comment implémenter

- **`src/services/db_service.py`** : nouvelle méthode `get_performance_timeline(db, model_name)`.
  1. Récupère toutes les versions du modèle : `SELECT name, version, created_at, training_stats FROM model_metadata WHERE name = ? ORDER BY created_at ASC`.
  2. Pour chaque version, appelle `get_performance_pairs(db, model_name, version)` (déjà implémenté)
     pour calculer accuracy/MAE sur les paires historiques.
  3. Extrait `trained_at` et `n_rows` du champ JSON `training_stats` existant.
- **`src/api/models.py`** : nouvel endpoint `GET /models/{name}/performance-timeline`.
- **`src/schemas/model.py`** : schéma `PerformanceTimelineResponse` + `VersionTimelineEntry`.

---

### 2.2 `GET /models/{name}/confidence-distribution` — Histogramme de confiance

#### Pourquoi

La confiance d'un modèle (= probabilité max de la classe prédite) est un signal d'alerte
précoce qui ne nécessite **aucune ground truth**. Si un modèle de classification était
confiant à 0.95+ la semaine passée et plafonne maintenant à 0.72, quelque chose a changé
dans la distribution des entrées — bien avant que les observed_results confirment la
dégradation. C'est particulièrement utile pour les modèles récents (< 30 jours en prod)
qui n'ont pas encore accumulé assez de ground truth pour calculer une accuracy fiable.
Le champ `probabilities` (JSON) est déjà stocké sur chaque prédiction — aucune donnée
supplémentaire à capturer.

#### Signature

```
GET /models/{name}/confidence-distribution?version=2.0.0&days=7

Réponse :
{
  "model_name":   "iris",
  "version":      "2.0.0",
  "period_days":  7,
  "sample_count": 3420,
  "mean_confidence":       0.84,
  "pct_high_confidence":   0.71,
  "pct_uncertain":         0.08,
  "histogram": [
    { "bin_min": 0.50, "bin_max": 0.60, "count": 120, "pct": 0.035 },
    { "bin_min": 0.60, "bin_max": 0.70, "count": 156, "pct": 0.046 },
    { "bin_min": 0.70, "bin_max": 0.80, "count": 412, "pct": 0.120 },
    { "bin_min": 0.80, "bin_max": 0.90, "count": 891, "pct": 0.260 },
    { "bin_min": 0.90, "bin_max": 1.00, "count": 1841,"pct": 0.538 }
  ]
}
```

- `pct_high_confidence` = fraction des prédictions avec max_prob > 0.80
- `pct_uncertain` = fraction avec max_prob < 0.60 (signal d'alerte)
- Seuils paramétrables via query params : `?high_threshold=0.8&uncertain_threshold=0.6`

#### Comment implémenter

- **`src/services/db_service.py`** : nouvelle méthode `get_confidence_distribution(db, model_name, version, days)`.
  Récupère les `probabilities` (JSON) des prédictions récentes via :
  ```sql
  SELECT probabilities FROM predictions
  WHERE model_name = ? AND model_version = ?
    AND timestamp >= now() - interval '? days'
    AND status = 'success' AND probabilities IS NOT NULL
  ```
  En Python : extrait `max(prob.values())` pour chaque ligne, construit l'histogramme
  avec `numpy.histogram(confidences, bins=10, range=(0.5, 1.0))`.
- **`src/api/models.py`** : nouvel endpoint `GET /models/{name}/confidence-distribution`.
- **`src/schemas/model.py`** : `ConfidenceDistributionResponse` + `ConfidenceBin`.

---

### 2.3 `GET /models/{name}/readiness` — Vérification opérationnelle avant mise en trafic

#### Pourquoi

Avant de mettre un modèle en production (via `PATCH /models/{name}/{version}` avec
`is_production=true`), plusieurs conditions doivent être remplies : le fichier .joblib doit
être accessible dans MinIO, le baseline doit être calculé (sinon le drift est aveugle),
et aucun drift critique ne doit être actif. Aujourd'hui, ces vérifications sont soit
manuelles, soit éparpillées dans plusieurs appels. Dans un pipeline CI/CD (GitHub Actions,
Terraform), un gate "est-ce que ce modèle est prêt ?" doit être scriptable en une seule
requête. `GET /models/{name}/readiness` retourne un résultat binaire avec les causes
détaillées d'un éventuel échec.

#### Signature

```
GET /models/{name}/readiness?version=2.0.0

Réponse (modèle prêt) :
{
  "model_name":   "iris",
  "version":      "2.0.0",
  "ready":        true,
  "checked_at":   "2026-04-25T10:00:00",
  "checks": {
    "is_production":      { "pass": true  },
    "file_accessible":    { "pass": true  },
    "baseline_computed":  { "pass": true  },
    "no_critical_drift":  { "pass": true  }
  }
}

Réponse (modèle non prêt) :
{
  "ready": false,
  "checks": {
    "is_production":      { "pass": false, "detail": "is_production=False" },
    "file_accessible":    { "pass": true  },
    "baseline_computed":  { "pass": false, "detail": "feature_baseline is null" },
    "no_critical_drift":  { "pass": true  }
  }
}
```

HTTP 200 dans les deux cas — `ready: false` n'est pas une erreur, c'est un état.

#### Comment implémenter

- **`src/api/models.py`** : nouvel endpoint `GET /models/{name}/readiness`.
  Exécute 4 checks en parallèle (`asyncio.gather`) :
  1. **is_production** : `model_meta.is_production` (déjà en mémoire)
  2. **file_accessible** : `minio_service.get_object_info(model_meta.object_key)` — un
     `try/except` suffit (déjà implémenté dans `MinIOService`)
  3. **baseline_computed** : `model_meta.feature_baseline is not None`
  4. **no_critical_drift** : appel interne `DriftService.summarize_drift()` sur les stats
     de production des dernières 24h — vérifie que `status != "critical"`
- **`src/schemas/model.py`** : schéma `ReadinessResponse` + `ReadinessCheck`.
- Pas de migration DB.

---

## CHUNK 3 — Priorité MOYENNE/BASSE · et ce qui n'a pas été retenu

---

## CHUNK 3A — Priorité MOYENNE · Difficulté FACILE

*Implémentables en 2–3 heures chacune.*

---

### 3.1 `GET /models/{name}/{version}/download` — Téléchargement du .joblib

#### Pourquoi

Pour déboguer localement un modèle en production, rejouer des prédictions dans un
notebook, ou l'analyser avec des outils externes (SkLearn-inspection, Evidently), le
Data Scientist a besoin du fichier .joblib. Aujourd'hui, il doit passer par la console MinIO
ou demander un accès direct au stockage objet — une friction qui sort du workflow API
et crée une dépendance à l'infra. Un endpoint de téléchargement direct garde tout dans
le contrat de l'API.

#### Signature

```
GET /models/{name}/{version}/download
Authorization: Bearer <admin_token>

Réponse : stream binaire, Content-Type: application/octet-stream
Content-Disposition: attachment; filename="iris_2.0.0.joblib"
```

Admin uniquement — le .joblib contient la logique interne du modèle.

#### Comment implémenter

- **`src/api/models.py`** : nouvel endpoint `GET /models/{name}/{version}/download`.
  1. Récupère les métadonnées pour vérifier que la version existe et que l'utilisateur
     est admin (`require_admin` déjà implémenté).
  2. `model_bytes = minio_service.download_model(object_key)` (déjà implémenté, retourne bytes).
  3. `return Response(content=model_bytes, media_type="application/octet-stream",
     headers={"Content-Disposition": f'attachment; filename="{name}_{version}.joblib"'})`.
- Aucun nouveau service, aucune migration, aucun nouveau schéma.

---

### 3.2 `GET /users/{user_id}/usage` — Statistiques d'usage par utilisateur

#### Pourquoi

En production multi-tenant, l'admin a besoin de savoir qui consomme quoi : quel utilisateur
frappe le plus souvent quel modèle, sur quelle période. Aujourd'hui, `GET /users/{id}/quota`
retourne uniquement le nombre d'appels restants aujourd'hui — pas de ventilation par modèle
ni d'historique. Pour la facturation interne, la détection d'abus, ou la planification des
quotas, un breakdown historique (appels/modèle/jour sur les N derniers jours) est nécessaire.
La table `predictions` contient `user_id`, `model_name` et `timestamp` — tout y est.

#### Signature

```
GET /users/{user_id}/usage?days=30

Réponse :
{
  "user_id": 42,
  "username": "alice",
  "period_days": 30,
  "total_calls": 4820,
  "by_model": [
    { "model_name": "iris",    "calls": 3200, "errors": 12, "avg_latency_ms": 21 },
    { "model_name": "churn",   "calls": 1620, "errors":  4, "avg_latency_ms": 45 }
  ],
  "by_day": [
    { "date": "2026-04-01", "calls": 180 },
    { "date": "2026-04-02", "calls": 210 }
  ]
}
```

Contrôle d'accès : un utilisateur voit ses propres stats ; un admin voit n'importe quel utilisateur.

#### Comment implémenter

- **`src/services/db_service.py`** : nouvelle méthode `get_user_usage(db, user_id, days)`.
  Deux requêtes GROUP BY sur la table `predictions` (déjà indexée sur `user_id` et `timestamp`) :
  ```sql
  -- Par modèle
  SELECT model_name, COUNT(*) as calls,
         SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors,
         AVG(response_time_ms) as avg_latency
  FROM predictions WHERE user_id = ? AND timestamp >= now() - interval '? days'
  GROUP BY model_name

  -- Par jour
  SELECT DATE(timestamp) as date, COUNT(*) as calls
  FROM predictions WHERE user_id = ?  AND timestamp >= now() - interval '? days'
  GROUP BY DATE(timestamp) ORDER BY date
  ```
- **`src/api/users.py`** : nouvel endpoint `GET /users/{user_id}/usage`.
- **`src/schemas/user.py`** : schéma `UserUsageResponse`.

---

## Tableau récapitulatif

| # | Feature | Endpoint | Priorité | Difficulté | Effort estimé |
|---|---------|----------|----------|------------|---------------|
| 1.1 | Comparaison multi-versions | `GET /models/{name}/compare` | **Haute** | Facile | ~4 h |
| 1.2 | Santé des dépendances | `GET /health/dependencies` | **Haute** | Facile | ~3 h |
| 1.3 | Filtres avancés sur les modèles | `GET /models?algorithm=X&min_accuracy=Y` | **Haute** | Facile | ~2 h |
| 2.1 | Timeline de performance | `GET /models/{name}/performance-timeline` | **Haute** | Moyenne | ~1 jour |
| 2.2 | Distribution de confiance | `GET /models/{name}/confidence-distribution` | Moyenne | Moyenne | ~1 jour |
| 2.3 | Vérification de readiness | `GET /models/{name}/readiness` | Moyenne | Moyenne | ~1 jour |
| 3.1 | Téléchargement du .joblib | `GET /models/{name}/{version}/download` | Moyenne | Facile | ~2 h |
| 3.2 | Usage par utilisateur | `GET /users/{id}/usage` | Moyenne | Facile | ~3 h |

---

## Ce qui n'a PAS été retenu

| Idée | Raison |
|------|--------|
| Support LIME | SHAP (tree + linear) couvre 90 % des cas d'usage ; LIME ajoute une dépendance lourde (`lime` package) sans valeur ajoutée différenciante |
| Streaming / Kafka | Hors périmètre du projet ; `POST /predict-batch` couvre le besoin des volumes élevés sans introduire un bus de messages |
| Intégration feature store (Feast, Tecton) | Dépendance externe critique ; concerne < 20 % des déploiements |
| Canary rollout progressif (10 % → 25 % → 50 %) | A/B testing avec `traffic_weight` + shadow mode couvrent le besoin de déploiement progressif |
| Arbre de lignée des modèles (parent → enfant) | `model_history` + `parent_version` suffisent pour l'usage courant ; un arbre de lignée complet est une complexité DB non justifiée |
| Comparaison cross-modèles (iris vs wine) | `GET /monitoring/overview` + filtres avancés (1.3) couvrent ce besoin sans ajouter un endpoint dédié |
| Annotations métier sur les prédictions (`context` dict) | Utile, mais nécessite une migration DB + changement de schéma pour un gain marginal ; le champ `id_obs` existant permet déjà de lier à un contexte externe |
| Replay d'historique sur nouveau modèle | Le shadow mode en pré-déploiement couvre ce besoin de manière plus propre |
