# ROADMAP V4 — predictml-api

> **Perspective** : Data Scientist / MLOps Engineer — Avril 2026

---

## État des lieux

L'API couvre déjà l'essentiel du cycle de vie ML en production :

- Prédiction simple et batch, routage A/B et shadow
- Versionnage, audit log et rollback des modèles
- Ré-entraînement manuel et planifié (cron), auto-promotion
- Drift détection (Z-score + PSI), feature importance (SHAP)
- Comparaison A/B avec significativité statistique (Chi-² / Mann-Whitney)
- Purge RGPD, export CSV/JSONL, rate limiting par utilisateur
- Dashboard Streamlit (supervision, stats, A/B, gestion modèles/users)
- Prometheus + email de supervision hebdomadaire

**Ce roadmap ne propose que des améliorations à valeur réelle pour ≥ 80 % des utilisateurs.** Les idées écartées sont justifiées en fin de document.

---

## CHUNK 1 — Priorité HAUTE · Difficulté FACILE

*Chaque feature est implémentable en moins d'une journée.*

---

### 1.1 `GET /users/me` + `GET /users/me/quota`

#### Pourquoi

Un utilisateur standard ne connaît pas son ID numérique : il ne peut donc pas appeler `GET /users/{id}` sans solliciter un admin. Pire, **il n'existe aucun moyen de consulter son quota restant** avant de se heurter à un 429. Ces deux endpoints transforment l'API en outil pleinement auto-service.

#### Endpoints

```
GET /users/me
Réponse : id, username, email, role, is_active, rate_limit_per_day,
           created_at, last_login, api_token

GET /users/me/quota
Réponse : rate_limit_per_day, used_today, remaining_today, reset_at
```

#### Comment — API (`src/api/users.py`)

- `GET /users/me` : retourne directement `current_user` via `Depends(verify_token)` → utilise le schéma `UserResponse` existant. **Zéro nouvelle requête DB.**
- `GET /users/me/quota` : appelle `await DBService.get_user_prediction_count_today(db, user.id)` — méthode **déjà présente** dans `db_service.py`. Nouveau schéma `QuotaResponse` (4 champs) dans `src/schemas/user.py`.
- `reset_at` = `datetime.combine(date.today() + timedelta(days=1), time.min, tzinfo=timezone.utc)` — calcul inline, pas de requête.

#### Comment — Streamlit

| Fichier | Modification |
|---------|-------------|
| `api_client.py` | Ajouter `get_me()` et `get_my_quota()` dans la section `# --- Users ---` |
| `app.py` | Sidebar post-login : bloc "Mon compte" visible par tous les rôles — `st.progress(used/limit)`, texte `"342 / 1000 aujourd'hui"`, `st.warning` si quota épuisé |
| `pages/1_Users.py` | Avant le `require_admin()`, ajouter un expander "Mon profil" pour les non-admins : token masqué (bouton révéler), quota, rôle |

---

### 1.2 `POST /models/{name}/{version}/compute-baseline`

#### Pourquoi

**Le système de drift est aveugle sans `feature_baseline`.** Aujourd'hui, `GET /models/{name}/drift` retourne systématiquement `"drift_summary": "no_baseline"` pour tout modèle dont le baseline n'a pas été fourni manuellement à l'upload — une tâche que la quasi-totalité des équipes saute.

La solution est déjà dans la base : `DBService.get_feature_production_stats()` calcule exactement `{mean, std, min, max, count, values}` par feature depuis les prédictions en production. Il suffit d'un endpoint pour persister ce résultat comme baseline.

#### Endpoint

```
POST /models/{name}/{version}/compute-baseline?days=30&dry_run=true

Réponse :
{
  "model_name": "iris",
  "version": "1.0.0",
  "predictions_used": 1240,
  "dry_run": false,
  "baseline": {
    "petal_length": {"mean": 3.76, "std": 1.77, "min": 1.0, "max": 6.9},
    ...
  }
}
```

`dry_run=true` par défaut — calcule et affiche sans sauvegarder, pour validation avant commit.

#### Comment — API (`src/api/models.py`)

1. Appel `await DBService.get_feature_production_stats(db, name, version, days)` — **déjà implémenté**
2. Strip des champs `values` et `count` → format `{mean, std, min, max}` attendu par le drift service
3. Si `dry_run=False` : `model_meta.feature_baseline = baseline_dict` + `await db.commit()` + `log_model_history(action=UPDATED, changed_fields=["feature_baseline"])`
4. Garde : lever 422 si `predictions_used < 100` avec message explicite ("Insuffisant pour un baseline fiable")
5. Nouveau schéma `ComputeBaselineResponse` dans `src/schemas/model.py`

#### Comment — Streamlit (`pages/2_Models.py`)

- Dans la vue détail d'un modèle, section admin, à côté du bouton "Ré-entraîner"
- Expander "📐 Calculer le baseline depuis la production" (admin uniquement)
  - Slider `days` (7–180, défaut 30)
  - Checkbox `dry_run` (cochée par défaut)
  - Bouton "Calculer"
  - Si `dry_run=True` : `st.json(baseline)` + message "Décochez dry_run pour sauvegarder"
  - Si `dry_run=False` et succès : `st.success("Baseline sauvegardé — le drift est maintenant actif")` + `st.cache_data.clear()`
- Dans la liste des modèles : badge **"Baseline ✓"** (vert) / **"No baseline ⚠️"** (orange) selon `feature_baseline is not None`
- `api_client.py` : ajouter `compute_baseline(name, version, days, dry_run)` → `self._post(..., params=...)`

---

## CHUNK 2 — Priorité HAUTE/MOYENNE · Difficulté MOYENNE

*Chaque feature est implémentable en 1–2 jours.*

---

### 2.1 `GET /models/{name}/calibration`

#### Pourquoi

RandomForest et GradientBoosting sont souvent **sur-confiants** : ils annoncent 92 % de confiance pour une précision réelle de 78 %. Sans ce diagnostic, les équipes métier s'appuient sur des scores de probabilité non calibrés pour des décisions à risque (scoring crédit, triage médical, alertes fraud).

Les données sont déjà disponibles : `probabilities` est stocké sur chaque prédiction, et `DBService.get_performance_pairs()` fait déjà la jointure avec `observed_results`.

#### Endpoint

```
GET /models/{name}/calibration?version=1.0.0&start=...&end=...&n_bins=10

Réponse :
{
  "model_name": "iris",
  "version": "1.0.0",
  "sample_size": 580,
  "brier_score": 0.082,
  "calibration_status": "overconfident",   # ok | overconfident | underconfident | insufficient_data
  "mean_confidence": 0.88,
  "mean_accuracy": 0.79,
  "overconfidence_gap": 0.09,
  "reliability": [
    {"confidence_bin": "0.7–0.8", "predicted_rate": 0.75, "observed_rate": 0.71, "count": 112},
    ...
  ]
}
```

Règles de statut :
| Condition | Statut |
|-----------|--------|
| `sample_size < 30` | `insufficient_data` |
| `\|mean_confidence − mean_accuracy\| < 0.05` | `ok` |
| `mean_confidence > mean_accuracy + 0.05` | `overconfident` |
| `mean_accuracy > mean_confidence + 0.05` | `underconfident` |

#### Comment — API (`src/api/models.py`)

1. Réutilise `await DBService.get_performance_pairs(db, name, start, end, version)` → tuples `(prediction_result, observed_result, probabilities, timestamp)`
2. `confidence = max(probabilities)` par prédiction
3. `correct = 1 if prediction_result == str(observed_result) else 0`
4. Brier score : `mean((confidence - correct)²)` — une ligne numpy
5. Reliability : buckets `[0.0, 0.1), ..., [0.9, 1.0]` → `predicted_rate = mean(confidence in bucket)`, `observed_rate = mean(correct in bucket)`
6. Garde : 422 si toutes les `probabilities` sont `null` (modèle sans `predict_proba`)
7. Nouveaux schémas `CalibrationResponse` + `ReliabilityBin` dans `src/schemas/model.py`
8. `api_client.py` : ajouter `get_model_calibration(name, version, start, end, n_bins)`

#### Comment — Streamlit (`pages/7_Supervision.py`)

- Section "📏 Calibration" dans la vue détail d'un modèle, après "Performance drift"
- KPI cards : Brier score · Gap confiance/précision · Statut coloré (🟢 / 🟡 / 🔴)
- **Courbe de calibration Plotly** :
  - Droite `y = x` en pointillé (calibration parfaite)
  - Courbe réelle avec marqueurs, taille de bulle proportionnelle au `count`
  - Zone grisée entre la courbe et la diagonale
- Message contextuel : si `overconfident` → *"Envisagez `CalibratedClassifierCV(method='isotonic')` lors du prochain retrain"*
- Si `insufficient_data` → `st.info("Soumettez des observed_results pour activer la calibration")`

---

### 2.2 `POST /observed-results/upload-csv`

#### Pourquoi

Le ground truth arrive **en CSV** depuis les outils BI, les tableurs ou les exports de bases métier. Aujourd'hui, chaque équipe écrit et maintient son propre script de conversion CSV → JSON avant d'appeler `POST /observed-results`. C'est une friction quotidienne qui ralentit le feedback loop prédiction → évaluation → drift.

La logique d'upsert dans `DBService.upsert_observed_results()` est déjà robuste — il suffit d'une nouvelle interface d'ingestion.

#### Endpoint

```
POST /observed-results/upload-csv   (multipart/form-data)
  file        : fichier CSV
  model_name  : string (optionnel — override la colonne CSV)

Format CSV attendu :
  id_obs, model_name, observed_result, date_time

Réponse :
{
  "upserted": 2450,
  "skipped_rows": 3,
  "parse_errors": [
    {"row": 14, "reason": "missing id_obs"},
    {"row": 87, "reason": "invalid date format"}
  ],
  "filename": "labels_q1.csv"
}
```

Comportement sur erreur : **partial success** — les lignes valides sont importées, les erreurs sont listées sans bloquer l'upload entier.

#### Comment — API (`src/api/observed_results.py`)

1. `UploadFile` + `Form(model_name=None)` optionnel — même pattern que `POST /models` avec `file=@...`
2. Vérification taille : `len(content) > 10 * 1024 * 1024` → 422 "Fichier trop volumineux (max 10 MB)"
3. Parse : `csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace")))`
4. Validation par ligne (id_obs requis, date_time en ISO 8601 ou `YYYY-MM-DD HH:MM:SS`, observed_result non vide) → collect d'erreurs sans interrompre
5. Appel `await DBService.upsert_observed_results(db, valid_records)` — **déjà existant**
6. Nouveau schéma `CSVUploadResponse` dans `src/schemas/prediction.py`
7. `api_client.py` : `upload_observed_results_csv(file_bytes, filename, model_name=None)` → `requests.post(..., files={"file": (filename, file_bytes)}, data={...})`

#### Comment — Streamlit (`pages/3_Predictions.py`)

- Expander "📤 Importer des résultats observés (CSV)" en haut de la page
- `st.file_uploader("Fichier CSV", type=["csv"])` + champ texte `model_name` override optionnel
- `st.download_button("Télécharger un template CSV", data="id_obs,model_name,observed_result,date_time\n")` — aide à la prise en main
- Après upload : `st.success(f"{result['upserted']} résultats importés")` 
- Si `parse_errors` : `st.warning` + `st.dataframe` du tableau des erreurs par ligne

---

### 2.3 `GET /models/{name}/confidence-trend`

#### Pourquoi

La confiance du modèle est un **signal précoce de drift** qui ne nécessite ni baseline ni observed_results. Un modèle voyant des inputs hors distribution voit sa confiance baisser avant que le feature Z-score franchisse un seuil ou que l'accuracy se dégrade (le ground truth arrivant toujours avec latence).

`probabilities` est stocké sur chaque prédiction en production. C'est une agrégation pure sur des données existantes.

#### Endpoint

```
GET /models/{name}/confidence-trend?version=1.0.0&days=30&granularity=day

Réponse :
{
  "model_name": "iris",
  "version": "1.0.0",
  "period_days": 30,
  "overall": {
    "mean_confidence": 0.84,
    "p25_confidence": 0.73,
    "p75_confidence": 0.96,
    "low_confidence_rate": 0.07
  },
  "trend": [
    {
      "date": "2026-03-21",
      "mean_confidence": 0.88,
      "p25": 0.79,
      "p75": 0.97,
      "predictions": 142,
      "low_confidence_count": 8
    },
    ...
  ]
}
```

`low_confidence_count` utilise le `confidence_threshold` du modèle (défaut 0.5).

#### Comment — API

**`src/services/db_service.py`** — nouvelle méthode `get_confidence_trend(db, name, version, days, granularity)` :
- Requête : `SELECT DATE(timestamp), probabilities FROM predictions WHERE model_name=... AND status='success' AND is_shadow=False AND timestamp > now() - interval 'N days'`
- Agrégation Python par jour : `confidence = max(probabilities)` par ligne, puis `mean / np.percentile([25, 75]) / count(< threshold)` — **même pattern que `get_accuracy_drift()`** déjà dans le service

**`src/api/models.py`** — nouvel endpoint, nouveau schéma `ConfidenceTrendResponse` dans `src/schemas/model.py`

**`api_client.py`** : `get_confidence_trend(name, version, days, granularity)`

#### Comment — Streamlit (`pages/7_Supervision.py`)

- Section "📉 Tendance de confiance" dans la vue détail, entre "Performance drift" et "Feature drift"
- **Graphique Plotly** : ligne `mean_confidence` + bande IQR (p25–p75) en fond semi-transparent + ligne horizontale en pointillé = `confidence_threshold`
- `st.metric` delta : `mean_confidence` actuelle vs semaine précédente (flèche verte/rouge)
- Si `low_confidence_rate > 0.15` → `st.warning(f"{low_confidence_rate*100:.0f} % des prédictions sous le seuil de confiance")`
- Si le modèle n'a pas de `predict_proba` (probabilities null) → `st.info("Ce modèle ne retourne pas de probabilités")`, pas d'erreur

---

## Tableau récapitulatif

| # | Feature | Endpoint(s) | Priorité | Difficulté | Effort estimé |
|---|---------|-------------|----------|------------|---------------|
| 1.1 | Profil + quota utilisateur | `GET /users/me`, `GET /users/me/quota` | **Haute** | Facile | ~2 h |
| 1.2 | Calcul automatique du baseline | `POST /models/{n}/{v}/compute-baseline` | **Haute** | Facile | ~3 h |
| 2.1 | Calibration du modèle | `GET /models/{n}/calibration` | **Haute** | Moyenne | ~1 jour |
| 2.2 | Import CSV observed results | `POST /observed-results/upload-csv` | **Haute** | Moyenne | ~1 jour |
| 2.3 | Tendance de confiance | `GET /models/{n}/confidence-trend` | Moyenne | Moyenne | ~1 jour |

---

## Fichiers à modifier

| Fichier | Modifications |
|---------|--------------|
| `src/api/users.py` | `GET /users/me`, `GET /users/me/quota` |
| `src/api/models.py` | `compute-baseline`, `calibration`, `confidence-trend` |
| `src/api/observed_results.py` | `POST /observed-results/upload-csv` |
| `src/services/db_service.py` | `get_confidence_trend()` |
| `src/schemas/user.py` | `QuotaResponse` |
| `src/schemas/model.py` | `ComputeBaselineResponse`, `CalibrationResponse`, `ReliabilityBin`, `ConfidenceTrendResponse` |
| `src/schemas/prediction.py` | `CSVUploadResponse` |
| `streamlit_app/utils/api_client.py` | 5 nouvelles méthodes |
| `streamlit_app/app.py` | Bloc quota dans la sidebar |
| `streamlit_app/pages/1_Users.py` | Section "Mon profil" pour non-admins |
| `streamlit_app/pages/2_Models.py` | Expander compute-baseline |
| `streamlit_app/pages/3_Predictions.py` | Expander upload CSV observed results |
| `streamlit_app/pages/7_Supervision.py` | Sections calibration + confidence-trend |

---

## Ce qui n'a PAS été retenu

| Idée | Raison |
|------|--------|
| Recherche texte sur les modèles | Table < 100 lignes — filtre Python côté Streamlit suffit |
| Alerting webhook sur drift | Email supervision toutes les 6 h déjà en place |
| Comparaison versions arbitraires | `ab-compare` regroupe déjà par version hors mode A/B |
| Endpoint "outlier predictions" | `low_confidence` déjà stocké — filtre sur `GET /predictions` suffit |
| Tests A/B Bayésiens | Chi-² + Mann-Whitney (power 80 %) déjà en place, suffisant |
| Async batch jobs | Nécessite une queue (Celery/Redis Streams), hors scope du projet |
