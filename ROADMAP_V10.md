# ROADMAP V10 — État des lieux & propositions ciblées

> **Perspective** : Data Scientist / MLOps Engineer utilisant cette plateforme au quotidien pour
> gérer des modèles sklearn en production.
>
> **Périmètre** : Ce document couvre uniquement ce qui est réellement absent après V1–V9.
> Tout ce qui est déjà implémenté n'est pas relisté.

---

## Constat préalable : le projet est fonctionnellement complet

Après inventaire exhaustif de V1 à V9, la plateforme couvre :

- **55 endpoints API** — prédiction, batch, SHAP, monitoring, drift, A/B avec significativité statistique, retrain planifié, auto-promotion par policy, purge RGPD, validation d'input stricte, leaderboard, calibration, model card, rollback, retrain-history, what-if explorer, scatter plot multi-modèles…
- **8 pages Streamlit** — utilisateurs, modèles, prédictions, stats, code example, A/B testing, supervision, retrain
- **Services complets** — drift (Z-score + PSI + null rate), SHAP (tree & linear), auto-promotion, webhooks, email SMTP, scheduler APScheduler, ab-significance, input validation

**Si aucune des trois propositions ci-dessous ne résout un problème réel et quotidien, le projet est à considérer comme terminé. Ne pas ajouter de fonctionnalités par principe.**

---

## Tableau de synthèse

| # | Domaine | Fonctionnalité | Priorité | Difficulté |
|---|---------|----------------|----------|------------|
| 1 | API + UX | Filtre de confiance sur `GET /predictions` | P1 | S |
| 2 | API + UX | `GET /predictions/anomalies` — prédictions avec input aberrant | P1 | S |
| 3 | API + UX | Golden test set — régression testing pré-déploiement | P2 | M |

Légende difficulté : **S** 1–4h · **M** 4–8h

---

## P1 — Quick wins à fort impact quotidien

---

### 1. Filtre de confiance sur `GET /predictions`

#### Pourquoi

Le endpoint `GET /predictions` permet de filtrer par `model_name`, `version`, `user`, dates, et `id_obs`.
Mais aucun filtre sur le niveau de confiance de la prédiction n'existe.

Or le workflow QA le plus fréquent pour un Data Scientist est précisément :
**"Montre-moi toutes les prédictions de la semaine où le modèle était incertain (< 70% de confiance)".**

Sans ce filtre, il faut exporter l'intégralité des prédictions, recalculer `max(probabilities)` en Python
et filtrer manuellement — une opération que personne ne fait systématiquement, ce qui laisse passer
des prédictions problématiques en silence.

Le champ `probabilities` est déjà stocké en JSON dans la table `predictions`. Il manque uniquement
une colonne dérivée indexée et son exposition comme filtre de requête.

#### Comment

**Côté base de données :**
- Ajouter une colonne `max_confidence FLOAT` dans `src/db/models/prediction.py`
- Calculée à l'insertion : `max(probabilities)` pour les modèles de classification,
  `None` pour la régression (pas de notion de probabilité)
- Migration Alembic : `alembic/versions/YYYYMMDD_add_max_confidence.py`
- Indexer la colonne : `Index("ix_predictions_max_confidence", "max_confidence")`

**Côté API — `GET /predictions` :**
- Ajouter deux paramètres optionnels : `min_confidence: Optional[float] = None` et
  `max_confidence: Optional[float] = None`
- Filtre SQL dans `db_service.get_predictions()` :
  ```python
  if min_confidence is not None:
      query = query.filter(Prediction.max_confidence >= min_confidence)
  if max_confidence is not None:
      query = query.filter(Prediction.max_confidence <= max_confidence)
  ```
- Inclure `max_confidence` dans le schéma de réponse `PredictionResponse`

**Côté Streamlit — page "3_Predictions.py" :**
- Ajouter deux `st.slider` dans la section filtres :
  `Confiance min` et `Confiance max` (0.0 → 1.0, step 0.05)
- Désactiver les sliders si le modèle sélectionné est un régresseur (max_confidence = null)

**Exemple d'usage :**
```bash
# Toutes les prédictions incertaines des 7 derniers jours
GET /predictions?model_name=iris&max_confidence=0.70&start=2026-04-21

# Seulement les prédictions très confiantes
GET /predictions?model_name=iris&min_confidence=0.95
```

**Fichiers à modifier :**
- `src/db/models/prediction.py` — colonne `max_confidence`
- `src/api/predict.py` — calcul au moment de l'insert + paramètres de filtre
- `src/services/db_service.py` — filtre SQL dans `get_predictions()`
- `src/schemas/prediction.py` — champ `max_confidence` dans `PredictionResponse`
- `alembic/versions/` — nouvelle migration
- `streamlit_app/pages/3_Predictions.py` — sliders de filtre

---

### 2. `GET /predictions/anomalies` — prédictions avec features aberrantes

#### Pourquoi

`GET /models/{name}/drift` calcule le drift **en agrégat** sur une fenêtre glissante : il répond à
"la distribution des features a-t-elle changé globalement ?". C'est utile pour détecter une tendance
mais insuffisant pour le debugging.

Quand un modèle dérive, la question suivante est toujours : **"Quelle prédiction concrète avait
des features anormales, et lesquelles ?"** Aujourd'hui, cette question oblige à exporter les données
et recalculer manuellement des z-scores feature par feature — une tâche longue qui n'est pas faite.

Ce endpoint comble l'écart entre drift agrégé (macro) et anomalie individuelle (micro) :

| | Drift (`/drift`) | Anomalies (`/predictions/anomalies`) |
|---|---|---|
| Granularité | Distribution globale | Prédiction individuelle |
| Question | "Le modèle dérive-t-il ?" | "Quelles prédictions avaient des inputs aberrants ?" |
| Utilité | Alerte | Debugging, root cause analysis |

#### Comment

**Logique :**
1. Charger les N dernières prédictions pour le modèle (limité à `limit`, fenêtre `days`)
2. Charger `feature_baseline` du modèle (si absent : retourner `{"error": "no_baseline"}`)
3. Pour chaque prédiction, calculer le z-score par feature :
   `z = |value - baseline_mean| / baseline_std`
4. Retourner uniquement les prédictions où ≥ 1 feature a `z_score >= z_threshold`
5. Inclure les features fautives avec leur valeur, z-score et les stats de baseline

**Zéro nouvelle colonne DB.** Entièrement applicatif, réutilise `drift_service.compute_feature_drift()`
qui implémente déjà le calcul de z-score.

**Signature :**
```
GET /predictions/anomalies
  ?model_name=iris          # requis
  &days=7                   # fenêtre temporelle (défaut: 7)
  &z_threshold=3.0          # seuil de détection (défaut: 3.0)
  &limit=100                # max prédictions à analyser (défaut: 200, max: 1000)
```

**Réponse :**
```json
{
  "model_name": "iris",
  "period_days": 7,
  "z_threshold": 3.0,
  "total_checked": 850,
  "anomalous_count": 12,
  "anomaly_rate": 0.014,
  "predictions": [
    {
      "prediction_id": "abc123",
      "timestamp": "2026-04-27T14:32:00",
      "prediction_result": "setosa",
      "max_confidence": 0.91,
      "anomalous_features": {
        "sepal_length": {
          "value": 12.1,
          "z_score": 4.7,
          "baseline_mean": 5.8,
          "baseline_std": 0.83
        }
      }
    }
  ]
}
```

**Côté Streamlit — page "7_Supervision.py" :**
- Ajouter un onglet "Prédictions anomales" dans la vue détail par modèle
- Tableau des prédictions flaggées avec liens vers la prédiction complète
- Slider pour ajuster `z_threshold` en temps réel

**Fichiers à modifier :**
- `src/api/predict.py` — nouveau handler `get_anomalous_predictions()`
- `src/services/db_service.py` — `get_predictions_with_features(model_name, days, limit)` (variante de `get_predictions()` retournant aussi `input_features`)
- `src/schemas/prediction.py` — `AnomalyPredictionEntry`, `AnomaliesResponse`
- `streamlit_app/pages/7_Supervision.py` — onglet anomalies
- `tests/test_predictions_anomalies.py` — nouveau fichier de tests

---

## P2 — Effort moyen, valeur MLOps réelle

---

### 3. Golden test set — régression testing pré-déploiement

#### Pourquoi

L'auto-promotion évalue `accuracy` sur `observed_results` et latence P95. C'est une mesure
statistique globale — bonne pour détecter une régression diffuse, mais insuffisante pour garantir
que le modèle ne régresse pas sur des **cas connus et importants**.

Exemple concret : un modèle iris retrained avec de nouvelles données peut avoir une accuracy globale
de 0.94 (critère satisfait) tout en classant mal `id_obs=patient_007` — un cas limite documenté
dans les incidents passés. L'auto-promotion va promouvoir ce modèle.

Le golden test set répond à un pattern MLOps établi : **"ces N exemples critiques doivent toujours
produire le bon output"**. C'est l'équivalent ML des tests de régression en développement logiciel.

Intérêt maximal pour les équipes qui font du retraining fréquent (schedulé ou manuel) :
chaque nouveau modèle doit passer le golden set avant d'être éligible à la promotion automatique.

#### Comment

**Nouvelle table DB `GoldenTest` :**
```python
# src/db/models/golden_test.py
class GoldenTest(Base):
    id: int (PK)
    model_name: str (FK → ModelMetadata.name)
    input_features: JSON          # {"sepal_length": 5.1, ...}
    expected_output: str/float    # "setosa" ou 0.87
    description: str (nullable)  # "cas limite identifié incident #42"
    created_at: datetime
    created_by_user_id: int
```

**Endpoints :**

```
POST   /models/{name}/golden-tests           # Créer un cas de test (admin)
POST   /models/{name}/golden-tests/upload-csv # Batch upload CSV (admin)
GET    /models/{name}/golden-tests           # Lister les cas (bearer auth)
DELETE /models/{name}/golden-tests/{id}      # Supprimer un cas (admin)
POST   /models/{name}/{version}/golden-tests/run  # Exécuter les tests (admin)
```

**Schéma de réponse de `/run` :**
```json
{
  "model_name": "iris",
  "version": "1.1.0",
  "total_tests": 20,
  "passed": 19,
  "failed": 1,
  "pass_rate": 0.95,
  "details": [
    {
      "test_id": 4,
      "description": "cas limite incident #42",
      "input": {"sepal_length": 4.3, "sepal_width": 3.0, "petal_length": 1.1, "petal_width": 0.1},
      "expected": "setosa",
      "actual": "versicolor",
      "passed": false
    }
  ]
}
```

**Intégration auto-promotion :**
- Ajouter `min_golden_test_pass_rate: Optional[float] = None` dans `PromotionPolicy`
  (dans `src/schemas/model.py` et `src/services/auto_promotion_service.py`)
- Si configuré, exécuter `/run` en interne lors de l'évaluation de la policy
- Si `pass_rate < min_golden_test_pass_rate` → non promu, avec `auto_promote_reason` explicite

**Côté Streamlit — page "2_Models.py" :**
- Nouvel onglet "Tests de régression" dans la section détail d'un modèle :
  - Formulaire d'ajout de cas de test (JSON input + expected output)
  - Bouton "Upload CSV"
  - Liste des cas existants
  - Bouton "Lancer les tests" sur la version sélectionnée avec résultat pass/fail coloré

**Fichiers à modifier / créer :**
- `src/db/models/golden_test.py` — nouveau modèle SQLAlchemy
- `src/api/models.py` — 5 nouveaux handlers
- `src/services/golden_test_service.py` — logique d'exécution des tests (nouveau)
- `src/schemas/golden_test.py` — schémas Pydantic (nouveau)
- `src/schemas/model.py` — `PromotionPolicy` + champ `min_golden_test_pass_rate`
- `src/services/auto_promotion_service.py` — critère golden test
- `alembic/versions/YYYYMMDD_add_golden_tests.py` — migration
- `streamlit_app/pages/2_Models.py` — onglet "Tests de régression"
- `tests/test_golden_tests.py` — nouveau fichier de tests

---

## Ce qui a été délibérément exclu

| Fonctionnalité | Raison d'exclusion |
|---|---|
| Refresh de baseline planifié | Contre-productif : masquerait le drift réel. La baseline doit rester ancrée à la distribution d'origine. Un modèle retrained repart avec une nouvelle baseline naturellement. |
| Rate limiting par modèle | Redondant : le rate limiting par user + la purge RGPD couvrent déjà le contrôle du volume. |
| Expiration automatique des tokens | Hors du scope MLOps central. La régénération manuelle existe déjà ; l'expiration relève de la politique de sécurité organisationnelle. |
| Sampling des prédictions (log 10%) | Perte de données d'audit. La purge RGPD (`DELETE /predictions/purge`) est le bon outil pour contrôler la taille de la table. |
| Métriques de fairness / biais | Nécessite des données démographiques absentes du système et une définition métier. Trop domain-specific pour être générique. |
| Intégration feature store externe | Scope creep. La plateforme est un serveur de modèles, pas un pipeline de données. |
| Comparaison cross-modèles (nouvel endpoint) | Le leaderboard `GET /models/leaderboard` + le scatter plot accuracy vs latency (V9) couvrent déjà ce besoin. |
| Notification en production sur predict | Ajoute de la latence sur le chemin chaud. Les webhooks post-prédiction (déjà implémentés) adressent ce besoin. |
| Multi-armed bandit | Cas d'usage trop spécialisé. Le A/B test standard avec significativité statistique couvre 95% des besoins. |

---

## Ordre d'implémentation recommandé

**Si une seule chose :** implémenter le filtre de confiance (#1) — effort minimal, utilisé immédiatement.

**Si deux choses :** ajouter les anomalies (#2) — complète le drift agrégé et active le debugging.

**Si le retraining automatique est activement utilisé :** implémenter le golden test set (#3).

---

## Verdict final

La plateforme predictml-api couvre l'ensemble du cycle de vie MLOps pour des modèles sklearn :
upload, serving, monitoring, drift, A/B testing, retraining automatique, governance et conformité.
Les trois propositions ci-dessus comblent des lacunes opérationnelles réelles — aucune n'est
une feature inventée pour le plaisir. Si elles ne correspondent pas à des douleurs actuelles,
**le projet est complet**.
