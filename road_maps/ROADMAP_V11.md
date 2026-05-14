# ROADMAP V11 — État des lieux & propositions finales

> **Perspective** : Data Scientist / MLOps Engineer utilisant cette plateforme au quotidien pour
> gérer des modèles sklearn en production.
>
> **Périmètre** : Ce document couvre uniquement ce qui est réellement absent après V1–V10.
> Tout ce qui est déjà implémenté n'est pas relisté.

---

## Constat préalable : le projet est fonctionnellement quasi-complet

Après inventaire exhaustif de V1 à V10, la plateforme couvre l'ensemble du cycle de vie MLOps :

- **55+ endpoints API** — prédiction, batch, SHAP inline, monitoring, drift input (Z-score + PSI +
  null-rate), A/B avec significativité statistique, retrain planifié, auto-promotion par policy,
  purge RGPD, validation d'input stricte, leaderboard, calibration, model card, golden tests,
  rollback, anomaly detection, confidence filtering, shadow deployment…
- **8 pages Streamlit** — utilisateurs, modèles, prédictions, stats, code example, A/B testing,
  supervision, retrain
- **Infrastructure complète** — PostgreSQL, MinIO, Redis, MLflow, APScheduler, Prometheus,
  OpenTelemetry, email SMTP, webhooks

**Ce document propose 3 lacunes réelles identifiées par analyse du code source.
Si elles ne correspondent pas à des douleurs concrètes, le projet est à considérer comme terminé.**

---

## Tableau de synthèse

| # | Domaine | Fonctionnalité | Priorité | Difficulté |
|---|---------|----------------|----------|------------|
| 1 | API + Monitoring | `GET /models/{name}/output-drift` — label shift sans ground truth | P1 | M |
| 2 | API + Dashboard | `GET /models/{name}/shadow-compare` — comparaison shadow enrichie | P2 | S |
| 3 | API + Sécurité | Auto-demotion policy (circuit breaker) *(optionnel)* | P2 | M |

Légende difficulté : **S** 1–4h · **M** 4–8h

---

## P1 — Output distribution drift (label shift monitoring)

---

### Pourquoi

`GET /models/{name}/drift` surveille la distribution des **features d'entrée** (Z-score, PSI,
null-rate). C'est nécessaire mais insuffisant.

Il existe un autre type de dérive silencieux : le **label shift** — le modèle commence à prédire
massivement une classe au détriment des autres, sans que les inputs ne changent. Exemple concret :
un modèle de scoring crédit prédit 90% "accepté" en décembre alors qu'il oscillait autour de 33%
en production stable. Le modèle n'est pas cassé, mais ses prédictions sont biaisées.

**Cette dérive est détectable sans ground truth**, en comparant la distribution récente des
`prediction_result` à la distribution d'entraînement (`label_distribution` dans `training_stats`).
C'est précisément la valeur de ce monitoring : détecter un problème **avant** d'avoir les étiquettes
réelles, souvent plusieurs semaines plus tôt.

| | Drift input (`/drift`) | Output drift (`/output-drift`) |
|---|---|---|
| Surveille | Features d'entrée | Distribution des prédictions |
| Baseline | `feature_baseline` | `training_stats.label_distribution` |
| Ground truth requis | Non | Non |
| Détecte | Changement de population | Biais de prédiction, dérive de concept |

### Comment

**Logique :**

1. Charger `training_stats.label_distribution` de la version de production (baseline, ex :
   `{"setosa": 0.33, "versicolor": 0.34, "virginica": 0.33}`).
2. Calculer la distribution des `prediction_result` sur les `period_days` derniers jours.
3. Calculer le **PSI** entre les deux distributions :
   - Classification : PSI sur les fréquences de classes (identique aux PSI existants sur features)
   - Régression : PSI sur binning en 10 intervalles des valeurs prédites
4. Si `training_stats.label_distribution` absent : réponse `{"status": "no_baseline"}`.
5. Seuils : PSI < 0.1 → `ok`, 0.1–0.2 → `warning`, ≥ 0.2 → `critical` (identiques input drift).

**Signature :**
```
GET /models/{name}/output-drift
  ?period_days=7          # fenêtre (défaut: 7)
  &model_version=1.0.0    # optionnel, version de production par défaut
```

**Réponse :**
```json
{
  "model_name": "iris",
  "model_version": "1.0.0",
  "period_days": 7,
  "predictions_analyzed": 850,
  "status": "warning",
  "psi": 0.14,
  "baseline_distribution": {
    "setosa": 0.33,
    "versicolor": 0.34,
    "virginica": 0.33
  },
  "current_distribution": {
    "setosa": 0.62,
    "versicolor": 0.21,
    "virginica": 0.17
  },
  "by_class": [
    { "label": "setosa",     "baseline_ratio": 0.33, "current_ratio": 0.62, "delta": +0.29 },
    { "label": "versicolor", "baseline_ratio": 0.34, "current_ratio": 0.21, "delta": -0.13 },
    { "label": "virginica",  "baseline_ratio": 0.33, "current_ratio": 0.17, "delta": -0.16 }
  ]
}
```

**Intégration supervision :**
- Ajouter l'évaluation du output drift dans le scheduler d'alertes existant (`src/tasks/`)
- Si `status == "critical"` → email + webhook identiques aux alertes de drift input actuelles

**Côté Streamlit :**
- Section "Drift de sortie" dans `2_Models.py` (détail modèle) : tableau `by_class` + jauge PSI
- Badge `output_drift_status` dans `7_Supervision.py` à côté du badge drift input existant

**Fichiers à modifier :**
- `src/services/drift_service.py` — ajouter `compute_output_drift(model_name, period_days, db)`
- `src/schemas/model.py` — `OutputDriftResponse`, `OutputDriftClassResult`
- `src/api/models.py` — handler `GET /models/{name}/output-drift`
- `src/services/db_service.py` — `get_prediction_label_distribution(name, days)` (agrégation SQL)
- `src/tasks/` (supervision scheduler) — appel `compute_output_drift` dans la boucle d'alerte
- `streamlit_app/pages/2_Models.py` — section output drift dans détail
- `streamlit_app/pages/7_Supervision.py` — badge output drift dans tableau de supervision
- `tests/test_drift.py` — cas output drift (no_baseline, ok, warning, critical)

---

## P2a — `GET /models/{name}/shadow-compare` enrichi

---

### Pourquoi

Le shadow deployment est implémenté et fonctionnel : les prédictions shadow sont persistées avec
`is_shadow=True`, `get_shadow_agreement_rate()` existe dans `db_service.py`, et `7_Supervision.py`
affiche un taux d'accord sommaire.

Cependant, il n'existe pas d'**endpoint REST dédié** pour consulter la comparaison shadow de façon
programmatique, et les statistiques disponibles dans le dashboard se limitent au taux d'accord
brut. Il manque les métriques qui permettent de décider si on promeut le shadow en production :
delta de confiance, delta de latence, et surtout **accuracy shadow vs production sur le même
ground truth**.

Sans ces données, un Data Scientist doit exporter et recroiser manuellement plusieurs sources
pour prendre la décision de promotion — une friction inutile quand l'infrastructure est déjà là.

### Comment

**Logique :**
1. Identifier la version en `deployment_mode="shadow"` pour le modèle.
2. Joindre les prédictions shadow et production sur `id_obs` communs (jointure existante dans
   `get_shadow_agreement_rate()`).
3. Calculer les métriques différentielles sur ces paires.
4. Si des `observed_results` existent pour ces `id_obs`, calculer l'accuracy de chaque version.

**Signature :**
```
GET /models/{name}/shadow-compare
  ?period_days=30    # fenêtre (défaut: 30)
```

**Réponse :**
```json
{
  "model_name": "iris",
  "shadow_version": "2.0.0",
  "production_version": "1.0.0",
  "period_days": 30,
  "n_comparable": 412,
  "agreement_rate": 0.87,
  "shadow_confidence_delta": +0.04,
  "shadow_latency_delta_ms": -12.3,
  "shadow_accuracy": 0.94,
  "production_accuracy": 0.91,
  "accuracy_available": true,
  "recommendation": "shadow_better"
}
```

- `recommendation` : `"shadow_better"` | `"production_better"` | `"equivalent"` | `"insufficient_data"`
- `accuracy_available: false` si aucun `observed_result` ne couvre les `id_obs` comparés

**Côté Streamlit — `6_AB_Testing.py` :**
- Onglet "Shadow" enrichi remplaçant l'affichage actuel dans `7_Supervision.py` :
  métriques en colonnes, jauge d'accord, bouton "Promouvoir le shadow" si `recommendation="shadow_better"`

**Fichiers à modifier :**
- `src/schemas/model.py` — `ShadowCompareResponse`
- `src/api/models.py` — handler `GET /models/{name}/shadow-compare`
- `src/services/db_service.py` — `get_shadow_comparison_stats(name, period_days)` (enrichit
  l'existant `get_shadow_agreement_rate()`)
- `streamlit_app/pages/6_AB_Testing.py` — onglet Shadow enrichi
- `tests/test_ab_shadow.py` — cas shadow compare endpoint

---

## P2b — Auto-demotion policy / circuit breaker *(optionnel)*

---

### Pourquoi

L'auto-promotion existe (`PATCH /models/{name}/policy`). Son complément logique est
l'**auto-demotion** : si un modèle en production dérive au-delà d'un seuil critique ou que son
accuracy chute sous un minimum, le retirer automatiquement de la production.

Les alertes email et webhook existent déjà, mais elles restent **passives** : elles notifient
sans agir. En dehors des heures de bureau, un modèle dégradé peut rester en production des heures.

**Pourquoi c'est optionnel :** le risque principal est le faux positif — une demotion automatique
sans version de fallback disponible met le service hors production. Il faut donc des garde-fous
stricts et une activation explicite opt-in.

### Comment

**Étendre `PATCH /models/{name}/policy`** avec des champs optionnels :

```json
{
  "auto_promote": true,
  "min_accuracy": 0.90,
  "auto_demote": true,
  "demote_on_drift": "critical",
  "demote_on_accuracy_below": 0.75,
  "demote_cooldown_hours": 24
}
```

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `auto_demote` | bool | false | Activer le circuit breaker |
| `demote_on_drift` | `"warning"` \| `"critical"` | `"critical"` | Niveau de drift déclencheur |
| `demote_on_accuracy_below` | float \| null | null | Accuracy minimale (nécessite observed_results) |
| `demote_cooldown_hours` | int | 24 | Délai minimal entre deux demotions automatiques |

**Garde-fous impératifs :**
- Ne démote que si ≥ 1 autre version active (`is_production=False`) existe comme fallback
- Si aucun fallback → pas de demotion, alerte critique à la place
- Cooldown : évite les oscillations production/demotion sur des alertes transitoires
- Loggé dans `ModelHistory` avec `action_type="auto_demote"` + raison détaillée
- Email + webhook envoyés immédiatement après la demotion

**Côté Streamlit — `7_Supervision.py` :**
- Toggle "Auto-demotion activée" + affichage des critères configurés
- Badge rouge "Auto-démis le [date] — raison : [drift_critical]" si demotion récente

**Fichiers à modifier :**
- `src/schemas/model.py` — enrichir `PromotionPolicy` avec les champs auto_demote
- `src/services/auto_promotion_service.py` — `evaluate_auto_demotion(model_name, db)`
- `src/tasks/` (supervision scheduler) — appel périodique `evaluate_auto_demotion`
- `src/api/models.py` — PATCH /models/{name}/policy (déjà existant, ajouter les champs)
- `streamlit_app/pages/7_Supervision.py` — toggle + statut auto-demotion
- `tests/test_auto_promotion_policy.py` — cas demotion (cooldown, no fallback, triggered)

---

## Ce qui a été délibérément exclu

| Fonctionnalité | Raison d'exclusion |
|---|---|
| Async batch scoring | Le batch synchrone couvre les cas ML standards (< quelques milliers de lignes). Implémenter un système de job async (job_id, polling, cleanup) ajoute une complexité infrastructure significative pour un bénéfice marginal. |
| Multi-armed bandit routing | 5 % des utilisateurs au maximum. Le A/B test avec significativité statistique (Chi-², Mann-Whitney, power analysis) couvre 95 % des besoins de comparaison. |
| PDP / ICE plots | Niche. SHAP global + what-if explorer (déjà en V9) couvrent les besoins d'explicabilité pour la grande majorité des Data Scientists. |
| Baseline auto-refresh planifié | Contre-productif : recalculer la baseline régulièrement masquerait le drift réel. La baseline doit rester ancrée à la distribution d'entraînement d'origine. |
| Feature store / multi-tenant | Hors scope total. La plateforme est un serveur de modèles, pas un pipeline de données ni une plateforme SaaS multi-organisation. |
| Métriques de fairness / biais | Nécessite des attributs protégés absents du système et une définition métier. Trop domain-specific pour être générique et utile à 80 % des utilisateurs. |
| ONNX export | Hors scope. Introduit une dépendance lourde pour un cas d'usage rare (déploiement edge / inférence batch GPU). |

---

## Ordre d'implémentation recommandé

**Si une seule chose :** implémenter le output drift (#1) — comble le vrai blind spot de monitoring
le plus fréquent en production ML.

**Si deux choses :** ajouter shadow compare (#2) — complète une feature déjà à 80 %, effort minimal
pour un gain direct sur le workflow de promotion.

**Si le retraining automatique est actif et l'équipe confiante dans ses seuils :**
activer l'auto-demotion (#3) — mais uniquement après avoir validé les garde-fous sur un
environnement non-critique.

---

## Verdict final

La plateforme predictml-api couvre l'ensemble du cycle de vie MLOps pour des modèles sklearn.
Les trois propositions ci-dessus comblent des lacunes opérationnelles identifiées dans le code
source — aucune n'est une feature inventée pour le plaisir.

**Si elles ne correspondent pas à des douleurs actuelles sur votre déploiement, le projet est
complet.** L'ajout de fonctionnalités supplémentaires risque de complexifier la maintenance
sans apporter de valeur proportionnelle aux 80 % des utilisateurs.
