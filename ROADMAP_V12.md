# ROADMAP V12 — État des lieux & propositions

> **Perspective** : Data Scientist / MLOps Engineer utilisant cette plateforme au quotidien.
>
> **Périmètre** : Ce document couvre uniquement ce qui est réellement absent après V1–V11.
> Les fonctionnalités déjà implémentées ne sont pas relistées.

---

## Bilan V11 : les 3 items sont implémentés

Depuis ROADMAP_V11, les trois lacunes identifiées ont été livrées et testées :

| Item V11 | Statut | Preuve |
|---|---|---|
| P1 — Output drift (`GET /models/{name}/output-drift`) | ✅ Implémenté | `drift_service.compute_output_drift()`, wired dans supervision_reporter, tests dans `test_drift.py` |
| P2a — Shadow-compare enrichi (`GET /models/{name}/shadow-compare`) | ✅ Implémenté | Retourne `shadow_accuracy`, `production_accuracy`, `confidence_delta`, `latency_delta`, `recommendation`. Section dédiée dans `6_AB_Testing.py`. |
| P2b — Auto-demotion / circuit breaker | ✅ Implémenté | `auto_promotion_service.evaluate_auto_demotion()`, `PromotionPolicy` avec `auto_demote` / `demote_on_drift` / `demote_cooldown_hours`, appelé dans la boucle supervision toutes les 6h. |

---

## Constat : le projet est fonctionnellement complet

Après inventaire exhaustif de V1 à V11, la plateforme couvre l'intégralité du cycle de vie
MLOps pour des modèles sklearn :

- **50+ endpoints API** — prédiction unitaire, batch (`/predict-batch`), SHAP inline/post-hoc,
  monitoring, drift input + output, A/B avec significativité statistique, shadow compare,
  retrain planifié cron, auto-promotion/demotion, purge RGPD, validation stricte d'input,
  leaderboard, calibration, model card, golden tests, rollback, anomaly detection, confidence
  trend/distribution, readiness checks, model download, model compare, webhooks, quotas…
- **8 pages Streamlit** — utilisateurs, modèles, prédictions, stats, code example, A/B testing,
  supervision, retrain
- **Infrastructure complète** — PostgreSQL, MinIO, Redis (FakeRedis en dev), MLflow,
  APScheduler, Prometheus, OpenTelemetry, email SMTP, webhooks

**Si aucune des deux propositions ci-dessous ne correspond à une douleur concrète, le projet
est à considérer comme terminé.**

---

## Propositions V12 (2 uniquement)

| # | Domaine | Fonctionnalité | Priorité | Difficulté |
|---|---------|----------------|----------|------------|
| 1 | API + Scheduler | Retrain déclenché par le drift | P1 | M |
| 2 | Streamlit | Page Golden Tests (API déjà existante) | P2 | S |

Légende difficulté : **S** 1–4h · **M** 4–8h

---

## P1 — Retrain déclenché par le drift

### Pourquoi

Le scheduler de retrain (`retrain_scheduler.py`) est entièrement **cron-based** : il se
déclenche à des intervalles fixes (ex. chaque lundi à 03h00). Or le drift ne respecte pas
les calendriers. Un incident de drift peut survenir un mardi à 14h00 — le modèle dégradé
restera en production jusqu'au prochain cycle cron, potentiellement plusieurs jours.

La boucle d'alertes dans `supervision_reporter.py` tourne déjà toutes les 6h, calcule le
drift input et output, envoie email + webhook. Elle a **tous les signaux** pour déclencher
un retrain — mais elle ne le fait pas encore.

L'auto-demotion (V11) coupe le modèle si les critères sont franchis. Le retrain drift-triggered
est son complément : il déclenche la remédiation active au lieu de simplement éteindre.

### Ce qui manque

Deux petites additions au `retrain_schedule` JSON :

```json
{
  "cron": "0 3 * * 1",
  "lookback_days": 30,
  "auto_promote": false,
  "enabled": true,
  "trigger_on_drift": "critical",
  "drift_retrain_cooldown_hours": 24
}
```

| Champ | Type | Défaut | Description |
|---|---|---|---|
| `trigger_on_drift` | `"warning"` \| `"critical"` \| null | null | Niveau de drift déclencheur (null = désactivé) |
| `drift_retrain_cooldown_hours` | int ≥ 1 | 24 | Délai minimal entre deux retrains drift-triggered (évite les boucles) |

**Comportement** : si `trigger_on_drift` est défini et que le drift calculé par
`supervision_reporter` atteint ou dépasse ce seuil, le reporter appelle `_do_retrain()`
(déjà implémenté dans `retrain_scheduler.py`) — à condition que :
1. Le modèle a un `train_script_object_key` (script `.py` disponible dans MinIO)
2. Le cooldown n'est pas actif (`last_run_at` + cooldown < now)

Le retrain planifié et le retrain drift-triggered sont complémentaires : le cron reste actif,
le drift-trigger ajoute un déclenchement réactif hors cycle.

### Signature

Aucune nouvelle route — l'endpoint existant suffit :
```
PATCH /models/{name}/{version}/schedule
Body: { "cron": "0 3 * * 1", "trigger_on_drift": "critical", "drift_retrain_cooldown_hours": 24 }
```

### Implémentation

- `src/schemas/model.py` — Ajouter `trigger_on_drift` et `drift_retrain_cooldown_hours` à
  `RetrainScheduleInput`
- `src/tasks/supervision_reporter.py` — Dans `run_alert_check()`, après calcul du drift,
  vérifier `schedule.get("trigger_on_drift")` et déclencher `_do_retrain()` si seuil atteint
  et cooldown expiré
- `src/api/models.py` — `update_retrain_schedule()` persiste déjà `schedule_dict` ; inclure
  les deux nouveaux champs
- `tests/test_scheduled_retraining.py` — Ajouter cas : trigger fire sur critical, pas de
  trigger si warning avec threshold=critical, cooldown bloque le second fire

---

## P2 — Page Streamlit Golden Tests

### Pourquoi

L'API Golden Tests est entièrement implémentée depuis V8 :
- `POST /models/{name}/golden-tests` — créer un cas de test (input + expected output)
- `GET /models/{name}/golden-tests` — lister les tests
- `POST /models/{name}/golden-tests/run` — exécuter et obtenir pass/fail

**Il n'existe aucune page Streamlit pour cette fonctionnalité.**

En pratique, cela signifie que les golden tests ne sont accessibles qu'en ligne de commande
(curl ou Python). La majorité des Data Scientists et MLOps Engineers qui utilisent le
dashboard n'utilisent donc pas cette fonctionnalité, alors qu'elle est particulièrement
précieuse pour les pipelines de retrain : vérifier qu'un modèle ré-entraîné produit toujours
les sorties attendues sur des cas de référence.

### Ce qui manque

Une page `streamlit_app/pages/9_Golden_Tests.py` suivant le pattern des pages existantes :

**Section 1 — Sélection du modèle** (selectbox)

**Section 2 — Cas de tests existants** (tableau)
- Colonnes : description, input, expected output, created_by, date
- Bouton "Lancer tous les tests" → appel `POST /models/{name}/golden-tests/run`
- Affichage résultats inline : ✅ PASS / ❌ FAIL + diff attendu/reçu si FAIL

**Section 3 — Ajouter un cas de test** (expandable form, admin uniquement)
- Champ JSON pour les features d'entrée
- Champ expected output
- Description libre
- Bouton "Enregistrer"

### Implémentation

- `streamlit_app/pages/9_Golden_Tests.py` — nouvelle page (pattern identique à 8_Retrain.py)
- `streamlit_app/utils/api_client.py` — ajouter `get_golden_tests(model_name)`,
  `create_golden_test(model_name, payload)`, `run_golden_tests(model_name)`

---

## Ce qui a été délibérément exclu (non-objectifs V12)

| Fonctionnalité | Raison d'exclusion |
|---|---|
| Streaming WebSocket / SSE | Niche (< 20 % des cas). Complexifie l'infrastructure pour un bénéfice marginal sur des workloads qui restent majoritairement request/response. |
| Multi-model ensemble | Haute complexité d'implémentation, cas d'usage rare. Le A/B testing et le shadow deployment couvrent les besoins de comparaison sans fusionner les sorties. |
| Métriques de fairness / biais | Nécessite des attributs protégés absents du système et une définition métier spécifique. Hors scope généraliste. |
| PDP / ICE plots | Déjà exclu en V11. SHAP global (feature importance) + SHAP par prédiction (explain endpoint) couvrent les besoins d'explicabilité. |
| Intégration MLflow plus profonde | Le tracking des retrains via MLflow est déjà optionnel (`MLFLOW_TRACKING_URI`). Pousser également les prédictions live ajouterait une dépendance forte pour un gain limité. |
| Feature store / multi-tenant | Hors scope. La plateforme est un serveur de modèles, pas un pipeline de données ni une plateforme SaaS. |
| ONNX export | Déjà exclu en V11. Dépendance lourde pour un cas d'usage edge/GPU rare. |
| Async batch scoring avec job_id | `POST /predict-batch` synchrone est déjà implémenté et couvre les besoins standards (quelques milliers de lignes). Un système job async (polling, TTL, cleanup) représente une complexité infrastructure disproportionnée. |
| Multi-armed bandit routing | Chi-², Mann-Whitney U, calcul de puissance statistique sont déjà en production. Le bandit ne serait utile qu'en exploitation continue avec récompenses temps-réel — hors scope. |

---

## Verdict final

La plateforme est **complète** au sens opérationnel du terme.

Les deux propositions ci-dessus ne comblent pas des manques structurels : elles ferment deux
petites imperfections dans des workflows déjà fonctionnels. Si le retrain automatique n'est
pas utilisé, ou si l'équipe gère ses golden tests en dehors du dashboard, aucune des deux
n'est nécessaire.

**Règle appliquée** : si une proposition ne résout pas une douleur concrète sur votre
déploiement, ne l'implémentez pas.
