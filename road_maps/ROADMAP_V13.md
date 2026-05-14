# ROADMAP V13 — Bilan d'exhaustivité & conclusion

> **Perspective** : Data Scientist / MLOps Engineer utilisant cette plateforme au quotidien.
>
> **Périmètre** : Audit exhaustif après V12. Ce document corrige l'état réel des deux
> propositions V12 et conclut sur la maturité du projet.

---

## Bilan V12 : les deux items sont déjà implémentés

ROADMAP_V12 listait deux propositions non encore livrées. Après audit du code source, les
deux sont présentes et fonctionnelles.

### P1 V12 — Retrain déclenché par drift

**Statut : ✅ Implémenté**

`src/tasks/supervision_reporter.py`, lignes 258–287 :

```python
# Retrain déclenché par drift
if prod_meta:
    sched = prod_meta.retrain_schedule
    if (
        sched
        and sched.get("trigger_on_drift")
        and prod_meta.train_script_object_key
    ):
        threshold_level = _DRIFT_LEVEL.get(sched["trigger_on_drift"], 999)
        detected_level = max(
            _DRIFT_LEVEL.get(_max_input_drift, 0),
            _DRIFT_LEVEL.get(_max_output_drift, 0),
        )
        if detected_level >= threshold_level:
            cooldown_hours = int(sched.get("drift_retrain_cooldown_hours", 24))
            ...
            if cooldown_ok:
                asyncio.create_task(_run_retrain_job(model_name, prod_meta.version))
```

Les champs `trigger_on_drift` et `drift_retrain_cooldown_hours` sont gérés par
`PATCH /models/{name}/{version}/schedule` et persistés dans `retrain_schedule`.
Le cooldown bloque les re-déclenchements consécutifs. La boucle `run_alert_check()`
(toutes les 6h) calcule `_max_input_drift` et `_max_output_drift` avant d'évaluer
le trigger.

### P2 V12 — Page Streamlit Golden Tests

**Statut : ✅ Implémenté**

`streamlit_app/pages/9_Golden_Tests.py` est complète avec :

- Sélection modèle + version
- Tableau des cas de test existants avec "Lancer tous les tests" (résultats inline
  PASS/FAIL + diff attendu/reçu)
- Suppression de cas (admin)
- Formulaire ajout (admin)
- Import CSV en lot (admin)

---

## Inventaire complet du projet à ce jour

Après audit de V1 à V12 (y compris les deux items ci-dessus), la plateforme couvre
l'intégralité du cycle de vie MLOps pour des modèles sklearn en production :

### API — 50+ endpoints

| Domaine | Couverture |
|---|---|
| Prédiction | Unitaire, batch, SHAP inline, SHAP post-hoc, explain par ID, validation stricte d'input |
| Gestion modèles | CRUD, versioning, rollback, deprecation, download, warmup, model card |
| Monitoring | Leaderboard, performance, timeline, calibration, confidence trend/distribution, readiness |
| Drift | Input (PSI + Z-test + null rate), output (label shift), drift-triggered retrain |
| A/B & Shadow | A/B compare avec significativité statistique, shadow compare enrichi |
| Retraining | Manuel, planifié cron, déclenché par drift, historique des retrains |
| Politiques | Auto-promotion, auto-demotion (circuit breaker), promotion policy CRUD |
| Golden tests | CRUD, CSV upload, run par version |
| Observed results | Upsert, CSV upload, export, stats |
| Prédictions | Historique filtré, export CSV, anomaly detection, purge RGPD |
| Utilisateurs | CRUD admin, quotas, usage 30j, rotation token |
| Infra | Health, health/dependencies, Prometheus metrics |

### Dashboard Streamlit — 9 pages

| Page | Contenu |
|---|---|
| 1 — Users | Gestion utilisateurs, quotas, création/suppression (admin) |
| 2 — Models | Liste, statut, déploiement, tags, suppression |
| 3 — Predictions | Historique filtré, anomalies, export |
| 4 — Stats | Agrégats, tendances, comparaison inter-modèles |
| 5 — Code Example | Exemples d'appels API (curl, Python) |
| 6 — AB Testing | Résultats A/B, shadow compare, test de significativité |
| 7 — Supervision | Monitoring global, drift, alertes, auto-demotion |
| 8 — Retrain | Planning cron, historique, déclenchement manuel, auto-promote |
| 9 — Golden Tests | CRUD cas de test, run par version, import CSV |

### Infrastructure complète

PostgreSQL · MinIO · Redis · MLflow · APScheduler · Prometheus · OpenTelemetry ·
Email SMTP · Webhooks · Docker Compose

### Qualité

70+ fichiers de tests couvrant : unitaire, intégration, E2E, smoke tests Docker.
Lint Ruff + formatage Black.

---

## Conclusion : le projet est terminé

Après douze itérations de roadmap, la plateforme couvre l'ensemble du cycle MLOps
pour son objectif initial — un serveur de modèles sklearn avec monitoring, retraining
automatisé et dashboard admin.

Il n'existe pas de fonctionnalité manquante qui toucherait 80 % des utilisateurs.
Les pistes suivantes ont été évaluées et délibérément écartées :

| Idée | Pourquoi non |
|---|---|
| Page Streamlit "Santé infra" (PostgreSQL, Redis…) | `/health/dependencies` existe. Un panneau Streamlit duplique l'info sans valeur opérationnelle supplémentaire pour un DS/MLOps qui a accès aux logs Docker. |
| Auto-reset de la baseline après retrain | La baseline est calculée sur des prédictions réelles passées. Juste après un retrain, la nouvelle version n'en a pas encore — un reset automatique n'aurait rien à calculer. |
| Streaming WebSocket / SSE | Niche (< 20 % des cas). Le modèle request/response couvre les besoins. |
| Feature importance historique | Stocker SHAP par prédiction est coûteux. Le trend de confidence et le drift de features adressent la même question. |
| Ensemble / multi-model fusion | Haute complexité, cas rare. Le shadow et l'A/B couvrent la comparaison. |
| Métriques de fairness / biais | Nécessite des attributs protégés et une définition métier. Hors scope généraliste. |
| Async batch scoring (job_id + polling) | `/predict-batch` synchrone couvre les volumes standards. Un système job async est une complexité infrastructure disproportionnée. |
| Multi-tenancy / feature store | Hors scope. |
| ONNX export | Dépendance lourde pour un cas edge/GPU rare. |

**Règle appliquée** : une proposition n'est pertinente que si elle résout une douleur
concrète sur votre déploiement. Aucune des idées ci-dessus ne satisfait ce critère
pour un projet de cette nature et taille.

---

> Le projet peut maintenant entrer en phase de maintenance : gestion des dépendances,
> mises à jour de sécurité, et évolutions guidées par des besoins opérationnels réels
> plutôt que par des fonctionnalités anticipées.
