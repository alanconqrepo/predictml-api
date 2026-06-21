# Auto-promotion — Fonctionnement et paramétrage

Ce document décrit le module d'auto-promotion de PredictML : comment il fonctionne, quel container l'exécute, comment le configurer et comment observer son activité.

---

## 1. Qu'est-ce que l'auto-promotion ?

L'auto-promotion est un mécanisme qui **promeut automatiquement une nouvelle version de modèle en production** après un ré-entraînement, à condition que des critères de qualité soient satisfaits — sans intervention humaine.

**Analogie** : après une mise à jour logicielle, des tests automatiques s'exécutent. Si tous passent, le déploiement se fait automatiquement. Sinon, la nouvelle version reste en attente.

**Ce qu'il fait concrètement :**
1. Un ré-entraînement (manuel ou planifié) produit une nouvelle version du modèle.
2. PredictML évalue si cette nouvelle version atteint les seuils configurés (accuracy, MAE, latence…).
3. Si **tous** les critères sont satisfaits → la nouvelle version passe automatiquement en `is_production=true`.
4. L'ancienne version de production est rétrogradée.

> ⚠️ L'auto-promotion évalue les **critères en logique ET** : tous doivent être satisfaits simultanément pour que la promotion ait lieu.

---

## 2. Quel container l'exécute ?

**Container : `retrain-worker`** — un seul réplica (pour éviter les exécutions en double).

```bash
# Démarrer le worker
docker-compose up -d retrain-worker

# Vérifier qu'il tourne
docker-compose ps retrain-worker

# Logs en temps réel
docker-compose logs -f retrain-worker
```

Ce container exécute un **worker ARQ** (queue de tâches asynchrone basée sur Redis) :
```
python -m arq src.tasks.arq_worker.WorkerSettings
```

C'est lui — et non l'API (`predictml-api`) — qui :
1. Exécute les scripts `train.py` en sous-processus isolé
2. Télécharge les nouveaux modèles dans MinIO
3. Crée la nouvelle version en base
4. Évalue la politique d'auto-promotion
5. Met à jour `is_production` si les critères sont satisfaits
6. Envoie les webhooks et emails de notification

---

## 3. Deux mécanismes distincts

### 3a. Auto-promotion post-retrain

| Déclencheur | Quand |
|---|---|
| Retrain **manuel** | `POST /models/{name}/{version}/retrain` |
| Retrain **planifié** | Cron APScheduler chargé depuis la DB |
| Retrain **sur drift** | Alert check détecte un drift critique |

**Condition d'évaluation :** `auto_promote: true` dans le `retrain_schedule` ET une `promotion_policy` définie sur le modèle.

**Fichiers concernés :**
- `src/services/retrain_service.py` — retrain manuel (via tâche ARQ)
- `src/tasks/retrain_scheduler.py` — retrain planifié (APScheduler)
- `src/services/auto_promotion_service.py` — logique d'évaluation (`evaluate_auto_promotion`)

### 3b. Circuit breaker (auto-demotion)

Le circuit breaker surveille en continu le modèle **en production** et le retire s'il se dégrade. C'est le mécanisme inverse de l'auto-promotion.

Voir **[CIRCUIT_BREAKER.md](./CIRCUIT_BREAKER.md)** pour la documentation complète.

---

## 4. Flux complet

```
Retrain déclenché
(manuel, planifié ou drift)
         │
         ▼
  Subprocess train.py
  (timeout 600 s)
         │
         ▼
  Métriques extraites du stdout
  (accuracy, f1_score, n_rows…)
         │
         ▼
  Nouveau ModelMetadata créé
  (is_production = false par défaut)
         │
         ▼
  auto_promote: true ?
  ET promotion_policy définie ?
         │
    ┌────┴─────┐
    │ OUI      │ NON
    ▼          ▼
evaluate_auto_promotion()   Fin (pas de promotion)
    │
    ├─ min_sample_validation atteint ?
    ├─ min_accuracy satisfaite ?
    ├─ max_mae satisfait ?
    ├─ max_latency_p95_ms satisfait ?
    └─ min_golden_test_pass_rate satisfait ?
         │
    ┌────┴─────────────────────┐
    │ TOUS satisfaits          │ Au moins un non satisfait
    ▼                          ▼
Promotion :               Rejet :
is_production = true      is_production = false
Old version → false       auto_promote_reason loggé
         │
    ┌────┴──────────────────────────────┐
    │ ✅ Webhook "model_promoted"        │
    │ ✅ Email (si ENABLE_EMAIL_ALERTS)  │
    │ ✅ MLflow tags mis à jour          │
    │ ✅ training_stats.auto_promoted    │
    └────────────────────────────────────┘
```

---

## 5. Critères de promotion

| Champ | Type | Description | Exemple |
|---|---|---|---|
| `min_accuracy` | float [0–1] | Accuracy minimale sur les paires de validation récentes | `0.90` |
| `max_mae` | float > 0 | MAE maximale (régression uniquement) | `5.0` |
| `max_latency_p95_ms` | float > 0 | P95 latence en ms | `200` |
| `min_sample_validation` | int ≥ 1 | Nombre minimum de paires (prédiction, résultat observé) | `50` |
| `min_golden_test_pass_rate` | float [0–1] | Taux de réussite des golden tests | `0.95` |
| `auto_promote` | bool | Active l'évaluation après chaque retrain | `true` |

**Règle :** Si un critère est à `null` (ou `0` dans l'interface), il est ignoré. Seuls les critères configurés sont évalués.

**Source des données de validation :** paires `(prediction, observed_result)` récentes en base — les résultats réels doivent avoir été enregistrés via `POST /observed-results`.

---

## 6. Configuration

### Via l'API

```bash
# Définir la politique d'auto-promotion
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

La politique s'applique à **toutes les versions actives** du modèle.

### Activer l'évaluation sur retrain planifié

```bash
curl -X PATCH http://localhost:8000/models/mon_modele/1.0.0/schedule \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "cron": "0 3 * * 1",
    "lookback_days": 30,
    "auto_promote": true,
    "enabled": true
  }'
```

> ℹ️ Sans `"auto_promote": true` dans le schedule, la politique est définie mais jamais évaluée lors du retrain planifié.

### Via Streamlit

**Deux accès disponibles :**

| Page | Section | Usage recommandé |
|---|---|---|
| **Modèles** → sélectionner un modèle → ⚙️ Politique de promotion | Formulaire complet (auto-promotion + circuit breaker) | Configuration initiale et avancée |
| **Supervision** → Zoom sur un modèle → ⚙️ Configuration → 🚀 Auto-promotion | Formulaire simplifié | Ajustements rapides en cours d'exploitation |

---

## 7. Notifications

### Webhook (toujours, si configuré)

Après chaque retrain, un webhook est envoyé à `webhook_url` du modèle :

```json
{
  "event": "retrain_completed",
  "model_name": "mon_modele",
  "version": "1.1.0-scheduled-20260620030000",
  "timestamp": "2026-06-20T03:00:00Z",
  "details": { "source_version": "1.0.0", "accuracy": 0.93, "f1_score": 0.91 }
}
```

Si la promotion est acceptée, un second webhook est envoyé :

```json
{
  "event": "model_promoted",
  "model_name": "mon_modele",
  "version": "1.1.0-scheduled-20260620030000",
  "timestamp": "2026-06-20T03:00:05Z",
  "details": { "reason": "All promotion criteria are satisfied." }
}
```

Configurer le webhook URL via `PATCH /models/{name}/{version}` :
```bash
curl -X PATCH http://localhost:8000/models/mon_modele/1.0.0 \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ"}'
```

### Email (si ENABLE_EMAIL_ALERTS=true)

Un email de confirmation est envoyé aux destinataires `ALERT_EMAIL_TO` :
- **Sujet :** `[PredictML] ✅ Auto-promotion — mon_modele v1.1.0`
- **Contenu :** modèle, nouvelle version, accuracy/f1, critères satisfaits, horodatage

---

## 8. Variables d'environnement

### Auto-promotion

Aucune variable spécifique — la politique est entièrement configurée en base via l'API.

### Notifications email

Ajouter dans `.env` (et redémarrer `retrain-worker`) :

```bash
# Active les emails d'alerte (auto-promotion + demotion + drift + erreurs)
ENABLE_EMAIL_ALERTS=true

# Serveur SMTP (exemple : Gmail avec mot de passe d'application)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_STARTTLS=true
SMTP_USER=votre-email@gmail.com
SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx  # Mot de passe d'application Gmail
SMTP_FROM=predictml-alerts@gmail.com

# Destinataires (séparés par des virgules)
ALERT_EMAIL_TO=admin@exemple.com,equipe-ml@exemple.com

# URL Streamlit (pour le lien dans les emails)
STREAMLIT_URL=http://localhost:8501
```

> ℹ️ Si `SMTP_HOST` n'est pas configuré, les méthodes email sont des no-ops silencieux — aucune erreur n'est levée.

---

## 9. Observabilité — Comment voir si une promotion a eu lieu ?

### 1. Logs du `retrain-worker`

```bash
docker-compose logs -f retrain-worker | grep "auto-promotion"
```

Entrées typiques :
```
INFO Auto-promotion evaluated model=mon_modele new_version=1.1.0 promoted=True reason="All promotion criteria are satisfied."
INFO Email sent subject="[PredictML] ✅ Auto-promotion — mon_modele v1.1.0"
```

### 2. Dashboard Streamlit

- **Page Retrain → Historique** : colonne `Auto-promotion` indique ✅/❌ et la raison
- **Page Modèles → sélectionner le modèle** : badge `🟢 Production` sur la nouvelle version

### 3. Champ `training_stats` de la version

```bash
curl http://localhost:8000/models/mon_modele/1.1.0 \
  -H "Authorization: Bearer <token>" | jq '.training_stats'
```

Réponse :
```json
{
  "auto_promoted": true,
  "auto_promote_reason": "All promotion criteria are satisfied.",
  "accuracy": 0.93,
  "n_rows": 12450
}
```

### 4. MLflow

Le run MLflow du retrain contient les tags :
- `auto_promoted` : `True` / `False`
- `auto_promote_reason` : raison
- `is_production` : `True` / `False`

### 5. Via l'API

```bash
# Vérifie quelle version est en production
curl http://localhost:8000/models/mon_modele \
  -H "Authorization: Bearer <token>" | jq '.[] | select(.is_production==true)'
```

---

## 10. Troubleshooting

### L'auto-promotion n'est jamais déclenchée

**Vérifications :**
1. Le schedule a-t-il `"auto_promote": true` ?
   ```bash
   curl http://localhost:8000/models/mon_modele/1.0.0 -H "Authorization: Bearer <token>" | jq '.retrain_schedule'
   ```
2. Une `promotion_policy` est-elle définie ?
   ```bash
   curl http://localhost:8000/models/mon_modele/1.0.0 -H "Authorization: Bearer <token>" | jq '.promotion_policy'
   ```
3. Le container `retrain-worker` tourne-t-il ?
   ```bash
   docker-compose ps retrain-worker
   ```

### La promotion est évaluée mais rejetée

Consulter la raison dans `training_stats.auto_promote_reason` (voir section 9.3) ou dans les logs :
```bash
docker-compose logs retrain-worker | grep "Auto-promotion evaluated"
```

Causes fréquentes :
- **Pas assez de paires de validation** : publier des résultats observés via `POST /observed-results`
- **Accuracy insuffisante** : baisser le seuil `min_accuracy` ou améliorer le script `train.py`
- **Latence trop élevée** : augmenter `max_latency_p95_ms` ou optimiser le modèle

### Aucun email reçu après une promotion

1. Vérifier `ENABLE_EMAIL_ALERTS=true` dans les variables du container `retrain-worker`
2. Vérifier que `SMTP_HOST` et `ALERT_EMAIL_TO` sont configurés
3. Chercher les erreurs SMTP dans les logs :
   ```bash
   docker-compose logs retrain-worker | grep -i "email\|smtp"
   ```

---

## 11. Relation avec le circuit breaker

L'auto-promotion et le circuit breaker sont **deux mécanismes complémentaires** :

| Mécanisme | Direction | Déclencheur | Fréquence |
|---|---|---|---|
| **Auto-promotion** | Non-production → Production | Fin de retrain | À chaque retrain |
| **Circuit breaker** | Production → Non-production | Dégradation des performances | Toutes les 6 h |

Ils opèrent indépendamment mais partagent le même champ `promotion_policy` en base.

→ Voir **[CIRCUIT_BREAKER.md](./CIRCUIT_BREAKER.md)** pour configurer la protection automatique contre la dégradation.
