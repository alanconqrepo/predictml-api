# Circuit Breaker — Fonctionnement et paramétrage

Ce document décrit le circuit breaker de PredictML : ce que c'est, quel container l'exécute, quand et pourquoi il se déclenche, comment le configurer, et comment l'observer.

---

## 1. Qu'est-ce que le circuit breaker ?

Le circuit breaker est un mécanisme de sécurité qui **retire automatiquement un modèle ML de la production** lorsque ses performances se dégradent — sans intervention humaine.

**Analogie** : comme un disjoncteur électrique qui coupe le courant quand il détecte une surcharge, le circuit breaker coupe le modèle de la production avant que de mauvaises prédictions n'atteignent les utilisateurs.

**Deux types de dégradation détectés** :
- **Drift de features** : la distribution des données d'entrée s'éloigne de ce que le modèle a appris (les utilisateurs envoient des valeurs inhabituelles)
- **Chute d'accuracy** : le modèle se trompe sur les dernières prédictions pour lesquelles on dispose du résultat réel

Quand le circuit breaker se déclenche :
1. La version en production passe en `is_production=false`
2. Un email d'alerte est envoyé (si SMTP configuré)
3. Un webhook est déclenché (si `webhook_url` configuré sur le modèle)
4. L'événement est tracé dans l'historique du modèle (`action: auto_demote`)

---

## 2. Quel container l'exécute ?

**Container : `retrain-worker`** (service Docker Compose).

```bash
# Démarrer le worker
docker-compose up -d retrain-worker

# Vérifier qu'il tourne
docker-compose ps retrain-worker

# Lire les logs en temps réel
docker-compose logs -f retrain-worker
```

Ce container exécute un **worker ARQ** :
```
python -m arq src.tasks.arq_worker.WorkerSettings
```

C'est lui — et non l'API (`predictml-api`) — qui :
1. Interroge PostgreSQL pour calculer les métriques (drift, accuracy…)
2. Évalue si le circuit breaker doit se déclencher
3. Modifie `is_production` en base
4. Envoie les emails et webhooks

Le worker nécessite accès à **PostgreSQL**, **Redis** et, si les emails sont activés, à un **serveur SMTP** externe.

---

## 3. Cadence d'exécution

| Job | Fréquence | Heures UTC |
|-----|-----------|------------|
| `alert_check_task` (inclut le circuit breaker) | Toutes les 6 h | 0 h, 6 h, 12 h, 18 h |

Le circuit breaker est évalué à chaque cycle de `alert_check_task`, défini dans `src/tasks/arq_worker.py`.

---

## 4. Flux d'exécution complet

```
Toutes les 6 h (0h, 6h, 12h, 18h UTC)
  │
  ▼
retrain-worker [Docker]
  │
  ▼
alert_check_task()          [src/tasks/arq_worker.py]
  │
  ▼
run_alert_check()           [src/tasks/supervision_reporter.py]
  │  Pour chaque modèle actif avec auto_demote=true :
  ▼
evaluate_auto_demotion()    [src/services/auto_promotion_service.py]
  │
  ├─ Vérifier qu'une version en production existe
  ├─ Vérifier qu'une version de secours (fallback) existe
  ├─ Vérifier que le cooldown est écoulé
  ├─ Calculer le drift de features (vs baseline) sur les 24 h passées
  ├─ Calculer le drift de sortie (label shift)
  ├─ Calculer l'accuracy sur les dernières paires (prédiction, résultat observé)
  │
  ├─ Si aucun critère déclenché → rien (log info)
  │
  └─ Si au moins un critère déclenché :
       ├─ model_metadata.is_production = False
       ├─ Enregistrer dans model_history (action=auto_demote, reason=...)
       ├─ email_service.send_auto_demotion_alert() [si ENABLE_EMAIL_ALERTS=true]
       └─ send_webhook(..., event_type="auto_demote") [si webhook_url configuré]
```

---

## 5. Conditions de déclenchement

Le circuit breaker se déclenche si **au moins un** des critères suivants est satisfait :

### 5.1 Drift de features

Requiert que le modèle ait un `feature_baseline` calculé (voir section 9).

La comparaison se fait sur les **24 dernières heures** de prédictions en production vs le baseline.

| Métrique | Seuil warning | Seuil critical |
|----------|---------------|----------------|
| Z-score | ≥ 2 | ≥ 3 |
| PSI | ≥ 0.1 | ≥ 0.2 |

Le paramètre `demote_on_drift` contrôle la sensibilité :
- `"warning"` → déclenche dès qu'une feature atteint le niveau warning
- `"critical"` → déclenche uniquement en cas de drift sévère (recommandé)

### 5.2 Drift de sortie (label shift)

PSI sur la distribution des labels prédits (toutes classes confondues). Mêmes seuils que le drift de features. Contrôlé par `demote_on_drift`.

### 5.3 Chute d'accuracy

Requiert que des résultats observés soient enregistrés via `POST /observed-results`.

- `demote_on_accuracy_below` : seuil d'accuracy minimale (ex : `0.80`)
- Évalué sur les `min_sample_validation` dernières paires (prédiction, résultat observé)
- Applicable aux **modèles de classification uniquement** ; pour la régression, utiliser `max_mae`

### 5.4 Guardrails (protections contre les faux positifs)

Le circuit breaker **ne se déclenche PAS** si :

| Condition | Comportement |
|-----------|-------------|
| Pas de version de secours (non-production, active, non-archivée) | Demotion bloquée, email d'avertissement envoyé |
| Cooldown actif (`demote_cooldown_hours` non écoulé depuis la dernière demotion) | Demotion ignorée |
| Pas de baseline calculé | Drift de features non évalué (les autres critères s'appliquent quand même) |

---

## 6. Notifications

### 6.1 Email

Envoyé si `ENABLE_EMAIL_ALERTS=true` ET SMTP configuré ET `ALERT_EMAIL_TO` non-vide.

**Sujet** : `[PredictML] 🔴 Auto-demotion — {modele} v{version}`

Le corps de l'email contient :
- Le nom du modèle et la version démise
- La/les raison(s) du déclenchement
- Un bouton "Open dashboard" vers le dashboard Streamlit

**Cas particulier (pas de fallback)** : si aucune version de secours n'existe, la demotion est bloquée mais un email d'avertissement est quand même envoyé : `"No fallback version available — the model has NOT been demoted. Manual action required."`

### 6.2 Webhook

Envoyé si `webhook_url` est configuré sur le modèle (indépendant de l'email).

```json
{
  "event_type": "auto_demote",
  "model_name": "mon_modele",
  "version": "1.2.0",
  "timestamp": "2026-06-20T06:00:00Z",
  "details": {
    "reason": "Feature drift critical detected. Accuracy 0.72 < 0.80."
  }
}
```

Pour configurer le webhook sur un modèle :

```bash
curl -X PATCH http://localhost:8000/models/mon_modele/1.2.0 \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ"}'
```

---

## 7. Guide de configuration

### 7.1 Via le dashboard Streamlit

**Depuis la page Modèles** :
1. Aller dans **Modèles** → sélectionner le modèle
2. Ouvrir le panneau **⚙️ Politique de promotion** (visible aux admins uniquement)
3. Le statut actuel est affiché en haut du panneau (🟢 actif / ⬜ inactif)
4. Cliquer sur **💡 Comment ça fonctionne ?** pour une explication détaillée
5. Dans la section **⚡ Circuit breaker** :
   - Cocher **Activer le circuit breaker**
   - Choisir le **Niveau de drift déclencheur** (Warning ou Critical)
   - Renseigner l'**Accuracy minimale** (0 = désactivé)
   - Ajuster le **Cooldown** (24 h recommandé)
6. Cliquer **💾 Sauvegarder la politique**

**Depuis la page Supervision** :
1. Aller dans **Supervision** → sélectionner un modèle en détail
2. Ouvrir **⚙️ Configuration** → onglet **⚡ Circuit breaker**
3. Le statut ACTIF/INACTIF et les paramètres actuels sont affichés en haut
4. Modifier les paramètres et cliquer **💾 Sauvegarder le circuit breaker**

### 7.2 Via l'API

```bash
curl -X POST http://localhost:8000/models/mon_modele/policy \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "auto_promote": false,
    "auto_demote": true,
    "demote_on_drift": "critical",
    "demote_on_accuracy_below": 0.80,
    "demote_cooldown_hours": 24,
    "min_sample_validation": 10
  }'
```

Pour **désactiver** le circuit breaker sans supprimer la configuration :

```bash
curl -X POST http://localhost:8000/models/mon_modele/policy \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"auto_demote": false}'
```

---

## 8. Référence des paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `auto_demote` | bool | `false` | **Interrupteur principal**. Doit être `true` pour activer le circuit breaker. |
| `demote_on_drift` | `"warning"` ou `"critical"` | `"critical"` | Niveau de drift minimum pour déclencher la demotion. `"critical"` = plus strict (recommandé pour débuter). |
| `demote_on_accuracy_below` | float [0–1] ou null | null (désactivé) | Démote si l'accuracy sur les paires récentes tombe sous ce seuil. Mettre `0` pour désactiver. |
| `demote_cooldown_hours` | int | 24 | Délai minimum (en heures) entre deux auto-demotions du même modèle. Évite les oscillations. |
| `min_sample_validation` | int | 10 | Nombre minimum de paires (prédiction, résultat observé) requises pour évaluer l'accuracy. |

Ces paramètres font partie de `promotion_policy` (JSON) stocké dans `model_metadata`. Ils s'appliquent à **toutes les versions actives** du modèle.

---

## 9. Prérequis : calculer un feature baseline

La détection de drift de features nécessite un **baseline** calculé à partir des prédictions de production passées.

Sans baseline, seuls les critères d'accuracy et de drift de sortie sont évalués.

**Via le dashboard** :
1. Modèles → sélectionner le modèle → **📐 Calculer le baseline depuis la production**
2. Choisir la fenêtre temporelle (7 à 180 jours)
3. Lancer en **dry-run** d'abord pour vérifier le résultat
4. Relancer sans dry-run pour sauvegarder

**Via l'API** :

```bash
# Simulation (dry_run=true par défaut)
curl -X POST "http://localhost:8000/models/mon_modele/1.0.0/baseline?days=30" \
  -H "Authorization: Bearer <admin_token>"

# Sauvegarder
curl -X POST "http://localhost:8000/models/mon_modele/1.0.0/baseline?days=30&dry_run=false" \
  -H "Authorization: Bearer <admin_token>"
```

---

## 10. Scénarios d'exemple

### Scénario A — Demotion par drift (cas nominal)

**Configuration** : `auto_demote=true`, `demote_on_drift="critical"`, `demote_on_accuracy_below=null`

**À 06:00 UTC** :
1. Le worker charge les statistiques des features sur les 24 h passées.
2. La feature `age` a un Z-score de 4.1 → drift `critical`.
3. Une version de secours v1.0.0 existe et est active.
4. Le cooldown (24 h) est écoulé.
5. `v1.1.0.is_production = false`.
6. Entrée dans `model_history` : `action=auto_demote`, `reason="Feature drift critical detected."`.
7. Email envoyé, webhook POST déclenché.

### Scénario B — Cooldown bloque la demotion

Une demotion s'est produite à 00:00 UTC. À 06:00 UTC, le drift est encore critique.

Le worker détecte que 6 h seulement se sont écoulées (cooldown = 24 h) → demotion ignorée. Aucun email, aucun webhook.

### Scénario C — Pas de version de secours

Le drift est critique, mais `v1.1.0` est la **seule** version active du modèle.

La demotion est bloquée (retirer la seule version de production laisserait le modèle sans aucun point d'entrée). Un email d'avertissement est envoyé si `ENABLE_EMAIL_ALERTS=true` : `"No fallback version available — the model has NOT been demoted."`. Aucun changement en base.

### Scénario D — Chute d'accuracy

**Configuration** : `auto_demote=true`, `demote_on_accuracy_below=0.85`, `min_sample_validation=50`

Après réception de 50+ résultats observés, l'accuracy sur les 50 dernières paires est de 0.78 < 0.85 → demotion déclenchée.

---

## 11. Dépannage / FAQ

**Q : Le circuit breaker est activé mais le modèle n'a jamais été démis malgré un drift visible.**

Vérifier dans l'ordre :
1. Le container `retrain-worker` tourne-t-il ? (`docker-compose ps`)
2. Le modèle a-t-il un `feature_baseline` ? (Modèles → 📐 Calculer le baseline)
3. Le `demote_on_drift` est-il au bon niveau par rapport au drift observé ?
4. Existe-t-il une version de secours non-production et active ?
5. Le cooldown est-il encore actif ? (Supervision → ⚙️ Configuration → ⚡ Circuit breaker — la bannière rouge montre le dernier événement)
6. Lire les logs du worker : `docker-compose logs -f retrain-worker`

---

**Q : Je reçois un email "no fallback available" mais rien ne change en production.**

C'est le comportement attendu. Le circuit breaker refuse de démette un modèle si aucune version de secours n'existe (cela rendrait le modèle complètement indisponible). Uploader ou réactiver une ancienne version avant que le circuit breaker puisse agir.

---

**Q : Le circuit breaker peut-il démette un modèle en mode A/B test ou shadow ?**

Non. Il agit uniquement sur les versions dont `is_production=true`.

---

**Q : Où voir l'historique des auto-demotions ?**

- **Onglet rapide** : Supervision → modèle → ⚙️ Configuration → ⚡ Circuit breaker (dernière event)
- **Historique complet** : Modèles → sélectionner le modèle → 📜 Historique des modifications (filtre `auto_demote`)

---

**Q : Comment tester le circuit breaker sans attendre 6 h ?**

Déclencher manuellement la tâche via l'API ARQ n'est pas exposé en UI. L'approche la plus simple est de baisser temporairement les seuils de déclenchement à des valeurs que les données actuelles dépassent déjà, puis d'attendre le prochain cycle (max 6 h).

Pour les tests automatisés, voir `tests/test_auto_promotion_policy.py` qui couvre `evaluate_auto_demotion()` avec des mocks.

---

## 12. Variables d'environnement liées

| Variable | Défaut | Description |
|----------|--------|-------------|
| `ENABLE_EMAIL_ALERTS` | `false` | Active l'envoi d'emails (inclus l'email auto-demotion) |
| `SMTP_HOST` | `""` | Serveur SMTP (vide = emails désactivés) |
| `SMTP_PORT` | `587` | Port SMTP |
| `SMTP_STARTTLS` | `true` | `true` = STARTTLS (port 587), `false` = SSL direct |
| `SMTP_USER` | `""` | Utilisateur SMTP |
| `SMTP_PASSWORD` | `""` | Mot de passe SMTP |
| `SMTP_FROM` | `""` | Adresse expéditeur |
| `ALERT_EMAIL_TO` | `""` | Destinataire(s), séparés par des virgules |
| `STREAMLIT_URL` | `http://localhost:8501` | URL dans les boutons des emails |

---

## 13. Fichiers clés

| Fichier | Rôle |
|---------|------|
| `src/services/auto_promotion_service.py` | Logique `evaluate_auto_demotion()` — critères, guardrails, demotion |
| `src/tasks/supervision_reporter.py` | Appelle `evaluate_auto_demotion()` dans `run_alert_check()` |
| `src/tasks/arq_worker.py` | Cron ARQ : `alert_check_task` toutes les 6 h |
| `src/services/email_service.py` | `send_auto_demotion_alert()` — template HTML email |
| `src/services/webhook_service.py` | Envoi webhook avec `event_type="auto_demote"` |
| `src/db/models/model_metadata.py` | Champ JSON `promotion_policy` (stocke la config circuit breaker) |
| `src/db/models/model_history.py` | `HistoryActionType.AUTO_DEMOTE` — audit trail |
| `src/schemas/model.py` | `ModelHistoryResponse` (champ `entries`, pas `history`) |
| `streamlit_app/pages/2_Models.py` | UI config : panneau "Politique de promotion" |
| `streamlit_app/pages/7_Supervision.py` | UI config : onglet "⚡ Circuit breaker" |
| `tests/test_auto_promotion_policy.py` | Tests unitaires `evaluate_auto_demotion()` |

---

## 14. Checklist de mise en service

- [ ] Calculer un feature baseline pour chaque modèle (`POST /models/{name}/{version}/baseline`)
- [ ] Uploader des résultats observés régulièrement (`POST /observed-results`) pour activer la détection d'accuracy
- [ ] S'assurer qu'au moins une version de secours est disponible (non-production, active)
- [ ] Configurer le circuit breaker via le dashboard ou l'API
- [ ] Définir `ENABLE_EMAIL_ALERTS=true` et configurer SMTP si des notifications email sont souhaitées
- [ ] Optionnel : configurer `webhook_url` sur les modèles critiques
- [ ] Vérifier que `retrain-worker` est démarré : `docker-compose up -d retrain-worker`
- [ ] Surveiller les logs au premier cycle : `docker-compose logs -f retrain-worker`
