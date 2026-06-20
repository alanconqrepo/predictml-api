# Alerting — Fonctionnement et paramétrage

Ce document décrit le système d'alertes automatiques de PredictML : quel container l'exécute, quelles conditions le déclenchent, comment configurer les notifications et quelle est la structure des emails envoyés.

---

## 1. Quel container exécute les alertes ?

**Container : `retrain-worker`** (service Docker Compose du même nom).

```
docker-compose up -d retrain-worker
```

Ce container exécute un **worker ARQ** (`python -m arq src.tasks.arq_worker.WorkerSettings`). C'est lui — et non l'API (`predictml-api`) — qui :

1. Interroge PostgreSQL pour calculer les métriques (taux d'erreur, accuracy, drift…)
2. Évalue les seuils d'alerte configurés par modèle
3. Envoie les emails via SMTP
4. Déclenche les webhooks configurés sur chaque modèle

Le worker nécessite accès à **PostgreSQL**, **Redis** et, si les emails sont activés, à un **serveur SMTP** externe. Ces connexions sont configurées via les mêmes variables d'environnement que l'API (fichier `.env`).

### Cadence d'exécution

| Job | Fréquence | Heure(s) UTC |
|-----|-----------|--------------|
| `alert_check_task` | Toutes les 6 h | 0 h, 6 h, 12 h, 18 h |
| `weekly_report_task` | Hebdomadaire | Lundi 8 h |

---

## 2. Logique de déclenchement — OU (pas ET)

Chaque type d'alerte est **indépendant**. Un même modèle peut déclencher 0, 1, 2, 3 ou 4 alertes différentes lors d'un même cycle de 6 h. Il n'y a **aucun seuil global** à atteindre pour activer les autres.

### Types d'alertes et conditions

| Type | Condition de déclenchement | Source de données |
|------|---------------------------|-------------------|
| **Error spike** | Taux d'erreur sur 24 h ≥ `error_rate_max` | Table `predictions` |
| **Performance drift** | Accuracy récente < `accuracy_min` **OU** chute relative ≥ 10 % | Table `observed_results` croisée avec `predictions` |
| **AUC below threshold** | AUC enregistré < `auc_min` (binaire uniquement) | Champ `auc` de `model_metadata` |
| **Feature drift** | Z-score ≥ 3 ou PSI ≥ 0.2 sur une feature en prod | Statistiques de production vs baseline |
| **Output drift** | PSI ≥ 0.2 sur la distribution des labels prédits | Statistiques de production |

> **Seuils par défaut** (appliqués si aucun seuil modèle-spécifique n'est configuré) :
> - `error_rate_max` global : `ERROR_RATE_ALERT_THRESHOLD` (défaut `0.10` = 10 %)
> - `performance_drift` relatif : `PERFORMANCE_DRIFT_ALERT_THRESHOLD` (défaut `0.10` = 10 points)

### Priorité des seuils

```
Seuil modèle-spécifique (alert_thresholds dans DB)
    ↓ si absent
Seuil global (variable d'environnement)
    ↓ si absent
Valeur par défaut codée en dur
```

---

## 3. Configuration des seuils par modèle

Via le dashboard Streamlit → **Zoom sur un modèle → ⚙️ Configuration → 🔔 Seuils d'alerte**, ou via l'API :

```bash
curl -X PATCH http://localhost:8000/models/mon_modele/1.0.0 \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "alert_thresholds": {
      "error_rate_max": 0.05,
      "accuracy_min": 0.90,
      "auc_min": 0.80,
      "drift_auto_alert": true
    }
  }'
```

| Champ | Type | Description |
|-------|------|-------------|
| `error_rate_max` | float [0–1] | Seuil de taux d'erreur. `0.05` = alerte si > 5 % d'erreurs sur 24 h |
| `accuracy_min` | float [0–1] | Accuracy minimale attendue. Déclenche si la moyenne récente passe en-dessous |
| `auc_min` | float [0–1] | AUC minimal (classifieurs binaires uniquement). Comparé à l'AUC enregistré lors du dernier entraînement |
| `drift_auto_alert` | bool | `true` = envoie email + webhook si drift critique (feature ou output) |

Supprimer tous les seuils d'un modèle : envoyer `"alert_thresholds": null`.

---

## 4. Canaux de notification

### 4.1 Email (global, tous les modèles)

Les emails sont envoyés via **SMTP** depuis le container `retrain-worker`. Toutes les variables d'env sont à définir dans `.env` :

```env
# Activer les alertes email
ENABLE_EMAIL_ALERTS=true

# Serveur SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_STARTTLS=true          # true = STARTTLS (port 587) ; false = SSL direct (port 465)
SMTP_USER=alerts@company.com
SMTP_PASSWORD=<mot_de_passe_ou_app_password>
SMTP_FROM=alerts@predictml.io   # optionnel, utilise SMTP_USER si absent

# Destinataires (séparés par des virgules)
ALERT_EMAIL_TO=admin@company.com,ops@company.com

# Rapport hebdomadaire (optionnel)
WEEKLY_REPORT_ENABLED=true
WEEKLY_REPORT_DAY=monday    # lundi par défaut
WEEKLY_REPORT_HOUR=8        # 8 h UTC par défaut

# URL du dashboard dans les emails (bouton "Open dashboard")
STREAMLIT_URL=https://predictml.company.com
```

> **Langue des emails** : anglais uniquement (templates HTML en dur dans `src/services/email_service.py`).

**Condition d'envoi** : `ENABLE_EMAIL_ALERTS=true` **ET** `SMTP_HOST` non-vide **ET** `ALERT_EMAIL_TO` non-vide. Si l'une de ces trois conditions n'est pas remplie, l'email est ignoré silencieusement (log `DEBUG`).

### 4.2 Webhook (par modèle)

Indépendant du canal email — se déclenche même si `ENABLE_EMAIL_ALERTS=false`.

Configurer un webhook sur une version de modèle :

```bash
curl -X PATCH http://localhost:8000/models/mon_modele/1.0.0 \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ"}'
```

Le webhook reçoit un POST JSON de la forme :

```json
{
  "event_type": "error_rate_threshold",
  "model_name": "mon_modele",
  "version": "1.0.0",
  "timestamp": "2026-06-20T06:00:00Z",
  "details": {
    "error_rate": 0.12,
    "threshold": 0.05
  }
}
```

| `event_type` | Condition |
|---|---|
| `error_rate_threshold` | Taux d'erreur > seuil |
| `drift_critical` | Drift critique sur une feature |
| `output_drift_critical` | Drift critique sur la distribution des labels |
| `auc_below_threshold` | AUC < `auc_min` |

---

## 5. Contenu des emails

### Error spike

**Sujet** : `[PredictML] 🔴 Error spike — {modele} ({taux}%)`

| Champ | Valeur |
|-------|--------|
| Current error rate | Taux calculé sur les 24 dernières heures |
| Configured threshold | Seuil modèle (`error_rate_max`) ou seuil global |
| Detected on | Timestamp UTC de la détection |

### Performance drift

**Sujet** : `[PredictML] 🔴 Performance drift — {modele}`

| Champ | Valeur |
|-------|--------|
| Baseline accuracy | Accuracy moyenne sur la 1re moitié de la fenêtre 24 h |
| Recent accuracy | Accuracy moyenne sur la 2e moitié |
| Drop | Écart absolu |

> Pour les modèles de régression, la métrique utilisée est le MAE (inversé) au lieu de l'accuracy.

### AUC below threshold (binaire)

**Sujet** : `[PredictML] 🔴 AUC below threshold — {modele} ({auc} < {min})`

| Champ | Valeur |
|-------|--------|
| Current AUC | AUC enregistré lors du dernier entraînement |
| Minimum required | Valeur configurée dans `auc_min` |
| Gap | Écart entre les deux |

### Feature drift

**Sujet** : `[PredictML] ⚠️ Feature drift — {modele} / {feature}`

| Champ | Valeur |
|-------|--------|
| Status | `critical` |
| Z-score | Valeur calculée (critique ≥ 3) |
| PSI | Valeur calculée (critique ≥ 0.2) |

### Rapport hebdomadaire

**Sujet** : `[PredictML] Weekly report — {date_debut} to {date_fin}`

Tableau récapitulatif de tous les modèles actifs : nombre de prédictions, taux d'erreur, latence, statut de drift et statut de santé global.

---

## 6. Variables d'environnement globales

| Variable | Défaut | Description |
|----------|--------|-------------|
| `ENABLE_EMAIL_ALERTS` | `false` | Active l'envoi d'emails d'alerte |
| `SMTP_HOST` | `""` | Serveur SMTP (vide = emails désactivés) |
| `SMTP_PORT` | `587` | Port SMTP |
| `SMTP_STARTTLS` | `true` | `true` = STARTTLS, `false` = SSL direct |
| `SMTP_USER` | `""` | Utilisateur SMTP |
| `SMTP_PASSWORD` | `""` | Mot de passe SMTP |
| `SMTP_FROM` | `""` | Adresse expéditeur (utilise `SMTP_USER` si vide) |
| `ALERT_EMAIL_TO` | `""` | Destinataire(s), séparés par des virgules |
| `WEEKLY_REPORT_ENABLED` | `false` | Active le rapport hebdomadaire |
| `WEEKLY_REPORT_DAY` | `monday` | Jour du rapport (nom anglais) |
| `WEEKLY_REPORT_HOUR` | `8` | Heure UTC du rapport |
| `STREAMLIT_URL` | `http://localhost:8501` | URL du dashboard dans les boutons des emails |
| `ERROR_RATE_ALERT_THRESHOLD` | `0.10` | Seuil global de taux d'erreur (si pas de seuil modèle) |
| `PERFORMANCE_DRIFT_ALERT_THRESHOLD` | `0.10` | Chute relative d'accuracy déclenchant une alerte |

---

## 7. Checklist de mise en service

- [ ] Définir `ENABLE_EMAIL_ALERTS=true` dans `.env`
- [ ] Renseigner `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- [ ] Renseigner `ALERT_EMAIL_TO` (au moins une adresse)
- [ ] Vérifier que `retrain-worker` est démarré (`docker-compose up -d retrain-worker`)
- [ ] Configurer `webhook_url` sur les modèles critiques si nécessaire
- [ ] Configurer les seuils par modèle via le dashboard ou l'API
- [ ] Vérifier les logs du worker au premier cycle de 6 h : `docker-compose logs -f retrain-worker`

---

## 8. Implémentation — fichiers clés

| Fichier | Rôle |
|---------|------|
| `src/tasks/arq_worker.py` | Définition des cron jobs ARQ (`alert_check_task`, `weekly_report_task`) |
| `src/tasks/supervision_reporter.py` | Logique de vérification des métriques et déclenchement des alertes |
| `src/services/email_service.py` | Templates HTML et envoi SMTP |
| `src/services/webhook_service.py` | Envoi des payloads webhook |
| `src/core/config.py` | Lecture des variables d'environnement SMTP |
