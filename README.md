# PredictML API

API de déploiement et de prédiction Machine Learning, construite avec FastAPI.  
Conçue pour mettre en production des modèles scikit-learn, les versionner, suivre chaque prédiction, et mesurer la dérive des modèles dans le temps.

**Version 2.0** — Multi-utilisateurs, stockage distribué, experiment tracking, dashboard admin.

[![Tests](https://github.com/alanconqrepo/predictml-api/actions/workflows/tests.yml/badge.svg)](https://github.com/alanconqrepo/predictml-api/actions/workflows/tests.yml)

---

## Pourquoi ce projet ?

Entraîner un modèle ML est une chose. Le rendre utilisable en production, sécurisé, versionné et observable en est une autre.

PredictML API résout ce problème :

- **Déploiement en une commande** — uploader un `.pkl`, l'API est prête
- **Multi-modèles et multi-versions** — chaque modèle a son cycle de vie (actif, production, déprécié)
- **Traçabilité complète** — chaque prédiction est loguée avec ses features, son résultat, sa latence et l'utilisateur
- **Évaluation continue** — les résultats observés peuvent être rapportés pour mesurer la précision réelle des modèles
- **A/B testing & shadow deployment** — router le trafic entre versions et comparer silencieusement
- **Drift detection** — détecter la dérive des features en production (Z-score + PSI)
- **Explicabilité SHAP** — comprendre pourquoi le modèle a fait une prédiction, et mesurer l'importance globale des features sur les prédictions récentes
- **Ré-entraînement automatique** — déclencher un retrain depuis l'API avec un script `train.py`, ou le planifier automatiquement via une expression cron
- **Supervision & alertes** — monitoring global, alertes email, rapports hebdomadaires
- **Gestion multi-utilisateurs** — tokens Bearer, rôles (admin/user/readonly), quotas journaliers
- **Dashboard admin** — interface Streamlit pour piloter tout ça sans code

---

## Cible

| Profil | Usage |
|---|---|
| Data Scientist | Déployer un modèle `.pkl` sans écrire de code serveur |
| Développeur back-end | Consommer l'API dans une application |
| MLOps | Versionner, monitorer, comparer, ré-entraîner des modèles en production |
| Administrateur | Gérer les utilisateurs, quotas et accès via le dashboard |

---

## Stack technique

| Composant | Technologie | Port |
|---|---|---|
| API | FastAPI (async) | 8000 |
| Dashboard admin | Streamlit | 8501 |
| Base de données | PostgreSQL 16 | 5433 |
| Stockage modèles | MinIO (compatible S3) | 9000 / console 9001 |
| Experiment tracking | MLflow | 5000 |
| Cache distribué | Redis 7 | 6379 |
| Observabilité | Grafana LGTM (Loki + Tempo + Prometheus) | 3000 |

---

## Prérequis & Installation

```bash
# Prérequis : Git, Docker Desktop (avec Docker Compose v2)
git clone https://github.com/alanconqrepo/predictml-api.git
cd predictml-api
```

---

## Démarrage rapide

```bash
# 1. Lancer tous les services
docker-compose up -d --build

# 2. Initialiser la base de données et l'utilisateur admin (premier déploiement uniquement)
docker exec predictml-api python init_data/init_db.py

# 3. Accéder au dashboard admin
open http://localhost:8501

# 4. Tester l'API
curl http://localhost:8000/health
```

**Credentials par défaut**

| Service | Identifiants |
|---|---|
| Token admin API | `<ADMIN_TOKEN>` |
| PostgreSQL | `postgres / postgres` |
| MinIO | `minioadmin / minioadmin` |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin / admin) |

---

## Fonctionnalités principales

### Gestion des modèles
- Upload d'un fichier `.pkl` via API ou référence à un run MLflow
- Versionnage (`name` + `version`)
- Flag `is_production` pour router automatiquement les prédictions
- Métadonnées riches : algorithme, accuracy, f1_score, features, classes, dataset
- Tags personnalisés et webhooks de notification
- Seuil de confiance configurable (`confidence_threshold`)
- Baseline de features pour la détection de dérive

### Validation du schéma d'entrée
- `POST /models/{name}/{version}/validate-input` — valider les features avant de prédire (sans consommer de quota)
- Détecte les **features manquantes**, les **features inattendues**, et les **coercitions de type** (`string` → `float`)
- Source de vérité : `feature_names_in_` du modèle sklearn (priorité) ou `feature_baseline` enregistrée en DB
- `POST /predict?strict_validation=true` — mode strict : rejette avec 422 les features inattendues

### Prédictions
- `POST /predict` — prédiction unitaire avec routage intelligent (A/B, shadow)
- `POST /predict-batch` — prédictions en lot (modèle chargé une seule fois)
- Sortie des probabilités de classe si disponibles (`predict_proba`)
- Flag `low_confidence` si la probabilité max est sous le seuil configuré
- Identifiant `id_obs` pour lier une prédiction à un résultat observé
- Cache Redis distribué pour les instances de modèles

### A/B Testing & Shadow Deployment
- `deployment_mode` : `"production"`, `"ab_test"`, ou `"shadow"`
- `traffic_weight` : fraction du trafic routée vers une version (0.0 – 1.0)
- Mode shadow : prédictions exécutées en arrière-plan sans impact client
- `GET /models/{name}/ab-compare` : rapport de comparaison côte à côte avec **test de significativité statistique** automatique (Chi-² sur le taux d'erreur, Mann-Whitney U sur la latence) — inclut `p_value`, `winner`, et `min_samples_needed` pour éviter de promouvoir du bruit
- `GET /models/{name}/shadow-compare` : rapport enrichi comparant le modèle shadow et le modèle de production — accuracy comparée, delta de confiance, delta de latence, taux de désaccord et recommandation de promotion automatique

### Explicabilité SHAP
- `POST /explain` : valeurs SHAP locales pour une observation
- `GET /models/{name}/feature-importance` : importance globale agrégée sur les N dernières prédictions
  - Paramètres : `version`, `last_n` (défaut 100, max 500), `days` (défaut 7)
  - Retourne `mean(|SHAP|)` par feature + classement — idéal pour détecter des dérives comportementales
- Support des modèles arborescents (TreeExplainer) et linéaires (LinearExplainer)
- Interprétation : contribution de chaque feature à la prédiction

### Dérive des données (input + output)
- `GET /models/{name}/drift` : rapport de dérive par feature (Z-score + PSI + null rate)
- `GET /models/{name}/output-drift` : dérive de la distribution des sorties (**label shift**) via PSI — compare la distribution récente des prédictions à la distribution de référence issue de `training_stats`
- Statuts : `ok`, `warning`, `critical`, `no_baseline`, `insufficient_data`
- **5 dimensions de monitoring** : dérive de distribution par feature, dérive de performance (accuracy/MAE), dérive de taux d'erreur, null rate par feature, et label shift en sortie

### Seuils d'alerte par modèle

- `alert_thresholds` configurable via `PATCH /models/{name}/{version}` pour définir des seuils spécifiques à chaque modèle (dérive, taux d'erreur, null rate)
- Surcharge les seuils globaux définis par variables d'environnement
- Utilisé par le service de supervision pour déclencher des alertes ciblées

### Tendance de confiance & Lignée

- `GET /models/{name}/confidence-trend` : évolution de la confiance moyenne des prédictions dans le temps (par fenêtre temporelle)
- `parent_version` : chaque nouvelle version issue d'un retrain stocke la référence vers sa version source (traçabilité de lignée complète)
- `training_stats` : snapshot des données d'entraînement sauvegardé automatiquement à chaque retrain (n_rows, feature_stats, label_distribution)

### Préchauffage du cache

- `POST /models/{name}/{version}/warmup` : précharge un modèle en mémoire (cache Redis) sans attendre la première requête de prédiction
- Réduit la latence à froid lors des déploiements

### Performance réelle
- `GET /models/{name}/performance` : métriques calculées via les résultats observés
- Classification : accuracy, précision, rappel, F1, matrice de confusion, métriques par classe
- Régression : MAE, MSE, RMSE, R²
- Agrégation temporelle configurable

### Ré-entraînement automatique
- Upload d'un `train.py` à l'enregistrement du modèle
- `POST /models/{name}/{version}/retrain` : déclenche l'entraînement sur une plage de dates
- `PATCH /models/{name}/{version}/schedule` : planifie un retrain automatique via une expression cron (ex : `"0 3 * * 1"` = chaque lundi à 3h UTC)
- **Retrain déclenché par drift** : configurer `trigger_on_drift: "critical"` (ou `"warning"`) dans le schedule pour qu'un retrain se lance automatiquement dès que le drift détecté atteint le seuil — `drift_retrain_cooldown_hours` évite les boucles de retrain
- Logs complets stdout/stderr retournés dans la réponse
- Nouvelle version enregistrée automatiquement dans MinIO
- Verrou Redis pour éviter les exécutions simultanées en multi-réplicas

### Auto-promotion & Auto-demotion (circuit breaker)
- `PATCH /models/{name}/policy` : définir la politique d'auto-promotion post-retrain (`min_accuracy`, `max_latency_p95_ms`, `min_sample_validation`, `auto_promote`, `min_golden_test_pass_rate`)
- **Auto-demotion** : configurer `auto_demote: true` avec `demote_on_drift` (`"warning"` ou `"critical"`) et/ou `demote_on_accuracy_below` (seuil flottant) — si les critères sont franchis, le modèle de production est automatiquement ramené à un statut inactif par le superviseur (toutes les 6h)
- `demote_cooldown_hours` : délai minimal entre deux démotions automatiques

### Model Card
- `GET /models/{name}/{version}/card` : fiche récapitulative d'un modèle en un seul appel — métadonnées, métriques de performance réelle, état du drift, calibration, top-5 features SHAP, infos de retrain et couverture ground truth
- Accepte `Accept: text/markdown` pour retourner un fichier `.md` prêt à être partagé

### Tests de régression (Golden Tests)
- CRUD de cas de test par modèle : `POST /models/{name}/golden-tests`, `GET /models/{name}/golden-tests`, `DELETE /models/{name}/golden-tests/{id}`
- Import en lot depuis un CSV : `POST /models/{name}/golden-tests/upload-csv`
- `POST /models/{name}/{version}/run-golden-tests` : exécuter tous les cas de test sur une version — retourne PASS/FAIL par cas avec le diff attendu/reçu
- Intégré dans la politique d'auto-promotion via `min_golden_test_pass_rate`
- Interface complète dans la **page 9 du dashboard Streamlit**

### Détection d'anomalies
- `GET /predictions/anomalies` : prédictions récentes dont au moins une feature présente un z-score anormal par rapport à la baseline (`z_threshold` configurable, défaut 3.0)
- Paramètres : `model_name`, `days` (défaut 7), `z_threshold`, `limit`

### Leaderboard & Comparaison multi-modèles
- `GET /models/leaderboard` : classement des modèles en production par métrique (`accuracy`, `f1_score`, `latency_p95_ms`, `predictions_count`) sur une fenêtre configurable, résultat mis en cache (TTL)
- Visualisé dans la page Stats du dashboard : onglet Leaderboard + scatter plot accuracy vs latency P95

### Calibration des probabilités
- `GET /models/{name}/calibration` : mesure si les probabilités prédites sont fiables (Brier score, courbe de reliability)
- Un modèle avec `brier_score < 0.1` est bien calibré ; un gap de surconfiance positif signale que le modèle surestime sa certitude

### Distribution de confiance
- `GET /models/{name}/confidence-distribution` : histogramme du niveau de confiance (`max(probabilities)`) sur les prédictions récentes
- Permet d'identifier la proportion de prédictions incertaines avant d'affiner le seuil `confidence_threshold`

### Rapport de performance consolidé
- `GET /models/{name}/performance-report` : agrège en un seul appel performance réelle + drift + feature importance + calibration + A/B
- Idéal pour les scripts de monitoring automatique, les alertes programmatiques, ou les intégrations Grafana

### Dépréciation & Cycle de vie complet
- `PATCH /models/{name}/{version}/deprecate` : marque une version comme dépréciée — les nouvelles prédictions retournent HTTP 410 Gone
- `GET /models/{name}/readiness` : vérifie qu'un modèle satisfait tous les prérequis avant production (fichier MinIO accessible, baseline calculée, pas de drift critique)
- `GET /models/{name}/retrain-history` : journal structuré de tous les retrains (version source → nouvelle version, accuracy, auto_promoted, trained_by, fenêtre d'entraînement)

### Historique & Rollback
- `GET /models/{name}/history` : journal de tous les changements
- `POST /models/{name}/{version}/rollback/{history_id}` : restaurer un état précédent

### Supervision & Alertes
- `GET /monitoring/overview` : tableau de bord global (toutes les erreurs, dérives, performances)
- `GET /monitoring/model/{name}` : détail par modèle (timeseries, A/B, drift, erreurs récentes)
- `GET /predictions/stats` : statistiques agrégées (volume, taux d'erreur, temps de réponse p50/p95)
- Alertes email configurables (dérive, taux d'erreur)
- Rapports hebdomadaires automatiques
- `GET /metrics` — endpoint Prometheus (scraping automatique par Grafana LGTM, auth optionnelle via `METRICS_TOKEN`)
- Traces OpenTelemetry vers Grafana LGTM (optionnel)

### Gestion des utilisateurs
- Création par un admin avec rôle et quota journalier
- Token Bearer unique par utilisateur
- Renouvellement de token via `PATCH /users/{id}` avec `{"regenerate_token": true}`
- Rate limiting automatique (HTTP 429 si quota dépassé)

---

## Endpoints API

| Méthode | Route | Auth | Description |
|---|---|---|---|
| GET | `/` | Non | Statut de l'API et modèles disponibles |
| GET | `/health` | Non | Health check (DB + cache Redis) |
| **Modèles** | | | |
| GET | `/models` | Non | Liste des modèles actifs (filtre par tag) |
| GET | `/models/cached` | Non | Modèles chargés en mémoire |
| GET | `/models/{name}/{version}` | Non | Détail complet d'un modèle |
| POST | `/models` | Oui | Uploader un modèle (.pkl ou MLflow) |
| PATCH | `/models/{name}/{version}` | Oui | Mettre à jour (production, A/B, tags, webhook…) |
| DELETE | `/models/{name}/{version}` | Oui | Supprimer une version |
| DELETE | `/models/{name}` | Oui | Supprimer toutes les versions |
| GET | `/models/{name}/performance` | Oui | Métriques réelles via résultats observés |
| GET | `/models/{name}/drift` | Oui | Rapport de dérive des features |
| GET | `/models/{name}/feature-importance` | Oui | Importance globale SHAP agrégée |
| GET | `/models/{name}/history` | Oui | Historique complet des changements |
| GET | `/models/{name}/{version}/history` | Oui | Historique d'une version spécifique |
| POST | `/models/{name}/{version}/rollback/{history_id}` | Admin | Rollback vers un état précédent |
| POST | `/models/{name}/{version}/retrain` | Admin | Ré-entraîner avec train.py |
| PATCH | `/models/{name}/{version}/schedule` | Admin | Configurer le planning cron de ré-entraînement |
| PATCH | `/models/{name}/policy` | Admin | Définir la politique d'auto-promotion post-retrain |
| GET | `/models/leaderboard` | Non | Classement des modèles en production par métrique |
| GET | `/models/{name}/performance-timeline` | Oui | Timeline de performance par version déployée |
| GET | `/models/{name}/calibration` | Oui | Calibration des probabilités (Brier score, reliability diagram) |
| GET | `/models/{name}/confidence-distribution` | Oui | Distribution de confiance (histogramme par bins) |
| GET | `/models/{name}/performance-report` | Oui | Rapport consolidé : performance + drift + SHAP + calibration |
| GET | `/models/{name}/readiness` | Oui | Vérification de disponibilité avant passage en production |
| GET | `/models/{name}/retrain-history` | Oui | Historique des événements de ré-entraînement |
| PATCH | `/models/{name}/{version}/deprecate` | Admin | Déprécier une version (bloque les prédictions avec HTTP 410) |
| POST | `/models/{name}/{version}/validate-input` | Oui | Valider le schéma de features sans consommer de quota |
| GET | `/models/{name}/{version}/download` | Oui | Télécharger le fichier .pkl depuis MinIO |
| GET | `/models/{name}/ab-compare` | Oui | Rapport de comparaison A/B avec significativité statistique |
| GET | `/models/{name}/shadow-compare` | Oui | Rapport enrichi shadow vs production (accuracy, latence, désaccord) |
| GET | `/models/{name}/output-drift` | Oui | Drift de distribution des sorties (label shift via PSI) |
| GET | `/models/{name}/{version}/card` | Oui | Model card consolidée (JSON ou Markdown) |
| GET | `/models/{name}/confidence-trend` | Oui | Tendance de confiance dans le temps |
| POST | `/models/{name}/{version}/warmup` | Oui | Préchauffer le modèle dans le cache Redis |
| GET | `/models/{name}/golden-tests` | Oui | Liste des cas de test golden |
| POST | `/models/{name}/golden-tests` | Oui | Créer un cas de test golden |
| DELETE | `/models/{name}/golden-tests/{id}` | Admin | Supprimer un cas de test golden |
| POST | `/models/{name}/golden-tests/upload-csv` | Admin | Import en lot de cas de test depuis CSV |
| POST | `/models/{name}/{version}/run-golden-tests` | Oui | Exécuter les golden tests sur une version |
| **Prédictions** | | | |
| POST | `/predict` | Oui | Prédiction unitaire (`?explain=true` pour SHAP inline) |
| POST | `/predict-batch` | Oui | Prédictions en lot |
| POST | `/explain` | Oui | Explicabilité SHAP locale |
| GET | `/predictions` | Oui | Historique des prédictions (pagination curseur) |
| GET | `/predictions/{id}` | Oui | Consulter une prédiction par son ID |
| GET | `/predictions/{id}/explain` | Oui | Explication SHAP post-hoc d'une prédiction existante |
| GET | `/predictions/stats` | Oui | Statistiques agrégées par modèle |
| GET | `/predictions/anomalies` | Oui | Prédictions avec features aberrantes (z-score vs baseline) |
| GET | `/predictions/export` | Admin | Export streaming CSV / JSONL / Parquet |
| DELETE | `/predictions/purge` | Admin | Purge RGPD des prédictions anciennes (`dry_run` par défaut) |
| **Résultats observés** | | | |
| POST | `/observed-results` | Oui | Enregistrer des résultats réels |
| GET | `/observed-results` | Oui | Consulter les résultats observés |
| GET | `/observed-results/export` | Oui | Exporter les résultats observés (CSV/JSON) |
| GET | `/observed-results/stats` | Oui | Statistiques de couverture ground truth |
| POST | `/observed-results/upload-csv` | Oui | Import en lot depuis un fichier CSV |
| **Utilisateurs** | | | |
| GET | `/users/me` | Oui | Profil de l'utilisateur courant |
| GET | `/users/me/quota` | Oui | Quota journalier (utilisé, restant, heure de reset) |
| GET | `/users/{id}/usage` | Oui | Statistiques d'utilisation par modèle et par jour |
| POST | `/users` | Admin | Créer un utilisateur |
| GET | `/users` | Admin | Lister tous les utilisateurs |
| GET | `/users/{id}` | Oui | Détail d'un utilisateur |
| PATCH | `/users/{id}` | Admin | Modifier rôle, statut, quota, token |
| DELETE | `/users/{id}` | Admin | Supprimer un utilisateur |
| **Monitoring** | | | |
| GET | `/monitoring/overview` | Oui | Tableau de bord global |
| GET | `/monitoring/model/{name}` | Oui | Détail monitoring d'un modèle |
| GET | `/health/dependencies` | Non | Santé détaillée de chaque dépendance (DB, Redis, MinIO, MLflow) |
| GET | `/metrics` | Optionnel | Métriques Prometheus (scraped par Grafana LGTM) |

---

## Exemple minimal

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "<ADMIN_TOKEN>"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Prédiction unitaire
response = requests.post(
    f"{BASE_URL}/predict",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-001",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
print(response.json())
# {"model_name": "iris_model", "model_version": "1.0", "prediction": 0,
#  "probability": [0.97, 0.02, 0.01], "low_confidence": false}

# Explicabilité SHAP
explain = requests.post(
    f"{BASE_URL}/explain",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                     "petal length (cm)": 1.4, "petal width (cm)": 0.2}
    }
)
print(explain.json()["shap_values"])
# {"petal length (cm)": -1.32, "petal width (cm)": -0.87, ...}
```

---

## Tests

```bash
# Tests automatisés (sans Docker — utilise TestClient FastAPI)
pytest tests/ -v

# Tests d'un fichier spécifique
pytest tests/test_api.py -v

# Smoke tests (nécessitent Docker démarré)
python smoke-tests/test_multimodel_api.py
```

---

## Documentation complète

| Document | Contenu |
|---|---|
| [documentation/BEGINNER_GUIDE.md](documentation/BEGINNER_GUIDE.md) | Guide complet pour débutant — tutoriel pas-à-pas avec Python |
| [documentation/QUICKSTART.md](documentation/QUICKSTART.md) | Guide de démarrage et workflow complet |
| [documentation/API_REFERENCE.md](documentation/API_REFERENCE.md) | Référence complète de tous les endpoints, schémas, exemples Python |
| [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) | Structure du projet, services et flux de données |
| [documentation/DOCKER.md](documentation/DOCKER.md) | Commandes Docker, services, variables d'environnement |
| [documentation/DATABASE.md](documentation/DATABASE.md) | Schéma SQL, requêtes utiles, connexion Python |

---

## Structure du projet

```
src/
├── api/                    # Endpoints FastAPI
│   ├── models.py           # CRUD modèles + drift + history + retrain + A/B
│   ├── predict.py          # Prédictions unitaires, batch, SHAP, stats
│   ├── users.py            # Gestion utilisateurs
│   ├── observed_results.py # Résultats observés
│   └── monitoring.py       # Tableau de bord global et par modèle
├── core/                   # Config, auth, télémétrie
│   ├── config.py
│   ├── security.py
│   └── telemetry.py
├── db/                     # ORM SQLAlchemy
│   ├── database.py
│   └── models/             # User, Prediction, ModelMetadata, ObservedResult, ModelHistory
├── services/               # Logique métier
│   ├── db_service.py               # Toutes les requêtes DB
│   ├── model_service.py            # Chargement, cache Redis, routage A/B/shadow
│   ├── minio_service.py            # Upload/download MinIO
│   ├── drift_service.py            # Calcul dérive Z-score + PSI + null rate (input + output)
│   ├── shap_service.py             # Explications SHAP locales
│   ├── ab_significance_service.py  # Tests statistiques A/B (Chi-², Mann-Whitney U)
│   ├── auto_promotion_service.py   # Auto-promotion + auto-demotion (circuit breaker)
│   ├── golden_test_service.py      # Tests de régression golden (CRUD + run)
│   ├── input_validation_service.py # Validation schéma de features d'entrée
│   ├── supervision_reporter.py     # Supervision 6h : drift, alertes, retrain réactif
│   ├── email_service.py            # Alertes email & rapports hebdomadaires
│   └── webhook_service.py          # Webhooks HTTP post-prédiction
├── schemas/                # Schémas Pydantic (validation I/O)
└── main.py

streamlit_app/              # Dashboard admin multipage (9 pages : Users, Models, Predictions, Stats, Code, A/B, Supervision, Retrain, Golden Tests)
tests/                      # Tests automatisés (pytest)
smoke-tests/                # Tests manuels contre Docker live
init_data/                  # Scripts d'initialisation one-shot
alembic/                    # Migrations base de données
```

---

## Qualité de code

```bash
# Lint
ruff check src/

# Formatage
black --check src/

# Correction automatique
ruff check src/ --fix && black src/
```
