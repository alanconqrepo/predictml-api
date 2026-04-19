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
- **Ré-entraînement automatique** — déclencher un retrain depuis l'API avec un script `train.py`
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
| Token admin API | `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA` |
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

### Explicabilité SHAP
- `POST /explain` : valeurs SHAP locales pour une observation
- `GET /models/{name}/feature-importance` : importance globale agrégée sur les N dernières prédictions
  - Paramètres : `version`, `last_n` (défaut 100, max 500), `days` (défaut 7)
  - Retourne `mean(|SHAP|)` par feature + classement — idéal pour détecter des dérives comportementales
- Support des modèles arborescents (TreeExplainer) et linéaires (LinearExplainer)
- Interprétation : contribution de chaque feature à la prédiction

### Dérive des données
- `GET /models/{name}/drift` : rapport de dérive par feature (Z-score + PSI)
- Statuts : `ok`, `warning`, `critical`, `no_baseline`, `insufficient_data`
- Basé sur la `feature_baseline` enregistrée à l'upload du modèle

### Performance réelle
- `GET /models/{name}/performance` : métriques calculées via les résultats observés
- Classification : accuracy, précision, rappel, F1, matrice de confusion, métriques par classe
- Régression : MAE, MSE, RMSE, R²
- Agrégation temporelle configurable

### Ré-entraînement automatique
- Upload d'un `train.py` à l'enregistrement du modèle
- `POST /models/{name}/{version}/retrain` : déclenche l'entraînement sur une plage de dates
- Logs complets stdout/stderr retournés dans la réponse
- Nouvelle version enregistrée automatiquement dans MinIO

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
| PATCH | `/models/{name}/policy` | Admin | Définir la politique d'auto-promotion post-retrain |
| GET | `/models/{name}/ab-compare` | Oui | Rapport de comparaison A/B |
| **Prédictions** | | | |
| POST | `/predict` | Oui | Prédiction unitaire |
| POST | `/predict-batch` | Oui | Prédictions en lot |
| POST | `/explain` | Oui | Explicabilité SHAP locale |
| GET | `/predictions` | Oui | Historique des prédictions (pagination curseur) |
| GET | `/predictions/stats` | Oui | Statistiques agrégées par modèle |
| **Résultats observés** | | | |
| POST | `/observed-results` | Oui | Enregistrer des résultats réels |
| GET | `/observed-results` | Oui | Consulter les résultats observés |
| **Utilisateurs** | | | |
| POST | `/users` | Admin | Créer un utilisateur |
| GET | `/users` | Admin | Lister tous les utilisateurs |
| GET | `/users/{id}` | Oui | Détail d'un utilisateur |
| PATCH | `/users/{id}` | Admin | Modifier rôle, statut, quota, token |
| DELETE | `/users/{id}` | Admin | Supprimer un utilisateur |
| **Monitoring** | | | |
| GET | `/monitoring/overview` | Oui | Tableau de bord global |
| GET | `/monitoring/model/{name}` | Oui | Détail monitoring d'un modèle |

---

## Exemple minimal

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"
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
│   ├── db_service.py           # Toutes les requêtes DB
│   ├── model_service.py        # Chargement, cache Redis, routage A/B/shadow
│   ├── minio_service.py        # Upload/download MinIO
│   ├── drift_service.py        # Calcul dérive Z-score + PSI
│   ├── shap_service.py         # Explications SHAP locales
│   ├── ab_significance_service.py  # Tests statistiques A/B (Chi-², Mann-Whitney U)
│   ├── email_service.py        # Alertes email & rapports hebdomadaires
│   └── webhook_service.py      # Webhooks HTTP post-prédiction
├── schemas/                # Schémas Pydantic (validation I/O)
└── main.py

streamlit_app/              # Dashboard admin multipage
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
