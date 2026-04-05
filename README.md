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
- **Traçabilité complète** — chaque prédiction est loguée avec ses features, son résultat, son latence et l'utilisateur
- **Évaluation continue** — les résultats observés peuvent être rapportés pour mesurer la précision réelle des modèles
- **Gestion multi-utilisateurs** — tokens Bearer, rôles (admin/user/readonly), quotas journaliers
- **Dashboard admin** — interface Streamlit pour piloter tout ça sans code

---

## Cible

| Profil | Usage |
|---|---|
| Data Scientist | Déployer un modèle `.pkl` sans écrire de code serveur |
| Développeur back-end | Consommer l'API dans une application |
| MLOps | Versionner, monitorer et comparer des modèles en production |
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

---

## Démarrage rapide

```bash
# 1. Lancer tous les services
docker-compose up -d --build

# 2. Initialiser la base de données et l'utilisateur admin (premier déploiement)
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

---

## Fonctionnalités principales

### Gestion des modèles
- Upload d'un fichier `.pkl` via API ou référence à un run MLflow
- Versionnage (`name` + `version`)
- Flag `is_production` pour router automatiquement les prédictions
- Métadonnées riches : algorithme, accuracy, f1_score, features, classes, dataset

### Prédictions
- Endpoint unique `POST /predict` pour tous les modèles
- Sélection automatique de la version de production si `model_version` non fourni
- Sortie des probabilités de classe si disponibles (`predict_proba`)
- Identifiant `id_obs` pour lier une prédiction à un résultat observé ultérieur

### Traçabilité et évaluation
- Toutes les prédictions sont stockées en base avec features, résultat, latence, IP, user-agent
- Les résultats réels peuvent être soumis via `POST /observed-results`
- Jointure possible via `id_obs` pour calculer les métriques réelles

### Gestion des utilisateurs
- Création par un admin avec rôle et quota journalier
- Token Bearer unique par utilisateur
- Renouvellement de token via `PATCH /users/{id}` avec `regenerate_token: true`
- Rate limiting automatique (HTTP 429 si quota dépassé)

---

## Endpoints API

| Méthode | Route | Auth | Description |
|---|---|---|---|
| GET | `/` | Non | Statut de l'API et modèles disponibles |
| GET | `/health` | Non | Health check (DB + cache modèles) |
| GET | `/models` | Non | Liste des modèles actifs |
| GET | `/models/cached` | Non | Modèles chargés en mémoire |
| GET | `/models/{name}/{version}` | Non | Détail complet d'un modèle |
| POST | `/models` | Oui | Uploader un modèle |
| PATCH | `/models/{name}/{version}` | Oui | Mettre à jour (production, métriques…) |
| DELETE | `/models/{name}/{version}` | Oui | Supprimer une version |
| DELETE | `/models/{name}` | Oui | Supprimer toutes les versions |
| POST | `/predict` | Oui | Faire une prédiction |
| GET | `/predictions` | Oui | Historique des prédictions avec filtres |
| POST | `/users` | Admin | Créer un utilisateur |
| GET | `/users` | Admin | Lister tous les utilisateurs |
| GET | `/users/{id}` | Oui | Détail d'un utilisateur |
| PATCH | `/users/{id}` | Admin | Modifier rôle, statut, quota, token |
| DELETE | `/users/{id}` | Admin | Supprimer un utilisateur |
| POST | `/observed-results` | Oui | Enregistrer des résultats observés |
| GET | `/observed-results` | Oui | Consulter les résultats observés |

---

## Exemple minimal

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Faire une prédiction
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
# {"model_name": "iris_model", "model_version": "1.0", "prediction": 0, "probability": [0.97, 0.02, 0.01]}
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
| [documentation/API_REFERENCE.md](documentation/API_REFERENCE.md) | Référence complète de tous les endpoints, schémas, exemples Python |
| [documentation/DATABASE.md](documentation/DATABASE.md) | Schéma SQL, requêtes utiles, connexion Python |
| [documentation/QUICKSTART.md](documentation/QUICKSTART.md) | Guide de démarrage et cas d'usage courants |
| [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) | Structure du projet et flux de données |
| [documentation/DOCKER.md](documentation/DOCKER.md) | Commandes Docker et dépannage |

---

## Structure du projet

```
src/
├── api/                    # Endpoints FastAPI
│   ├── models.py           # CRUD modèles
│   ├── predict.py          # Prédictions
│   ├── users.py            # Gestion utilisateurs
│   └── observed_results.py # Résultats observés
├── core/                   # Config, auth, télémétrie
│   ├── config.py
│   ├── security.py
│   └── telemetry.py
├── db/                     # ORM SQLAlchemy
│   ├── database.py
│   └── models/
├── services/               # Logique métier
│   ├── db_service.py
│   ├── model_service.py
│   └── minio_service.py
├── schemas/                # Schémas Pydantic
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
