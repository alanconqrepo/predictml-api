# Guide débutant — PredictML API

Ce guide est destiné à quelqu'un qui découvre PredictML API. Il explique le concept, l'architecture, et guide pas-à-pas à travers un workflow ML complet en Python.

---

## Qu'est-ce que PredictML API ?

PredictML API est une plateforme qui permet de **mettre un modèle Machine Learning en production en quelques minutes**.

Le problème qu'elle résout : vous avez entraîné un modèle scikit-learn sur votre ordinateur. Comment le rendre utilisable par une application web ? Comment le versionner, tracer chaque prédiction, mesurer sa précision réelle, et le mettre à jour ? PredictML API répond à tout cela.

**Ce que vous faites :**
1. Entraîner un modèle localement (`model.fit(...)`)
2. Le sauvegarder en `.pkl` (`pickle.dump(...)`)
3. L'uploader via l'API
4. Faire des prédictions via HTTP depuis n'importe quelle application

**Ce que l'API gère pour vous :**
- Stocker le modèle dans MinIO (stockage objet compatible S3)
- Versionner les modèles (v1.0, v2.0…)
- Logger chaque prédiction avec ses features, son résultat et sa latence
- Calculer les métriques réelles quand vous envoyez les résultats observés
- Détecter si les données de production dérivent par rapport aux données d'entraînement
- Expliquer pourquoi le modèle a prédit ce qu'il a prédit (SHAP)
- Tester silencieusement une nouvelle version avant de la mettre en production (shadow)

---

## Architecture expliquée simplement

```
Votre script Python  ──upload .pkl──▶  PredictML API  ──stocke──▶  MinIO
                                            │
                                            │  ──log──▶  PostgreSQL
                                            │
Votre application    ──POST /predict──▶  API  ──retourne──▶  { prediction: 0 }
```

L'API tourne dans Docker avec 7 services :
- **FastAPI** (port 8000) — l'API principale
- **PostgreSQL** (port 5433) — stocke les prédictions et métadonnées
- **MinIO** (port 9000) — stocke les fichiers `.pkl`
- **Redis** (port 6379) — cache les modèles en mémoire pour des prédictions rapides
- **MLflow** (port 5000) — experiment tracking optionnel
- **Streamlit** (port 8501) — dashboard d'administration
- **Grafana** (port 3000) — observabilité (logs, traces, métriques)

---

## Installation

```bash
# Prérequis : Git + Docker Desktop
git clone https://github.com/alanconqrepo/predictml-api.git
cd predictml-api

docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py

# Vérifier
curl http://localhost:8000/health
# {"status": "ok", "models_available": 2, "models_cached": 1}
```

Installez les dépendances Python pour les exemples ci-dessous :

```bash
pip install requests scikit-learn pandas numpy shap
```

---

## Tutoriel complet — Iris classifier

### Étape 1 : Entraîner et sauvegarder un modèle

```python
# 1_train.py
import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Charger le dataset Iris (150 fleurs, 4 features, 3 classes)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

X = df[iris.feature_names]
y = df["target"]

# Entraîner
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Sauvegarder
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Modèle sauvegardé dans iris_model.pkl")
```

```bash
python 1_train.py
# Accuracy : 1.0000
# F1 Score : 1.0000
# Modèle sauvegardé dans iris_model.pkl
```

### Étape 2 : Uploader le modèle

```python
# 2_upload.py
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

with open("iris_model.pkl", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/models",
        headers=HEADERS,
        files={"file": ("iris_model.pkl", f, "application/octet-stream")},
        data={
            "name": "iris_model",
            "version": "1.0.0",
            "description": "Classifieur Iris — RandomForest 100 arbres",
            "algorithm": "RandomForestClassifier",
            "accuracy": "1.0",
            "f1_score": "1.0",
            "features_count": "4",
            "classes": '["setosa", "versicolor", "virginica"]',
        },
    )

print(f"Status : {response.status_code}")   # 201
model = response.json()
print(f"ID : {model['id']}, version : {model['version']}")
```

### Étape 3 : Passer en production

```python
# 3_set_production.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

response = requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0",
    headers=HEADERS,
    json={"is_production": True}
)
print(f"En production : {response.json()['is_production']}")  # True
```

### Étape 4 : Faire une prédiction

```python
# 4_predict.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

# Une fleur à identifier
response = requests.post(
    f"{BASE_URL}/predict",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "id_obs": "obs-001",          # identifiant pour lier au résultat réel plus tard
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)

result = response.json()
classes = ["setosa", "versicolor", "virginica"]
print(f"Prédiction : {classes[result['prediction']]} (classe {result['prediction']})")
print(f"Probabilités : {result['probability']}")
# Prédiction : setosa (classe 0)
# Probabilités : [0.97, 0.02, 0.01]
```

### Étape 5 : Prédictions en lot (batch)

Plus efficace que des appels individuels : le modèle est chargé une seule fois.

```python
# 5_batch.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

observations = [
    {"id_obs": "obs-001", "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}},
    {"id_obs": "obs-002", "features": {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.3}},
    {"id_obs": "obs-003", "features": {"sepal length (cm)": 5.8, "sepal width (cm)": 2.7, "petal length (cm)": 4.1, "petal width (cm)": 1.0}},
]

response = requests.post(
    f"{BASE_URL}/predict-batch",
    headers=HEADERS,
    json={"model_name": "iris_model", "inputs": observations}
)

classes = ["setosa", "versicolor", "virginica"]
for item in response.json()["predictions"]:
    print(f"  {item['id_obs']} → {classes[item['prediction']]} "
          f"(conf: {max(item['probability']):.0%})")
# obs-001 → setosa (conf: 97%)
# obs-002 → virginica (conf: 94%)
# obs-003 → versicolor (conf: 71%)
```

### Étape 6 : Explication SHAP

Comprendre pourquoi le modèle a fait cette prédiction.

```python
# 6_explain.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

response = requests.post(
    f"{BASE_URL}/explain",
    headers=HEADERS,
    json={
        "model_name": "iris_model",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)

data = response.json()
print(f"Prédiction : {data['prediction']}")
print(f"Base value : {data['base_value']:.4f}")
print("\nContributions SHAP (+ = vers la classe prédite, - = à l'opposé) :")
for feat, val in sorted(data["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
    bar = "█" * int(abs(val) * 5) if abs(val) > 0.05 else "·"
    print(f"  {'↑' if val > 0 else '↓'} {feat:<25} {val:+.4f}  {bar}")
```

### Étape 7 : Enregistrer les résultats observés

Après avoir obtenu le vrai résultat, envoyez-le pour calculer la performance réelle.

```python
# 7_feedback.py
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

# Les vrais résultats (ce que vous avez constaté après la prédiction)
observed = [
    {"id_obs": "obs-001", "model_name": "iris_model",
     "date_time": datetime.now().isoformat(), "observed_result": 0},   # setosa ✓
    {"id_obs": "obs-002", "model_name": "iris_model",
     "date_time": datetime.now().isoformat(), "observed_result": 2},   # virginica ✓
    {"id_obs": "obs-003", "model_name": "iris_model",
     "date_time": datetime.now().isoformat(), "observed_result": 1},   # versicolor ✓
]

response = requests.post(
    f"{BASE_URL}/observed-results",
    headers=HEADERS,
    json={"data": observed}
)
print(response.json())  # {"upserted": 3}
```

### Étape 8 : Consulter la performance réelle

```python
# 8_performance.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

response = requests.get(
    f"{BASE_URL}/models/iris_model/performance",
    headers=HEADERS,
    params={"start": "2025-01-01T00:00:00", "end": "2025-12-31T23:59:59"}
)

data = response.json()
print(f"Prédictions total : {data['total_predictions']}")
print(f"Avec résultat réel : {data['matched_predictions']}")
print(f"Accuracy réelle : {data.get('accuracy', 'N/A')}")
print(f"F1 réel : {data.get('f1_weighted', 'N/A')}")
```

### Étape 9 : Détecter la dérive des données

Configurer d'abord une baseline (stats des features à l'entraînement) :

```python
# 9_drift.py
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

# 1. Définir la baseline (stats des features en entraînement)
requests.patch(
    f"{BASE_URL}/models/iris_model/1.0.0",
    headers=HEADERS,
    json={
        "feature_baseline": {
            "sepal length (cm)": {"mean": 5.84, "std": 0.83, "min": 4.3, "max": 7.9},
            "sepal width (cm)":  {"mean": 3.05, "std": 0.43, "min": 2.0, "max": 4.4},
            "petal length (cm)": {"mean": 3.76, "std": 1.76, "min": 1.0, "max": 6.9},
            "petal width (cm)":  {"mean": 1.20, "std": 0.76, "min": 0.1, "max": 2.5},
        }
    }
)

# 2. Consulter le rapport de dérive (après accumulation de prédictions)
response = requests.get(
    f"{BASE_URL}/models/iris_model/drift",
    headers=HEADERS,
    params={"days": 30}
)
data = response.json()
print(f"Résumé dérive : {data['drift_summary']}")
for feat, info in data["features"].items():
    status = info["drift_status"]
    z = info.get("z_score")
    print(f"  {feat:<25} {status:<15} z={z:.2f}" if z else f"  {feat:<25} {status}")
```

**Interprétation des statuts :**
- `ok` — aucune dérive détectée
- `warning` — dérive modérée, à surveiller
- `critical` — dérive forte, ré-entraînement probablement nécessaire
- `no_baseline` — vous n'avez pas encore configuré de baseline

---

## Fonctionnalités avancées

### A/B Testing

Tester une nouvelle version sur 20% du trafic sans risque :

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

# 1. Uploader la nouvelle version
with open("iris_model_v2.pkl", "rb") as f:
    requests.post(f"{BASE_URL}/models", headers=HEADERS,
                  files={"file": ("iris_model_v2.pkl", f, "application/octet-stream")},
                  data={"name": "iris_model", "version": "2.0.0"})

# 2. Configurer l'A/B test (20% du trafic vers v2)
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=HEADERS,
               json={"deployment_mode": "ab_test", "traffic_weight": 0.2})

# 3. Les prédictions normales seront routées automatiquement (80% v1 / 20% v2)
response = requests.post(f"{BASE_URL}/predict", headers=HEADERS,
                         json={"model_name": "iris_model",
                               "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                                            "petal length (cm)": 1.4, "petal width (cm)": 0.2}})
print(f"Version utilisée : {response.json()['selected_version']}")

# 4. Comparer les résultats après quelques jours
compare = requests.get(f"{BASE_URL}/models/iris_model/ab-compare",
                       headers=HEADERS, params={"days": 7})
data = compare.json()

for v in data["versions"]:
    print(f"  v{v['version']}: {v['total_predictions']} preds, erreur={v['error_rate']:.1%}")

# 5. Interpréter la significativité statistique
sig = data.get("ab_significance")
if sig:
    if sig["significant"]:
        print(f"\n✅ Différence statistiquement significative (p={sig['p_value']:.4f})")
        print(f"   Gagnant : {sig['winner']} — basé sur {sig['metric']}")
    else:
        print(f"\n⚠️  Différence NON significative (p={sig['p_value']:.4f})")
        print(f"   Il faut ~{sig['min_samples_needed']} observations/version pour conclure")
        print("   Ne promotez pas encore — accumulez plus de données")
```

### Shadow Deployment

Tester silencieusement une nouvelle version sans jamais l'exposer aux clients :

```python
# La v2 reçoit les mêmes inputs que la v1 en arrière-plan
# mais ses résultats ne sont jamais retournés aux clients
requests.patch(f"{BASE_URL}/models/iris_model/2.0.0", headers=HEADERS,
               json={"deployment_mode": "shadow"})

# Après accumulation, comparer les taux de concordance
compare = requests.get(f"{BASE_URL}/models/iris_model/ab-compare", headers=HEADERS)
for v in compare.json()["versions"]:
    if v["deployment_mode"] == "shadow":
        print(f"Concordance shadow vs prod : {v['agreement_rate']:.1%}")
```

### Seuil de confiance

Marquer les prédictions incertaines avec `low_confidence: true` :

```python
# Configurer un seuil de confiance à 80%
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=HEADERS,
               json={"confidence_threshold": 0.8})

# Les prédictions avec probabilité max < 80% auront low_confidence: true
result = requests.post(f"{BASE_URL}/predict", headers=HEADERS,
                       json={"model_name": "iris_model",
                             "features": {"sepal length (cm)": 5.8, "sepal width (cm)": 2.7,
                                          "petal length (cm)": 4.1, "petal width (cm)": 1.0}})
data = result.json()
if data.get("low_confidence"):
    print("⚠ Prédiction incertaine — révision manuelle recommandée")
```

### Ré-entraînement automatique

Si vous avez fourni un script `train.py` à l'upload, vous pouvez déclencher un ré-entraînement :

```python
response = requests.post(
    f"{BASE_URL}/models/iris_model/1.0.0/retrain",
    headers=HEADERS,
    json={
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "new_version": "1.1.0",
        "set_production": False
    }
)
data = response.json()
if data["success"]:
    print(f"Nouvelle version créée : {data['new_version']}")
    print(data["stdout"][-500:])   # derniers logs
else:
    print(f"Erreur : {data['error']}")
```

### Explication SHAP inline sur /predict

Obtenir les valeurs SHAP directement dans la réponse de prédiction sans appel supplémentaire :

```python
response = requests.post(
    f"{BASE_URL}/predict",
    headers=HEADERS,
    params={"explain": "true"},          # paramètre de requête
    json={
        "model_name": "iris_model",
        "features": {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    }
)
data = response.json()
print(f"Prédiction : {data['prediction']}")
# La réponse contient un champ "explanation" avec les valeurs SHAP
if data.get("explanation"):
    for feat, val in data["explanation"]["shap_values"].items():
        print(f"  {feat}: {val:+.4f}")
```

### Consulter une prédiction par son ID

Retrouver le détail d'une prédiction passée (features, résultat, latence) depuis son identifiant interne :

```python
prediction_id = 42   # ID retourné par /predict ou trouvé dans /predictions

response = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}",
    headers=HEADERS
)
data = response.json()
print(f"Modèle : {data['model_name']} v{data['model_version']}")
print(f"Résultat : {data['prediction_result']}")
print(f"Latence : {data['response_time_ms']} ms")

# Récupérer l'explication SHAP post-hoc (si le modèle supporte SHAP)
explain = requests.get(
    f"{BASE_URL}/predictions/{prediction_id}/explain",
    headers=HEADERS
)
print(explain.json()["shap_values"])
```

### Import CSV de résultats observés

Plutôt que d'envoyer les résultats un par un, importez-les en lot depuis un fichier CSV :

```python
import io
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

# Format attendu : colonnes id_obs, model_name, date_time, observed_result
csv_content = """id_obs,model_name,date_time,observed_result
obs-001,iris_model,2026-01-15T10:00:00,0
obs-002,iris_model,2026-01-15T10:01:00,2
obs-003,iris_model,2026-01-15T10:02:00,1
"""

response = requests.post(
    f"{BASE_URL}/observed-results/upload-csv",
    headers=HEADERS,
    files={"file": ("results.csv", io.StringIO(csv_content), "text/csv")}
)
print(response.json())  # {"upserted": 3, "errors": []}
```

Vous pouvez également exporter les résultats observés et vérifier la couverture :

```python
# Statistiques de couverture (combien de prédictions ont un résultat observé)
stats = requests.get(f"{BASE_URL}/observed-results/stats", headers=HEADERS,
                     params={"model_name": "iris_model"})
data = stats.json()
print(f"Prédictions avec ground truth : {data['matched_count']} / {data['total_predictions']}")
print(f"Couverture : {data['coverage_rate']:.1%}")

# Export CSV des résultats observés
export = requests.get(f"{BASE_URL}/observed-results/export", headers=HEADERS,
                      params={"model_name": "iris_model", "format": "csv"})
with open("observed_results_export.csv", "wb") as f:
    f.write(export.content)
```

### Webhooks

Recevoir une notification après chaque prédiction :

```python
requests.patch(f"{BASE_URL}/models/iris_model/1.0.0", headers=HEADERS,
               json={"webhook_url": "https://your-app.com/hooks/predictions"})
# L'API enverra un POST JSON après chaque prédiction sur ce modèle
```

---

## Dashboard Streamlit

Ouvrez **http://localhost:8501** et connectez-vous avec le token admin.

**Pages disponibles :**

| Page | Fonctionnalités |
|---|---|
| Accueil | Vue d'ensemble, liens vers les services |
| Utilisateurs | Créer/désactiver des comptes, renouveler les tokens |
| Modèles | Détails, passer en production, lien MLflow |
| Prédictions | Historique filtrable par modèle, date, version |
| Stats | Graphiques : volume, temps de réponse, distribution |
| Code Example | Exemples Python générés pour MLflow + API |
| A/B Testing | Shadow mode, comparaison statistique, décision de promotion |
| Supervision | Monitoring global, drift, alertes, tendances de performance |
| Retrain | Planifier, déclencher et suivre les ré-entraînements |

---

## Gestion des utilisateurs

```python
import requests

BASE_URL = "http://localhost:8000"
ADMIN_HEADERS = {"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"}

# Créer un utilisateur pour votre application
response = requests.post(
    f"{BASE_URL}/users",
    headers=ADMIN_HEADERS,
    json={
        "username": "mon_app",
        "email": "app@example.com",
        "role": "user",
        "rate_limit": 10000    # 10 000 prédictions/jour
    }
)
user = response.json()
app_token = user["api_token"]
print(f"Token de l'app : {app_token}")  # À stocker en sécurité

# Utiliser le token de l'app pour les prédictions
app_headers = {"Authorization": f"Bearer {app_token}"}
result = requests.post(f"{BASE_URL}/predict", headers=app_headers,
                       json={"model_name": "iris_model",
                             "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                                          "petal length (cm)": 1.4, "petal width (cm)": 0.2}})
```

**Rôles :**
- `admin` — accès complet (gestion utilisateurs, modèles, tout)
- `user` — peut faire des prédictions et voir ses propres données
- `readonly` — peut seulement lire (pas de prédiction)

---

## Récapitulatif des URL importantes

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Documentation interactive | http://localhost:8000/docs |
| Dashboard admin | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| MinIO (gestion fichiers) | http://localhost:9001 |
| Grafana (observabilité) | http://localhost:3000 |

---

## Prochaines étapes

- [QUICKSTART.md](QUICKSTART.md) — récapitulatif concis des commandes essentielles
- [API_REFERENCE.md](API_REFERENCE.md) — référence complète de tous les endpoints
- [ARCHITECTURE.md](ARCHITECTURE.md) — comment les services interagissent
- `http://localhost:8000/docs` — interface Swagger interactive pour tester l'API en direct
