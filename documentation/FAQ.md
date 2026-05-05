# FAQ — Questions Fréquentes sur PredictML

---

## Installation & démarrage

### Comment démarrer PredictML pour la première fois ?

```bash
git clone https://github.com/alanconqrepo/predictml-api.git
cd predictml-api
docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py
```

Vérifiez que tout fonctionne :
```bash
curl http://localhost:8000/health
# {"status": "healthy", "database": "connected", ...}
```

Le dashboard est sur http://localhost:8501, le token admin par défaut est `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA`.

---

### Les containers ne démarrent pas. Que faire ?

```bash
# Voir les logs en détail
docker-compose logs api
docker-compose logs postgres

# Vérifier que les ports ne sont pas déjà utilisés
netstat -tlnp | grep -E '8000|8501|5433|9000|5000'

# Rebuild complet
docker-compose down && docker-compose up -d --build
```

---

### Comment reconstruire le container Streamlit après avoir modifié une page ?

```bash
docker-compose up -d --build streamlit
```

---

## Authentification & Utilisateurs

### Comment obtenir mon token d'API ?

Le token admin par défaut est `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA`.

Pour créer un token pour un autre utilisateur :
```python
response = requests.post(
    "http://localhost:8000/users",
    headers={"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"},
    json={"username": "alice", "email": "alice@example.com", "role": "user", "rate_limit": 1000}
)
token = response.json()["api_token"]
```

---

### J'ai perdu mon token admin. Comment le retrouver ?

```bash
docker exec -it predictml-postgres psql -U postgres -d sklearn_api \
  -c "SELECT username, api_token FROM users WHERE role='admin';"
```

---

### Comment renouveler un token ?

```python
# Via l'API (admin requis)
requests.patch(
    "http://localhost:8000/users/2",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    json={"regenerate_token": True}
)
```

Via le dashboard : page **Utilisateurs** → bouton **Régénérer** sur l'utilisateur concerné.

---

### Pourquoi est-ce que j'obtiens HTTP 429 ?

Votre quota journalier de prédictions est épuisé. Vérifiez :
```python
r = requests.get("http://localhost:8000/users/me/quota",
                 headers={"Authorization": f"Bearer {TOKEN}"})
print(r.json())
# {"daily_limit": 1000, "used_today": 1000, "remaining": 0, "reset_at": "2026-01-16T00:00:00"}
```

Solution : augmenter le quota via `PATCH /users/{id}` (admin) ou attendre le reset à minuit UTC.

---

## Gestion des modèles

### Comment uploader mon premier modèle ?

```python
import pickle, requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Entraîner
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sauvegarder
with open("mon_modele.pkl", "wb") as f:
    pickle.dump(model, f)

# Uploader
with open("mon_modele.pkl", "rb") as f:
    r = requests.post(
        "http://localhost:8000/models",
        headers={"Authorization": "Bearer ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"},
        files={"file": ("mon_modele.pkl", f, "application/octet-stream")},
        data={
            "name": "mon_modele", "version": "1.0.0",
            "accuracy": str(accuracy_score(y_test, y_pred)),
            "f1_score": str(f1_score(y_test, y_pred, average="weighted")),
        }
    )
print(r.json())  # {"id": 1, "name": "mon_modele", "version": "1.0.0", ...}
```

---

### Comment passer un modèle en production ?

```python
requests.patch(
    "http://localhost:8000/models/mon_modele/1.0.0",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"is_production": True}
)
```

Un seul modèle par `name` peut être en production à la fois. Passer v2.0 en production retire automatiquement v1.0 du statut production.

---

### Quelle est la différence entre `is_active` et `is_production` ?

- `is_active: true` → le modèle existe et peut recevoir des prédictions (si explicitement ciblé ou en A/B/shadow)
- `is_production: true` → le modèle reçoit le trafic par défaut quand `POST /predict` ne précise pas de version

---

### Comment comparer deux versions d'un modèle ?

```python
# 1. Configurer l'A/B test
requests.patch("http://localhost:8000/models/mon_modele/2.0.0",
               headers={"Authorization": f"Bearer {TOKEN}"},
               json={"deployment_mode": "ab_test", "traffic_weight": 0.2})

# 2. Après accumulation de trafic, comparer
r = requests.get("http://localhost:8000/models/mon_modele/ab-compare",
                 headers={"Authorization": f"Bearer {TOKEN}"})
data = r.json()
sig = data.get("ab_significance")
if sig and sig["significant"]:
    print(f"Winner : {sig['winner']} (p={sig['p_value']:.4f})")
```

---

### Comment détecter la dérive de mes données ?

1. D'abord, configurez une baseline (stats des features à l'entraînement) :
```python
requests.patch(
    "http://localhost:8000/models/mon_modele/1.0.0",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"feature_baseline": {
        "age":    {"mean": 35.5, "std": 10.2, "min": 18, "max": 80},
        "income": {"mean": 45000, "std": 15000, "min": 10000, "max": 200000},
    }}
)
```

2. Après accumulation de prédictions en production, consultez le drift :
```python
r = requests.get("http://localhost:8000/models/mon_modele/drift",
                 headers={"Authorization": f"Bearer {TOKEN}"},
                 params={"days": 7})
```

3. Via le dashboard : page **Supervision** → section Drift.

---

## Prédictions

### Comment faire une prédiction ?

```python
r = requests.post(
    "http://localhost:8000/predict",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "model_name": "mon_modele",
        "id_obs":     "obs-001",
        "features":   {"age": 35, "income": 50000, "score": 720}
    }
)
print(r.json())
# {"prediction": 1, "probability": [0.12, 0.88], "low_confidence": false, ...}
```

---

### Qu'est-ce que `id_obs` ?

Un identifiant de votre choix pour lier la prédiction à un résultat observé ultérieur. Exemple : identifiant client, numéro de dossier, UUID de session.

Si vous n'en avez pas besoin, omettez-le (la valeur sera `null`).

---

### Comment enregistrer le résultat réel (feedback) ?

```python
requests.post(
    "http://localhost:8000/observed-results",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"data": [
        {"id_obs": "obs-001", "model_name": "mon_modele",
         "date_time": "2026-01-15T10:00:00", "observed_result": 1}
    ]}
)
```

Cela permet de calculer la performance réelle du modèle (accuracy, F1, matrice de confusion).

---

### Comment faire des prédictions en lot (batch) ?

```python
r = requests.post(
    "http://localhost:8000/predict-batch",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "model_name": "mon_modele",
        "inputs": [
            {"id_obs": "obs-001", "features": {"age": 35, "income": 50000}},
            {"id_obs": "obs-002", "features": {"age": 28, "income": 35000}},
            {"id_obs": "obs-003", "features": {"age": 52, "income": 85000}},
        ]
    }
)
for item in r.json()["predictions"]:
    print(f"{item['id_obs']} → {item['prediction']}")
```

Plus efficace que des appels unitaires : le modèle est chargé une seule fois.

---

### Comment obtenir une explication SHAP ?

```python
# Explication d'une seule prédiction
r = requests.post(
    "http://localhost:8000/explain",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={"model_name": "mon_modele", "features": {"age": 35, "income": 50000}}
)
for feat, val in r.json()["shap_values"].items():
    print(f"  {feat}: {val:+.4f}")

# SHAP inline dans /predict
r = requests.post(
    "http://localhost:8000/predict",
    headers={"Authorization": f"Bearer {TOKEN}"},
    params={"explain": "true"},
    json={"model_name": "mon_modele", "features": {"age": 35, "income": 50000}}
)
print(r.json()["explanation"]["shap_values"])
```

---

## Ré-entraînement

### Mon train.py est rejeté à l'upload. Pourquoi ?

L'API vérifie que votre script :
1. A une syntaxe Python valide
2. Lit `os.environ["TRAIN_START_DATE"]`
3. Lit `os.environ["TRAIN_END_DATE"]`
4. Lit `os.environ["OUTPUT_MODEL_PATH"]`
5. Appelle `pickle.dump()`, `joblib.dump()` ou `save_model()`

Vérifiez que ces 5 éléments sont bien présents.

---

### Le ré-entraînement échoue avec "No data for date range". Que faire ?

Votre script ne trouve pas de données pour la plage de dates demandée. Solutions :
1. Vérifiez votre source de données (CSV, BDD) et le filtrage sur la date
2. Testez manuellement : `TRAIN_START_DATE=2025-01-01 TRAIN_END_DATE=2025-12-31 OUTPUT_MODEL_PATH=/tmp/test.pkl python train.py`
3. Ajoutez un check dans votre script et terminez avec `sys.exit(1)` + message JSON d'erreur si aucune donnée

---

### Comment voir les logs du dernier ré-entraînement ?

Via l'API :
```python
r = requests.get("http://localhost:8000/models/mon_modele/retrain-history",
                 headers={"Authorization": f"Bearer {TOKEN}"})
for entry in r.json():
    print(f"{entry['trained_at']} v{entry['source_version']} → v{entry['new_version']}")
```

Via le dashboard : page **Retrain** → onglet **Historique**.

---

## Dashboard Streamlit

### Je ne vois pas tous les menus dans la barre latérale. Pourquoi ?

Certaines pages (notamment **Utilisateurs** et les actions admin dans d'autres pages) ne sont visibles qu'aux utilisateurs avec le rôle `admin`. Connectez-vous avec un token admin.

---

### Comment exporter mes prédictions ?

Page **Prédictions** → bouton **Exporter** → choisissez le format (CSV, JSONL, Parquet).

Ou via l'API :
```python
r = requests.get(
    "http://localhost:8000/predictions/export",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    params={"model_name": "mon_modele", "format": "csv"}
)
with open("export.csv", "wb") as f:
    f.write(r.content)
```

---

### Que fait le "What-If Explorer" sur la page Modèles ?

Il vous permet de modifier les valeurs des features avec des sliders et de voir la prédiction changer en temps réel, sans créer de prédiction loggée. Utile pour comprendre les zones de décision du modèle.

---

## Variables d'environnement importantes

| Variable | Description | Défaut |
|---|---|---|
| `ANTHROPIC_API_KEY` | Clé API Anthropic pour le chatbot d'aide | `` (désactivé) |
| `ADMIN_TOKEN` | Token admin personnalisé | `` (généré auto) |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | Credentials PostgreSQL | `postgres` / `postgres` |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | Credentials MinIO | `minioadmin` / `minioadmin` |
| `REDIS_CACHE_TTL` | TTL du cache des modèles en secondes | `3600` |
| `ENABLE_OTEL` | Activer les traces OpenTelemetry vers Grafana | `false` |
| `ENABLE_EMAIL_ALERTS` | Activer les alertes email | `false` |
| `ANTHROPIC_API_KEY` | Clé pour le chatbot Claude (page Aide) | `` |

Ces variables se définissent dans le fichier `.env` à la racine du projet :
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
ADMIN_TOKEN=mon-token-securise
```

---

## Codes d'erreur HTTP

| Code | Signification |
|---|---|
| 200 | Succès |
| 201 | Ressource créée (nouveau modèle, utilisateur) |
| 400 | Requête invalide (paramètre manquant ou incorrect) |
| 401 | Token absent ou invalide |
| 403 | Rôle insuffisant (admin requis) |
| 404 | Ressource introuvable (modèle, utilisateur) |
| 409 | Conflit (modèle nom+version déjà existant) |
| 410 | Gone — modèle déprécié (prédictions bloquées) |
| 422 | Erreur de validation Pydantic (schéma de features) |
| 429 | Quota journalier épuisé |
| 500 | Erreur serveur interne |
