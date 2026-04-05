"""
Exemple de code complet : MLflow + API
"""
import os
import streamlit as st
from utils.auth import require_auth

st.set_page_config(page_title="Code Example — PredictML", page_icon="💡", layout="wide")
require_auth()

st.title("💡 Exemple de code : MLflow + API")
st.markdown("Exemple complet pour entraîner un modèle, le tracker avec MLflow, puis l'uploader et l'utiliser via l'API.")

API_URL = st.session_state.get("api_url", "http://localhost:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")

# ============================================================
# SECTION 1 — Entraîner et tracker avec MLflow
# ============================================================
st.subheader("1. Entraîner un modèle et le tracker avec MLflow")
st.markdown("Installez les dépendances : `pip install scikit-learn mlflow boto3`")

code_mlflow = f'''\
import mlflow
import mlflow.sklearn
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ── Configuration ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = "{MLFLOW_URL}"
EXPERIMENT_NAME = "iris_classification"
MODEL_NAME = "iris_model"
MODEL_VERSION = "1.0.0"

# ── Setup MLflow ────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Données ─────────────────────────────────────────────────────
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# ── Entraînement avec tracking MLflow ──────────────────────────
with mlflow.start_run() as run:
    params = {{"n_estimators": 100, "max_depth": 5, "random_state": 42}}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id
    print(f"Run ID : {{run_id}}")
    print(f"Accuracy : {{acc:.4f}} | F1 : {{f1:.4f}}")

# ── Sauvegarder le .pkl localement ──────────────────────────────
with open(f"{{MODEL_NAME}}_v{{MODEL_VERSION}}.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle sauvegardé.")
'''
st.code(code_mlflow, language="python")

# ============================================================
# SECTION 2 — Uploader via l'API
# ============================================================
st.subheader("2. Uploader le modèle via l'API")

code_upload = f'''\
import requests

# ── Configuration ──────────────────────────────────────────────
API_URL = "{API_URL}"
API_TOKEN = "VOTRE_TOKEN_ICI"       # Token Bearer (admin ou user)
MODEL_NAME = "iris_model"
MODEL_VERSION = "1.0.0"
PKL_FILE = f"{{MODEL_NAME}}_v{{MODEL_VERSION}}.pkl"

headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

# ── Upload du modèle ────────────────────────────────────────────
with open(PKL_FILE, "rb") as f:
    response = requests.post(
        f"{{API_URL}}/models",
        headers=headers,
        files={{"file": (PKL_FILE, f, "application/octet-stream")}},
        data={{
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "description": "Random Forest sur Iris dataset",
            "algorithm": "RandomForestClassifier",
            "accuracy": str(acc),           # variable du step 1
            "f1_score": str(f1),            # variable du step 1
            "mlflow_run_id": run_id,        # variable du step 1
            "features_count": "4",
            "classes": "[0, 1, 2]",
            "training_params": '{{"n_estimators": 100, "max_depth": 5}}',
        }},
    )

response.raise_for_status()
model_data = response.json()
print(f"Modèle uploadé : {{model_data['name']}} v{{model_data['version']}}")
'''
st.code(code_upload, language="python")

# ============================================================
# SECTION 3 — Faire une prédiction
# ============================================================
st.subheader("3. Faire une prédiction")

code_predict = f'''\
import requests

API_URL = "{API_URL}"
API_TOKEN = "VOTRE_TOKEN_ICI"
headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

# ── Prédiction avec le modèle par défaut (version en production) ─
payload = {{
    "model_name": "iris_model",
    "features": {{
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }},
    "id_obs": "obs_001",       # optionnel — identifiant de l\'observation
}}

response = requests.post(f"{{API_URL}}/predict", headers=headers, json=payload)
response.raise_for_status()

result = response.json()
print(f"Prédiction : {{result[\'prediction\']}}")
print(f"Probabilités : {{result.get(\'probability\')}}")

# ── Passer en production (admin) ─────────────────────────────────
requests.patch(
    f"{{API_URL}}/models/iris_model/1.0.0",
    headers=headers,
    json={{"is_production": True}},
).raise_for_status()
print("Version 1.0.0 passée en production.")
'''
st.code(code_predict, language="python")

# ============================================================
# SECTION 4 — Enregistrer les résultats observés
# ============================================================
st.subheader("4. Enregistrer les résultats observés (optionnel)")
st.markdown("Permet de comparer les prédictions avec les vraies valeurs.")

code_observed = f'''\
import requests
from datetime import datetime

API_URL = "{API_URL}"
API_TOKEN = "VOTRE_TOKEN_ICI"
headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

payload = {{
    "data": [
        {{
            "id_obs": "obs_001",
            "model_name": "iris_model",
            "date_time": datetime.utcnow().isoformat(),
            "observed_result": 0,    # vraie classe observée
        }}
    ]
}}

response = requests.post(f"{{API_URL}}/observed-results", headers=headers, json=payload)
response.raise_for_status()
print(f"{{response.json()[\'upserted\']}} résultat(s) enregistré(s).")
'''
st.code(code_observed, language="python")

st.divider()
st.caption(f"API : `{API_URL}` — MLflow : `{MLFLOW_URL}`")
