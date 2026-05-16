"""
Exemple de code complet : MLflow + API
"""

import os

import streamlit as st
from utils.auth import require_auth

st.set_page_config(page_title="Code Example — PredictML", page_icon="💡", layout="wide")
require_auth()

st.title("💡 Exemple de code : MLflow + API")
st.markdown(
    "Exemple complet pour entraîner un modèle, le tracker avec MLflow, puis l'uploader et l'utiliser via l'API."
)

API_URL = st.session_state.get("api_url", "http://localhost:8000")
TOKEN = st.session_state.get("api_token", "VOTRE_TOKEN_ICI")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")

tab_python, tab_curl, tab_js = st.tabs(["Python", "curl / bash", "JavaScript"])

# ============================================================
# TAB PYTHON
# ============================================================
with tab_python:
    # SECTION 1 — Entraîner et tracker avec MLflow
    st.subheader("1. Entraîner un modèle et le tracker avec MLflow")
    st.markdown("Installez les dépendances : `pip install scikit-learn mlflow boto3`")

    code_mlflow = f"""\
import joblib
import mlflow
import mlflow.sklearn
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

# ── Sauvegarder le modèle localement ───────────────────────────
joblib.dump(model, f"{{MODEL_NAME}}_v{{MODEL_VERSION}}.joblib")

print("Modèle sauvegardé.")
"""
    st.code(code_mlflow, language="python")

    # SECTION 2 — Uploader via l'API
    st.subheader("2. Uploader le modèle via l'API")

    code_upload = f"""\
import requests

# ── Configuration ──────────────────────────────────────────────
API_URL = "{API_URL}"
API_TOKEN = "{TOKEN}"
MODEL_NAME = "iris_model"
MODEL_VERSION = "1.0.0"
MODEL_FILE = f"{{MODEL_NAME}}_v{{MODEL_VERSION}}.joblib"

headers = {{"Authorization": f"Bearer {{API_TOKEN}}"}}

# ── Upload du modèle ────────────────────────────────────────────
with open(MODEL_FILE, "rb") as f:
    response = requests.post(
        f"{{API_URL}}/models",
        headers=headers,
        files={{"file": (MODEL_FILE, f, "application/octet-stream")}},
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
"""
    st.code(code_upload, language="python")

    # SECTION 3 — Faire une prédiction
    st.subheader("3. Faire une prédiction")

    code_predict = f"""\
import requests

API_URL = "{API_URL}"
API_TOKEN = "{TOKEN}"
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
"""
    st.code(code_predict, language="python")

    # SECTION 4 — Enregistrer les résultats observés
    st.subheader("4. Enregistrer les résultats observés (optionnel)")
    st.markdown("Permet de comparer les prédictions avec les vraies valeurs.")

    code_observed = f"""\
import requests
from datetime import datetime

API_URL = "{API_URL}"
API_TOKEN = "{TOKEN}"
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
"""
    st.code(code_observed, language="python")

# ============================================================
# TAB CURL / BASH
# ============================================================
with tab_curl:
    st.subheader("1. Uploader le modèle")

    code_curl_upload = f"""\
#!/usr/bin/env bash
API_URL="{API_URL}"
TOKEN="{TOKEN}"

curl -X POST "$API_URL/models" \\
  -H "Authorization: Bearer $TOKEN" \\
  -F "file=@iris_model_v1.0.0.joblib;type=application/octet-stream" \\
  -F "name=iris_model" \\
  -F "version=1.0.0" \\
  -F "description=Random Forest sur Iris dataset" \\
  -F "algorithm=RandomForestClassifier" \\
  -F "accuracy=0.9667" \\
  -F "f1_score=0.9667" \\
  -F "features_count=4" \\
  -F "classes=[0, 1, 2]"
"""
    st.code(code_curl_upload, language="bash")

    st.subheader("2. Faire une prédiction")

    code_curl_predict = f"""\
API_URL="{API_URL}"
TOKEN="{TOKEN}"

curl -X POST "$API_URL/predict" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model_name": "iris_model",
    "features": {{
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }},
    "id_obs": "obs_001"
  }}'
"""
    st.code(code_curl_predict, language="bash")

    st.subheader("3. Récupérer l'historique des prédictions")

    code_curl_history = f"""\
API_URL="{API_URL}"
TOKEN="{TOKEN}"

curl -G "$API_URL/predictions" \\
  -H "Authorization: Bearer $TOKEN" \\
  --data-urlencode "model_name=iris_model" \\
  --data-urlencode "limit=10"
"""
    st.code(code_curl_history, language="bash")

    st.subheader("4. Soumettre un résultat observé")

    code_curl_observed = f"""\
API_URL="{API_URL}"
TOKEN="{TOKEN}"

curl -X POST "$API_URL/observed-results" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "data": [
      {{
        "id_obs": "obs_001",
        "model_name": "iris_model",
        "date_time": "2026-04-28T12:00:00",
        "observed_result": 0
      }}
    ]
  }}'
"""
    st.code(code_curl_observed, language="bash")

# ============================================================
# TAB JAVASCRIPT
# ============================================================
with tab_js:
    st.subheader("1. Uploader le modèle")
    st.markdown(
        "_Depuis un navigateur, le fichier `.joblib` doit provenir d'un `<input type=\"file\">`._"
    )

    code_js_upload = f"""\
const API_URL = "{API_URL}";
const TOKEN = "{TOKEN}";

// fileInput est un élément <input type="file"> dans votre page HTML
const pklFile = fileInput.files[0];

const formData = new FormData();
formData.append("file", pklFile, "iris_model_v1.0.0.joblib");
formData.append("name", "iris_model");
formData.append("version", "1.0.0");
formData.append("description", "Random Forest sur Iris dataset");
formData.append("algorithm", "RandomForestClassifier");
formData.append("accuracy", "0.9667");
formData.append("f1_score", "0.9667");
formData.append("features_count", "4");
formData.append("classes", "[0, 1, 2]");

const response = await fetch(`${{API_URL}}/models`, {{
  method: "POST",
  headers: {{ "Authorization": `Bearer ${{TOKEN}}` }},
  body: formData,
}});
const model = await response.json();
console.log(`Modèle uploadé : ${{model.name}} v${{model.version}}`);
"""
    st.code(code_js_upload, language="javascript")

    st.subheader("2. Faire une prédiction")

    code_js_predict = f"""\
const API_URL = "{API_URL}";
const TOKEN = "{TOKEN}";

const response = await fetch(`${{API_URL}}/predict`, {{
  method: "POST",
  headers: {{
    "Authorization": `Bearer ${{TOKEN}}`,
    "Content-Type": "application/json",
  }},
  body: JSON.stringify({{
    model_name: "iris_model",
    features: {{
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2,
    }},
    id_obs: "obs_001",
  }}),
}});
const result = await response.json();
console.log(`Prédiction : ${{result.prediction}}`);
console.log(`Probabilités :`, result.probability);
"""
    st.code(code_js_predict, language="javascript")

    st.subheader("3. Récupérer l'historique des prédictions")

    code_js_history = f"""\
const API_URL = "{API_URL}";
const TOKEN = "{TOKEN}";

const params = new URLSearchParams({{ model_name: "iris_model", limit: "10" }});
const response = await fetch(`${{API_URL}}/predictions?${{params}}`, {{
  headers: {{ "Authorization": `Bearer ${{TOKEN}}` }},
}});
const history = await response.json();
console.log(`${{history.length}} prédiction(s) trouvée(s).`);
"""
    st.code(code_js_history, language="javascript")

    st.subheader("4. Soumettre un résultat observé")

    code_js_observed = f"""\
const API_URL = "{API_URL}";
const TOKEN = "{TOKEN}";

const response = await fetch(`${{API_URL}}/observed-results`, {{
  method: "POST",
  headers: {{
    "Authorization": `Bearer ${{TOKEN}}`,
    "Content-Type": "application/json",
  }},
  body: JSON.stringify({{
    data: [
      {{
        id_obs: "obs_001",
        model_name: "iris_model",
        date_time: new Date().toISOString(),
        observed_result: 0,
      }},
    ],
  }}),
}});
const result = await response.json();
console.log(`${{result.upserted}} résultat(s) enregistré(s).`);
"""
    st.code(code_js_observed, language="javascript")

st.divider()
st.caption(f"API : `{API_URL}` — MLflow : `{MLFLOW_URL}`")
