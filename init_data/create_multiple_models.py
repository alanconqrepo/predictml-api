"""
Script pour créer plusieurs modèles d'exemple avec MLflow.

Chaque modèle est :
- Entraîné directement (sans Pipeline avancé)
- Loggé dans MLflow (params, métriques, artefact dans s3://mlflow/)
- Enregistré via POST /models avec mlflow_run_id (pas d'upload MinIO séparé)

Ordre d'exécution :
    docker-compose up -d
    python init_data/create_multiple_models.py
"""
import os
import sys

# Fix encoding sur Windows (cp1252 ne supporte pas les emojis MLflow)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn
from minio import Minio

MLFLOW_TRACKING_URI = "http://localhost:5000"
MINIO_ENDPOINT = "localhost:9002"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minio_secure_password_123"
MODEL_VERSION = "1.0.0"

API_URL = "http://localhost:8000"
API_TOKEN = "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA"


def configure_mlflow():
    """Configure MLflow avec MinIO comme backend d'artefacts."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MINIO_ENDPOINT}"
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                   secret_key=MINIO_SECRET_KEY, secure=False)
    if not client.bucket_exists("mlflow"):
        client.make_bucket("mlflow")
        print("   Bucket 'mlflow' cree dans MinIO")


def train_and_register(name, model, X_train, X_test, y_train, y_test, params, description, classes):
    """Entraîne le modèle, logue dans MLflow, enregistre via POST /models."""
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"{name}_v{MODEL_VERSION}"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.set_tag("description", description)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=name)

        run_id = mlflow.active_run().info.run_id
        print(f"   Accuracy : {acc:.4f}  |  F1 : {f1:.4f}")
        print(f"   MLflow run ID : {run_id}")

    # Enregistrer via POST /models (mlflow_run_id uniquement, pas d'upload MinIO séparé)
    import json
    response = requests.post(
        f"{API_URL}/models",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        data={
            "name": name,
            "version": MODEL_VERSION,
            "description": description,
            "algorithm": params.get("algorithm", ""),
            "mlflow_run_id": run_id,
            "accuracy": acc,
            "f1_score": f1,
            "features_count": params.get("features_count"),
            "classes": json.dumps(classes) if classes else None,
            "training_params": json.dumps(params),
            "training_dataset": params.get("dataset", ""),
        },
    )

    if response.status_code == 201:
        print(f"   Modele enregistre via API (id={response.json()['id']})")
    elif response.status_code == 409:
        print(f"   Modele '{name}' existe deja — ignore")
    else:
        print(f"   ERREUR API {response.status_code}: {response.text}")
        raise RuntimeError(f"Echec enregistrement {name}: {response.status_code}")


def create_iris():
    print("\n[1/3] iris_model — RandomForest")
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_and_register(
        name="iris_model",
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        params={"algorithm": "RandomForestClassifier", "n_estimators": 100, "dataset": "iris", "features_count": 4},
        description="Iris dataset — RandomForest",
        classes=[0, 1, 2],
    )


def create_wine():
    print("\n[2/3] wine_model — LogisticRegression")
    X, y = load_wine(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_and_register(
        name="wine_model",
        model=LogisticRegression(max_iter=10000, random_state=42),
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        params={"algorithm": "LogisticRegression", "max_iter": 10000, "dataset": "wine", "features_count": 13},
        description="Wine dataset — LogisticRegression",
        classes=[0, 1, 2],
    )


def create_cancer():
    print("\n[3/3] cancer_model — DecisionTree")
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_and_register(
        name="cancer_model",
        model=DecisionTreeClassifier(max_depth=10, random_state=42),
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        params={"algorithm": "DecisionTreeClassifier", "max_depth": 10, "dataset": "breast_cancer", "features_count": 30},
        description="Breast cancer dataset — DecisionTree",
        classes=[0, 1],
    )


def main():
    print("=" * 60)
    print("Creation des modeles avec MLflow")
    print(f"MLflow : {MLFLOW_TRACKING_URI}")
    print(f"API    : {API_URL}")
    print("=" * 60)

    print("\nConfiguration MLflow + MinIO...")
    configure_mlflow()

    create_iris()
    create_wine()
    create_cancer()

    print("\n" + "=" * 60)
    print("Tous les modeles ont ete crees, logues dans MLflow et enregistres via l'API.")
    print("=" * 60)


if __name__ == "__main__":
    main()
