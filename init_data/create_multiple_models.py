"""
Script to create several example models with MLflow.

Each model is:
- Trained directly (without an advanced Pipeline)
- Logged to MLflow (params, metrics, artifact in s3://mlflow/)
- Registered via POST /models with mlflow_run_id (no separate MinIO upload)
- PATCH'd after registration to store training_stats (label_distribution + n_rows)

Execution order:
    docker-compose up -d
    python init_data/create_multiple_models.py
"""
import json
import os
import sys

# Fix encoding on Windows (cp1252 does not support MLflow emojis)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn
from minio import Minio

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MODEL_VERSION = "1.0.0"

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("API_TOKEN", "")


def configure_mlflow():
    """Configure MLflow with MinIO as the artifact backend."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MINIO_ENDPOINT}"
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                   secret_key=MINIO_SECRET_KEY, secure=False)
    if not client.bucket_exists("mlflow"):
        client.make_bucket("mlflow")
        print("   Bucket 'mlflow' created in MinIO")


def _patch_training_stats(name: str, y_train) -> None:
    """PATCH the model to store label_distribution and n_rows in training_stats."""
    total = len(y_train)
    label_distribution = {
        str(int(cls)): round(float(np.sum(y_train == cls)) / total, 4)
        for cls in np.unique(y_train)
    }
    patch_body = {
        "training_stats": {
            "label_distribution": label_distribution,
            "n_rows": total,
        }
    }
    resp = requests.patch(
        f"{API_URL}/models/{name}/{MODEL_VERSION}",
        headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
        json=patch_body,
        timeout=30,
    )
    if resp.status_code == 200:
        print(f"   training_stats stored (label_distribution: {label_distribution})")
    else:
        print(f"   [WARN] PATCH training_stats failed ({resp.status_code}): {resp.text[:120]}")


def train_and_register(name, model, X_train, X_test, y_train, y_test, params, description, classes):
    """Train the model, log to MLflow, register via POST /models."""
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"{name}_v{MODEL_VERSION}"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        auc = None
        if classes and len(classes) == 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        if auc is not None:
            mlflow.log_metric("auc", auc)
        mlflow.set_tag("description", description)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=name)

        run_id = mlflow.active_run().info.run_id
        auc_str = f"  |  AUC : {auc:.4f}" if auc is not None else ""
        print(f"   Accuracy : {acc:.4f}  |  F1 : {f1:.4f}{auc_str}")
        print(f"   MLflow run ID : {run_id}")

    # Register via POST /models (mlflow_run_id only, no separate MinIO upload)
    data = {
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
    }
    if auc is not None:
        data["auc"] = auc
    response = requests.post(
        f"{API_URL}/models",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        data=data,
    )

    if response.status_code == 201:
        print(f"   Model registered via API (id={response.json()['id']})")
        _patch_training_stats(name, y_train)
    elif response.status_code == 409:
        print(f"   Model '{name}' already exists — skipped")
    else:
        print(f"   API ERROR {response.status_code}: {response.text}")
        raise RuntimeError(f"Registration failed for {name}: {response.status_code}")


def create_iris():
    print("\n[1/3] iris_model — RandomForest")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
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
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
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
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
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
    print("Creating models with MLflow")
    print(f"MLflow : {MLFLOW_TRACKING_URI}")
    print(f"API    : {API_URL}")
    print("=" * 60)

    print("\nConfiguring MLflow + MinIO...")
    configure_mlflow()

    create_iris()
    create_wine()
    create_cancer()

    print("\n" + "=" * 60)
    print("All models have been created, logged in MLflow and registered via the API.")
    print("=" * 60)


if __name__ == "__main__":
    main()
