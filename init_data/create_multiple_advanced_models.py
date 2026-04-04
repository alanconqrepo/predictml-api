"""
Script de création de modèles avancés avec sklearn Pipelines et MLflow.

Chaque modèle est :
- Entraîné dans un Pipeline sklearn (StandardScaler + éventuellement OneHotEncoder)
- Loggé dans MLflow (params, métriques, artefact dans s3://mlflow/)
- Uploadé via POST /models (s3://models/ + base de données)

Ordre d'exécution :
    docker-compose up -d
    python init_data/create_multiple_advanced_models.py
"""
import json
import os
import sys
from pathlib import Path

# Fix encoding sur Windows (cp1252 ne supporte pas les emojis MLflow)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import requests
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def configure_mlflow():
    """Configure les variables d'environnement S3 et l'URI MLflow."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MINIO_ENDPOINT}"
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                   secret_key=MINIO_SECRET_KEY, secure=False)
    if not client.bucket_exists("mlflow"):
        client.make_bucket("mlflow")
        print("   Bucket 'mlflow' cree dans MinIO")


# ---------------------------------------------------------------------------
# Builders de pipelines
# ---------------------------------------------------------------------------

def make_numeric_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])


def make_mixed_pipeline(numeric_cols: list, categorical_cols: list, estimator) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


# ---------------------------------------------------------------------------
# Entraînement + logging MLflow + upload via API
# ---------------------------------------------------------------------------

def train_and_register(
    name: str,
    pipeline: Pipeline,
    X_train, X_test, y_train, y_test,
    params: dict,
    description: str,
    classes: list = None,
) -> None:
    """
    Entraîne le pipeline, logue dans MLflow, puis envoie le modèle via POST /models.
    """
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"{name}_v{MODEL_VERSION}"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.set_tag("description", description)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=name)

        run_id = mlflow.active_run().info.run_id
        print(f"   Accuracy : {acc:.4f}  |  F1 : {f1:.4f}")
        print(f"   MLflow run ID : {run_id}")

    # Envoyer via POST /models (mlflow_run_id uniquement, pas d'upload MinIO séparé)
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
            "features_count": params.get("n_features") or params.get("n_numeric_features"),
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


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

def create_iris_advanced():
    print("\n[1/4] iris_advanced_model — RandomForest + StandardScaler")
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = make_numeric_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    train_and_register(
        name="iris_advanced_model",
        pipeline=pipeline,
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        params={
            "algorithm": "RandomForestClassifier",
            "n_estimators": 200,
            "preprocessing": "StandardScaler",
            "dataset": "iris",
            "n_features": 4,
            "n_classes": 3,
        },
        description="Iris dataset — RandomForest avec StandardScaler",
        classes=[0, 1, 2],
    )


def create_wine_advanced():
    print("\n[2/4] wine_advanced_model — LogisticRegression + StandardScaler")
    X, y = load_wine(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = make_numeric_pipeline(
        LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    )
    train_and_register(
        name="wine_advanced_model",
        pipeline=pipeline,
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        params={
            "algorithm": "LogisticRegression",
            "C": 1.0,
            "max_iter": 5000,
            "solver": "lbfgs",
            "preprocessing": "StandardScaler",
            "dataset": "wine",
            "n_features": 13,
            "n_classes": 3,
        },
        description="Wine dataset — LogisticRegression avec StandardScaler",
        classes=[0, 1, 2],
    )


def create_cancer_advanced():
    print("\n[3/4] cancer_advanced_model — GradientBoosting + StandardScaler")
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = make_numeric_pipeline(
        GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    )
    train_and_register(
        name="cancer_advanced_model",
        pipeline=pipeline,
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        params={
            "algorithm": "GradientBoostingClassifier",
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "preprocessing": "StandardScaler",
            "dataset": "breast_cancer",
            "n_features": 30,
            "n_classes": 2,
        },
        description="Breast cancer dataset — GradientBoosting avec StandardScaler",
        classes=[0, 1],
    )


def create_loan_model():
    print("\n[4/4] loan_model — GradientBoosting + ColumnTransformer (num + cat)")

    rng = np.random.default_rng(42)
    n = 300

    age = rng.integers(22, 65, n).astype(float)
    income = rng.integers(20000, 120000, n).astype(float)
    loan_amount = rng.integers(5000, 50000, n).astype(float)
    employment = rng.choice(["employed", "self_employed", "unemployed"], n)
    education = rng.choice(["high_school", "bachelor", "master", "phd"], n)
    credit = rng.choice(["good", "fair", "poor"], n)

    score = (income / 120000) * 0.4
    score += np.where(credit == "good", 0.4, np.where(credit == "fair", 0.2, 0.0))
    score += np.where(np.isin(education, ["master", "phd"]), 0.2, 0.1)
    score += rng.uniform(-0.1, 0.1, n)
    approved = (score > 0.5).astype(int)

    X = np.column_stack([age, income, loan_amount, employment, education, credit])
    y = approved

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = make_mixed_pipeline(
        numeric_cols=[0, 1, 2],
        categorical_cols=[3, 4, 5],
        estimator=GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        ),
    )

    train_and_register(
        name="loan_model",
        pipeline=pipeline,
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        params={
            "algorithm": "GradientBoostingClassifier",
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "preprocessing": "ColumnTransformer(StandardScaler + OneHotEncoder)",
            "dataset": "synthetic_loan",
            "n_numeric_features": 3,
            "n_categorical_features": 3,
            "n_samples": n,
        },
        description="Dataset synthetique (pret bancaire) — features mixtes num + cat",
        classes=[0, 1],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Creation des modeles avances avec Pipelines + MLflow")
    print(f"MLflow : {MLFLOW_TRACKING_URI}")
    print(f"API    : {API_URL}")
    print("=" * 60)

    print("\nConfiguration MLflow + MinIO...")
    configure_mlflow()

    create_iris_advanced()
    create_wine_advanced()
    create_cancer_advanced()
    create_loan_model()

    print("\n" + "=" * 60)
    print("Tous les modeles ont ete crees, logues dans MLflow et enregistres via l'API.")
    print("=" * 60)


if __name__ == "__main__":
    main()
