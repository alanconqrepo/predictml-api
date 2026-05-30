"""
Script for creating advanced models with sklearn Pipelines and MLflow.

Each model is:
- Trained in a sklearn Pipeline (StandardScaler + optionally OneHotEncoder)
- Logged to MLflow (params, metrics, artifact in s3://mlflow/)
- Uploaded via POST /models (s3://models/ + database)

Execution order:
    docker-compose up -d
    python init_data/create_multiple_advanced_models.py
"""

import json
import os
import sys

# Fix encoding on Windows (cp1252 does not support MLflow emojis)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from minio import Minio
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MODEL_VERSION = "1.0.0"

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("API_TOKEN", "")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_mlflow():
    """Configure S3 environment variables and the MLflow URI."""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MINIO_ENDPOINT}"
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = Minio(
        MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
    )
    if not client.bucket_exists("mlflow"):
        client.make_bucket("mlflow")
        print("   Bucket 'mlflow' created in MinIO")


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def make_numeric_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def make_mixed_pipeline(numeric_cols: list, categorical_cols: list, estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Training + MLflow logging + upload via API
# ---------------------------------------------------------------------------


def train_and_register(
    name: str,
    pipeline: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    params: dict,
    description: str,
    classes: list = None,
) -> None:
    """
    Trains the pipeline, logs to MLflow, then sends the model via POST /models.
    """
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"{name}_v{MODEL_VERSION}"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        auc = None
        if classes and len(classes) == 2 and hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        if auc is not None:
            mlflow.log_metric("auc", auc)
        mlflow.set_tag("description", description)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=name)

        run_id = mlflow.active_run().info.run_id
        auc_str = f"  |  AUC : {auc:.4f}" if auc is not None else ""
        print(f"   Accuracy : {acc:.4f}  |  F1 : {f1:.4f}{auc_str}")
        print(f"   MLflow run ID : {run_id}")

    # Send via POST /models (mlflow_run_id only, no separate MinIO upload)
    data = {
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
    elif response.status_code == 409:
        print(f"   Model '{name}' already exists — skipped")
    else:
        print(f"   API ERROR {response.status_code}: {response.text}")
        raise RuntimeError(f"Registration failed for {name}: {response.status_code}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def create_iris_advanced():
    print("\n[1/4] iris_advanced_model — RandomForest + StandardScaler")
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = make_numeric_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    train_and_register(
        name="iris_advanced_model",
        pipeline=pipeline,
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        params={
            "algorithm": "RandomForestClassifier",
            "n_estimators": 200,
            "preprocessing": "StandardScaler",
            "dataset": "iris",
            "n_features": 4,
            "n_classes": 3,
        },
        description="Iris dataset — RandomForest with StandardScaler",
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
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
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
        description="Wine dataset — LogisticRegression with StandardScaler",
        classes=[0, 1, 2],
    )


def create_cancer_advanced():
    print("\n[3/4] cancer_advanced_model — GradientBoosting + StandardScaler")
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = make_numeric_pipeline(
        GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
    )
    train_and_register(
        name="cancer_advanced_model",
        pipeline=pipeline,
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
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
        description="Breast cancer dataset — GradientBoosting with StandardScaler",
        classes=[0, 1],
    )


def create_titanic_model():
    print("\n[4/5] titanic-survival — GradientBoosting + ColumnTransformer (num + cat)")
    print("      Features : age, fare, parch, sibsp (num) + pclass, sex, embarked (cat)")

    rng = np.random.default_rng(42)
    n = 891  # reference: actual size of the Titanic dataset

    # Categorical features
    pclass_vals = rng.choice(["1st", "2nd", "3rd"], size=n, p=[0.24, 0.21, 0.55])
    sex_vals = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    embarked_vals = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    # Numeric features
    age_mu = np.where(pclass_vals == "1st", 39.0, np.where(pclass_vals == "2nd", 29.0, 25.0))
    age_vals = rng.normal(age_mu, 14.0).clip(1.0, 80.0).round(1)
    fare_mu = np.where(pclass_vals == "1st", 87.0, np.where(pclass_vals == "2nd", 21.0, 13.0))
    fare_vals = np.exp(rng.normal(np.log(fare_mu), 0.6)).clip(5.0, 512.0).round(2)
    sibsp_vals = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.68, 0.16, 0.10, 0.04, 0.02])
    parch_vals = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.76, 0.13, 0.07, 0.03, 0.01])

    # Target: probabilistic survival
    base_surv = np.where(sex_vals == "female", 0.74, 0.19)
    class_bonus = np.where(pclass_vals == "1st", 0.15, np.where(pclass_vals == "3rd", -0.10, 0.0))
    age_bonus = np.where(age_vals < 12, 0.15, 0.0)
    survival_prob = np.clip(base_surv + class_bonus + age_bonus, 0.05, 0.95)
    survived = (rng.uniform(size=n) < survival_prob).astype(int)

    # pandas DataFrame -> pipeline.feature_names_in_ will contain the 7 column names
    NUMERIC_FEATURES = ["age", "fare", "parch", "sibsp"]
    CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]
    df = pd.DataFrame(
        {
            "age": age_vals,
            "fare": fare_vals,
            "parch": parch_vals.astype(float),
            "sibsp": sibsp_vals.astype(float),
            "pclass": pclass_vals,
            "sex": sex_vals,
            "embarked": embarked_vals,
        }
    )
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = survived

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Use column names (not indices) so that feature_names_in_ is defined
    pipeline = make_mixed_pipeline(
        numeric_cols=NUMERIC_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
        estimator=GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
        ),
    )

    train_and_register(
        name="titanic-survival",
        pipeline=pipeline,
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        params={
            "algorithm": "GradientBoostingClassifier",
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.08,
            "preprocessing": "ColumnTransformer(StandardScaler + OneHotEncoder)",
            "dataset": "synthetic_titanic",
            "n_numeric_features": len(NUMERIC_FEATURES),
            "n_categorical_features": len(CATEGORICAL_FEATURES),
            "n_samples": n,
        },
        description=(
            "Synthetic dataset inspired by the Titanic — mixed num + cat features. "
            "Predicts survival (0=deceased, 1=survivor). "
            "Numeric: age, fare, parch, sibsp. "
            "Categorical: pclass (1st/2nd/3rd), sex (male/female), embarked (S/C/Q)."
        ),
        classes=[0, 1],
    )


def create_loan_model():
    print("\n[5/5] loan_model — GradientBoosting + ColumnTransformer (num + cat)")

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
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
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
        description="Synthetic dataset (bank loan) — mixed num + cat features",
        classes=[0, 1],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Creating advanced models with Pipelines + MLflow")
    print(f"MLflow : {MLFLOW_TRACKING_URI}")
    print(f"API    : {API_URL}")
    print("=" * 60)

    print("\nConfiguring MLflow + MinIO...")
    configure_mlflow()

    create_iris_advanced()
    create_wine_advanced()
    create_cancer_advanced()
    create_titanic_model()
    create_loan_model()

    print("\n" + "=" * 60)
    print("All models have been created, logged in MLflow and registered via the API.")
    print("=" * 60)


if __name__ == "__main__":
    main()
