"""
Couverture des fonctionnalités pour les modèles de classification multiclasse (3+ classes).

Vérifie que les comportements suivants sont corrects pour les classifieurs multiclasses :
- POST /predict          → probability a exactement N_classes éléments
- POST /predict-batch    → idem pour chaque item du batch
- POST /explain          → SHAP fonctionne (tree et linear) sans erreur de format 3D/liste
- GET  /feature-importance → importances valides pour un classifieur multiclasse
- POST /validate-input   → validation de schéma fonctionne comme pour les autres types
- POST /predict?explain=true → SHAP inline non-null en multiclasse
"""

import asyncio
import pickle
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-mc-admin-x4z9"
USER_TOKEN = "test-token-mc-user-x4z9"

MC_LR_MODEL = "mc_lr_iris"
MC_RF_MODEL = "mc_rf_iris"
MODEL_VERSION = "1.0.0"

IRIS_FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
SAMPLE_INPUT = {f: v for f, v in zip(IRIS_FEATURES, [5.1, 3.5, 1.4, 0.2])}
IRIS_CLASSES = [0, 1, 2]


# ---------------------------------------------------------------------------
# Model builders — iris, 3 classes
# ---------------------------------------------------------------------------


def _make_lr_iris() -> LogisticRegression:
    X, y = load_iris(return_X_y=True, as_frame=True)
    return LogisticRegression(max_iter=1000).fit(X, y)


def _make_rf_iris() -> RandomForestClassifier:
    X, y = load_iris(return_X_y=True, as_frame=True)
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)


def _inject_cache(model_name: str, version: str, model, feature_baseline=None) -> str:
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
            feature_baseline=feature_baseline,
        ),
    }
    asyncio.run(model_service._redis.set(f"model:{key}", pickle.dumps(data)))
    return key


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        for token, username, email, role in [
            (ADMIN_TOKEN, "mc_admin", "mc_admin@test.com", "admin"),
            (USER_TOKEN, "mc_user", "mc_user@test.com", "user"),
        ]:
            if not await DBService.get_user_by_token(db, token):
                await DBService.create_user(
                    db,
                    username=username,
                    email=email,
                    api_token=token,
                    role=role,
                    rate_limit=99999,
                )

        for name in [MC_LR_MODEL, MC_RF_MODEL]:
            if not await DBService.get_model_metadata(db, name, MODEL_VERSION):
                await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=MODEL_VERSION,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{MODEL_VERSION}.pkl",
                    is_active=True,
                    is_production=True,
                )

        # Seed predictions for feature-importance tests
        user = await DBService.get_user_by_token(db, USER_TOKEN)
        for _ in range(6):
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=MC_RF_MODEL,
                model_version=MODEL_VERSION,
                input_features=SAMPLE_INPUT,
                prediction_result=0,
                probabilities=[0.9, 0.07, 0.03],
                response_time_ms=8.0,
            )


asyncio.run(_setup())

AUTH = {"Authorization": f"Bearer {USER_TOKEN}"}
ADMIN_AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


# ---------------------------------------------------------------------------
# POST /predict — multiclasse
# ---------------------------------------------------------------------------


class TestMulticlassPredict:
    def test_lr_predict_probability_has_3_elements(self):
        """LogisticRegression iris → probability contient exactement 3 valeurs."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict",
                headers=AUTH,
                json={"model_name": MC_LR_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["probability"] is not None
            assert len(data["probability"]) == 3
            assert abs(sum(data["probability"]) - 1.0) < 1e-6
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_lr_predict_prediction_is_valid_class(self):
        """La prédiction est une classe iris valide (0, 1 ou 2)."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict",
                headers=AUTH,
                json={"model_name": MC_LR_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            pred = r.json()["prediction"]
            assert pred in IRIS_CLASSES
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_rf_predict_probability_has_3_elements(self):
        """RandomForestClassifier iris → probability contient exactement 3 valeurs."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict",
                headers=AUTH,
                json={"model_name": MC_RF_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            prob = r.json()["probability"]
            assert prob is not None
            assert len(prob) == 3
            assert abs(sum(prob) - 1.0) < 1e-6
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_rf_predict_probabilities_are_floats(self):
        """Les probabilités d'un RF iris sont bien des floats entre 0 et 1."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict",
                headers=AUTH,
                json={"model_name": MC_RF_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            prob = r.json()["probability"]
            for p in prob:
                assert isinstance(p, float)
                assert 0.0 <= p <= 1.0
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# POST /predict-batch — multiclasse
# ---------------------------------------------------------------------------


class TestMulticlassBatchPredict:
    def test_batch_predict_each_item_has_3_probabilities(self):
        """batch predict sur LR iris → chaque résultat a 3 probabilités."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        samples = [
            {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
            {"sepal length (cm)": 6.3, "sepal width (cm)": 3.3, "petal length (cm)": 4.7, "petal width (cm)": 1.6},
            {"sepal length (cm)": 6.5, "sepal width (cm)": 3.0, "petal length (cm)": 5.2, "petal width (cm)": 2.0},
        ]
        try:
            r = client.post(
                "/predict-batch",
                headers=AUTH,
                json={
                    "model_name": MC_LR_MODEL,
                    "inputs": [{"features": s} for s in samples],
                },
            )
            assert r.status_code == 200
            predictions = r.json()["predictions"]
            assert len(predictions) == 3
            for item in predictions:
                prob = item["probability"]
                assert prob is not None
                assert len(prob) == 3
                assert abs(sum(prob) - 1.0) < 1e-6
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_batch_predict_rf_no_probability_none(self):
        """RF multiclasse batch → aucun item ne retourne probability=None."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict-batch",
                headers=AUTH,
                json={
                    "model_name": MC_RF_MODEL,
                    "inputs": [{"features": SAMPLE_INPUT}, {"features": SAMPLE_INPUT}],
                },
            )
            assert r.status_code == 200
            for item in r.json()["predictions"]:
                assert item["probability"] is not None
                assert len(item["probability"]) == 3
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# POST /explain — SHAP multiclasse
# ---------------------------------------------------------------------------


class TestMulticlassSHAP:
    def test_explain_rf_multiclass_returns_200(self):
        """POST /explain sur RF iris (tree, 3 classes) → 200, shap_values complet."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/explain",
                headers=AUTH,
                json={"model_name": MC_RF_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["model_type"] == "tree"
            assert isinstance(data["shap_values"], dict)
            assert set(data["shap_values"].keys()) == set(IRIS_FEATURES)
            for v in data["shap_values"].values():
                assert isinstance(v, float)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_explain_lr_multiclass_returns_200(self):
        """POST /explain sur LR iris (linear, 3 classes) → 200, model_type=linear."""
        model = _make_lr_iris()
        baseline = {f: {"mean": 5.0, "std": 1.0, "min": 4.0, "max": 8.0} for f in IRIS_FEATURES}
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model, feature_baseline=baseline)
        try:
            r = client.post(
                "/explain",
                headers=AUTH,
                json={"model_name": MC_LR_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["model_type"] == "linear"
            assert isinstance(data["shap_values"], dict)
            assert set(data["shap_values"].keys()) == set(IRIS_FEATURES)
            for v in data["shap_values"].values():
                assert isinstance(v, float)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_explain_rf_multiclass_base_value_is_float(self):
        """base_value retourné par SHAP pour un RF multiclasse est bien un float scalaire."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/explain",
                headers=AUTH,
                json={"model_name": MC_RF_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            assert isinstance(r.json()["base_value"], float)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_explain_lr_multiclass_base_value_is_float(self):
        """base_value retourné par SHAP pour un LR multiclasse est bien un float scalaire."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/explain",
                headers=AUTH,
                json={"model_name": MC_LR_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            assert isinstance(r.json()["base_value"], float)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_explain_rf_multiclass_all_classes_sample(self):
        """SHAP fonctionne pour les 3 classes iris (setosa, versicolor, virginica)."""
        model = _make_rf_iris()
        # One representative sample per class
        samples = [
            # setosa (class 0)
            {"sepal length (cm)": 5.0, "sepal width (cm)": 3.6, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
            # versicolor (class 1)
            {"sepal length (cm)": 6.4, "sepal width (cm)": 3.2, "petal length (cm)": 4.5, "petal width (cm)": 1.5},
            # virginica (class 2)
            {"sepal length (cm)": 7.2, "sepal width (cm)": 3.6, "petal length (cm)": 6.1, "petal width (cm)": 2.5},
        ]
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            for sample in samples:
                r = client.post(
                    "/explain",
                    headers=AUTH,
                    json={"model_name": MC_RF_MODEL, "features": sample},
                )
                assert r.status_code == 200, f"Échec SHAP pour sample {sample}: {r.text}"
                assert isinstance(r.json()["shap_values"], dict)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# GET /models/{name}/feature-importance — multiclasse
# ---------------------------------------------------------------------------


class TestMulticlassFeatureImportance:
    def test_feature_importance_rf_multiclass_returns_200(self):
        """GET /feature-importance sur RF iris → 200 avec importances valides."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.get(
                f"/models/{MC_RF_MODEL}/feature-importance",
                headers=AUTH,
                params={"last_n": 10, "days": 30},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["model_name"] == MC_RF_MODEL
            assert data["sample_size"] > 0
            fi = data["feature_importance"]
            assert set(fi.keys()) == set(IRIS_FEATURES)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_feature_importance_rf_multiclass_ranks_are_valid(self):
        """Rangs consécutifs de 1 à N dans feature_importance multiclasse."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.get(
                f"/models/{MC_RF_MODEL}/feature-importance",
                headers=AUTH,
                params={"last_n": 10, "days": 30},
            )
            assert r.status_code == 200
            fi = r.json()["feature_importance"]
            ranks = sorted(v["rank"] for v in fi.values())
            assert ranks == list(range(1, len(fi) + 1))
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_feature_importance_rf_multiclass_mean_abs_shap_nonnegative(self):
        """mean_abs_shap ≥ 0 pour toutes les features en multiclasse."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.get(
                f"/models/{MC_RF_MODEL}/feature-importance",
                headers=AUTH,
                params={"last_n": 10, "days": 30},
            )
            assert r.status_code == 200
            fi = r.json()["feature_importance"]
            for feat, info in fi.items():
                assert info["mean_abs_shap"] >= 0.0, f"mean_abs_shap négatif pour {feat}"
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# POST /models/{name}/{version}/validate-input — multiclasse
# ---------------------------------------------------------------------------


class TestMulticlassInputValidation:
    def test_validate_input_all_features_present_returns_valid(self):
        """validate-input avec toutes les features iris → valid=true."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                f"/models/{MC_LR_MODEL}/{MODEL_VERSION}/validate-input",
                headers=AUTH,
                json=SAMPLE_INPUT,
            )
            assert r.status_code == 200
            data = r.json()
            assert data["valid"] is True
            assert data["errors"] == []
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_validate_input_missing_feature_returns_invalid(self):
        """validate-input avec une feature manquante → valid=false, erreur missing_feature."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        incomplete = {k: v for k, v in SAMPLE_INPUT.items() if k != "petal length (cm)"}
        try:
            r = client.post(
                f"/models/{MC_LR_MODEL}/{MODEL_VERSION}/validate-input",
                headers=AUTH,
                json=incomplete,
            )
            assert r.status_code == 200
            data = r.json()
            assert data["valid"] is False
            error_types = [e["type"] for e in data["errors"]]
            assert "missing_feature" in error_types
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_validate_input_unexpected_feature_reported(self):
        """validate-input avec une feature supplémentaire → unexpected_feature dans errors."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        extra = {**SAMPLE_INPUT, "extra_col": 99.0}
        try:
            r = client.post(
                f"/models/{MC_LR_MODEL}/{MODEL_VERSION}/validate-input",
                headers=AUTH,
                json=extra,
            )
            assert r.status_code == 200
            data = r.json()
            error_types = [e["type"] for e in data["errors"]]
            assert "unexpected_feature" in error_types
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_validate_input_expected_features_lists_all_iris_features(self):
        """expected_features contient les 4 features iris."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                f"/models/{MC_LR_MODEL}/{MODEL_VERSION}/validate-input",
                headers=AUTH,
                json=SAMPLE_INPUT,
            )
            assert r.status_code == 200
            expected = r.json()["expected_features"]
            assert expected is not None
            assert set(expected) == set(IRIS_FEATURES)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))


# ---------------------------------------------------------------------------
# POST /predict?explain=true — SHAP inline multiclasse
# ---------------------------------------------------------------------------


class TestMulticlassPredictInlineExplain:
    def test_predict_explain_true_lr_multiclass_returns_shap_values(self):
        """POST /predict?explain=true sur LR iris → shap_values non-null dans la réponse."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict",
                headers=AUTH,
                params={"explain": "true"},
                json={"model_name": MC_LR_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["shap_values"] is not None
            assert isinstance(data["shap_values"], dict)
            assert set(data["shap_values"].keys()) == set(IRIS_FEATURES)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_predict_explain_true_rf_multiclass_returns_shap_values(self):
        """POST /predict?explain=true sur RF iris → shap_values non-null dans la réponse."""
        model = _make_rf_iris()
        key = _inject_cache(MC_RF_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict",
                headers=AUTH,
                params={"explain": "true"},
                json={"model_name": MC_RF_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["shap_values"] is not None
            assert isinstance(data["shap_values"], dict)
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))

    def test_predict_explain_false_no_shap_values(self):
        """POST /predict sans explain → shap_values est null (comportement par défaut)."""
        model = _make_lr_iris()
        key = _inject_cache(MC_LR_MODEL, MODEL_VERSION, model)
        try:
            r = client.post(
                "/predict",
                headers=AUTH,
                json={"model_name": MC_LR_MODEL, "features": SAMPLE_INPUT},
            )
            assert r.status_code == 200
            data = r.json()
            assert data.get("shap_values") is None
        finally:
            asyncio.run(model_service._redis.delete(f"model:{key}"))
