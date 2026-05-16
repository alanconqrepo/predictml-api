"""
Tests pour POST /models/{name}/{version}/validate-input et ?strict_validation sur /predict.

Stratégie :
  - Injecter les modèles dans le cache Redis via model_service._redis
  - Créer les entrées ModelMetadata en DB dans _setup()
  - Nettoyer le cache dans try/finally pour éviter les interférences
  - SQLite en mémoire + FakeRedis — aucun Docker requis
"""

import asyncio
import io
import joblib
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-input-validation-v1"
IV_MODEL = "iv_model_features"  # modèle avec feature_names_in_
IV_MODEL_NOFEAT = "iv_model_nofeat"  # modèle sans feature_names_in_ ni baseline
IV_MODEL_BASELINE = "iv_model_base"  # modèle sans feature_names_in_, avec feature_baseline
MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers — construction de modèles
# ---------------------------------------------------------------------------


def _make_model_with_features() -> LogisticRegression:
    """LogisticRegression sur DataFrame → feature_names_in_ disponible."""
    x_train = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.3, 5.8],
            "sepal_width": [3.5, 3.0, 2.9, 2.7],
            "petal_length": [1.4, 1.4, 5.6, 5.1],
            "petal_width": [0.2, 0.2, 1.8, 1.9],
        }
    )
    y = [0, 0, 1, 1]
    return LogisticRegression(max_iter=1000).fit(x_train, y)


def _make_model_no_feature_names() -> LogisticRegression:
    """LogisticRegression sur numpy array → PAS de feature_names_in_."""
    x_arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(x_arr, y)


def _inject_cache(model_name: str, version: str, model, confidence_threshold=None) -> str:
    """Injecte un modèle dans le cache Redis ; retourne la clé pour le nettoyage."""
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=confidence_threshold,
            webhook_url=None,
            feature_baseline=None,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


async def _delete_cache(key: str):
    await model_service._redis.delete(f"model:{key}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_input_validation_user",
                email="test_iv@test.com",
                api_token=TEST_TOKEN,
                role="user",
                rate_limit=10000,
            )

        for name, baseline in [
            (IV_MODEL, None),
            (IV_MODEL_NOFEAT, None),
            (
                IV_MODEL_BASELINE,
                {
                    "sepal_length": {"mean": 5.5, "std": 0.5, "min": 4.3, "max": 7.9},
                    "sepal_width": {"mean": 3.0, "std": 0.4, "min": 2.0, "max": 4.4},
                },
            ),
        ]:
            existing = await DBService.get_model_metadata(db, name, MODEL_VERSION)
            if not existing:
                meta = await DBService.create_model_metadata(
                    db,
                    name=name,
                    version=MODEL_VERSION,
                    minio_bucket="models",
                    minio_object_key=f"{name}/v{MODEL_VERSION}.pkl",
                    is_active=True,
                    is_production=True,
                )
                if baseline:
                    meta.feature_baseline = baseline
                    await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Tests — POST /models/{name}/{version}/validate-input
# ---------------------------------------------------------------------------


class TestValidateInput:

    def test_model_not_found_returns_404(self):
        """Modèle inexistant → 404."""
        response = client.post(
            "/models/inexistant_model/9.9.9/validate-input",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"sepal_length": 5.1},
        )
        assert response.status_code == 404

    def test_requires_auth(self):
        """Sans token → 401/403."""
        response = client.post(
            f"/models/{IV_MODEL}/{MODEL_VERSION}/validate-input",
            json={"sepal_length": 5.1},
        )
        assert response.status_code in [401, 403]

    def test_valid_input_all_features_present(self):
        """Toutes les features attendues présentes → valid=true, pas d'erreur."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                f"/models/{IV_MODEL}/{MODEL_VERSION}/validate-input",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["errors"] == []
            assert data["warnings"] == []
            assert sorted(data["expected_features"]) == [
                "petal_length",
                "petal_width",
                "sepal_length",
                "sepal_width",
            ]
        finally:
            asyncio.run(_delete_cache(key))

    def test_missing_feature_detected(self):
        """Feature manquante → valid=false, erreur missing_feature."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                f"/models/{IV_MODEL}/{MODEL_VERSION}/validate-input",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "sepal_length": 5.1,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                    # sepal_width est manquant
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            error_types = {e["type"] for e in data["errors"]}
            error_features = {e["feature"] for e in data["errors"]}
            assert "missing_feature" in error_types
            assert "sepal_width" in error_features
        finally:
            asyncio.run(_delete_cache(key))

    def test_unexpected_feature_detected(self):
        """Feature inattendue → valid=false, erreur unexpected_feature."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                f"/models/{IV_MODEL}/{MODEL_VERSION}/validate-input",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                    "petal_width_squared": 0.04,  # feature inconnue
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            error_types = {e["type"] for e in data["errors"]}
            error_features = {e["feature"] for e in data["errors"]}
            assert "unexpected_feature" in error_types
            assert "petal_width_squared" in error_features
        finally:
            asyncio.run(_delete_cache(key))

    def test_multiple_errors_missing_and_unexpected(self):
        """Features manquantes ET inattendues → plusieurs erreurs."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                f"/models/{IV_MODEL}/{MODEL_VERSION}/validate-input",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "sepal_length": 5.1,
                    # sepal_width manquant
                    "petal_length": 1.4,
                    # petal_width manquant
                    "extra_feature": 99.0,  # inattendue
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False

            types = [e["type"] for e in data["errors"]]
            features = [e["feature"] for e in data["errors"]]
            assert "missing_feature" in types
            assert "unexpected_feature" in types
            assert "sepal_width" in features
            assert "petal_width" in features
            assert "extra_feature" in features
        finally:
            asyncio.run(_delete_cache(key))

    def test_type_coercion_warning_for_string_float(self):
        """Valeur string convertible en float → avertissement type_coercion."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                f"/models/{IV_MODEL}/{MODEL_VERSION}/validate-input",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "sepal_length": "5.1",  # string → coercion warning
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True  # warning n'invalide pas
            assert len(data["warnings"]) == 1
            w = data["warnings"][0]
            assert w["type"] == "type_coercion"
            assert w["feature"] == "sepal_length"
            assert w["from_type"] == "string"
            assert w["to_type"] == "float"
        finally:
            asyncio.run(_delete_cache(key))

    def test_no_schema_returns_valid_true_with_null_features(self):
        """Modèle sans feature_names_in_ ni baseline → valid=true, expected_features=null."""
        model = _make_model_no_feature_names()
        key = _inject_cache(IV_MODEL_NOFEAT, MODEL_VERSION, model)
        try:
            response = client.post(
                f"/models/{IV_MODEL_NOFEAT}/{MODEL_VERSION}/validate-input",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={"f1": 1.0, "f2": 2.0},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["errors"] == []
            assert data["expected_features"] is None
        finally:
            asyncio.run(_delete_cache(key))

    def test_fallback_to_feature_baseline_when_model_not_loaded(self):
        """Sans modèle en cache, fallback sur feature_baseline de la DB."""
        # IV_MODEL_BASELINE n'a pas de modèle en cache → fallback baseline
        response = client.post(
            f"/models/{IV_MODEL_BASELINE}/{MODEL_VERSION}/validate-input",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"sepal_length": 5.5, "sepal_width": 3.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert sorted(data["expected_features"]) == ["sepal_length", "sepal_width"]

    def test_fallback_baseline_detects_missing(self):
        """Fallback sur baseline : feature manquante → erreur."""
        response = client.post(
            f"/models/{IV_MODEL_BASELINE}/{MODEL_VERSION}/validate-input",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"sepal_length": 5.5},  # sepal_width manquant
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        features = [e["feature"] for e in data["errors"]]
        assert "sepal_width" in features

    def test_expected_features_sorted_in_response(self):
        """expected_features retournées dans l'ordre alphabétique."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                f"/models/{IV_MODEL}/{MODEL_VERSION}/validate-input",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
            )
            data = response.json()
            feats = data["expected_features"]
            assert feats == sorted(feats)
        finally:
            asyncio.run(_delete_cache(key))


# ---------------------------------------------------------------------------
# Tests — POST /predict?strict_validation=true
# ---------------------------------------------------------------------------


class TestStrictValidationPredict:

    def test_strict_validation_false_allows_unexpected_features(self):
        """strict_validation=false (défaut) : features inattendues ignorées silencieusement."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                "/predict",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "model_name": IV_MODEL,
                    "model_version": MODEL_VERSION,
                    "features": {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                        "extra_col": 999.0,  # inattendue — pas bloquante sans strict
                    },
                },
            )
            assert response.status_code == 200
        finally:
            asyncio.run(_delete_cache(key))

    def test_strict_validation_true_rejects_unexpected_features(self):
        """strict_validation=true : features inattendues → 422."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                "/predict?strict_validation=true",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "model_name": IV_MODEL,
                    "model_version": MODEL_VERSION,
                    "features": {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                        "extra_col": 999.0,  # inattendue → bloquante en mode strict
                    },
                },
            )
            assert response.status_code == 422
            detail = response.json()["detail"]
            assert detail["valid"] is False
            error_types = {e["type"] for e in detail["errors"]}
            assert "unexpected_feature" in error_types
        finally:
            asyncio.run(_delete_cache(key))

    def test_strict_validation_true_accepts_exact_match(self):
        """strict_validation=true : toutes les features exactement correspondantes → 200."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                "/predict?strict_validation=true",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "model_name": IV_MODEL,
                    "model_version": MODEL_VERSION,
                    "features": {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    },
                },
            )
            assert response.status_code == 200
        finally:
            asyncio.run(_delete_cache(key))

    def test_strict_validation_true_still_rejects_missing_features(self):
        """strict_validation=true : features manquantes toujours rejetées (comportement de base)."""
        model = _make_model_with_features()
        key = _inject_cache(IV_MODEL, MODEL_VERSION, model)
        try:
            response = client.post(
                "/predict?strict_validation=true",
                headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                json={
                    "model_name": IV_MODEL,
                    "model_version": MODEL_VERSION,
                    "features": {
                        "sepal_length": 5.1,
                        # sepal_width, petal_length, petal_width manquants
                    },
                },
            )
            assert response.status_code == 422
        finally:
            asyncio.run(_delete_cache(key))


# ---------------------------------------------------------------------------
# Tests unitaires — service de validation
# ---------------------------------------------------------------------------


class TestValidationService:

    def test_no_errors_no_warnings_for_exact_match(self):
        from src.services.input_validation_service import validate_input_features

        errors, warnings = validate_input_features({"a": 1.0, "b": 2.0}, ["a", "b"])
        assert errors == []
        assert warnings == []

    def test_detects_missing_feature(self):
        from src.services.input_validation_service import validate_input_features

        errors, _ = validate_input_features({"a": 1.0}, ["a", "b"])
        assert any(e.type == "missing_feature" and e.feature == "b" for e in errors)

    def test_detects_unexpected_feature(self):
        from src.services.input_validation_service import validate_input_features

        errors, _ = validate_input_features({"a": 1.0, "c": 3.0}, ["a"])
        assert any(e.type == "unexpected_feature" and e.feature == "c" for e in errors)

    def test_detects_type_coercion_warning(self):
        from src.services.input_validation_service import validate_input_features

        _, warnings = validate_input_features({"a": "3.14"}, ["a"])
        assert any(w.type == "type_coercion" and w.feature == "a" for w in warnings)

    def test_non_numeric_string_no_warning(self):
        from src.services.input_validation_service import validate_input_features

        _, warnings = validate_input_features({"a": "not_a_number"}, ["a"])
        assert warnings == []

    def test_resolve_expected_features_from_model(self):
        from src.services.input_validation_service import resolve_expected_features

        x_train = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})
        model = LogisticRegression().fit(x_train, [0, 1])
        result = resolve_expected_features(model, None)
        assert result == ["x1", "x2"]

    def test_resolve_expected_features_from_baseline(self):
        from src.services.input_validation_service import resolve_expected_features

        class NoFeatModel:
            pass

        baseline = {"f1": {"mean": 1.0}, "f2": {"mean": 2.0}}
        result = resolve_expected_features(NoFeatModel(), baseline)
        assert sorted(result) == ["f1", "f2"]

    def test_resolve_expected_features_none_when_no_source(self):
        from src.services.input_validation_service import resolve_expected_features

        class NoFeatModel:
            pass

        result = resolve_expected_features(NoFeatModel(), None)
        assert result is None
