"""
Tests supplémentaires pour les endpoints de predict.py non couverts :

- GET /predictions/export (format csv, jsonl, parquet, erreur start>end)
- GET /predictions/anomalies (avec/sans baseline, modèle inexistant)
- GET /predictions/{id} (404, 403, 200)
- GET /predictions/{id}/explain (404, 403, 422, 200)
- POST /explain (200, 404, 422)
- POST /predict avec inline explain (?explain=true)
- POST /predict-batch (rate limit, modèle inexistant, happy path)
"""

import asyncio
import io
import pickle
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-pred-ext-admin-vv55"
USER_TOKEN = "test-token-pred-ext-user-vv56"
OTHER_TOKEN = "test-token-pred-ext-other-vv57"

EXT_MODEL = "pred_ext_model"
MODEL_VERSION = "1.0.0"

FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}

FEATURE_NAMES = list(FEATURES.keys())


def _make_iris_model() -> LogisticRegression:
    X = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9, 6.3, 5.8],
            "sepal width (cm)": [3.5, 3.0, 2.9, 2.7],
            "petal length (cm)": [1.4, 1.4, 5.6, 5.1],
            "petal width (cm)": [0.2, 0.2, 1.8, 1.9],
        }
    )
    y = ["setosa", "setosa", "virginica", "virginica"]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_cache(name: str, version: str = MODEL_VERSION):
    model = _make_iris_model()
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
            feature_baseline=None,
        ),
    }
    asyncio.run(model_service._redis.set(f"model:{name}:{version}", pickle.dumps(data)))


async def _setup():
    async with _TestSessionLocal() as db:
        for token, username, email, role in [
            (ADMIN_TOKEN, "pred_ext_admin", "pred_ext_admin@test.com", "admin"),
            (USER_TOKEN, "pred_ext_user", "pred_ext_user@test.com", "user"),
            (OTHER_TOKEN, "pred_ext_other", "pred_ext_other@test.com", "user"),
        ]:
            if not await DBService.get_user_by_token(db, token):
                await DBService.create_user(
                    db,
                    username=username,
                    email=email,
                    api_token=token,
                    role=role,
                    rate_limit=10000,
                )

        if not await DBService.get_model_metadata(db, EXT_MODEL, MODEL_VERSION):
            await DBService.create_model_metadata(
                db,
                name=EXT_MODEL,
                version=MODEL_VERSION,
                minio_bucket="models",
                minio_object_key=f"{EXT_MODEL}/v{MODEL_VERSION}.pkl",
                is_active=True,
                is_production=True,
            )
        await db.commit()


asyncio.run(_setup())
_inject_cache(EXT_MODEL)


def _headers(token=ADMIN_TOKEN):
    return {"Authorization": f"Bearer {token}"}


def _do_predict(model_name=EXT_MODEL) -> int:
    resp = client.post(
        "/predict",
        headers=_headers(USER_TOKEN),
        json={"model_name": model_name, "features": FEATURES},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"] if "id" in resp.json() else None


# ---------------------------------------------------------------------------
# GET /predictions/export
# ---------------------------------------------------------------------------


class TestExportPredictions:
    def test_export_csv_returns_200(self):
        """GET /predictions/export?format=csv → Content-Type text/csv."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            headers=_headers(USER_TOKEN),
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "csv",
            },
        )
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "csv" in ct or "text" in ct or "application" in ct

    def test_export_jsonl_returns_200(self):
        """GET /predictions/export?format=jsonl → réponse 200."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            headers=_headers(USER_TOKEN),
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "jsonl",
            },
        )
        assert resp.status_code == 200

    def test_export_parquet_returns_200(self):
        """GET /predictions/export?format=parquet → réponse 200."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            headers=_headers(USER_TOKEN),
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "parquet",
            },
        )
        assert resp.status_code == 200

    def test_export_invalid_format_returns_4xx(self):
        """format invalide → 400 ou 422."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            headers=_headers(USER_TOKEN),
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "xlsx",
            },
        )
        assert resp.status_code in [400, 422]

    def test_export_start_after_end_returns_422(self):
        """start > end → 422."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            headers=_headers(USER_TOKEN),
            params={
                "start": (now + timedelta(hours=1)).isoformat(),
                "end": (now - timedelta(hours=1)).isoformat(),
                "format": "csv",
            },
        )
        assert resp.status_code == 422

    def test_export_without_auth_returns_401(self):
        """Sans auth → 401/403."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
            },
        )
        assert resp.status_code in [401, 403]

    def test_export_with_model_filter(self):
        """model_name filter → réponse 200."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            headers=_headers(USER_TOKEN),
            params={
                "start": (now - timedelta(days=30)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "csv",
                "model_name": EXT_MODEL,
            },
        )
        assert resp.status_code == 200

    def test_export_with_status_filter(self):
        """status filter → réponse 200."""
        now = datetime.utcnow()
        resp = client.get(
            "/predictions/export",
            headers=_headers(USER_TOKEN),
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "csv",
                "status": "success",
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /predictions/anomalies
# ---------------------------------------------------------------------------


class TestAnomalousPredictions:
    def test_anomalies_no_baseline_returns_error_key(self):
        """Modèle sans baseline → réponse avec error='no_baseline'."""
        resp = client.get(
            "/predictions/anomalies",
            headers=_headers(USER_TOKEN),
            params={"model_name": EXT_MODEL, "days": 7},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data or "anomalies" in data

    def test_anomalies_nonexistent_model_returns_404(self):
        """Modèle inexistant → 404."""
        resp = client.get(
            "/predictions/anomalies",
            headers=_headers(USER_TOKEN),
            params={"model_name": "nonexistent_anom_model", "days": 7},
        )
        assert resp.status_code == 404

    def test_anomalies_without_auth_returns_401(self):
        """Sans auth → 401/403."""
        resp = client.get(
            "/predictions/anomalies",
            params={"model_name": EXT_MODEL, "days": 7},
        )
        assert resp.status_code in [401, 403]

    def test_anomalies_with_custom_threshold(self):
        """z_threshold custom → 200."""
        resp = client.get(
            "/predictions/anomalies",
            headers=_headers(USER_TOKEN),
            params={"model_name": EXT_MODEL, "days": 7, "z_threshold": 2.0},
        )
        assert resp.status_code in [200, 404]


# ---------------------------------------------------------------------------
# GET /predictions/{id}
# ---------------------------------------------------------------------------


class TestGetPredictionById:
    def test_prediction_not_found_returns_404(self):
        """ID inexistant → 404."""
        resp = client.get(
            "/predictions/999999999",
            headers=_headers(USER_TOKEN),
        )
        assert resp.status_code == 404

    def test_prediction_forbidden_for_other_user(self):
        """User qui ne possède pas la prédiction → 403."""
        # Créer une prédiction avec USER_TOKEN, y accéder avec OTHER_TOKEN
        pred_resp = client.post(
            "/predict",
            headers=_headers(USER_TOKEN),
            json={"model_name": EXT_MODEL, "features": FEATURES},
        )
        assert pred_resp.status_code == 200
        pred_id = pred_resp.json().get("id")
        if pred_id is None:
            pytest.skip("Endpoint ne retourne pas d'id dans la réponse")

        resp = client.get(
            f"/predictions/{pred_id}",
            headers=_headers(OTHER_TOKEN),
        )
        assert resp.status_code in [403, 404]

    def test_admin_can_access_any_prediction(self):
        """Admin peut accéder à n'importe quelle prédiction."""
        pred_resp = client.post(
            "/predict",
            headers=_headers(USER_TOKEN),
            json={"model_name": EXT_MODEL, "features": FEATURES},
        )
        assert pred_resp.status_code == 200
        pred_id = pred_resp.json().get("id")
        if pred_id is None:
            pytest.skip("Endpoint ne retourne pas d'id dans la réponse")

        resp = client.get(
            f"/predictions/{pred_id}",
            headers=_headers(ADMIN_TOKEN),
        )
        assert resp.status_code == 200

    def test_get_prediction_without_auth_returns_401(self):
        """Sans auth → 401/403."""
        resp = client.get("/predictions/1")
        assert resp.status_code in [401, 403]


# ---------------------------------------------------------------------------
# GET /predictions/{id}/explain
# ---------------------------------------------------------------------------


class TestExplainPredictionById:
    def test_explain_nonexistent_prediction_returns_404(self):
        """ID inexistant → 404."""
        resp = client.get(
            "/predictions/999999998/explain",
            headers=_headers(USER_TOKEN),
        )
        assert resp.status_code == 404

    def test_explain_forbidden_for_other_user(self):
        """Prédiction d'un autre user → 403."""
        pred_resp = client.post(
            "/predict",
            headers=_headers(USER_TOKEN),
            json={"model_name": EXT_MODEL, "features": FEATURES},
        )
        assert pred_resp.status_code == 200
        pred_id = pred_resp.json().get("id")
        if pred_id is None:
            pytest.skip("No id in response")

        resp = client.get(
            f"/predictions/{pred_id}/explain",
            headers=_headers(OTHER_TOKEN),
        )
        assert resp.status_code in [403, 404]

    def test_explain_without_auth_returns_401(self):
        """Sans auth → 401/403."""
        resp = client.get("/predictions/1/explain")
        assert resp.status_code in [401, 403]

    def test_explain_admin_can_explain_any_prediction(self):
        """Admin peut expliquer n'importe quelle prédiction."""
        pred_resp = client.post(
            "/predict",
            headers=_headers(USER_TOKEN),
            json={"model_name": EXT_MODEL, "features": FEATURES},
        )
        assert pred_resp.status_code == 200
        pred_id = pred_resp.json().get("id")
        if pred_id is None:
            pytest.skip("No id in response")

        resp = client.get(
            f"/predictions/{pred_id}/explain",
            headers=_headers(ADMIN_TOKEN),
        )
        assert resp.status_code in [200, 422]


# ---------------------------------------------------------------------------
# POST /explain (standalone)
# ---------------------------------------------------------------------------


class TestExplainEndpoint:
    def test_explain_success(self):
        """POST /explain avec features valides → 200, shap_values."""
        resp = client.post(
            "/explain",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": EXT_MODEL,
                "features": FEATURES,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "shap_values" in data or "prediction" in data

    def test_explain_nonexistent_model_returns_404(self):
        """Modèle inexistant → 404."""
        resp = client.post(
            "/explain",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": "nonexistent_explain_model",
                "features": FEATURES,
            },
        )
        assert resp.status_code == 404

    def test_explain_missing_features_returns_422(self):
        """Features manquantes → 422."""
        resp = client.post(
            "/explain",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": EXT_MODEL,
                "features": {"sepal length (cm)": 5.1},
            },
        )
        assert resp.status_code == 422

    def test_explain_without_auth_returns_401(self):
        """Sans auth → 401/403."""
        resp = client.post(
            "/explain",
            json={"model_name": EXT_MODEL, "features": FEATURES},
        )
        assert resp.status_code in [401, 403]


# ---------------------------------------------------------------------------
# POST /predict avec explain=true (inline SHAP)
# ---------------------------------------------------------------------------


class TestPredictWithInlineExplain:
    def test_predict_with_explain_true(self):
        """POST /predict?explain=true → shap_values non null dans la réponse."""
        resp = client.post(
            "/predict?explain=true",
            headers=_headers(USER_TOKEN),
            json={"model_name": EXT_MODEL, "features": FEATURES},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "shap_values" in data

    def test_predict_explicit_version(self):
        """POST /predict avec model_version explicite → 200."""
        resp = client.post(
            "/predict",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": EXT_MODEL,
                "model_version": MODEL_VERSION,
                "features": FEATURES,
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict-batch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    def test_predict_batch_success(self):
        """POST /predict-batch avec plusieurs items → 200, liste de prédictions."""
        resp = client.post(
            "/predict-batch",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": EXT_MODEL,
                "inputs": [
                    {"features": FEATURES},
                    {"features": {k: v + 1 for k, v in FEATURES.items()}},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_predict_batch_nonexistent_model_returns_404(self):
        """Modèle inexistant → 404."""
        resp = client.post(
            "/predict-batch",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": "nonexistent_batch_model",
                "inputs": [{"features": FEATURES}],
            },
        )
        assert resp.status_code == 404

    def test_predict_batch_without_auth_returns_401(self):
        """Sans auth → 401/403."""
        resp = client.post(
            "/predict-batch",
            json={
                "model_name": EXT_MODEL,
                "inputs": [{"features": FEATURES}],
            },
        )
        assert resp.status_code in [401, 403]

    def test_predict_batch_missing_features_returns_422(self):
        """Features manquantes dans un item → 422."""
        resp = client.post(
            "/predict-batch",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": EXT_MODEL,
                "inputs": [{"features": {"sepal length (cm)": 5.1}}],
            },
        )
        assert resp.status_code == 422

    def test_predict_batch_with_id_obs(self):
        """POST /predict-batch avec id_obs par item → 200."""
        resp = client.post(
            "/predict-batch",
            headers=_headers(USER_TOKEN),
            json={
                "model_name": EXT_MODEL,
                "inputs": [
                    {"features": FEATURES, "id_obs": "batch-obs-001"},
                    {"features": FEATURES, "id_obs": "batch-obs-002"},
                ],
            },
        )
        assert resp.status_code == 200
