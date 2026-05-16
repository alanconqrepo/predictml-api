"""
Tests pour le Golden Test Set.

Couvre :
- POST /models/{name}/golden-tests          (création, auth)
- POST /models/{name}/golden-tests/upload-csv (import CSV, admin)
- GET  /models/{name}/golden-tests          (liste, auth)
- DELETE /models/{name}/golden-tests/{id}   (suppression, admin)
- POST /models/{name}/{version}/golden-tests/run (exécution, admin)
- Intégration dans evaluate_auto_promotion (min_golden_test_pass_rate)
"""

import asyncio
import io
import joblib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.schemas.golden_test import GoldenTestRunResponse
from src.services.auto_promotion_service import evaluate_auto_promotion
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-gt-admin-cc33"
USER_TOKEN = "test-token-gt-user-dd44"
GT_MODEL = "gt_iris_model"
GT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iris_model() -> LogisticRegression:
    """LogisticRegression sur DataFrame iris → feature_names_in_ disponible."""
    x_train = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.3, 5.8, 4.6, 7.0],
            "sepal_width": [3.5, 3.0, 2.9, 2.7, 3.1, 3.2],
            "petal_length": [1.4, 1.4, 5.6, 5.1, 1.5, 4.7],
            "petal_width": [0.2, 0.2, 1.8, 1.9, 0.2, 1.4],
        }
    )
    y = ["setosa", "setosa", "virginica", "virginica", "setosa", "versicolor"]
    return LogisticRegression(max_iter=1000).fit(x_train, y)


def _inject_model_cache(name: str, version: str, model) -> str:
    """Injecte un modèle dans le cache FakeRedis ; retourne la clé."""
    key = f"{name}:{version}"
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
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


async def _clear_cache(key: str):
    await model_service._redis.delete(f"model:{key}")


def _admin_headers():
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def _user_headers():
    return {"Authorization": f"Bearer {USER_TOKEN}"}


# ---------------------------------------------------------------------------
# Setup DB
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="gt_admin_user",
                email="gt_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=100000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="gt_regular_user",
                email="gt_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )
        existing = await DBService.get_model_metadata(db, GT_MODEL, GT_VERSION)
        if not existing:
            await DBService.create_model_metadata(
                db,
                name=GT_MODEL,
                version=GT_VERSION,
                minio_bucket="models",
                minio_object_key=f"{GT_MODEL}/v{GT_VERSION}.pkl",
                is_active=True,
                is_production=True,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# TestGoldenTestCRUD
# ---------------------------------------------------------------------------


class TestGoldenTestCRUD:
    """Tests CRUD des endpoints golden-tests."""

    def test_list_requires_auth(self):
        r = client.get(f"/models/{GT_MODEL}/golden-tests")
        assert r.status_code in [401, 403]

    def test_list_returns_empty_initially(self):
        r = client.get(f"/models/{GT_MODEL}/golden-tests", headers=_user_headers())
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_create_requires_admin(self):
        r = client.post(
            f"/models/{GT_MODEL}/golden-tests",
            headers=_user_headers(),
            json={
                "input_features": {"sepal_length": 5.1},
                "expected_output": "setosa",
            },
        )
        assert r.status_code == 403

    def test_create_requires_auth(self):
        r = client.post(
            f"/models/{GT_MODEL}/golden-tests",
            json={
                "input_features": {"sepal_length": 5.1},
                "expected_output": "setosa",
            },
        )
        assert r.status_code in [401, 403]

    def test_create_success(self):
        r = client.post(
            f"/models/{GT_MODEL}/golden-tests",
            headers=_admin_headers(),
            json={
                "input_features": {"sepal_length": 5.1, "sepal_width": 3.5},
                "expected_output": "setosa",
                "description": "cas normal",
            },
        )
        assert r.status_code == 201
        data = r.json()
        assert data["model_name"] == GT_MODEL
        assert data["expected_output"] == "setosa"
        assert data["description"] == "cas normal"
        assert data["input_features"] == {"sepal_length": 5.1, "sepal_width": 3.5}
        assert "id" in data

    def test_list_returns_created_tests(self):
        r_create = client.post(
            f"/models/{GT_MODEL}/golden-tests",
            headers=_admin_headers(),
            json={
                "input_features": {"f1": 1.0},
                "expected_output": "class_a",
                "description": "list test",
            },
        )
        assert r_create.status_code == 201
        created_id = r_create.json()["id"]

        r_list = client.get(f"/models/{GT_MODEL}/golden-tests", headers=_user_headers())
        assert r_list.status_code == 200
        ids = [t["id"] for t in r_list.json()]
        assert created_id in ids

    def test_delete_requires_admin(self):
        r_create = client.post(
            f"/models/{GT_MODEL}/golden-tests",
            headers=_admin_headers(),
            json={"input_features": {"f1": 1.0}, "expected_output": "x"},
        )
        assert r_create.status_code == 201
        test_id = r_create.json()["id"]

        r_del = client.delete(
            f"/models/{GT_MODEL}/golden-tests/{test_id}",
            headers=_user_headers(),
        )
        assert r_del.status_code == 403

    def test_delete_success(self):
        r_create = client.post(
            f"/models/{GT_MODEL}/golden-tests",
            headers=_admin_headers(),
            json={"input_features": {"f1": 2.0}, "expected_output": "y"},
        )
        assert r_create.status_code == 201
        test_id = r_create.json()["id"]

        r_del = client.delete(
            f"/models/{GT_MODEL}/golden-tests/{test_id}",
            headers=_admin_headers(),
        )
        assert r_del.status_code == 204

        r_list = client.get(f"/models/{GT_MODEL}/golden-tests", headers=_user_headers())
        ids = [t["id"] for t in r_list.json()]
        assert test_id not in ids

    def test_delete_wrong_model_returns_404(self):
        r_create = client.post(
            f"/models/{GT_MODEL}/golden-tests",
            headers=_admin_headers(),
            json={"input_features": {"f1": 3.0}, "expected_output": "z"},
        )
        assert r_create.status_code == 201
        test_id = r_create.json()["id"]

        r_del = client.delete(
            f"/models/wrong_model_name/golden-tests/{test_id}",
            headers=_admin_headers(),
        )
        assert r_del.status_code == 404

    def test_delete_nonexistent_returns_404(self):
        r = client.delete(
            f"/models/{GT_MODEL}/golden-tests/999999",
            headers=_admin_headers(),
        )
        assert r.status_code == 404

    def test_upload_csv_requires_admin(self):
        csv_bytes = b"sepal_length,expected_output\n5.1,setosa\n"
        r = client.post(
            f"/models/{GT_MODEL}/golden-tests/upload-csv",
            headers=_user_headers(),
            files={"file": ("tests.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 403

    def test_upload_csv_success(self):
        csv_content = (
            "sepal_length,sepal_width,petal_length,petal_width,expected_output,description\n"
            "5.1,3.5,1.4,0.2,setosa,cas normal\n"
            "6.3,2.9,5.6,1.8,virginica,\n"
            "5.8,2.7,5.1,1.9,virginica,cas limite\n"
        )
        r = client.post(
            f"/models/{GT_MODEL}/golden-tests/upload-csv",
            headers=_admin_headers(),
            files={"file": ("batch.csv", csv_content.encode(), "text/csv")},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["created"] == 3
        assert data["errors"] == []

    def test_upload_csv_missing_expected_output_column(self):
        csv_content = "sepal_length,sepal_width\n5.1,3.5\n"
        r = client.post(
            f"/models/{GT_MODEL}/golden-tests/upload-csv",
            headers=_admin_headers(),
            files={"file": ("bad.csv", csv_content.encode(), "text/csv")},
        )
        assert r.status_code == 422

    def test_min_golden_test_pass_rate_validation(self):
        """PATCH policy with pass_rate > 1.0 → 422."""
        r = client.patch(
            f"/models/{GT_MODEL}/policy",
            headers=_admin_headers(),
            json={"min_golden_test_pass_rate": 1.5, "auto_promote": False},
        )
        assert r.status_code == 422

    def test_min_golden_test_pass_rate_stored_in_policy(self):
        """PATCH policy with valid pass_rate → stored correctly."""
        r = client.patch(
            f"/models/{GT_MODEL}/policy",
            headers=_admin_headers(),
            json={"min_golden_test_pass_rate": 0.9, "auto_promote": False},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["promotion_policy"]["min_golden_test_pass_rate"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestGoldenTestRun
# ---------------------------------------------------------------------------


class TestGoldenTestRun:
    """Tests de POST /models/{name}/{version}/golden-tests/run."""

    GT_RUN_MODEL = "gt_run_model"
    GT_RUN_VERSION = "2.0.0"

    @classmethod
    def setup_class(cls):
        async def _init():
            async with _TestSessionLocal() as db:
                if not await DBService.get_model_metadata(db, cls.GT_RUN_MODEL, cls.GT_RUN_VERSION):
                    await DBService.create_model_metadata(
                        db,
                        name=cls.GT_RUN_MODEL,
                        version=cls.GT_RUN_VERSION,
                        minio_bucket="models",
                        minio_object_key=f"{cls.GT_RUN_MODEL}/v{cls.GT_RUN_VERSION}.pkl",
                        is_active=True,
                        is_production=True,
                    )

        asyncio.run(_init())

    def test_run_requires_admin(self):
        r = client.post(
            f"/models/{self.GT_RUN_MODEL}/{self.GT_RUN_VERSION}/golden-tests/run",
            headers=_user_headers(),
        )
        assert r.status_code == 403

    def test_run_no_tests_returns_empty(self):
        model = _make_iris_model()
        key = _inject_model_cache(self.GT_RUN_MODEL, self.GT_RUN_VERSION, model)
        try:
            r = client.post(
                f"/models/{self.GT_RUN_MODEL}/{self.GT_RUN_VERSION}/golden-tests/run",
                headers=_admin_headers(),
            )
            assert r.status_code == 200
            data = r.json()
            assert data["total_tests"] == 0
            assert data["passed"] == 0
            assert data["failed"] == 0
            assert data["details"] == []
        finally:
            asyncio.run(_clear_cache(key))

    def test_run_all_pass(self):
        model = _make_iris_model()
        key = _inject_model_cache(self.GT_RUN_MODEL, self.GT_RUN_VERSION, model)

        # Déterminer le résultat réel du modèle pour construire des tests qui passent
        import numpy as np

        x = np.array(
            [[5.1, 3.5, 1.4, 0.2]],
            dtype=object,
        )
        expected_pred = str(model.predict(x)[0])

        r_create = client.post(
            f"/models/{self.GT_RUN_MODEL}/golden-tests",
            headers=_admin_headers(),
            json={
                "input_features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
                "expected_output": expected_pred,
                "description": "cas setosa simple",
            },
        )
        assert r_create.status_code == 201
        created_id = r_create.json()["id"]

        try:
            r = client.post(
                f"/models/{self.GT_RUN_MODEL}/{self.GT_RUN_VERSION}/golden-tests/run",
                headers=_admin_headers(),
            )
            assert r.status_code == 200
            data = r.json()
            assert data["total_tests"] >= 1
            assert data["pass_rate"] == pytest.approx(1.0)
            assert data["failed"] == 0
            detail_ids = [d["test_id"] for d in data["details"]]
            assert created_id in detail_ids
        finally:
            asyncio.run(_clear_cache(key))
            client.delete(
                f"/models/{self.GT_RUN_MODEL}/golden-tests/{created_id}",
                headers=_admin_headers(),
            )

    def test_run_some_fail(self):
        model = _make_iris_model()
        key = _inject_model_cache(self.GT_RUN_MODEL, self.GT_RUN_VERSION, model)

        r_create = client.post(
            f"/models/{self.GT_RUN_MODEL}/golden-tests",
            headers=_admin_headers(),
            json={
                "input_features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
                "expected_output": "wrong_class_zzz",
                "description": "intentionally wrong",
            },
        )
        assert r_create.status_code == 201
        created_id = r_create.json()["id"]

        try:
            r = client.post(
                f"/models/{self.GT_RUN_MODEL}/{self.GT_RUN_VERSION}/golden-tests/run",
                headers=_admin_headers(),
            )
            assert r.status_code == 200
            data = r.json()
            # Au moins ce test doit échouer
            failing = [d for d in data["details"] if d["test_id"] == created_id]
            assert len(failing) == 1
            assert failing[0]["passed"] is False
            assert failing[0]["expected"] == "wrong_class_zzz"
        finally:
            asyncio.run(_clear_cache(key))
            client.delete(
                f"/models/{self.GT_RUN_MODEL}/golden-tests/{created_id}",
                headers=_admin_headers(),
            )

    def test_run_model_not_in_db_returns_404(self):
        r = client.post(
            "/models/nonexistent_model_xyz/9.9.9/golden-tests/run",
            headers=_admin_headers(),
        )
        assert r.status_code == 404

    def test_run_response_structure(self):
        model = _make_iris_model()
        key = _inject_model_cache(self.GT_RUN_MODEL, self.GT_RUN_VERSION, model)
        try:
            r = client.post(
                f"/models/{self.GT_RUN_MODEL}/{self.GT_RUN_VERSION}/golden-tests/run",
                headers=_admin_headers(),
            )
            assert r.status_code == 200
            data = r.json()
            assert "model_name" in data
            assert "version" in data
            assert "total_tests" in data
            assert "passed" in data
            assert "failed" in data
            assert "pass_rate" in data
            assert "details" in data
            assert data["model_name"] == self.GT_RUN_MODEL
            assert data["version"] == self.GT_RUN_VERSION
        finally:
            asyncio.run(_clear_cache(key))


# ---------------------------------------------------------------------------
# TestGoldenTestAutoPromotion
# ---------------------------------------------------------------------------


class TestGoldenTestAutoPromotion:
    """Tests de l'intégration golden tests dans evaluate_auto_promotion."""

    async def _run_evaluate(self, policy: dict, version: str = "1.0.0"):
        async with _TestSessionLocal() as db:
            return await evaluate_auto_promotion(db, GT_MODEL, policy, version=version)

    def test_no_golden_test_field_skips_check(self):
        """Policy sans min_golden_test_pass_rate → pas de vérification des golden tests."""
        # min_sample_validation=0 pour éviter le blocage "échantillons insuffisants"
        policy = {"min_sample_validation": 0}
        should, reason = asyncio.run(self._run_evaluate(policy))
        # Sans min_golden_test_pass_rate, le résultat dépend d'autres critères
        assert isinstance(should, bool)
        assert "régression" not in reason

    def test_golden_tests_no_version_skips_check(self):
        """Sans version → check golden tests ignoré même si policy définie."""
        policy = {"min_golden_test_pass_rate": 1.0, "auto_promote": True, "min_sample_validation": 1}

        async def _run():
            async with _TestSessionLocal() as db:
                # version=None → le check golden tests est ignoré
                return await evaluate_auto_promotion(db, GT_MODEL, policy, version=None)

        should, reason = asyncio.run(_run())
        # Pas de vérification golden tests, résultat déterminé par autres critères
        assert isinstance(should, bool)

    def test_golden_tests_pass_does_not_block(self):
        """pass_rate >= seuil → promotion non bloquée par les golden tests."""
        # min_sample_validation=0 pour bypasser le check "échantillons insuffisants"
        policy = {"min_golden_test_pass_rate": 0.8, "min_sample_validation": 0}
        mock_result = GoldenTestRunResponse(
            model_name=GT_MODEL,
            version="1.0.0",
            total_tests=10,
            passed=9,
            failed=1,
            pass_rate=0.9,
            details=[],
        )
        with patch(
            "src.services.golden_test_service.GoldenTestService.run_tests",
            new=AsyncMock(return_value=mock_result),
        ):
            should, reason = asyncio.run(self._run_evaluate(policy))
        # pass_rate=0.9 >= 0.8 → golden tests ne bloquent pas
        assert "régression" not in reason

    def test_golden_tests_fail_blocks_promotion(self):
        """pass_rate < seuil → auto_promote_reason mentionne les golden tests."""
        # min_sample_validation=0 pour bypasser le check "échantillons insuffisants"
        policy = {"min_golden_test_pass_rate": 0.95, "min_sample_validation": 0}
        mock_result = GoldenTestRunResponse(
            model_name=GT_MODEL,
            version="1.0.0",
            total_tests=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
            details=[],
        )
        with patch(
            "src.services.golden_test_service.GoldenTestService.run_tests",
            new=AsyncMock(return_value=mock_result),
        ):
            should, reason = asyncio.run(self._run_evaluate(policy))
        assert should is False
        assert "régression" in reason
        assert "80.00%" in reason
        assert "95.00%" in reason

    def test_golden_tests_zero_total_does_not_block(self):
        """total_tests=0 → check ignoré, golden tests ne bloquent pas."""
        # min_sample_validation=0 pour bypasser le check "échantillons insuffisants"
        policy = {"min_golden_test_pass_rate": 1.0, "min_sample_validation": 0}
        mock_result = GoldenTestRunResponse(
            model_name=GT_MODEL,
            version="1.0.0",
            total_tests=0,
            passed=0,
            failed=0,
            pass_rate=1.0,
            details=[],
        )
        with patch(
            "src.services.golden_test_service.GoldenTestService.run_tests",
            new=AsyncMock(return_value=mock_result),
        ):
            should, reason = asyncio.run(self._run_evaluate(policy))
        # total_tests=0 → golden test check ignoré, ne bloque pas
        assert "régression" not in reason
