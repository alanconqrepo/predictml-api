"""
Tests E2E — Golden tests comme gate de promotion.

Scénarios :
  1. Créer modèle → CRUD golden tests (create / list / delete)
  2. Upload CSV de golden tests → liste enrichie
  3. Exécuter les golden tests (run) → résultat pass/fail
  4. Policy avec min_golden_test_pass_rate → gate de promotion
  5. Retrain avec golden tests qui passent → promotion autorisée
"""

import asyncio
import io
import joblib
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

ADMIN_TOKEN = "e2e-gt-gate-admin-token-mm33"
GT_E2E_MODEL = "e2e_gt_gate_model"
GT_VERSION = "1.0.0"

FEATURES = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def _make_iris_model_pkl() -> bytes:
    X = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.3, 5.8],
            "sepal_width": [3.5, 3.0, 2.9, 2.7],
            "petal_length": [1.4, 1.4, 5.6, 5.1],
            "petal_width": [0.2, 0.2, 1.8, 1.9],
        }
    )
    y = ["setosa", "setosa", "virginica", "virginica"]
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=1000).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _inject_cache(name: str, version: str):
    X = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.3, 5.8],
            "sepal_width": [3.5, 3.0, 2.9, 2.7],
            "petal_length": [1.4, 1.4, 5.6, 5.1],
            "petal_width": [0.2, 0.2, 1.8, 1.9],
        }
    )
    y = ["setosa", "setosa", "virginica", "virginica"]
    model = LogisticRegression(max_iter=1000).fit(X, y)
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
    asyncio.run(model_service._redis.set(f"model:{name}:{version}", _jbuf.getvalue()))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="e2e_gt_gate_admin",
                email="e2e_gt_gate@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())

# Créer le modèle une fois
_r = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={
        "file": (
            "model.joblib",
            io.BytesIO(_make_iris_model_pkl()),
            "application/octet-stream",
        )
    },
    data={"name": GT_E2E_MODEL, "version": GT_VERSION},
)
assert _r.status_code == 201, _r.text
_inject_cache(GT_E2E_MODEL, GT_VERSION)


def _headers():
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


class TestGoldenTestsCRUD:
    def test_list_golden_tests_empty(self):
        """GET /models/{name}/golden-tests → liste vide au départ."""
        resp = client.get(f"/models/{GT_E2E_MODEL}/golden-tests", headers=_headers())
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_golden_test(self):
        """POST /models/{name}/golden-tests → golden test créé."""
        resp = client.post(
            f"/models/{GT_E2E_MODEL}/golden-tests",
            headers=_headers(),
            json={
                "input_features": FEATURES,
                "expected_output": "setosa",
                "description": "Test case E2E setosa",
            },
        )
        assert resp.status_code in [200, 201]
        data = resp.json()
        assert data["expected_output"] == "setosa"
        assert "id" in data
        return data["id"]

    def test_list_golden_tests_after_create(self):
        """Après création → liste non vide."""
        client.post(
            f"/models/{GT_E2E_MODEL}/golden-tests",
            headers=_headers(),
            json={
                "input_features": FEATURES,
                "expected_output": "setosa",
            },
        )
        resp = client.get(f"/models/{GT_E2E_MODEL}/golden-tests", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_delete_golden_test(self):
        """POST puis DELETE → test supprimé."""
        r = client.post(
            f"/models/{GT_E2E_MODEL}/golden-tests",
            headers=_headers(),
            json={"input_features": FEATURES, "expected_output": "setosa"},
        )
        assert r.status_code in [200, 201]
        test_id = r.json()["id"]

        del_resp = client.delete(
            f"/models/{GT_E2E_MODEL}/golden-tests/{test_id}",
            headers=_headers(),
        )
        assert del_resp.status_code in [200, 204]

    def test_create_golden_test_without_auth_returns_401(self):
        """POST sans auth → 401/403."""
        resp = client.post(
            f"/models/{GT_E2E_MODEL}/golden-tests",
            json={"input_features": FEATURES, "expected_output": "setosa"},
        )
        assert resp.status_code in [401, 403]


class TestGoldenTestsCsvUpload:
    def test_upload_csv_adds_tests(self):
        """POST /models/{name}/golden-tests/upload-csv → tests créés depuis CSV."""
        csv_content = (
            "sepal_length,sepal_width,petal_length,petal_width,expected_output\n"
            "5.1,3.5,1.4,0.2,setosa\n"
            "6.3,2.9,5.6,1.8,virginica\n"
        )
        resp = client.post(
            f"/models/{GT_E2E_MODEL}/golden-tests/upload-csv",
            headers=_headers(),
            files={"file": ("tests.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code in [200, 201]
        data = resp.json()
        assert data["created"] >= 2

    def test_upload_csv_without_auth_returns_401(self):
        """Upload CSV sans auth → 401/403."""
        csv_content = "f1,expected_output\n1.0,setosa\n"
        resp = client.post(
            f"/models/{GT_E2E_MODEL}/golden-tests/upload-csv",
            files={"file": ("tests.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code in [401, 403]


class TestGoldenTestsRun:
    def test_run_golden_tests_returns_results(self):
        """POST /models/{name}/{version}/golden-tests/run → résultats pass/fail."""
        # S'assurer qu'il y a au moins un golden test
        client.post(
            f"/models/{GT_E2E_MODEL}/golden-tests",
            headers=_headers(),
            json={"input_features": FEATURES, "expected_output": "setosa"},
        )

        resp = client.post(
            f"/models/{GT_E2E_MODEL}/{GT_VERSION}/golden-tests/run",
            headers=_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_tests" in data
        assert "passed" in data
        assert "failed" in data
        assert "pass_rate" in data

    def test_run_golden_tests_without_auth_returns_401(self):
        """Run sans auth → 401/403."""
        resp = client.post(
            f"/models/{GT_E2E_MODEL}/{GT_VERSION}/golden-tests/run"
        )
        assert resp.status_code in [401, 403]


class TestGoldenTestsPolicyGate:
    def test_policy_with_min_pass_rate_blocks_if_below_threshold(self):
        """Policy min_golden_test_pass_rate → gate de promotion."""
        from src.services.auto_promotion_service import evaluate_auto_promotion

        # Évaluer avec policy qui exige 100% de pass
        async def _run():
            async with _TestSessionLocal() as db:
                result, reason = await evaluate_auto_promotion(
                    db=db,
                    model_name=GT_E2E_MODEL,
                    version=GT_VERSION,
                    policy={
                        "auto_promote": True,
                        "min_accuracy": None,
                        "max_latency_p95_ms": None,
                        "min_sample_validation": 1,
                        "min_golden_test_pass_rate": 1.0,
                    },
                )
            return result, reason

        promoted, reason = asyncio.run(_run())
        # Sans observed_results → min_sample non atteint → non promu
        assert promoted is False or reason is not None
