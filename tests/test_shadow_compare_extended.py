"""
Tests pour GET /models/{name}/shadow-compare.

Couvre :
- Auth (sans token, token invalide)
- Modèle inexistant → 404
- Modèle sans version shadow → réponse structurée avec n_comparable=0
- Modèle existant → structure de réponse complète
- Paramètre period_days
"""

import asyncio
import io
import joblib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

USER_TOKEN = "test-token-shadowcmp-user-cc03"
SHADOW_MODEL = "shadow_compare_model"


def _make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="shadow_cmp_user",
                email="shadow_cmp@test.com",
                api_token=USER_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())


class TestShadowCompareAuth:
    def test_shadow_compare_without_auth_returns_401(self):
        """GET /models/{name}/shadow-compare sans auth → 401."""
        resp = client.get(f"/models/{SHADOW_MODEL}/shadow-compare")
        assert resp.status_code in [401, 403]

    def test_shadow_compare_invalid_token_returns_401(self):
        """GET /models/{name}/shadow-compare avec token invalide → 401."""
        resp = client.get(
            f"/models/{SHADOW_MODEL}/shadow-compare",
            headers={"Authorization": "Bearer invalid-token-xyz"},
        )
        assert resp.status_code == 401


class TestShadowCompareNotFound:
    def test_shadow_compare_nonexistent_model_returns_404(self):
        """GET /models/nonexistent/shadow-compare → 404."""
        resp = client.get(
            "/models/nonexistent_shadow_xyz/shadow-compare",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert resp.status_code == 404


class TestShadowCompareNoShadowVersion:
    def test_shadow_compare_no_shadow_returns_empty_stats(self):
        """Modèle sans version shadow → n_comparable=0, agreement_rate=null."""
        # Créer un modèle simple sans shadow
        client.post(
            "/models",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            files={
                "file": (
                    "model.pkl",
                    io.BytesIO(_make_pkl_bytes()),
                    "application/octet-stream",
                )
            },
            data={"name": SHADOW_MODEL, "version": "1.0.0"},
        )

        # Patcher get_shadow_comparison_stats pour retourner des stats vides
        empty_stats = {
            "shadow_version": None,
            "production_version": "1.0.0",
            "n_comparable": 0,
            "agreement_rate": None,
            "shadow_confidence_delta": None,
            "shadow_latency_delta_ms": None,
            "shadow_accuracy": None,
            "production_accuracy": None,
            "accuracy_available": False,
        }

        with patch(
            "src.services.db_service.DBService.get_shadow_comparison_stats",
            new_callable=AsyncMock,
            return_value=empty_stats,
        ):
            resp = client.get(
                f"/models/{SHADOW_MODEL}/shadow-compare",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == SHADOW_MODEL
        assert data["n_comparable"] == 0
        assert data["agreement_rate"] is None
        assert data["recommendation"] in [
            "insufficient_data",
            "shadow_better",
            "production_better",
            "equivalent",
        ]

    def test_shadow_compare_response_has_expected_fields(self):
        """La réponse contient tous les champs attendus du schéma."""
        stats = {
            "shadow_version": "2.0.0",
            "production_version": "1.0.0",
            "n_comparable": 50,
            "agreement_rate": 0.92,
            "shadow_confidence_delta": 0.05,
            "shadow_latency_delta_ms": -10.0,
            "shadow_accuracy": 0.95,
            "production_accuracy": 0.91,
            "accuracy_available": True,
        }

        with patch(
            "src.services.db_service.DBService.get_shadow_comparison_stats",
            new_callable=AsyncMock,
            return_value=stats,
        ):
            resp = client.get(
                f"/models/{SHADOW_MODEL}/shadow-compare",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        expected_keys = {
            "model_name",
            "shadow_version",
            "production_version",
            "period_days",
            "n_comparable",
            "agreement_rate",
            "shadow_confidence_delta",
            "shadow_latency_delta_ms",
            "shadow_accuracy",
            "production_accuracy",
            "accuracy_available",
            "recommendation",
        }
        assert expected_keys.issubset(data.keys())

    def test_shadow_compare_period_days_param(self):
        """Le paramètre period_days est bien pris en compte."""
        empty_stats = {
            "shadow_version": None,
            "production_version": "1.0.0",
            "n_comparable": 0,
            "agreement_rate": None,
            "shadow_confidence_delta": None,
            "shadow_latency_delta_ms": None,
            "shadow_accuracy": None,
            "production_accuracy": None,
            "accuracy_available": False,
        }

        with patch(
            "src.services.db_service.DBService.get_shadow_comparison_stats",
            new_callable=AsyncMock,
            return_value=empty_stats,
        ) as mock_stats:
            resp = client.get(
                f"/models/{SHADOW_MODEL}/shadow-compare?period_days=7",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["period_days"] == 7


class TestShadowRecommendation:
    """Tests de la logique _shadow_recommendation via l'endpoint."""

    def test_recommendation_insufficient_data_when_n_comparable_zero(self):
        """n_comparable=0 → recommendation=insufficient_data."""
        stats = {
            "shadow_version": "2.0.0",
            "production_version": "1.0.0",
            "n_comparable": 0,
            "agreement_rate": None,
            "shadow_confidence_delta": None,
            "shadow_latency_delta_ms": None,
            "shadow_accuracy": None,
            "production_accuracy": None,
            "accuracy_available": False,
        }

        with patch(
            "src.services.db_service.DBService.get_shadow_comparison_stats",
            new_callable=AsyncMock,
            return_value=stats,
        ):
            resp = client.get(
                f"/models/{SHADOW_MODEL}/shadow-compare",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        assert resp.json()["recommendation"] == "insufficient_data"

    def test_recommendation_shadow_better_when_higher_accuracy(self):
        """shadow_accuracy > production_accuracy → shadow_better."""
        stats = {
            "shadow_version": "2.0.0",
            "production_version": "1.0.0",
            "n_comparable": 200,
            "agreement_rate": 0.85,
            "shadow_confidence_delta": 0.10,
            "shadow_latency_delta_ms": 5.0,
            "shadow_accuracy": 0.95,
            "production_accuracy": 0.82,
            "accuracy_available": True,
        }

        with patch(
            "src.services.db_service.DBService.get_shadow_comparison_stats",
            new_callable=AsyncMock,
            return_value=stats,
        ):
            resp = client.get(
                f"/models/{SHADOW_MODEL}/shadow-compare",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        assert resp.json()["recommendation"] == "shadow_better"
