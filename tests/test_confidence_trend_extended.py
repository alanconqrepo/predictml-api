"""
Tests pour GET /models/{name}/confidence-trend
et GET /models/{name}/confidence-distribution.

Couvre :
- Auth (sans token → 401)
- Modèle inexistant → 404
- Trend vide (aucune prédiction) → has_data=False, trend=[]
- Trend avec données → structure complète
- Distribution vide → histogram vide
- Filtre version dans confidence-trend
"""

import asyncio
import io
import pickle
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

USER_TOKEN = "test-token-conf-trend-user-dd04"
CONF_MODEL = "conf_trend_model"


def _make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="conf_trend_user",
                email="conf_trend@test.com",
                api_token=USER_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())


def _create_conf_model(name=CONF_MODEL, version="1.0.0"):
    return client.post(
        "/models",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
        files={
            "file": (
                "model.pkl",
                io.BytesIO(_make_pkl_bytes()),
                "application/octet-stream",
            )
        },
        data={"name": name, "version": version},
    )


class TestConfidenceTrendAuth:
    def test_trend_without_auth_returns_401(self):
        """GET /models/{name}/confidence-trend sans auth → 401."""
        resp = client.get(f"/models/{CONF_MODEL}/confidence-trend")
        assert resp.status_code in [401, 403]

    def test_trend_nonexistent_model_returns_404(self):
        """Modèle inexistant → 404."""
        resp = client.get(
            "/models/nonexistent_conf_xyz/confidence-trend",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert resp.status_code == 404


class TestConfidenceTrendEmpty:
    def test_trend_empty_when_no_predictions(self):
        """Aucune prédiction → trend=[], overall.mean_confidence=0."""
        _create_conf_model(name=f"{CONF_MODEL}_empty")

        empty_result = {
            "has_data": False,
            "overall": None,
            "trend": [],
        }

        with patch(
            "src.services.db_service.DBService.get_confidence_trend",
            new_callable=AsyncMock,
            return_value=empty_result,
        ):
            resp = client.get(
                f"/models/{CONF_MODEL}_empty/confidence-trend",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["trend"] == []
        assert data["overall"]["mean_confidence"] == 0.0

    def test_trend_with_data_returns_points(self):
        """Données présentes → liste de points non vide."""
        _create_conf_model(name=f"{CONF_MODEL}_data")

        mock_result = {
            "has_data": True,
            "overall": {
                "mean_confidence": 0.87,
                "p25_confidence": 0.75,
                "p75_confidence": 0.95,
                "low_confidence_rate": 0.05,
            },
            "trend": [
                {
                    "date": "2025-05-01",
                    "mean_confidence": 0.87,
                    "p25": 0.75,
                    "p75": 0.95,
                    "predictions": 100,
                    "low_confidence_count": 5,
                }
            ],
        }

        with patch(
            "src.services.db_service.DBService.get_confidence_trend",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.get(
                f"/models/{CONF_MODEL}_data/confidence-trend",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["trend"]) == 1
        assert data["overall"]["mean_confidence"] == pytest.approx(0.87)

    def test_trend_version_filter_passed_to_service(self):
        """Le paramètre version est transmis à DBService.get_confidence_trend."""
        _create_conf_model(name=f"{CONF_MODEL}_vf", version="2.0.0")

        empty_result = {"has_data": False, "overall": None, "trend": []}

        with patch(
            "src.services.db_service.DBService.get_confidence_trend",
            new_callable=AsyncMock,
            return_value=empty_result,
        ) as mock_fn:
            client.get(
                f"/models/{CONF_MODEL}_vf/confidence-trend?version=2.0.0",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )
            call_kwargs = mock_fn.call_args.kwargs
            assert call_kwargs.get("version") == "2.0.0"


class TestConfidenceDistribution:
    def test_distribution_without_auth_returns_401(self):
        """GET /models/{name}/confidence-distribution sans auth → 401."""
        resp = client.get(f"/models/{CONF_MODEL}/confidence-distribution")
        assert resp.status_code in [401, 403]

    def test_distribution_nonexistent_model_returns_404(self):
        """Modèle inexistant → 404."""
        resp = client.get(
            "/models/nonexistent_dist_xyz/confidence-distribution",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert resp.status_code == 404

    def test_distribution_empty_when_no_data(self):
        """Aucune prédiction → histogram vide."""
        _create_conf_model(name=f"{CONF_MODEL}_dist")

        empty_result = {
            "has_data": False,
            "histogram": [],
            "total_predictions": 0,
            "high_confidence_rate": 0.0,
            "uncertain_rate": 0.0,
            "mean_confidence": 0.0,
        }

        with patch(
            "src.services.db_service.DBService.get_confidence_distribution",
            new_callable=AsyncMock,
            return_value=empty_result,
        ):
            resp = client.get(
                f"/models/{CONF_MODEL}_dist/confidence-distribution",
                headers={"Authorization": f"Bearer {USER_TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == f"{CONF_MODEL}_dist"
        assert data["histogram"] == []
