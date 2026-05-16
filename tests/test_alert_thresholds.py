"""
Tests pour les seuils d'alerte par modèle (alert_thresholds).

Couvre :
- AlertThresholds : validation Pydantic (valeurs limites, champs optionnels)
- PATCH /models/{name}/{version} : stockage et lecture de alert_thresholds

Les tests de _get_model_threshold() et run_alert_check() sont dans
test_supervision_thresholds.py pour éviter une pollution du module
supervision_reporter avant que test_config.py recharge src.core.config.
"""

import asyncio
import io
import joblib

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.schemas.model import AlertThresholds
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TEST_TOKEN = "test-token-alert-thresholds-001"
MODEL_PREFIX = "alert_thresh_model"


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": name, "version": version},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_alert_thresholds",
                email="test_alert@test.com",
                api_token=TEST_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# AlertThresholds Pydantic schema
# ---------------------------------------------------------------------------


class TestAlertThresholdsSchema:
    def test_all_fields_optional(self):
        """AlertThresholds peut être instancié sans aucun champ."""
        obj = AlertThresholds()
        assert obj.accuracy_min is None
        assert obj.error_rate_max is None
        assert obj.drift_auto_alert is None

    def test_valid_full_thresholds(self):
        """Tous les champs fournis avec des valeurs valides."""
        obj = AlertThresholds(accuracy_min=0.90, error_rate_max=0.05, drift_auto_alert=True)
        assert obj.accuracy_min == pytest.approx(0.90)
        assert obj.error_rate_max == pytest.approx(0.05)
        assert obj.drift_auto_alert is True

    def test_accuracy_min_boundary_zero(self):
        """accuracy_min=0.0 est accepté (borne inférieure inclusive)."""
        obj = AlertThresholds(accuracy_min=0.0)
        assert obj.accuracy_min == pytest.approx(0.0)

    def test_accuracy_min_boundary_one(self):
        """accuracy_min=1.0 est accepté (borne supérieure inclusive)."""
        obj = AlertThresholds(accuracy_min=1.0)
        assert obj.accuracy_min == pytest.approx(1.0)

    def test_accuracy_min_above_one_rejected(self):
        """accuracy_min > 1.0 lève une ValidationError."""
        with pytest.raises(ValidationError):
            AlertThresholds(accuracy_min=1.1)

    def test_accuracy_min_below_zero_rejected(self):
        """accuracy_min < 0.0 lève une ValidationError."""
        with pytest.raises(ValidationError):
            AlertThresholds(accuracy_min=-0.01)

    def test_error_rate_max_boundary_zero(self):
        """error_rate_max=0.0 est accepté."""
        obj = AlertThresholds(error_rate_max=0.0)
        assert obj.error_rate_max == pytest.approx(0.0)

    def test_error_rate_max_above_one_rejected(self):
        """error_rate_max > 1.0 lève une ValidationError."""
        with pytest.raises(ValidationError):
            AlertThresholds(error_rate_max=1.5)

    def test_drift_auto_alert_false(self):
        """drift_auto_alert=False est accepté."""
        obj = AlertThresholds(drift_auto_alert=False)
        assert obj.drift_auto_alert is False

    def test_serializes_to_dict(self):
        """model_dump() produit un dict pur (requis pour la colonne JSON ORM)."""
        obj = AlertThresholds(accuracy_min=0.85, error_rate_max=0.10, drift_auto_alert=True)
        d = obj.model_dump()
        assert isinstance(d, dict)
        assert d["accuracy_min"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# PATCH endpoint : stockage et lecture de alert_thresholds
# ---------------------------------------------------------------------------


class TestAlertThresholdsPatchEndpoint:
    def test_patch_sets_alert_thresholds(self):
        """PATCH avec alert_thresholds valides → stocké et retourné dans la réponse."""
        _create_model(f"{MODEL_PREFIX}_patch1")
        r = client.patch(
            f"/models/{MODEL_PREFIX}_patch1/1.0.0",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "alert_thresholds": {
                    "accuracy_min": 0.88,
                    "error_rate_max": 0.07,
                    "drift_auto_alert": False,
                }
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["alert_thresholds"]["accuracy_min"] == pytest.approx(0.88)
        assert data["alert_thresholds"]["error_rate_max"] == pytest.approx(0.07)
        assert data["alert_thresholds"]["drift_auto_alert"] is False

    def test_patch_alert_thresholds_partial(self):
        """PATCH avec seulement error_rate_max → les autres champs sont None."""
        _create_model(f"{MODEL_PREFIX}_patch2")
        r = client.patch(
            f"/models/{MODEL_PREFIX}_patch2/1.0.0",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"alert_thresholds": {"error_rate_max": 0.03}},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["alert_thresholds"]["error_rate_max"] == pytest.approx(0.03)
        assert data["alert_thresholds"].get("accuracy_min") is None
        assert data["alert_thresholds"].get("drift_auto_alert") is None

    def test_patch_alert_thresholds_null_clears(self):
        """PATCH avec alert_thresholds=null efface les seuils précédents."""
        _create_model(f"{MODEL_PREFIX}_patch3")
        client.patch(
            f"/models/{MODEL_PREFIX}_patch3/1.0.0",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"alert_thresholds": {"error_rate_max": 0.05}},
        )
        r = client.patch(
            f"/models/{MODEL_PREFIX}_patch3/1.0.0",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"alert_thresholds": None},
        )
        assert r.status_code == 200, r.text
        assert r.json().get("alert_thresholds") is None

    def test_patch_alert_thresholds_invalid_accuracy_min(self):
        """PATCH avec accuracy_min=2.0 → 422 (hors plage)."""
        _create_model(f"{MODEL_PREFIX}_patch4")
        r = client.patch(
            f"/models/{MODEL_PREFIX}_patch4/1.0.0",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"alert_thresholds": {"accuracy_min": 2.0}},
        )
        assert r.status_code == 422

    def test_patch_preserves_other_fields(self):
        """PATCH alert_thresholds seul ne modifie pas les autres champs."""
        _create_model(f"{MODEL_PREFIX}_patch5")
        client.patch(
            f"/models/{MODEL_PREFIX}_patch5/1.0.0",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"description": "stable desc"},
        )
        r = client.patch(
            f"/models/{MODEL_PREFIX}_patch5/1.0.0",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"alert_thresholds": {"error_rate_max": 0.08}},
        )
        assert r.status_code == 200
        assert r.json()["description"] == "stable desc"
