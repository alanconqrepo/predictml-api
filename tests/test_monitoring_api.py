"""
Tests pour les endpoints de monitoring et leurs helpers internes.

Couvre :
- _worst_health(), _error_rate_status(), _performance_drift_status() (helpers purs)
- GET /monitoring/overview  — auth, validation période, résultat vide, résultat avec données
- GET /monitoring/model/{name} — auth, modèle inexistant, résultat complet
"""

import asyncio
import io
import pickle
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.api.monitoring import (
    _error_rate_status,
    _performance_drift_status,
    _worst_health,
)
from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

ADMIN_TOKEN = "test-token-monitor-admin-bb11"
MODEL_NAME = "monitor_test_model"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _make_pkl() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="monitor_admin",
                email="monitor_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        await db.commit()


asyncio.run(_setup())


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("m.pkl", io.BytesIO(_make_pkl()), "application/octet-stream")},
        data={"name": name, "version": version, "accuracy": "0.92"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _inject_model_in_cache(model_name: str, version: str = "1.0.0"):
    """Injecte un modèle LogisticRegression dans le cache Redis de test.

    La clé Redis doit correspondre au format utilisé par model_service : model:{name}:{version}
    """
    from types import SimpleNamespace
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    model.feature_names_in_ = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
        ),
    }
    asyncio.run(
        model_service._redis.set(f"model:{model_name}:{version}", pickle.dumps(data))
    )


def _predict(model_name: str):
    """Effectue une prédiction simple pour générer des données de monitoring."""
    _inject_model_in_cache(model_name)
    r = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={
            "model_name": model_name,
            "features": {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2,
            },
        },
    )
    return r


# ---------------------------------------------------------------------------
# Helper functions — tests unitaires purs (pas de HTTP)
# ---------------------------------------------------------------------------


def test_worst_health_returns_critical_over_warning():
    assert _worst_health("ok", "critical", "warning") == "critical"


def test_worst_health_returns_warning_over_ok():
    assert _worst_health("ok", "warning") == "warning"


def test_worst_health_ok_when_all_ok():
    assert _worst_health("ok", "ok") == "ok"


def test_worst_health_no_data_same_level_as_ok():
    """no_data et ok sont au même niveau (0) — le premier retourné dépend de l'ordre."""
    result = _worst_health("no_data", "ok")
    # Les deux ont le même poids (0), le résultat est l'un des deux
    assert result in ("no_data", "ok")


def test_error_rate_status_ok_below_5_percent():
    assert _error_rate_status(0.04) == "ok"


def test_error_rate_status_warning_between_5_and_10():
    assert _error_rate_status(0.07) == "warning"


def test_error_rate_status_critical_above_10():
    assert _error_rate_status(0.15) == "critical"


def test_error_rate_status_exactly_at_warning_threshold():
    assert _error_rate_status(0.05) == "warning"


def test_error_rate_status_exactly_at_critical_threshold():
    assert _error_rate_status(0.10) == "critical"


def test_performance_drift_status_insufficient_data_below_4():
    """Moins de 4 points → no_data."""
    data = [{"accuracy": 0.9, "matched_count": 10}] * 3
    assert _performance_drift_status(data) == "no_data"


def test_performance_drift_status_ok_stable():
    """Accuracy stable → ok."""
    data = [{"accuracy": 0.90, "matched_count": 10}] * 8
    assert _performance_drift_status(data) == "ok"


def test_performance_drift_status_warning_5_percent_drop():
    """Baisse de ~5 % sur la 2e moitié → warning."""
    first_half = [{"accuracy": 0.90, "matched_count": 10}] * 4
    second_half = [{"accuracy": 0.84, "matched_count": 10}] * 4
    result = _performance_drift_status(first_half + second_half)
    assert result in ("warning", "critical")


def test_performance_drift_status_critical_10_percent_drop():
    """Baisse de ~10 % sur la 2e moitié → critical."""
    first_half = [{"accuracy": 0.95, "matched_count": 10}] * 4
    second_half = [{"accuracy": 0.80, "matched_count": 10}] * 4
    assert _performance_drift_status(first_half + second_half) == "critical"


def test_performance_drift_status_no_matched_count():
    """matched_count = 0 partout → no_data."""
    data = [{"accuracy": 0.9, "matched_count": 0}] * 8
    assert _performance_drift_status(data) == "no_data"


# --- Tests drift régression (MAE) ---


def test_performance_drift_status_regression_mae_stable():
    """Régression : MAE stable → ok."""
    data = [{"accuracy": 0.0, "mae": 1.0, "matched_count": 10}] * 8
    assert _performance_drift_status(data) == "ok"


def test_performance_drift_status_regression_mae_critical():
    """Régression : MAE augmente de plus de 10 % → critical."""
    first_half = [{"accuracy": 0.0, "mae": 1.0, "matched_count": 10}] * 4
    second_half = [{"accuracy": 0.0, "mae": 1.12, "matched_count": 10}] * 4
    # drop = -(-1.0) - (-(-1.12)) = 1.0 - 1.12 = -0.12 en termes de -MAE
    # avg_first = -1.0, avg_second = -1.12 → drop = -1.0 - (-1.12) = 0.12 > 0.10
    assert _performance_drift_status(first_half + second_half) == "critical"


def test_performance_drift_status_regression_mae_warning():
    """Régression : MAE augmente entre 5 % et 10 % → warning."""
    first_half = [{"accuracy": 0.0, "mae": 1.0, "matched_count": 10}] * 4
    second_half = [{"accuracy": 0.0, "mae": 1.06, "matched_count": 10}] * 4
    result = _performance_drift_status(first_half + second_half)
    assert result in ("warning", "critical")


def test_performance_drift_status_regression_mae_improves():
    """Régression : MAE diminue → ok (amélioration)."""
    first_half = [{"accuracy": 0.0, "mae": 2.0, "matched_count": 10}] * 4
    second_half = [{"accuracy": 0.0, "mae": 1.0, "matched_count": 10}] * 4
    assert _performance_drift_status(first_half + second_half) == "ok"


def test_performance_drift_status_regression_mixed_mae_none():
    """Régression : certains points ont mae=None → ignorés dans le calcul."""
    data = [
        {"accuracy": 0.0, "mae": 1.0, "matched_count": 5},
        {"accuracy": 0.0, "mae": 1.0, "matched_count": 5},
        {"accuracy": 0.0, "mae": None, "matched_count": 0},
        {"accuracy": 0.0, "mae": 1.0, "matched_count": 5},
        {"accuracy": 0.0, "mae": 1.0, "matched_count": 5},
        {"accuracy": 0.0, "mae": None, "matched_count": 0},
        {"accuracy": 0.0, "mae": 1.0, "matched_count": 5},
        {"accuracy": 0.0, "mae": 1.0, "matched_count": 5},
    ]
    result = _performance_drift_status(data)
    assert result in ("ok", "no_data")


# ---------------------------------------------------------------------------
# GET /monitoring/overview
# ---------------------------------------------------------------------------


class TestMonitoringOverview:
    def test_monitoring_overview_requires_auth(self):
        """Sans Authorization → 401 ou 403."""
        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            params={
                "start": (now - timedelta(days=7)).isoformat(),
                "end": now.isoformat(),
            },
        )
        assert r.status_code in (401, 403)

    def test_monitoring_overview_invalid_period_end_before_start(self):
        """end <= start → 422."""
        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": now.isoformat(),
                "end": (now - timedelta(hours=1)).isoformat(),
            },
        )
        assert r.status_code == 422

    def test_monitoring_overview_empty_period_returns_zero_stats(self):
        """Période sans prédictions → stats à zéro et liste modèles vide."""
        future = datetime.utcnow() + timedelta(days=500)
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": future.isoformat(),
                "end": (future + timedelta(days=1)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["global_stats"]["total_predictions"] == 0
        assert data["models"] == []

    def test_monitoring_overview_with_predictions_returns_model_summaries(self):
        """Après avoir créé un modèle et fait des prédictions → overview non vide."""
        model = f"{MODEL_NAME}_overview"
        _create_model(model)
        _predict(model)

        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        model_names = [m["model_name"] for m in data["models"]]
        assert model in model_names
        summary = next(m for m in data["models"] if m["model_name"] == model)
        assert summary["total_predictions"] >= 1
        assert "health_status" in summary
        assert summary["health_status"] in ("ok", "warning", "critical", "no_data")


# ---------------------------------------------------------------------------
# GET /monitoring/model/{name}
# ---------------------------------------------------------------------------


class TestMonitoringModelDetail:
    def test_monitoring_model_detail_requires_auth(self):
        """Sans Authorization → 401 ou 403."""
        now = datetime.utcnow()
        r = client.get(
            f"/monitoring/model/{MODEL_NAME}_detail",
            params={
                "start": (now - timedelta(days=7)).isoformat(),
                "end": now.isoformat(),
            },
        )
        assert r.status_code in (401, 403)

    def test_monitoring_model_detail_invalid_period(self):
        """end <= start → 422."""
        now = datetime.utcnow()
        r = client.get(
            f"/monitoring/model/{MODEL_NAME}_detail",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": now.isoformat(),
                "end": (now - timedelta(hours=1)).isoformat(),
            },
        )
        assert r.status_code == 422

    def test_monitoring_model_detail_not_found_returns_404(self):
        """Modèle inexistant → 404."""
        now = datetime.utcnow()
        r = client.get(
            "/monitoring/model/completely_nonexistent_model_xyz789",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(days=7)).isoformat(),
                "end": now.isoformat(),
            },
        )
        assert r.status_code == 404

    def test_monitoring_model_detail_success(self):
        """Modèle existant → réponse complète avec les champs attendus."""
        model = f"{MODEL_NAME}_detail"
        _create_model(model)
        _predict(model)

        now = datetime.utcnow()
        r = client.get(
            f"/monitoring/model/{model}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == model
        assert "per_version_stats" in data
        assert "timeseries" in data
        assert "performance_by_day" in data
        assert "feature_drift" in data
        assert "recent_errors" in data


# ---------------------------------------------------------------------------
# GET /monitoring/overview?format=csv
# ---------------------------------------------------------------------------


class TestMonitoringOverviewCSV:
    def _params(self, extra: dict | None = None) -> dict:
        now = datetime.utcnow()
        p = {
            "start": (now - timedelta(hours=1)).isoformat(),
            "end": (now + timedelta(hours=1)).isoformat(),
            "format": "csv",
        }
        if extra:
            p.update(extra)
        return p

    def test_csv_format_returns_text_csv_content_type(self):
        """format=csv → Content-Type text/csv."""
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params=self._params(),
        )
        assert r.status_code == 200
        assert "text/csv" in r.headers.get("content-type", "")

    def test_csv_format_has_content_disposition_attachment(self):
        """format=csv → Content-Disposition avec attachment et filename .csv."""
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params=self._params(),
        )
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".csv" in cd

    def test_csv_format_header_contains_required_columns(self):
        """CSV retourné contient les colonnes spécifiées."""
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params=self._params(),
        )
        assert r.status_code == 200
        header = r.text.strip().splitlines()[0].split(",")
        for col in ("model_name", "status", "predictions_7d", "error_rate", "latency_p95", "drift_status"):
            assert col in header, f"colonne manquante : {col}"

    def test_csv_empty_period_returns_header_only(self):
        """Période sans prédictions → CSV avec uniquement la ligne d'en-tête."""
        future = datetime.utcnow() + timedelta(days=700)
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": future.isoformat(),
                "end": (future + timedelta(days=1)).isoformat(),
                "format": "csv",
            },
        )
        assert r.status_code == 200
        lines = [ln for ln in r.text.strip().splitlines() if ln]
        assert len(lines) == 1  # uniquement le header

    def test_csv_contains_model_row_after_prediction(self):
        """Après une prédiction → le modèle apparaît dans le CSV."""
        model = f"{MODEL_NAME}_csv"
        _create_model(model)
        _predict(model)

        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "csv",
            },
        )
        assert r.status_code == 200
        assert model in r.text

    def test_csv_model_row_has_correct_values(self):
        """La ligne du modèle dans le CSV contient un taux d'erreur et statut cohérents."""
        model = f"{MODEL_NAME}_csv2"
        _create_model(model)
        _predict(model)

        now = datetime.utcnow()
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "format": "csv",
            },
        )
        assert r.status_code == 200
        import csv as csv_mod

        reader = csv_mod.DictReader(r.text.splitlines())
        rows = {row["model_name"]: row for row in reader}
        assert model in rows
        row = rows[model]
        assert row["status"] in ("ok", "warning", "critical", "no_data")
        assert int(row["predictions_7d"]) >= 1
        assert float(row["error_rate"]) >= 0.0

    def test_csv_invalid_format_returns_422(self):
        """format=xml → 422 (valeur non autorisée)."""
        r = client.get(
            "/monitoring/overview",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            params={
                **self._params(),
                "format": "xml",
            },
        )
        assert r.status_code == 422

    def test_csv_requires_auth(self):
        """Sans token → 401 ou 403, même avec format=csv."""
        r = client.get(
            "/monitoring/overview",
            params=self._params(),
        )
        assert r.status_code in (401, 403)
