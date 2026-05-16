"""
Tests pour les webhooks sur événements modèle.

Couvre :
- send_webhook : nouveau paramètre event_type, retry exponentiel (3×), timeout 5 s
- retrain_completed : webhook déclenché après un ré-entraînement réussi
- model_promoted   : webhook déclenché après une auto-promotion
- drift_critical   : webhook déclenché par supervision_reporter (drift critique)
- error_rate_threshold : webhook déclenché par supervision_reporter (erreurs > seuil)
"""

import asyncio
import io
import joblib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal, _minio_mock

client = TestClient(app)

ADMIN_TOKEN = "test-token-whevt-admin-aa11"
MODEL_PREFIX = "whevt"

VALID_TRAIN_SCRIPT = """\
import os
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    joblib.dump(model, f)

print(json.dumps({"accuracy": 0.95, "f1_score": 0.93}))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={
            "file": ("model.joblib", io.BytesIO(_make_pkl_bytes()), "application/octet-stream"),
            "train_file": ("train.py", io.BytesIO(VALID_TRAIN_SCRIPT.encode()), "text/x-python"),
        },
        data={"name": name, "version": version},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _patch_webhook_url(name: str, version: str, url: str) -> None:
    r = client.patch(
        f"/models/{name}/{version}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"webhook_url": url},
    )
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


async def _setup() -> None:
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="whevt_admin",
                email="whevt_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )


asyncio.run(_setup())

_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


async def _mock_exec_success(*args, **kwargs):
    env = kwargs.get("env", {})
    output_path = env.get("OUTPUT_MODEL_PATH", "")
    if output_path:
        X, y = load_iris(return_X_y=True)
        with open(output_path, "wb") as f:
            joblib.dump(LogisticRegression(max_iter=200).fit(X, y), f)
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(
        return_value=(b'{"accuracy": 0.95, "f1_score": 0.93}\n', b"")
    )
    proc.kill = MagicMock()
    return proc


# ===========================================================================
# Part 1 — webhook_service : event_type + retry
# ===========================================================================


def _make_async_client_mock(response: MagicMock):
    client_mock = MagicMock()
    client_mock.post = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_mock)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, client_mock


def _make_response(status_code: int) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    return r


def _run_sync(coro):
    """Exécute une coroutine dans un event loop dédié (Python 3.11+ compatible)."""
    return asyncio.run(coro)


class TestSendWebhookEnhanced:
    """Tests unitaires pour les nouveaux comportements de send_webhook."""

    def test_event_type_added_as_event_key(self):
        """event_type fourni → clé 'event' préfixée dans le payload POST."""
        from src.services.webhook_service import send_webhook

        cm, client_mock = _make_async_client_mock(_make_response(200))
        with patch("src.services.webhook_service.httpx.AsyncClient", return_value=cm):
            _run_sync(
                send_webhook(
                    "http://example.com/hook",
                    {"model_name": "iris"},
                    event_type="retrain_completed",
                )
            )

        call_kwargs = client_mock.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
        assert sent_payload["event"] == "retrain_completed"
        assert sent_payload["model_name"] == "iris"

    def test_no_event_type_payload_unchanged(self):
        """event_type=None → payload transmis sans modification."""
        from src.services.webhook_service import send_webhook

        original = {"prediction": 1, "model_name": "iris"}
        cm, client_mock = _make_async_client_mock(_make_response(200))
        with patch("src.services.webhook_service.httpx.AsyncClient", return_value=cm):
            _run_sync(send_webhook("http://example.com/hook", original))

        call_kwargs = client_mock.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
        assert "event" not in sent_payload
        assert sent_payload == original

    def test_retry_on_network_failure_exhausts_three_retries(self):
        """Erreur réseau persistante → 4 tentatives (1 initial + 3 retries), puis abandon."""
        import httpx
        from src.services.webhook_service import send_webhook

        attempt_count = 0

        async def _failing_post(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise httpx.ConnectError("unreachable")

        cm = MagicMock()
        failing_client = MagicMock()
        failing_client.post = _failing_post
        cm.__aenter__ = AsyncMock(return_value=failing_client)
        cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.services.webhook_service.httpx.AsyncClient", return_value=cm),
            patch("src.services.webhook_service.asyncio.sleep", new=AsyncMock()),
        ):
            _run_sync(send_webhook("http://dead.example.com/hook", {"x": 1}))

        assert attempt_count == 4  # 1 tentative initiale + 3 retries

    def test_retry_succeeds_on_second_attempt(self):
        """Succès à la 2e tentative → 2 appels POST, pas de log d'échec final."""
        import httpx
        from src.services.webhook_service import send_webhook

        call_count = 0

        async def _flaky_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("temporary failure")
            return _make_response(200)

        cm = MagicMock()
        flaky_client = MagicMock()
        flaky_client.post = _flaky_post
        cm.__aenter__ = AsyncMock(return_value=flaky_client)
        cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.services.webhook_service.httpx.AsyncClient", return_value=cm),
            patch("src.services.webhook_service.asyncio.sleep", new=AsyncMock()),
        ):
            _run_sync(send_webhook("http://example.com/hook", {"x": 1}))

        assert call_count == 2

    def test_timeout_is_five_seconds(self):
        """Le timeout passé à AsyncClient est 5.0 s."""
        from src.services.webhook_service import send_webhook

        captured = {}

        def _fake_client(timeout=None):
            captured["timeout"] = timeout
            cm = MagicMock()
            client_mock = MagicMock()
            client_mock.post = AsyncMock(return_value=_make_response(200))
            cm.__aenter__ = AsyncMock(return_value=client_mock)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        with patch("src.services.webhook_service.httpx.AsyncClient", side_effect=_fake_client):
            _run_sync(send_webhook("http://example.com/hook", {}))

        assert captured["timeout"] == 5.0


# ===========================================================================
# Part 2 — retrain endpoint : retrain_completed + model_promoted
# ===========================================================================


class TestRetrainWebhooks:
    """Vérifie que les webhooks sont déclenchés depuis POST /models/{name}/{ver}/retrain."""

    WEBHOOK_URL = "https://hooks.example.com/events"
    MODEL_WITH_WH = f"{MODEL_PREFIX}_retrain_wh"
    MODEL_NO_WH = f"{MODEL_PREFIX}_retrain_no_wh"
    MODEL_PROMOTED = f"{MODEL_PREFIX}_retrain_promo"

    @classmethod
    def setup_class(cls):
        _create_model(cls.MODEL_WITH_WH)
        _patch_webhook_url(cls.MODEL_WITH_WH, "1.0.0", cls.WEBHOOK_URL)

        _create_model(cls.MODEL_NO_WH)

        _create_model(cls.MODEL_PROMOTED)
        _patch_webhook_url(cls.MODEL_PROMOTED, "1.0.0", cls.WEBHOOK_URL)
        # Activer la policy auto_promote
        client.patch(
            f"/models/{cls.MODEL_PROMOTED}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"auto_promote": True, "min_sample_validation": 1},
        )

    def test_retrain_completed_webhook_fired(self):
        """POST /retrain réussi + webhook_url configuré → retrain_completed envoyé."""
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
            patch("src.api.models.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            r = client.post(
                f"/models/{self.MODEL_WITH_WH}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2025-01-01", "end_date": "2025-12-31", "new_version": "2.0.0"},
            )

        assert r.status_code == 200
        assert r.json()["success"] is True
        mock_wh.assert_called()
        # Au moins un appel doit être retrain_completed
        event_types = [c.kwargs.get("event_type") or c.args[2] for c in mock_wh.call_args_list]
        assert "retrain_completed" in event_types

    def test_retrain_completed_payload_contains_expected_fields(self):
        """Payload retrain_completed contient model_name, version, details avec accuracy."""
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
            patch("src.api.models.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            r = client.post(
                f"/models/{self.MODEL_WITH_WH}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2024-01-01", "end_date": "2024-12-31", "new_version": "2.1.0"},
            )

        assert r.status_code == 200
        # Trouver l'appel retrain_completed
        completed_call = next(
            (c for c in mock_wh.call_args_list if c.kwargs.get("event_type") == "retrain_completed"),
            None,
        )
        assert completed_call is not None
        payload = completed_call.args[1]
        assert payload["model_name"] == self.MODEL_WITH_WH
        assert "version" in payload
        assert "timestamp" in payload
        assert "details" in payload
        assert "accuracy" in payload["details"]

    def test_no_webhook_when_url_not_set(self):
        """POST /retrain sur modèle sans webhook_url → send_webhook non appelé."""
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
            patch("src.api.models.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            r = client.post(
                f"/models/{self.MODEL_NO_WH}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2025-06-01", "end_date": "2025-12-31", "new_version": "2.0.0"},
            )

        assert r.status_code == 200
        mock_wh.assert_not_called()

    def test_model_promoted_webhook_fired_on_auto_promotion(self):
        """Auto-promotion déclenchée → model_promoted envoyé en plus de retrain_completed."""
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
            patch(
                "src.api.models.evaluate_auto_promotion",
                new=AsyncMock(return_value=(True, "Critères satisfaits.")),
            ),
            patch("src.api.models.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            r = client.post(
                f"/models/{self.MODEL_PROMOTED}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2025-01-01", "end_date": "2025-12-31", "new_version": "2.0.0"},
            )

        assert r.status_code == 200
        assert r.json()["auto_promoted"] is True
        event_types = [c.kwargs.get("event_type") for c in mock_wh.call_args_list]
        assert "retrain_completed" in event_types
        assert "model_promoted" in event_types

    def test_model_promoted_not_fired_when_auto_promotion_fails(self):
        """Auto-promotion évaluée mais critères non satisfaits → model_promoted non envoyé."""
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
            patch(
                "src.api.models.evaluate_auto_promotion",
                new=AsyncMock(return_value=(False, "Précision insuffisante.")),
            ),
            patch("src.api.models.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            r = client.post(
                f"/models/{self.MODEL_PROMOTED}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2024-06-01", "end_date": "2024-12-31", "new_version": "2.1.0"},
            )

        assert r.status_code == 200
        assert r.json()["auto_promoted"] is False
        event_types = [c.kwargs.get("event_type") for c in mock_wh.call_args_list]
        assert "model_promoted" not in event_types


# ===========================================================================
# Part 3 — supervision_reporter : drift_critical + error_rate_threshold
# ===========================================================================


def _make_session_context_manager():
    """Mock AsyncSessionLocal() → context manager renvoyant un mock de session."""
    mock_session = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_model_meta(webhook_url: str | None = None, feature_baseline=None):
    return SimpleNamespace(
        name="iris",
        version="1.0.0",
        is_production=True,
        webhook_url=webhook_url,
        alert_thresholds=None,
        feature_baseline=feature_baseline or {"sepal_length": {"mean": 5.0, "std": 0.5}},
    )


async def _run_alert_check_with_tasks():
    """Exécute run_alert_check() et donne une chance aux tâches planifiées de s'exécuter."""
    from src.tasks.supervision_reporter import run_alert_check

    await run_alert_check()
    await asyncio.sleep(0)  # flush pending create_task coroutines


class TestSupervisionWebhooks:
    """Vérifie que les webhooks sont déclenchés depuis run_alert_check()."""

    _STAT = {
        "model_name": "iris",
        "error_rate": 0.01,
        "total_predictions": 200,
        "error_count": 2,
        "shadow_predictions": 0,
        "avg_latency_ms": 45.0,
    }

    def test_drift_critical_fires_webhook(self):
        """Drift critique détecté → drift_critical envoyé au webhook_url du modèle."""
        meta = _make_model_meta(webhook_url="https://hooks.example.com/drift")

        feat_result = SimpleNamespace(drift_status="critical", psi=0.28, z_score=3.5)

        with (
            patch("src.db.database.AsyncSessionLocal", return_value=_make_session_context_manager()),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new=AsyncMock(return_value=[self._STAT]),
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new=AsyncMock(return_value=[meta]),
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "src.services.db_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={"sepal_length": {"mean": 6.5, "std": 0.5, "count": 50}}),
            ),
            patch(
                "src.services.drift_service.compute_feature_drift",
                return_value={"sepal_length": feat_result},
            ),
            patch("src.tasks.supervision_reporter.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            asyncio.run(_run_alert_check_with_tasks())

        mock_wh.assert_called_once()
        call = mock_wh.call_args
        assert call.kwargs.get("event_type") == "drift_critical"
        payload = call.args[1]
        assert payload["model_name"] == "iris"
        assert payload["details"]["feature"] == "sepal_length"
        assert payload["details"]["psi"] == pytest.approx(0.28)
        assert payload["details"]["status"] == "critical"

    def test_drift_critical_not_fired_when_no_webhook_url(self):
        """Drift critique mais webhook_url absent → send_webhook non appelé."""
        meta = _make_model_meta(webhook_url=None)
        feat_result = SimpleNamespace(drift_status="critical", psi=0.28, z_score=3.5)

        with (
            patch("src.db.database.AsyncSessionLocal", return_value=_make_session_context_manager()),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new=AsyncMock(return_value=[self._STAT]),
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new=AsyncMock(return_value=[meta]),
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "src.services.db_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={"sepal_length": {"mean": 6.5, "std": 0.5, "count": 50}}),
            ),
            patch(
                "src.services.drift_service.compute_feature_drift",
                return_value={"sepal_length": feat_result},
            ),
            patch("src.tasks.supervision_reporter.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            asyncio.run(_run_alert_check_with_tasks())

        mock_wh.assert_not_called()

    def test_error_rate_threshold_fires_webhook(self):
        """Taux d'erreur > seuil + webhook_url → error_rate_threshold envoyé."""
        meta = _make_model_meta(webhook_url="https://hooks.example.com/errors", feature_baseline=None)
        high_error_stat = {**self._STAT, "error_rate": 0.20}  # > seuil par défaut (0.10)

        with (
            patch("src.db.database.AsyncSessionLocal", return_value=_make_session_context_manager()),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new=AsyncMock(return_value=[high_error_stat]),
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new=AsyncMock(return_value=[meta]),
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new=AsyncMock(return_value=[]),
            ),
            patch("src.tasks.supervision_reporter.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            asyncio.run(_run_alert_check_with_tasks())

        mock_wh.assert_called_once()
        call = mock_wh.call_args
        assert call.kwargs.get("event_type") == "error_rate_threshold"
        payload = call.args[1]
        assert payload["model_name"] == "iris"
        assert payload["details"]["error_rate"] == pytest.approx(0.20)
        assert "threshold" in payload["details"]

    def test_error_rate_threshold_not_fired_when_below_threshold(self):
        """Taux d'erreur < seuil → send_webhook non appelé."""
        meta = _make_model_meta(webhook_url="https://hooks.example.com/errors", feature_baseline=None)
        low_error_stat = {**self._STAT, "error_rate": 0.01}  # < seuil

        with (
            patch("src.db.database.AsyncSessionLocal", return_value=_make_session_context_manager()),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new=AsyncMock(return_value=[low_error_stat]),
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new=AsyncMock(return_value=[meta]),
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new=AsyncMock(return_value=[]),
            ),
            patch("src.tasks.supervision_reporter.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            asyncio.run(_run_alert_check_with_tasks())

        mock_wh.assert_not_called()

    def test_error_rate_threshold_not_fired_when_no_webhook_url(self):
        """Taux d'erreur > seuil mais webhook_url absent → send_webhook non appelé."""
        meta = _make_model_meta(webhook_url=None, feature_baseline=None)
        high_error_stat = {**self._STAT, "error_rate": 0.20}

        with (
            patch("src.db.database.AsyncSessionLocal", return_value=_make_session_context_manager()),
            patch(
                "src.services.db_service.DBService.get_global_monitoring_stats",
                new=AsyncMock(return_value=[high_error_stat]),
            ),
            patch(
                "src.services.db_service.DBService.get_all_active_models",
                new=AsyncMock(return_value=[meta]),
            ),
            patch(
                "src.services.db_service.DBService.get_accuracy_drift",
                new=AsyncMock(return_value=[]),
            ),
            patch("src.tasks.supervision_reporter.send_webhook", new_callable=AsyncMock) as mock_wh,
        ):
            asyncio.run(_run_alert_check_with_tasks())

        mock_wh.assert_not_called()
