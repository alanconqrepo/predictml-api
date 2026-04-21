"""
Tests pour la fonctionnalité d'auto-promotion post-retrain.

Couvre :
- PATCH /models/{name}/policy
  - Auth / permissions
  - Modèle inexistant
  - Succès : policy stockée sur toutes les versions actives
  - Validation des champs (min_accuracy, max_latency_p95_ms)
  - Désactivation (auto_promote=False)
- POST /models/{name}/{version}/retrain avec auto-promotion
  - Sans policy → auto_promoted=None
  - Policy auto_promote=False → non évaluée
  - Policy auto_promote=True, échantillons insuffisants → auto_promoted=False
  - Policy auto_promote=True, accuracy OK → auto_promoted=True, promu
  - Policy auto_promote=True, accuracy insuffisante → auto_promoted=False
  - Policy auto_promote=True, latence P95 trop élevée → auto_promoted=False
  - set_production=True (manuel) prend la main sur l'auto-promotion
- evaluate_auto_promotion() (unit tests)
"""

import asyncio
import io
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.auto_promotion_service import evaluate_auto_promotion
from src.services.db_service import DBService
from tests.conftest import _minio_mock, _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-policy-admin-aa11"
USER_TOKEN = "test-token-policy-user-bb22"
MODEL_PREFIX = "policy_model"

VALID_TRAIN_SCRIPT = """\
import os
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

TRAIN_START_DATE = os.environ["TRAIN_START_DATE"]
TRAIN_END_DATE = os.environ["TRAIN_END_DATE"]
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(json.dumps({"accuracy": 0.97, "f1_score": 0.96}))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)  # noqa: N806
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


def _create_model(name: str, version: str = "1.0.0", with_train_script: bool = False) -> dict:
    files: dict = {
        "file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
    }
    if with_train_script:
        files["train_file"] = (
            "train.py",
            io.BytesIO(VALID_TRAIN_SCRIPT.encode()),
            "text/x-python",
        )
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files=files,
        data={"name": name, "version": version, "accuracy": "0.90", "f1_score": "0.89"},
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Setup utilisateurs
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="policy_admin",
                email="policy_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="policy_user",
                email="policy_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())

_minio_mock.download_file_bytes.return_value = VALID_TRAIN_SCRIPT.encode()
_minio_mock.upload_file_bytes.return_value = {
    "bucket": "models",
    "object_name": "mock_train.py",
    "size": len(VALID_TRAIN_SCRIPT),
}


# ---------------------------------------------------------------------------
# Subprocess mock
# ---------------------------------------------------------------------------


async def _mock_exec_success(*args, **kwargs):
    env = kwargs.get("env", {})
    output_path = env.get("OUTPUT_MODEL_PATH", "")
    if output_path:
        X, y = load_iris(return_X_y=True)  # noqa: N806
        model = LogisticRegression(max_iter=200).fit(X, y)
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(
        return_value=(
            b'Training done\n{"accuracy": 0.95, "f1_score": 0.93}\n',
            b"",
        )
    )
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Tests — PATCH /models/{name}/policy
# ---------------------------------------------------------------------------


class TestPolicyEndpoint:
    """Tests pour PATCH /models/{name}/policy."""

    @classmethod
    def setup_class(cls):
        cls.model_name = f"{MODEL_PREFIX}_policy_ep"
        _create_model(cls.model_name, "1.0.0")
        _create_model(cls.model_name, "2.0.0")

    def test_patch_policy_without_auth_returns_401(self):
        r = client.patch(
            f"/models/{self.model_name}/policy",
            json={"auto_promote": True, "min_accuracy": 0.9, "min_sample_validation": 5},
        )
        assert r.status_code in [401, 403]

    def test_patch_policy_with_user_token_returns_403(self):
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"auto_promote": True, "min_accuracy": 0.9},
        )
        assert r.status_code == 403

    def test_patch_policy_model_not_found_returns_404(self):
        r = client.patch(
            "/models/nonexistent_model_xyz/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"auto_promote": True},
        )
        assert r.status_code == 404

    def test_patch_policy_success_returns_200(self):
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={
                "min_accuracy": 0.90,
                "max_latency_p95_ms": 200.0,
                "min_sample_validation": 50,
                "auto_promote": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == self.model_name
        assert data["promotion_policy"]["min_accuracy"] == pytest.approx(0.90)
        assert data["promotion_policy"]["max_latency_p95_ms"] == pytest.approx(200.0)
        assert data["promotion_policy"]["min_sample_validation"] == 50
        assert data["promotion_policy"]["auto_promote"] is True

    def test_patch_policy_updates_all_active_versions(self):
        """La policy doit être propagée à toutes les versions actives."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 0.85, "min_sample_validation": 20, "auto_promote": True},
        )
        assert r.status_code == 200
        assert r.json()["updated_versions"] == 2

    def test_patch_policy_min_accuracy_out_of_range_returns_422(self):
        """min_accuracy doit être dans [0.0, 1.0]."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 1.5},
        )
        assert r.status_code == 422

    def test_patch_policy_max_latency_negative_returns_422(self):
        """max_latency_p95_ms doit être > 0."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"max_latency_p95_ms": -50.0},
        )
        assert r.status_code == 422

    def test_patch_policy_min_sample_validation_zero_returns_422(self):
        """min_sample_validation doit être >= 1."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_sample_validation": 0},
        )
        assert r.status_code == 422

    def test_patch_policy_defaults_auto_promote_false(self):
        """Sans auto_promote explicite, la valeur par défaut est False."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 0.80},
        )
        assert r.status_code == 200
        assert r.json()["promotion_policy"]["auto_promote"] is False

    def test_patch_policy_persisted_on_model_get(self):
        """La policy est visible dans GET /models/{name}/{version}."""
        client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 0.75, "auto_promote": True, "min_sample_validation": 5},
        )
        r = client.get(
            f"/models/{self.model_name}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        policy = r.json().get("promotion_policy")
        assert policy is not None
        assert policy["min_accuracy"] == pytest.approx(0.75)
        assert policy["auto_promote"] is True


# ---------------------------------------------------------------------------
# Tests — auto-promotion dans retrain
# ---------------------------------------------------------------------------


class TestAutoPromotionInRetrain:
    """Tests de l'auto-promotion déclenchée par POST /models/{name}/{version}/retrain."""

    @classmethod
    def setup_class(cls):
        cls.model_no_policy = f"{MODEL_PREFIX}_no_policy"
        cls.model_policy_off = f"{MODEL_PREFIX}_policy_off"
        cls.model_policy_on = f"{MODEL_PREFIX}_policy_on"

        _create_model(cls.model_no_policy, "1.0.0", with_train_script=True)
        _create_model(cls.model_policy_off, "1.0.0", with_train_script=True)
        _create_model(cls.model_policy_on, "1.0.0", with_train_script=True)

        # Activer la policy sur model_policy_on
        client.patch(
            f"/models/{cls.model_policy_on}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": True},
        )
        # Policy présente mais auto_promote=False
        client.patch(
            f"/models/{cls.model_policy_off}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": False},
        )

    def test_retrain_without_policy_auto_promoted_is_none(self):
        """Sans policy, auto_promoted n'est pas renseigné (None)."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{self.model_no_policy}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2025-01-01", "end_date": "2025-12-31", "new_version": "2.0.0"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["auto_promoted"] is None

    def test_retrain_policy_auto_promote_false_not_evaluated(self):
        """Policy présente mais auto_promote=False → auto_promoted=None (non déclenché)."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{self.model_policy_off}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2025-01-01", "end_date": "2025-12-31", "new_version": "2.0.0"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        # auto_promote=False → branche non déclenchée, pas de promotion → auto_promoted=False
        assert data["auto_promoted"] is False

    def test_retrain_auto_promote_insufficient_samples(self):
        """Policy active mais pas assez d'observed_results → auto_promoted=False."""
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_mock_exec_success),
            ),
            patch(
                "src.api.models.evaluate_auto_promotion",
                new=AsyncMock(return_value=(False, "Échantillons insuffisants : 0/3 requis.")),
            ),
        ):
            r = client.post(
                f"/models/{self.model_policy_on}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2025-01-01", "end_date": "2025-12-31", "new_version": "2.0.0"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["auto_promoted"] is False
        assert "insuffisant" in data["auto_promote_reason"].lower()
        assert data["new_model_metadata"]["is_production"] is False

    def test_retrain_auto_promote_success(self):
        """Policy active, critères satisfaits → auto_promoted=True, version en production."""
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_mock_exec_success),
            ),
            patch(
                "src.api.models.evaluate_auto_promotion",
                new=AsyncMock(
                    return_value=(True, "Tous les critères de promotion sont satisfaits.")
                ),
            ),
        ):
            r = client.post(
                f"/models/{self.model_policy_on}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2024-01-01", "end_date": "2024-12-31", "new_version": "3.0.0"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["auto_promoted"] is True
        assert "satisfait" in data["auto_promote_reason"].lower()
        assert data["new_model_metadata"]["is_production"] is True

    def test_retrain_auto_promote_accuracy_fails(self):
        """Policy active, accuracy insuffisante → auto_promoted=False, non promu."""
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_mock_exec_success),
            ),
            patch(
                "src.api.models.evaluate_auto_promotion",
                new=AsyncMock(
                    return_value=(False, "Précision insuffisante : 0.7500 < 0.8000 requis.")
                ),
            ),
        ):
            r = client.post(
                f"/models/{self.model_policy_on}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2023-01-01", "end_date": "2023-12-31", "new_version": "4.0.0"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["auto_promoted"] is False
        assert "précision" in data["auto_promote_reason"].lower()
        assert data["new_model_metadata"]["is_production"] is False

    def test_retrain_auto_promote_latency_fails(self):
        """Policy active, latence P95 trop élevée → auto_promoted=False."""
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_mock_exec_success),
            ),
            patch(
                "src.api.models.evaluate_auto_promotion",
                new=AsyncMock(
                    return_value=(False, "Latence P95 trop élevée : 350.0ms > 200.0ms max.")
                ),
            ),
        ):
            r = client.post(
                f"/models/{self.model_policy_on}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={"start_date": "2022-01-01", "end_date": "2022-12-31", "new_version": "5.0.0"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["auto_promoted"] is False
        assert "latence" in data["auto_promote_reason"].lower()

    def test_retrain_set_production_overrides_auto_promotion(self):
        """set_production=True : promotion manuelle, auto_promoted non calculé."""
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=_mock_exec_success),
        ):
            r = client.post(
                f"/models/{self.model_policy_on}/1.0.0/retrain",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                json={
                    "start_date": "2021-01-01",
                    "end_date": "2021-12-31",
                    "new_version": "6.0.0",
                    "set_production": True,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["new_model_metadata"]["is_production"] is True
        # set_production=True → branche auto-promotion non empruntée
        assert data["auto_promoted"] is None


# ---------------------------------------------------------------------------
# Tests unitaires — evaluate_auto_promotion()
# ---------------------------------------------------------------------------


class TestEvaluateAutoPromotion:
    """Tests unitaires pour src/services/auto_promotion_service.evaluate_auto_promotion."""

    def _make_pairs(self, n: int, correct_ratio: float = 1.0):
        """Génère n paires (prediction, observed_result, probs, timestamp).
        correct_ratio : fraction de prédictions correctes (pred == obs)."""
        pairs = []
        for i in range(n):
            pred = "A" if i / n < correct_ratio else "B"
            obs = "A"
            pairs.append((pred, obs, None, None))
        return pairs

    def _make_db(self, pairs, latency_times=None):
        """Crée un mock AsyncSession pour evaluate_auto_promotion."""
        db = MagicMock()
        # Mock get_performance_pairs
        get_pairs_mock = AsyncMock(return_value=pairs)

        # Mock db.execute for latency query
        if latency_times is not None:
            execute_result = MagicMock()
            execute_result.all.return_value = [(t,) for t in latency_times]
            db.execute = AsyncMock(return_value=execute_result)
        else:
            execute_result = MagicMock()
            execute_result.all.return_value = []
            db.execute = AsyncMock(return_value=execute_result)

        return db, get_pairs_mock

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_insufficient_samples_returns_false(self):
        pairs = self._make_pairs(2)
        db, pairs_mock = self._make_db(pairs)
        policy = {"min_accuracy": 0.80, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is False
        assert "insuffisant" in reason.lower()

    def test_sufficient_samples_accuracy_ok_returns_true(self):
        pairs = self._make_pairs(10, correct_ratio=1.0)  # 100% accuracy
        db, pairs_mock = self._make_db(pairs)
        policy = {"min_accuracy": 0.80, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    def test_accuracy_below_threshold_returns_false(self):
        pairs = self._make_pairs(10, correct_ratio=0.5)  # 50% accuracy
        db, pairs_mock = self._make_db(pairs)
        policy = {"min_accuracy": 0.80, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is False
        assert "précision" in reason.lower()

    def test_no_accuracy_threshold_skips_accuracy_check(self):
        """Sans min_accuracy, pas de vérification d'accuracy."""
        pairs = self._make_pairs(10, correct_ratio=0.1)  # accuracy très basse
        db, pairs_mock = self._make_db(pairs)
        policy = {"min_sample_validation": 5, "auto_promote": True}  # pas de min_accuracy
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True  # Aucun critère à vérifier → promotion acceptée

    def test_latency_ok_returns_true(self):
        pairs = self._make_pairs(10, correct_ratio=1.0)
        latency_times = [50.0] * 20  # p95 = 50ms
        db, pairs_mock = self._make_db(pairs, latency_times)
        policy = {
            "min_accuracy": 0.80,
            "max_latency_p95_ms": 200.0,
            "min_sample_validation": 5,
            "auto_promote": True,
        }
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    def test_latency_too_high_returns_false(self):
        pairs = self._make_pairs(10, correct_ratio=1.0)
        latency_times = [500.0] * 20  # p95 = 500ms
        db, pairs_mock = self._make_db(pairs, latency_times)
        policy = {
            "min_accuracy": 0.80,
            "max_latency_p95_ms": 200.0,
            "min_sample_validation": 5,
            "auto_promote": True,
        }
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is False
        assert "latence" in reason.lower()

    def test_no_latency_data_skips_latency_check(self):
        """Aucune donnée de latence → critère latence ignoré."""
        pairs = self._make_pairs(10, correct_ratio=1.0)
        db, pairs_mock = self._make_db(pairs, latency_times=[])  # liste vide
        policy = {
            "min_accuracy": 0.80,
            "max_latency_p95_ms": 200.0,
            "min_sample_validation": 5,
            "auto_promote": True,
        }
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    def test_uses_most_recent_n_pairs_for_accuracy(self):
        """L'accuracy est calculée sur les min_sample_validation paires les plus récentes."""
        # 10 paires : les 5 premières fausses, les 5 dernières correctes
        pairs = [("B", "A", None, None)] * 5 + [("A", "A", None, None)] * 5
        db, pairs_mock = self._make_db(pairs)
        # min_sample_validation=5 → utilise les 5 dernières (toutes correctes → accuracy=1.0)
        policy = {"min_accuracy": 0.90, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    # --- Tests régression ---

    def _make_regression_pairs(self, n: int, base: float = 10.0, noise: float = 0.5):
        """Génère n paires (prediction, observed_result) avec valeurs float (régression)."""
        import random
        random.seed(42)
        pairs = []
        for _ in range(n):
            pred = base + random.uniform(-noise, noise)
            obs = base + random.uniform(-noise, noise)
            pairs.append((str(pred), str(obs), None, None))
        return pairs

    def test_regression_max_mae_ok_returns_true(self):
        """Régression : MAE < seuil → promu."""
        pairs = self._make_regression_pairs(10, base=10.0, noise=0.3)
        db, pairs_mock = self._make_db(pairs)
        # Avec noise=0.3, MAE attendu ≈ 0.3 → max_mae=1.0 largement suffisant
        policy = {"max_mae": 1.0, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    def test_regression_max_mae_too_high_returns_false(self):
        """Régression : MAE > seuil → non promu."""
        pairs = self._make_regression_pairs(10, base=10.0, noise=2.0)
        db, pairs_mock = self._make_db(pairs)
        # Avec noise=2.0, MAE attendu ≈ 1.3 → max_mae=0.1 trop strict
        policy = {"max_mae": 0.1, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is False
        assert "mae" in reason.lower()

    def test_regression_no_mae_policy_returns_true(self):
        """Régression : sans max_mae dans la policy → promu (pas de critère à vérifier)."""
        pairs = self._make_regression_pairs(10, base=10.0, noise=5.0)
        db, pairs_mock = self._make_db(pairs)
        policy = {"min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    def test_regression_min_accuracy_ignored_for_float_values(self):
        """Régression : min_accuracy ignoré pour des floats → ne bloque pas la promotion.

        Des floats non-entiers déclenchent la branche régression, donc min_accuracy
        est ignoré même si défini (on n'a pas de max_mae ici).
        """
        pairs = self._make_regression_pairs(10, base=100.0, noise=10.0)
        db, pairs_mock = self._make_db(pairs)
        # min_accuracy avec valeur impossible pour régression, mais ignoré
        policy = {"min_accuracy": 0.99, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        # La branche régression est détectée → min_accuracy ignoré → promu
        assert ok is True
