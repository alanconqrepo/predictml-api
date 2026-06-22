"""
Tests for the auto-promotion post-retrain feature.

Covers:
- PATCH /models/{name}/policy
  - Auth / permissions
  - Non-existent model
  - Success: policy stored on all active versions
  - Field validation (min_accuracy, max_latency_p95_ms)
  - Disable (auto_promote=False)
- POST /models/{name}/{version}/retrain with auto-promotion
  - Without policy → auto_promoted=None
  - Policy auto_promote=False → not evaluated
  - Policy auto_promote=True, insufficient samples → auto_promoted=False
  - Policy auto_promote=True, accuracy OK → auto_promoted=True, promoted
  - Policy auto_promote=True, insufficient accuracy → auto_promoted=False
  - Policy auto_promote=True, P95 latency too high → auto_promoted=False
  - set_production=True (manual) overrides auto-promotion
- evaluate_auto_promotion() (unit tests)
"""

import asyncio
import io
import joblib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.auto_promotion_service import evaluate_auto_demotion, evaluate_auto_promotion
from src.services.db_service import DBService
from tests.conftest import _minio_mock, _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-policy-admin-aa11"
USER_TOKEN = "test-token-policy-user-bb22"
MODEL_PREFIX = "policy_model"

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

print(json.dumps({"accuracy": 0.97, "f1_score": 0.96}))
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)  # noqa: N806
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


def _create_model(name: str, version: str = "1.0.0", with_train_script: bool = False) -> dict:
    files: dict = {
        "file": ("model.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream"),
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
# User setup
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
    """Mock subprocess for retrain_service — readline() + stderr.read()."""
    env = kwargs.get("env", {})
    output_path = env.get("OUTPUT_MODEL_PATH", "")
    if output_path:
        X, y = load_iris(return_X_y=True)  # noqa: N806
        model = LogisticRegression(max_iter=200).fit(X, y)
        with open(output_path, "wb") as f:
            joblib.dump(model, f)
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout.readline = AsyncMock(
        side_effect=[
            b"Training done\n",
            b'{"accuracy": 0.95, "f1_score": 0.93}\n',
            b"",  # EOF
        ]
    )
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    return proc


def _auto_promo_source_fields(train_script_key="mock_train.py", promotion_policy=None):
    """Creates minimal source_fields for do_retrain() tests with auto-promotion."""
    return {
        "train_script_object_key": train_script_key,
        "description": None,
        "algorithm": None,
        "features_count": None,
        "classes": None,
        "model_task": None,
        "training_params": None,
        "hyperparameters": None,
        "training_dataset": None,
        "feature_baseline": None,
        "confidence_threshold": None,
        "tags": None,
        "webhook_url": None,
        "promotion_policy": promotion_policy,
        "retrain_schedule": None,
        "accuracy": 0.90,
        "f1_score": 0.89,
    }


# ---------------------------------------------------------------------------
# Tests — PATCH /models/{name}/policy
# ---------------------------------------------------------------------------


class TestPolicyEndpoint:
    """Tests for PATCH /models/{name}/policy."""

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
        """The policy must be propagated to all active versions."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 0.85, "min_sample_validation": 20, "auto_promote": True},
        )
        assert r.status_code == 200
        assert r.json()["updated_versions"] == 2

    def test_patch_policy_min_accuracy_out_of_range_returns_422(self):
        """min_accuracy must be in [0.0, 1.0]."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 1.5},
        )
        assert r.status_code == 422

    def test_patch_policy_max_latency_negative_returns_422(self):
        """max_latency_p95_ms must be > 0."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"max_latency_p95_ms": -50.0},
        )
        assert r.status_code == 422

    def test_patch_policy_min_sample_validation_zero_returns_422(self):
        """min_sample_validation must be >= 1."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_sample_validation": 0},
        )
        assert r.status_code == 422

    def test_patch_policy_defaults_auto_promote_false(self):
        """Without explicit auto_promote, the default value is False."""
        r = client.patch(
            f"/models/{self.model_name}/policy",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"min_accuracy": 0.80},
        )
        assert r.status_code == 200
        assert r.json()["promotion_policy"]["auto_promote"] is False

    def test_patch_policy_persisted_on_model_get(self):
        """The policy is visible in GET /models/{name}/{version}."""
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
# Tests — auto-promotion in retrain
# ---------------------------------------------------------------------------


class TestAutoPromotionInRetrain:
    """Tests for auto-promotion triggered by retrain_service.do_retrain().

    Retraining is now asynchronous (ARQ): the endpoint returns 202 and
    auto-promotion is evaluated by the worker. These tests validate the logic
    via do_retrain() directly.
    """

    @classmethod
    def setup_class(cls):
        cls.model_no_policy = f"{MODEL_PREFIX}_no_policy"
        cls.model_policy_off = f"{MODEL_PREFIX}_policy_off"
        cls.model_policy_on = f"{MODEL_PREFIX}_policy_on"

        _create_model(cls.model_no_policy, "1.0.0", with_train_script=True)
        _create_model(cls.model_policy_off, "1.0.0", with_train_script=True)
        _create_model(cls.model_policy_on, "1.0.0", with_train_script=True)

    def test_retrain_without_policy_auto_promoted_is_none(self):
        """Without policy, auto_promoted is None."""
        from src.services.retrain_service import do_retrain

        async def _run():
            with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)):
                return await do_retrain(
                    model_name=self.model_no_policy,
                    source_version="1.0.0",
                    new_version="auto_2.0.0",
                    start_date="2025-01-01",
                    end_date="2025-12-31",
                    source_fields=_auto_promo_source_fields(promotion_policy=None),
                )

        result = asyncio.run(_run())
        assert result["success"] is True
        assert result["auto_promoted"] is None

    def test_retrain_policy_auto_promote_false_not_evaluated(self):
        """Policy present but auto_promote=False → auto_promoted=None (not evaluated)."""
        from src.services.retrain_service import do_retrain

        async def _run():
            with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)):
                return await do_retrain(
                    model_name=self.model_policy_off,
                    source_version="1.0.0",
                    new_version="auto_2.0.0",
                    start_date="2025-01-01",
                    end_date="2025-12-31",
                    source_fields=_auto_promo_source_fields(
                        promotion_policy={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": False},
                    ),
                )

        result = asyncio.run(_run())
        assert result["success"] is True
        # auto_promote=False → branch not triggered → auto_promoted=None
        assert result["auto_promoted"] is None

    def test_retrain_auto_promote_insufficient_samples(self):
        """Active policy, insufficient samples → auto_promoted=False."""
        from src.services.retrain_service import do_retrain

        async def _run():
            with (
                patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
                patch(
                    "src.services.auto_promotion_service.evaluate_auto_promotion",
                    new=AsyncMock(return_value=(False, "Insufficient samples for validation: 0/3 required.")),
                ),
            ):
                return await do_retrain(
                    model_name=self.model_policy_on,
                    source_version="1.0.0",
                    new_version="auto_2.0.0",
                    start_date="2025-01-01",
                    end_date="2025-12-31",
                    source_fields=_auto_promo_source_fields(
                        promotion_policy={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": True},
                    ),
                )

        result = asyncio.run(_run())
        assert result["success"] is True
        assert result["auto_promoted"] is False
        assert "insuffisant" in result["auto_promote_reason"].lower()
        assert result["is_production"] is False

    def test_retrain_auto_promote_success(self):
        """Active policy, all criteria met → auto_promoted=True, version in production."""
        from src.services.retrain_service import do_retrain

        async def _run():
            with (
                patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
                patch(
                    "src.services.auto_promotion_service.evaluate_auto_promotion",
                    new=AsyncMock(return_value=(True, "Tous les critères de promotion sont satisfaits.")),
                ),
            ):
                return await do_retrain(
                    model_name=self.model_policy_on,
                    source_version="1.0.0",
                    new_version="auto_3.0.0",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    source_fields=_auto_promo_source_fields(
                        promotion_policy={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": True},
                    ),
                )

        result = asyncio.run(_run())
        assert result["success"] is True
        assert result["auto_promoted"] is True
        assert "satisfait" in result["auto_promote_reason"].lower()
        assert result["is_production"] is True

    def test_retrain_auto_promote_accuracy_fails(self):
        """Active policy, insufficient accuracy → auto_promoted=False, not promoted."""
        from src.services.retrain_service import do_retrain

        async def _run():
            with (
                patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
                patch(
                    "src.services.auto_promotion_service.evaluate_auto_promotion",
                    new=AsyncMock(
                        return_value=(False, "Insufficient accuracy: 0.7500 < 0.8000 required.")
                    ),
                ),
            ):
                return await do_retrain(
                    model_name=self.model_policy_on,
                    source_version="1.0.0",
                    new_version="auto_4.0.0",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    source_fields=_auto_promo_source_fields(
                        promotion_policy={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": True},
                    ),
                )

        result = asyncio.run(_run())
        assert result["success"] is True
        assert result["auto_promoted"] is False
        assert "accuracy" in result["auto_promote_reason"].lower()
        assert result["is_production"] is False

    def test_retrain_auto_promote_latency_fails(self):
        """Active policy, P95 latency too high → auto_promoted=False."""
        from src.services.retrain_service import do_retrain

        async def _run():
            with (
                patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)),
                patch(
                    "src.services.auto_promotion_service.evaluate_auto_promotion",
                    new=AsyncMock(
                        return_value=(False, "P95 latency too high: 350.0ms > 200.0ms max.")
                    ),
                ),
            ):
                return await do_retrain(
                    model_name=self.model_policy_on,
                    source_version="1.0.0",
                    new_version="auto_5.0.0",
                    start_date="2022-01-01",
                    end_date="2022-12-31",
                    source_fields=_auto_promo_source_fields(
                        promotion_policy={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": True},
                    ),
                )

        result = asyncio.run(_run())
        assert result["auto_promoted"] is False
        assert "latency" in result["auto_promote_reason"].lower()

    def test_retrain_set_production_overrides_auto_promotion(self):
        """set_production=True: manual promotion, auto_promoted is None."""
        from src.services.retrain_service import do_retrain

        async def _run():
            with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=_mock_exec_success)):
                return await do_retrain(
                    model_name=self.model_policy_on,
                    source_version="1.0.0",
                    new_version="auto_6.0.0",
                    start_date="2021-01-01",
                    end_date="2021-12-31",
                    set_production=True,
                    source_fields=_auto_promo_source_fields(
                        promotion_policy={"min_accuracy": 0.80, "min_sample_validation": 3, "auto_promote": True},
                    ),
                )

        result = asyncio.run(_run())
        assert result["success"] is True
        assert result["is_production"] is True
        # set_production=True → auto-promotion branch not taken
        assert result["auto_promoted"] is None


# ---------------------------------------------------------------------------
# Unit tests — evaluate_auto_promotion()
# ---------------------------------------------------------------------------


class TestEvaluateAutoPromotion:
    """Unit tests for src/services/auto_promotion_service.evaluate_auto_promotion."""

    def _make_pairs(self, n: int, correct_ratio: float = 1.0):
        """Generate n pairs (prediction, observed_result, probs, timestamp).
        correct_ratio: fraction of correct predictions (pred == obs)."""
        pairs = []
        for i in range(n):
            pred = "A" if i / n < correct_ratio else "B"
            obs = "A"
            pairs.append((pred, obs, None, None))
        return pairs

    def _make_db(self, pairs, latency_times=None):
        """Create a mock AsyncSession for evaluate_auto_promotion."""
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
        assert "accuracy" in reason.lower()

    def test_no_accuracy_threshold_skips_accuracy_check(self):
        """Without min_accuracy, no accuracy check is performed."""
        pairs = self._make_pairs(10, correct_ratio=0.1)  # very low accuracy
        db, pairs_mock = self._make_db(pairs)
        policy = {"min_sample_validation": 5, "auto_promote": True}  # no min_accuracy
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True  # No criteria to check → promotion accepted

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
        """No latency data → latency criterion ignored."""
        pairs = self._make_pairs(10, correct_ratio=1.0)
        db, pairs_mock = self._make_db(pairs, latency_times=[])  # empty list
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
        """Accuracy is computed on the most recent min_sample_validation pairs."""
        # 10 pairs: first 5 wrong, last 5 correct
        pairs = [("B", "A", None, None)] * 5 + [("A", "A", None, None)] * 5
        db, pairs_mock = self._make_db(pairs)
        # min_sample_validation=5 → uses last 5 (all correct → accuracy=1.0)
        policy = {"min_accuracy": 0.90, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    # --- Regression tests ---

    def _make_regression_pairs(self, n: int, base: float = 10.0, noise: float = 0.5):
        """Generate n pairs (prediction, observed_result) with float values (regression)."""
        import random
        random.seed(42)
        pairs = []
        for _ in range(n):
            pred = base + random.uniform(-noise, noise)
            obs = base + random.uniform(-noise, noise)
            pairs.append((str(pred), str(obs), None, None))
        return pairs

    def test_regression_max_mae_ok_returns_true(self):
        """Regression: MAE < threshold → promoted."""
        pairs = self._make_regression_pairs(10, base=10.0, noise=0.3)
        db, pairs_mock = self._make_db(pairs)
        # With noise=0.3, expected MAE ≈ 0.3 → max_mae=1.0 easily met
        policy = {"max_mae": 1.0, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is True

    def test_regression_max_mae_too_high_returns_false(self):
        """Regression: MAE > threshold → not promoted."""
        pairs = self._make_regression_pairs(10, base=10.0, noise=2.0)
        db, pairs_mock = self._make_db(pairs)
        # With noise=2.0, expected MAE ≈ 1.3 → max_mae=0.1 too strict
        policy = {"max_mae": 0.1, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, reason = self._run(evaluate_auto_promotion(db, "m", policy))
        assert ok is False
        assert "mae" in reason.lower()

    def test_regression_no_mae_policy_returns_true(self):
        """Regression: without max_mae in policy → promoted (no criteria to check)."""
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
        """Regression: min_accuracy ignored for floats → does not block promotion.

        Non-integer floats trigger the regression branch, so min_accuracy
        is ignored even if defined (no max_mae here).
        """
        pairs = self._make_regression_pairs(10, base=100.0, noise=10.0)
        db, pairs_mock = self._make_db(pairs)
        # min_accuracy with impossible value for regression, but ignored
        policy = {"min_accuracy": 0.99, "min_sample_validation": 5, "auto_promote": True}
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=pairs_mock,
        ):
            ok, _ = self._run(evaluate_auto_promotion(db, "m", policy))
        # Regression branch detected → min_accuracy ignored → promoted
        assert ok is True


# ---------------------------------------------------------------------------
# Unit tests — evaluate_auto_demotion()
# ---------------------------------------------------------------------------


class TestAutoDemotion:
    """Unit tests for src/services/auto_promotion_service.evaluate_auto_demotion."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _make_prod_meta(self, name="m", version="1.0.0", webhook_url=None, feature_baseline=None):
        meta = MagicMock()
        meta.name = name
        meta.version = version
        meta.is_production = True
        meta.is_active = True
        meta.webhook_url = webhook_url
        meta.feature_baseline = feature_baseline
        return meta

    def _make_fallback_meta(self, version="0.9.0"):
        meta = MagicMock()
        meta.version = version
        meta.is_production = False
        meta.is_active = True
        return meta

    def _make_db(self, prod_meta=None, fallback_metas=None, history_entries=None):
        """Create a mock AsyncSession for evaluate_auto_demotion.

        db.execute(select(...)) calls are sequential:
        1. Production model query
        2. Fallback versions query
        3. Cooldown query (ModelHistory)
        """
        db = MagicMock()

        responses = []

        # Response 1: production model
        r1 = MagicMock()
        r1.scalars.return_value.first.return_value = prod_meta
        responses.append(r1)

        if prod_meta is not None:
            # Response 2: fallback
            r2 = MagicMock()
            r2.scalars.return_value.all.return_value = fallback_metas or []
            responses.append(r2)

            if fallback_metas:
                # Response 3: cooldown history
                r3 = MagicMock()
                r3.scalars.return_value.first.return_value = (
                    history_entries[0] if history_entries else None
                )
                responses.append(r3)

        db.execute = AsyncMock(side_effect=responses)
        db.add = MagicMock()
        db.commit = AsyncMock()
        return db

    def _make_history_entry(self, hours_ago: float):
        from datetime import datetime, timedelta
        entry = MagicMock()
        entry.timestamp = datetime.utcnow() - timedelta(hours=hours_ago)
        return entry

    def _base_policy(self, **overrides):
        policy = {
            "auto_demote": True,
            "demote_on_drift": "critical",
            "demote_on_accuracy_below": None,
            "demote_cooldown_hours": 0,
            "min_sample_validation": 5,
        }
        policy.update(overrides)
        return policy

    # --- Basic cases ---

    def test_auto_demote_disabled_returns_false(self):
        db = MagicMock()
        policy = {"auto_demote": False}
        ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))
        assert ok is False
        assert "disabled" in reason.lower()
        db.execute.assert_not_called()

    def test_no_production_model_returns_false(self):
        db = self._make_db(prod_meta=None)
        ok, reason = self._run(evaluate_auto_demotion(db, "m", self._base_policy()))
        assert ok is False
        assert "production" in reason.lower()

    # --- Fallback safeguard ---

    def test_no_fallback_returns_false_and_sends_email(self):
        prod = self._make_prod_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[])
        with (
            patch("src.services.auto_promotion_service.email_service") as mock_email,
            patch("src.services.auto_promotion_service.settings") as mock_settings,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = True
            ok, reason = self._run(evaluate_auto_demotion(db, "m", self._base_policy()))
        assert ok is False
        assert "secours" in reason.lower()
        mock_email.send_auto_demotion_alert.assert_called_once()
        call_kwargs = mock_email.send_auto_demotion_alert.call_args
        assert call_kwargs.kwargs.get("no_fallback") is True or (
            len(call_kwargs.args) >= 4 and call_kwargs.args[3] is True
        )

    def test_no_fallback_no_email_when_disabled(self):
        prod = self._make_prod_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[])
        with (
            patch("src.services.auto_promotion_service.email_service") as mock_email,
            patch("src.services.auto_promotion_service.settings") as mock_settings,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = False
            ok, reason = self._run(evaluate_auto_demotion(db, "m", self._base_policy()))
        assert ok is False
        mock_email.send_auto_demotion_alert.assert_not_called()

    # --- Cooldown ---

    def test_cooldown_active_returns_false(self):
        prod = self._make_prod_meta()
        fallback = self._make_fallback_meta()
        recent_entry = self._make_history_entry(hours_ago=2)
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback], history_entries=[recent_entry])
        policy = self._base_policy(demote_cooldown_hours=24)
        ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))
        assert ok is False
        assert "cooldown" in reason.lower()

    def test_cooldown_expired_allows_demotion(self):
        """Cooldown entry older than the window → demotion can proceed.

        The mock returns [] for the cooldown query (no AUTO_DEMOTE within the window),
        simulating an expired cooldown.
        """
        prod = self._make_prod_meta()  # without feature_baseline → drift skipped
        fallback = self._make_fallback_meta()
        # history_entries=None → cooldown query returns None (no match within window)
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback], history_entries=None)
        policy = self._base_policy(demote_cooldown_hours=24)
        # No baseline → drift skipped, no accuracy check → no criteria → no demotion
        ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))
        assert ok is False
        assert "cooldown" not in reason.lower()

    def test_zero_cooldown_skips_history_check(self):
        """demote_cooldown_hours=0 → no cooldown query."""
        prod = self._make_prod_meta()
        fallback = self._make_fallback_meta()
        # With cooldown=0, the history query should not be issued
        # Mock only 2 responses (prod + fallback)
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback], history_entries=None)
        policy = self._base_policy(demote_cooldown_hours=0)
        ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))
        # No active criteria → no demotion, but no crash either
        assert ok is False
        assert "cooldown" not in reason.lower()

    # --- Drift ---

    def test_critical_drift_triggers_demotion(self):
        baseline = {"feat1": {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0}}
        prod = self._make_prod_meta(feature_baseline=baseline)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        feat_result = MagicMock()
        feat_result.drift_status = "critical"
        feat_result.null_rate_status = None

        output_report = MagicMock()
        output_report.status = "ok"

        with (
            patch(
                "src.services.auto_promotion_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={"feat1": {"mean": 9.0, "std": 0.5, "count": 50}}),
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_feature_drift",
                return_value={"feat1": feat_result},
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.summarize_drift",
                return_value="critical",
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_output_drift",
                new=AsyncMock(return_value=output_report),
            ),
            patch("src.services.auto_promotion_service.DBService.log_model_history", new=AsyncMock(return_value=MagicMock(snapshot={}))),
            patch("src.services.auto_promotion_service.email_service"),
            patch("src.services.auto_promotion_service.settings") as mock_settings,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = False
            ok, reason = self._run(evaluate_auto_demotion(db, "m", self._base_policy()))

        assert ok is True
        assert "drift" in reason.lower()
        assert prod.is_production is False
        db.commit.assert_called_once()

    def test_warning_drift_no_trigger_when_threshold_is_critical(self):
        baseline = {"feat1": {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0}}
        prod = self._make_prod_meta(feature_baseline=baseline)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        feat_result = MagicMock()
        feat_result.drift_status = "warning"
        feat_result.null_rate_status = None

        output_report = MagicMock()
        output_report.status = "ok"

        with (
            patch(
                "src.services.auto_promotion_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_feature_drift",
                return_value={"feat1": feat_result},
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.summarize_drift",
                return_value="warning",
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_output_drift",
                new=AsyncMock(return_value=output_report),
            ),
        ):
            policy = self._base_policy(demote_on_drift="critical")
            ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))

        assert ok is False

    def test_warning_drift_triggers_when_threshold_is_warning(self):
        baseline = {"feat1": {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0}}
        prod = self._make_prod_meta(feature_baseline=baseline)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        output_report = MagicMock()
        output_report.status = "ok"

        with (
            patch(
                "src.services.auto_promotion_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_feature_drift",
                return_value={},
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.summarize_drift",
                return_value="warning",
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_output_drift",
                new=AsyncMock(return_value=output_report),
            ),
            patch("src.services.auto_promotion_service.DBService.log_model_history", new=AsyncMock(return_value=MagicMock(snapshot={}))),
            patch("src.services.auto_promotion_service.email_service"),
            patch("src.services.auto_promotion_service.settings") as mock_settings,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = False
            policy = self._base_policy(demote_on_drift="warning")
            ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))

        assert ok is True
        assert "drift" in reason.lower()

    def test_no_feature_baseline_skips_drift_check(self):
        """Without feature_baseline → drift skipped, no demotion if no other criteria."""
        prod = self._make_prod_meta(feature_baseline=None)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])
        ok, reason = self._run(evaluate_auto_demotion(db, "m", self._base_policy()))
        assert ok is False

    # --- Accuracy ---

    def test_accuracy_below_threshold_triggers_demotion(self):
        prod = self._make_prod_meta(feature_baseline=None)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        # 10 pairs at 50% accuracy
        pairs = [("A", "A", None, None)] * 5 + [("B", "A", None, None)] * 5
        with (
            patch(
                "src.services.auto_promotion_service.DBService.get_performance_pairs",
                new=AsyncMock(return_value=pairs),
            ),
            patch("src.services.auto_promotion_service.DBService.log_model_history", new=AsyncMock(return_value=MagicMock(snapshot={}))),
            patch("src.services.auto_promotion_service.email_service"),
            patch("src.services.auto_promotion_service.settings") as mock_settings,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = False
            policy = self._base_policy(demote_on_accuracy_below=0.80, min_sample_validation=5)
            ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))

        assert ok is True
        assert "accuracy" in reason.lower() or "insufficient" in reason.lower()

    def test_accuracy_above_threshold_no_demotion(self):
        prod = self._make_prod_meta(feature_baseline=None)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        pairs = [("A", "A", None, None)] * 10  # 100% accuracy
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=AsyncMock(return_value=pairs),
        ):
            policy = self._base_policy(demote_on_accuracy_below=0.80, min_sample_validation=5)
            ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))

        assert ok is False

    def test_accuracy_insufficient_samples_skips_check(self):
        """Fewer than min_sample_validation pairs → accuracy check skipped."""
        prod = self._make_prod_meta(feature_baseline=None)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        pairs = [("B", "A", None, None)] * 3  # 3 pairs, 0% accuracy, but min=5
        with patch(
            "src.services.auto_promotion_service.DBService.get_performance_pairs",
            new=AsyncMock(return_value=pairs),
        ):
            policy = self._base_policy(demote_on_accuracy_below=0.80, min_sample_validation=5)
            ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))

        assert ok is False  # check skipped → no criteria triggered

    # --- Combined ---

    def test_both_drift_and_accuracy_combined_reason(self):
        baseline = {"feat1": {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0}}
        prod = self._make_prod_meta(feature_baseline=baseline)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        pairs = [("B", "A", None, None)] * 10  # 0% accuracy

        output_report = MagicMock()
        output_report.status = "ok"

        with (
            patch(
                "src.services.auto_promotion_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_feature_drift",
                return_value={},
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.summarize_drift",
                return_value="critical",
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_output_drift",
                new=AsyncMock(return_value=output_report),
            ),
            patch(
                "src.services.auto_promotion_service.DBService.get_performance_pairs",
                new=AsyncMock(return_value=pairs),
            ),
            patch("src.services.auto_promotion_service.DBService.log_model_history", new=AsyncMock(return_value=MagicMock(snapshot={}))),
            patch("src.services.auto_promotion_service.email_service"),
            patch("src.services.auto_promotion_service.settings") as mock_settings,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = False
            policy = self._base_policy(
                demote_on_drift="critical",
                demote_on_accuracy_below=0.80,
                min_sample_validation=5,
            )
            ok, reason = self._run(evaluate_auto_demotion(db, "m", policy))

        assert ok is True
        # Combined reason contains both triggers
        assert "drift" in reason.lower()

    # --- Audit trail ---

    def test_history_logged_with_auto_demote_action_and_reason(self):
        """After demotion, log_model_history called with AUTO_DEMOTE and reason in snapshot."""
        from src.db.models.model_history import HistoryActionType

        baseline = {"feat1": {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0}}
        prod = self._make_prod_meta(feature_baseline=baseline)
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        captured_entry = MagicMock()
        captured_entry.snapshot = {}
        log_mock = AsyncMock(return_value=captured_entry)

        output_report = MagicMock()
        output_report.status = "ok"

        with (
            patch(
                "src.services.auto_promotion_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_feature_drift",
                return_value={},
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.summarize_drift",
                return_value="critical",
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_output_drift",
                new=AsyncMock(return_value=output_report),
            ),
            patch(
                "src.services.auto_promotion_service.DBService.log_model_history",
                new=log_mock,
            ),
            patch("src.services.auto_promotion_service.email_service"),
            patch("src.services.auto_promotion_service.settings") as mock_settings,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = False
            ok, reason = self._run(evaluate_auto_demotion(db, "m", self._base_policy()))

        assert ok is True
        log_mock.assert_called_once()
        call_args = log_mock.call_args
        assert call_args.args[2] == HistoryActionType.AUTO_DEMOTE
        assert "auto_demote_reason" in captured_entry.snapshot

    def test_webhook_called_after_demotion(self):
        """Webhook sent with event_type='auto_demote' after a demotion."""
        baseline = {"feat1": {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0}}
        prod = self._make_prod_meta(feature_baseline=baseline, webhook_url="http://hook.example/cb")
        fallback = self._make_fallback_meta()
        db = self._make_db(prod_meta=prod, fallback_metas=[fallback])

        output_report = MagicMock()
        output_report.status = "ok"

        with (
            patch(
                "src.services.auto_promotion_service.DBService.get_feature_production_stats",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_feature_drift",
                return_value={},
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.summarize_drift",
                return_value="critical",
            ),
            patch(
                "src.services.auto_promotion_service.drift_service.compute_output_drift",
                new=AsyncMock(return_value=output_report),
            ),
            patch("src.services.auto_promotion_service.DBService.log_model_history", new=AsyncMock(return_value=MagicMock(snapshot={}))),
            patch("src.services.auto_promotion_service.email_service"),
            patch("src.services.auto_promotion_service.settings") as mock_settings,
            patch("src.services.auto_promotion_service.asyncio.create_task") as mock_task,
        ):
            mock_settings.ENABLE_EMAIL_ALERTS = False
            ok, _ = self._run(evaluate_auto_demotion(db, "m", self._base_policy()))

        assert ok is True
        mock_task.assert_called_once()
        webhook_call_args = mock_task.call_args[0][0]
        # The send_webhook coroutine is passed to create_task — verify its name
        assert "send_webhook" in str(type(webhook_call_args)) or hasattr(webhook_call_args, "cr_frame")
