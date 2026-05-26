"""
A/B Testing & Shadow Deployment Tests.

Strategy:
  - SQLite in-memory (conftest) + FakeRedis (model_service._redis)
  - sklearn models created on the fly, injected into the cache
  - The shadow background task is patched to use _TestSessionLocal
    (production AsyncSessionLocal points to PostgreSQL — unavailable in tests)
  - Each test cleans up the cache with try/finally
"""
import asyncio
import io
import joblib
from collections import Counter
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

# Tokens and names unique to this file
TEST_TOKEN = "test-token-ab-shadow-xr9k"
AB_MODEL = "ab_test_model"
SHADOW_MODEL = "shadow_test_model"
V1 = "1.0.0"
V2 = "2.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lr_model() -> LogisticRegression:
    """Create a minimal LogisticRegression with feature_names_in_."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]})
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_cache(model_name: str, version: str, model) -> str:
    """Inject a sklearn model into the FakeRedis cache."""
    key = f"{model_name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=model_name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


def _headers():
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


# ---------------------------------------------------------------------------
# Setup: user + ModelMetadata
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        # Test user
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_ab_shadow_user",
                email="ab_shadow@test.com",
                api_token=TEST_TOKEN,
                role="admin",
                rate_limit=99999,
            )

        # A/B versions for AB_MODEL
        for ver, mode, weight, is_prod in [
            (V1, "ab_test", 0.8, False),
            (V2, "ab_test", 0.2, False),
        ]:
            if not await DBService.get_model_metadata(db, AB_MODEL, ver):
                await DBService.create_model_metadata(
                    db,
                    name=AB_MODEL,
                    version=ver,
                    minio_bucket="models",
                    minio_object_key=f"{AB_MODEL}/v{ver}.joblib",
                    is_active=True,
                    is_production=is_prod,
                    deployment_mode=mode,
                    traffic_weight=weight,
                )

        # Production + shadow versions for SHADOW_MODEL
        for ver, mode, weight, is_prod in [
            (V1, None, None, True),    # production (legacy)
            (V2, "shadow", None, False),
        ]:
            if not await DBService.get_model_metadata(db, SHADOW_MODEL, ver):
                await DBService.create_model_metadata(
                    db,
                    name=SHADOW_MODEL,
                    version=ver,
                    minio_bucket="models",
                    minio_object_key=f"{SHADOW_MODEL}/v{ver}.joblib",
                    is_active=True,
                    is_production=is_prod,
                    deployment_mode=mode,
                    traffic_weight=weight,
                )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Test 1 — A/B routing: weighted traffic distribution
# ---------------------------------------------------------------------------


def test_ab_routing_distributes_traffic():
    """
    With v1 (80%) and v2 (20%), over N calls without explicit version,
    both versions must be selected with the expected ratio.
    Tolerance ±20 percentage points (probabilistic).
    """
    model = _make_lr_model()
    key_v1 = _inject_cache(AB_MODEL, V1, model)
    key_v2 = _inject_cache(AB_MODEL, V2, model)
    try:
        version_counts: Counter = Counter()
        n = 100
        for _ in range(n):
            r = client.post(
                "/predict",
                headers=_headers(),
                json={"model_name": AB_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
            )
            assert r.status_code == 200
            data = r.json()
            version_counts[data["model_version"]] += 1

        # Both versions must be selected
        assert V1 in version_counts, "v1 was never selected"
        assert V2 in version_counts, "v2 was never selected"

        # Approximate ratio (±25 pp tolerance for N=100)
        v1_ratio = version_counts[V1] / n
        assert 0.55 <= v1_ratio <= 1.0, f"v1 ratio too far from 80%: {v1_ratio:.0%}"
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 2 — A/B routing: selected_version is present in the response
# ---------------------------------------------------------------------------


def test_ab_routing_returns_selected_version():
    """During A/B routing, selected_version is populated in the response."""
    model = _make_lr_model()
    key_v1 = _inject_cache(AB_MODEL, V1, model)
    key_v2 = _inject_cache(AB_MODEL, V2, model)
    try:
        r = client.post(
            "/predict",
            headers=_headers(),
            json={"model_name": AB_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
        )
        assert r.status_code == 200
        data = r.json()
        # selected_version must be present (non-None) during automatic routing
        assert data.get("selected_version") is not None
        assert data["selected_version"] in [V1, V2]
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 3 — Explicit version bypasses A/B routing
# ---------------------------------------------------------------------------


def test_explicit_version_bypasses_ab_routing():
    """POST /predict with explicit model_version → always the requested version."""
    model = _make_lr_model()
    key_v1 = _inject_cache(AB_MODEL, V1, model)
    key_v2 = _inject_cache(AB_MODEL, V2, model)
    try:
        for _ in range(10):
            r = client.post(
                "/predict",
                headers=_headers(),
                json={
                    "model_name": AB_MODEL,
                    "model_version": V1,
                    "features": {"f1": 1.0, "f2": 2.0},
                },
            )
            assert r.status_code == 200
            assert r.json()["model_version"] == V1
            # selected_version is None when version is explicit
            assert r.json()["selected_version"] is None
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 4 — Shadow: response comes from the production model
# ---------------------------------------------------------------------------


def test_shadow_primary_response_is_production():
    """
    With v1 (production) and v2 (shadow), POST /predict without version
    → the response comes from v1 (not the shadow).
    """
    model = _make_lr_model()
    key_v1 = _inject_cache(SHADOW_MODEL, V1, model)
    key_v2 = _inject_cache(SHADOW_MODEL, V2, model)
    try:
        # Patch AsyncSessionLocal so the shadow task uses the test DB
        with patch("src.api.predict.AsyncSessionLocal", _TestSessionLocal):
            r = client.post(
                "/predict",
                headers=_headers(),
                json={"model_name": SHADOW_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
            )
        assert r.status_code == 200
        data = r.json()
        # Client response must come from v1 (production / is_production=True)
        assert data["model_version"] == V1
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 5 — Shadow: shadow prediction logged with is_shadow=True
# ---------------------------------------------------------------------------


def test_shadow_prediction_logged_with_is_shadow_flag():
    """
    After a call with active shadow, a prediction with is_shadow=True is in DB for v2.
    """
    model = _make_lr_model()
    key_v1 = _inject_cache(SHADOW_MODEL, V1, model)
    key_v2 = _inject_cache(SHADOW_MODEL, V2, model)
    id_obs_val = f"shadow-obs-{datetime.utcnow().strftime('%f')}"
    try:
        with patch("src.api.predict.AsyncSessionLocal", _TestSessionLocal):
            r = client.post(
                "/predict",
                headers=_headers(),
                json={
                    "model_name": SHADOW_MODEL,
                    "id_obs": id_obs_val,
                    "features": {"f1": 1.0, "f2": 2.0},
                },
            )
        assert r.status_code == 200
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))

    # Verify in DB: a shadow prediction for v2 must exist
    async def _check():
        async with _TestSessionLocal() as db:
            start = datetime.now(timezone.utc) - timedelta(minutes=5)
            end = datetime.now(timezone.utc) + timedelta(minutes=5)
            rows, total = await DBService.get_predictions(
                db=db,
                model_name=SHADOW_MODEL,
                start=start,
                end=end,
                id_obs=id_obs_val,
                limit=20,
            )
            return rows

    rows = asyncio.run(_check())
    shadow_rows = [r for r in rows if r.is_shadow]
    prod_rows = [r for r in rows if not r.is_shadow]

    assert len(prod_rows) >= 1, "Production prediction missing from DB"
    assert len(shadow_rows) >= 1, "Shadow prediction missing from DB"
    assert prod_rows[0].model_version == V1
    assert shadow_rows[0].model_version == V2


# ---------------------------------------------------------------------------
# Test 6 — Weight validation: sum > 1.0 → 422
# ---------------------------------------------------------------------------


def test_patch_ab_weight_sum_exceeds_one_returns_422():
    """
    PATCH of a version with traffic_weight that would exceed 1.0 → 422.
    v1 is already at 0.8 → trying to set v2 to 0.9 must fail.
    """
    r = client.patch(
        f"/models/{AB_MODEL}/{V2}",
        headers=_headers(),
        json={"deployment_mode": "ab_test", "traffic_weight": 0.9},
    )
    assert r.status_code == 422
    assert "1.0" in r.json()["detail"] or "dépasse" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Test 7 — Valid PATCH: update deployment_mode
# ---------------------------------------------------------------------------


def test_patch_deployment_mode_valid():
    """PATCH with traffic_weight ≤ 1.0 − 0.8 = 0.2 → 200."""
    # v1 weighs 0.8, set v2 to 0.2 → sum = 1.0: acceptable
    r = client.patch(
        f"/models/{AB_MODEL}/{V2}",
        headers=_headers(),
        json={"deployment_mode": "ab_test", "traffic_weight": 0.2},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["deployment_mode"] == "ab_test"
    assert abs(data["traffic_weight"] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Test 8 — GET /models/{name}/ab-compare: response structure
# ---------------------------------------------------------------------------


def test_get_ab_compare_structure():
    """GET /models/{name}/ab-compare → ABCompareResponse with expected fields."""
    r = client.get(
        f"/models/{AB_MODEL}/ab-compare",
        headers=_headers(),
        params={"days": 30},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == AB_MODEL
    assert "period_days" in data
    assert "versions" in data
    assert isinstance(data["versions"], list)
    # Each version must have the expected fields
    for vs in data["versions"]:
        assert "version" in vs
        assert "total_predictions" in vs
        assert "shadow_predictions" in vs
        assert "error_rate" in vs
        assert "prediction_distribution" in vs
    # The ab_significance field must be present (None or valid object)
    assert "ab_significance" in data
    sig = data["ab_significance"]
    if sig is not None:
        assert "metric" in sig
        assert "test" in sig
        assert "p_value" in sig
        assert "significant" in sig
        assert "current_samples" in sig


# ---------------------------------------------------------------------------
# Test 9 — GET /models/{name}/ab-compare: non-existent model → 404
# ---------------------------------------------------------------------------


def test_get_ab_compare_unknown_model_returns_404():
    """GET /models/{name}/ab-compare for a non-existent model → 404."""
    r = client.get(
        "/models/nonexistent_model_ab_xyz/ab-compare",
        headers=_headers(),
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 10 — Legacy fallback: no mode configured → is_production used
# ---------------------------------------------------------------------------


async def _create_legacy_model():
    """Create a model with legacy behavior (is_production, no deployment_mode)."""
    async with _TestSessionLocal() as db:
        name = "legacy_model_ab_test"
        if not await DBService.get_model_metadata(db, name, V1):
            await DBService.create_model_metadata(
                db,
                name=name,
                version=V1,
                minio_bucket="models",
                minio_object_key=f"{name}/v{V1}.joblib",
                is_active=True,
                is_production=True,
                deployment_mode=None,
                traffic_weight=None,
            )
        return name


def test_legacy_fallback_uses_production_version():
    """
    Without configured deployment_mode, fallback routing uses is_production=True.
    """
    legacy_model_name = asyncio.run(_create_legacy_model())
    model = _make_lr_model()
    key = _inject_cache(legacy_model_name, V1, model)
    try:
        r = client.post(
            "/predict",
            headers=_headers(),
            json={"model_name": legacy_model_name, "features": {"f1": 1.0, "f2": 2.0}},
        )
        assert r.status_code == 200
        assert r.json()["model_version"] == V1
    finally:
        asyncio.run(model_service.clear_cache(key))


# ---------------------------------------------------------------------------
# Test 11 — GET /models/{name}/shadow-compare: response structure
# ---------------------------------------------------------------------------


def test_get_shadow_compare_structure():
    """GET /models/{name}/shadow-compare → ShadowCompareResponse with expected fields."""
    r = client.get(
        f"/models/{SHADOW_MODEL}/shadow-compare",
        headers=_headers(),
        params={"period_days": 30},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == SHADOW_MODEL
    assert "period_days" in data
    assert data["period_days"] == 30
    assert "n_comparable" in data
    assert isinstance(data["n_comparable"], int)
    assert "agreement_rate" in data
    assert "shadow_confidence_delta" in data
    assert "shadow_latency_delta_ms" in data
    assert "shadow_accuracy" in data
    assert "production_accuracy" in data
    assert "accuracy_available" in data
    assert isinstance(data["accuracy_available"], bool)
    assert "recommendation" in data
    assert data["recommendation"] in (
        "shadow_better",
        "production_better",
        "equivalent",
        "insufficient_data",
    )
    assert data["shadow_version"] == V2
    assert data["production_version"] == V1


# ---------------------------------------------------------------------------
# Test 12 — GET /models/{name}/shadow-compare: non-existent model → 404
# ---------------------------------------------------------------------------


def test_get_shadow_compare_unknown_model_returns_404():
    """GET /models/{name}/shadow-compare for a non-existent model → 404."""
    r = client.get(
        "/models/nonexistent_shadow_model_xyz/shadow-compare",
        headers=_headers(),
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 13 — GET /models/{name}/shadow-compare: no shadow version → n_comparable=0
# ---------------------------------------------------------------------------


def test_get_shadow_compare_no_shadow_version():
    """Model without shadow version → shadow_version=None, n_comparable=0, insufficient_data."""
    r = client.get(
        f"/models/{AB_MODEL}/shadow-compare",
        headers=_headers(),
        params={"period_days": 30},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["shadow_version"] is None
    assert data["n_comparable"] == 0
    assert data["agreement_rate"] is None
    assert data["recommendation"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Test 14 — GET /models/{name}/shadow-compare: with pairs in DB → computed metrics
# ---------------------------------------------------------------------------


SHADOW_PAIRS_MODEL = "shadow_pairs_test_model"


async def _setup_shadow_pairs_model():
    """Create a dedicated model with shadow + production versions for the pairs test."""
    async with _TestSessionLocal() as db:
        for ver, mode, is_prod in [
            (V1, None, True),
            (V2, "shadow", False),
        ]:
            if not await DBService.get_model_metadata(db, SHADOW_PAIRS_MODEL, ver):
                await DBService.create_model_metadata(
                    db,
                    name=SHADOW_PAIRS_MODEL,
                    version=ver,
                    minio_bucket="models",
                    minio_object_key=f"{SHADOW_PAIRS_MODEL}/v{ver}.joblib",
                    is_active=True,
                    is_production=is_prod,
                    deployment_mode=mode,
                    traffic_weight=None,
                )

        user = await DBService.get_user_by_token(db, TEST_TOKEN)
        for i in range(15):
            id_obs = f"sc-pairs-{i:04d}"
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=SHADOW_PAIRS_MODEL,
                model_version=V1,
                input_features={"f1": float(i), "f2": float(i + 1)},
                prediction_result=0,
                probabilities=[0.75, 0.25],
                response_time_ms=10.0,
                id_obs=id_obs,
                is_shadow=False,
                max_confidence=0.75,
            )
            await DBService.create_prediction(
                db,
                user_id=user.id,
                model_name=SHADOW_PAIRS_MODEL,
                model_version=V2,
                input_features={"f1": float(i), "f2": float(i + 1)},
                prediction_result=0,
                probabilities=[0.90, 0.10],
                response_time_ms=8.0,
                id_obs=id_obs,
                is_shadow=True,
                max_confidence=0.90,
            )


asyncio.run(_setup_shadow_pairs_model())


def test_get_shadow_compare_with_pairs():
    """With 15 pairs in DB → n_comparable=15, agreement_rate computed, recommendation."""
    r = client.get(
        f"/models/{SHADOW_PAIRS_MODEL}/shadow-compare",
        headers=_headers(),
        params={"period_days": 30},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["n_comparable"] == 15
    assert data["agreement_rate"] is not None
    assert 0.0 <= data["agreement_rate"] <= 1.0
    # Both predict 0 → 100% agreement
    assert data["agreement_rate"] == 1.0
    # shadow_confidence_delta = 0.90 - 0.75 = +0.15
    assert data["shadow_confidence_delta"] is not None
    assert abs(data["shadow_confidence_delta"] - 0.15) < 0.01
    # shadow_latency_delta_ms = 8 - 10 = -2.0
    assert data["shadow_latency_delta_ms"] is not None
    assert abs(data["shadow_latency_delta_ms"] - (-2.0)) < 0.5
    # No observed_results → accuracy_available=False
    assert data["accuracy_available"] is False
    assert data["shadow_accuracy"] is None
    assert data["production_accuracy"] is None
    # recommendation: confidence_delta = 0.15 > 0.05 → shadow_better
    assert data["recommendation"] == "shadow_better"
