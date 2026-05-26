"""
Tests for the A/B statistical significance service.

Strategy:
  - Unit tests for compute_ab_significance service (no DB, no network)
  - Minimal integration test via the /ab-compare endpoint (SQLite in-memory)
"""

import asyncio

from fastapi.testclient import TestClient

from src.main import app
from src.services.ab_significance_service import (
    _cohen_d,
    _cohen_h,
    _min_samples_continuous,
    _min_samples_proportions,
    compute_ab_significance,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_cohen_h_equal_proportions():
    """Cohen h = 0 when both proportions are equal."""
    assert abs(_cohen_h(0.1, 0.1)) < 1e-10


def test_cohen_h_different_proportions():
    """Cohen h > 0 when proportions differ."""
    assert abs(_cohen_h(0.1, 0.3)) > 0


def test_cohen_d_equal_means():
    """Cohen d = 0 when means are identical."""
    assert _cohen_d(10.0, 10.0, 2.0, 2.0, 20, 20) == 0.0


def test_cohen_d_different_means():
    """Cohen d > 0 when means differ."""
    assert _cohen_d(10.0, 15.0, 2.0, 2.0, 20, 20) > 0.0


def test_min_samples_proportions_equal():
    """Equal proportions → 0 samples (no effect to detect)."""
    assert _min_samples_proportions(0.1, 0.1) == 0


def test_min_samples_proportions_large_gap():
    """Large difference between proportions → small minimum sample."""
    n = _min_samples_proportions(0.05, 0.30)
    assert 0 < n < 100


def test_min_samples_continuous_zero_effect():
    """Cohen d = 0 → 0 samples."""
    assert _min_samples_continuous(0.0) == 0


def test_min_samples_continuous_small_effect():
    """Small effect (d=0.2) → reasonable minimum sample (> 100)."""
    n = _min_samples_continuous(0.2)
    assert n > 100


# ---------------------------------------------------------------------------
# compute_ab_significance — Chi-squared cases
# ---------------------------------------------------------------------------


def _make_stats(version, n_total, n_errors, times=None):
    return {
        "version": version,
        "total_predictions": n_total,
        "error_count": n_errors,
        "response_times": times or [],
    }


def test_significance_returns_none_with_single_version():
    """Single version → None (test impossible)."""
    result = compute_ab_significance([_make_stats("v1", 100, 5)])
    assert result is None


def test_significance_returns_none_with_no_data():
    """No version with predictions → None."""
    result = compute_ab_significance([_make_stats("v1", 0, 0), _make_stats("v2", 0, 0)])
    assert result is None


def test_significance_chi2_with_clear_winner():
    """
    v1 has 5% error rate, v2 has 40% error rate over 500 predictions each.
    The Chi-squared test must be significant and designate v1 as winner.
    """
    stats = [
        _make_stats("v1", 500, 25),  # 5% error rate
        _make_stats("v2", 500, 200),  # 40% error rate
    ]
    result = compute_ab_significance(stats)

    assert result is not None
    assert result["test"] == "chi2"
    assert result["metric"] == "error_rate"
    assert result["significant"] is True
    assert result["winner"] == "v1"
    assert result["p_value"] < 0.05
    assert result["current_samples"]["v1"] == 500
    assert result["current_samples"]["v2"] == 500
    assert isinstance(result["min_samples_needed"], int)


def test_significance_chi2_no_errors_fallback():
    """
    No errors in either group → no Chi-squared, fallback to Mann-Whitney U if times available.
    """
    times_v1 = [10.0, 11.0, 10.5, 9.8, 10.2]
    times_v2 = [50.0, 55.0, 52.0, 48.0, 51.0]
    stats = [
        _make_stats("v1", 5, 0, times_v1),
        _make_stats("v2", 5, 0, times_v2),
    ]
    result = compute_ab_significance(stats)

    assert result is not None
    assert result["test"] == "mann_whitney_u"
    assert result["metric"] == "response_time_ms"


def test_significance_no_data_at_all_returns_none():
    """No errors and no response times → None."""
    stats = [_make_stats("v1", 5, 0), _make_stats("v2", 5, 0)]
    result = compute_ab_significance(stats)
    assert result is None


def test_significance_picks_two_most_active_versions():
    """
    Three versions available: the test uses the two with the most predictions.
    """
    stats = [
        _make_stats("v1", 50, 2),
        _make_stats("v2", 500, 200),
        _make_stats("v3", 100, 40),
    ]
    result = compute_ab_significance(stats)
    assert result is not None
    assert "v2" in result["current_samples"]
    assert "v3" in result["current_samples"]
    assert "v1" not in result["current_samples"]


def test_significance_confidence_level_respected():
    """The confidence threshold is reflected in the response."""
    stats = [_make_stats("v1", 200, 10), _make_stats("v2", 200, 80)]
    result = compute_ab_significance(stats, confidence_level=0.99)
    assert result is not None
    assert result["confidence_level"] == 0.99


def test_significance_tied_winner_is_none():
    """Identical error rates → winner=None."""
    stats = [
        _make_stats("v1", 200, 20),  # 10%
        _make_stats("v2", 200, 20),  # 10%
    ]
    result = compute_ab_significance(stats)
    # May be None (test impossible with h=0) or winner=None
    if result is not None:
        assert result["winner"] is None


# ---------------------------------------------------------------------------
# compute_ab_significance — Mann-Whitney U
# ---------------------------------------------------------------------------


def test_significance_mann_whitney_significant():
    """Two very different distributions → Mann-Whitney significant."""
    times_a = [10.0] * 30
    times_b = [100.0] * 30
    stats = [
        _make_stats("v1", 30, 0, times_a),
        _make_stats("v2", 30, 0, times_b),
    ]
    result = compute_ab_significance(stats)
    assert result is not None
    assert result["test"] == "mann_whitney_u"
    assert result["significant"] is True
    assert result["winner"] == "v1"
    assert result["p_value"] < 0.05


def test_significance_mann_whitney_not_significant():
    """Identical distributions → Mann-Whitney not significant."""
    times = [10.0, 11.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3]
    stats = [
        _make_stats("v1", 8, 0, list(times)),
        _make_stats("v2", 8, 0, list(times)),
    ]
    result = compute_ab_significance(stats)
    if result is not None:
        assert result["significant"] is False


def test_significance_response_fields_complete():
    """The response contains all required fields."""
    stats = [
        _make_stats("v1", 300, 15),
        _make_stats("v2", 300, 60),
    ]
    result = compute_ab_significance(stats)
    assert result is not None
    required_keys = {
        "metric",
        "test",
        "p_value",
        "significant",
        "confidence_level",
        "winner",
        "min_samples_needed",
        "current_samples",
    }
    assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Integration test: GET /models/{name}/ab-compare endpoint
# ---------------------------------------------------------------------------

_SIG_TOKEN = "test-token-ab-sig-e1hQw"
_SIG_MODEL = "sig_test_model"
_client = TestClient(app)


async def _create_sig_fixtures():
    from tests.conftest import _TestSessionLocal

    from src.services.db_service import DBService

    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, _SIG_TOKEN):
            await DBService.create_user(
                db,
                username="test_ab_sig_user",
                email="ab_sig@test.com",
                api_token=_SIG_TOKEN,
                role="admin",
                rate_limit=99999,
            )
        for ver, mode, weight in [("1.0.0", "ab_test", 0.6), ("2.0.0", "ab_test", 0.4)]:
            if not await DBService.get_model_metadata(db, _SIG_MODEL, ver):
                await DBService.create_model_metadata(
                    db,
                    name=_SIG_MODEL,
                    version=ver,
                    minio_bucket="models",
                    minio_object_key=f"{_SIG_MODEL}/v{ver}.joblib",
                    is_active=True,
                    is_production=False,
                    deployment_mode=mode,
                    traffic_weight=weight,
                )


asyncio.run(_create_sig_fixtures())


def test_ab_compare_endpoint_includes_significance_field():
    """GET /ab-compare → response contains the ab_significance field (None or valid object)."""
    headers = {"Authorization": f"Bearer {_SIG_TOKEN}"}
    r = _client.get(f"/models/{_SIG_MODEL}/ab-compare", headers=headers, params={"days": 30})
    assert r.status_code == 200
    data = r.json()
    assert "ab_significance" in data
    sig = data["ab_significance"]
    if sig is not None:
        assert "metric" in sig
        assert "test" in sig
        assert "p_value" in sig
        assert "significant" in sig
        assert "confidence_level" in sig
        assert "current_samples" in sig
        assert "min_samples_needed" in sig


# ---------------------------------------------------------------------------
# compute_ab_significance — regression path (prediction residuals)
# ---------------------------------------------------------------------------


def _make_stats_regression(version, n_total, prediction_errors, times=None):
    """Create a version_stats dict with prediction_errors to test regression."""
    return {
        "version": version,
        "total_predictions": n_total,
        "error_count": 0,
        "response_times": times or [],
        "prediction_errors": prediction_errors,
    }


def test_significance_regression_mae_significant():
    """Regression: very different residuals → Mann-Whitney U significant, metric=mae."""
    # v1: low errors, v2: high errors
    errors_v1 = [0.1] * 30
    errors_v2 = [5.0] * 30
    stats = [
        _make_stats_regression("v1", 30, errors_v1),
        _make_stats_regression("v2", 30, errors_v2),
    ]
    result = compute_ab_significance(stats)
    assert result is not None
    assert result["metric"] == "mae"
    assert result["test"] == "mann_whitney_u"
    assert result["significant"] is True
    assert result["winner"] == "v1"  # v1 has lower MAE
    assert result["p_value"] < 0.05
    assert result["current_samples"]["v1"] == 30
    assert result["current_samples"]["v2"] == 30


def test_significance_regression_mae_not_significant():
    """Regression: identical residuals → Mann-Whitney not significant."""
    errors = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.0]
    stats = [
        _make_stats_regression("v1", 8, list(errors)),
        _make_stats_regression("v2", 8, list(errors)),
    ]
    result = compute_ab_significance(stats)
    if result is not None:
        assert result["metric"] == "mae"
        assert result["significant"] is False


def test_significance_regression_prioritized_over_response_times():
    """If prediction_errors available, they take priority over response_times."""
    errors_v1 = [0.1] * 20
    errors_v2 = [5.0] * 20
    # Also very different response times (must not be used)
    times_v1 = [100.0] * 20
    times_v2 = [10.0] * 20
    stats = [
        _make_stats_regression("v1", 20, errors_v1, times=times_v1),
        _make_stats_regression("v2", 20, errors_v2, times=times_v2),
    ]
    result = compute_ab_significance(stats)
    assert result is not None
    assert result["metric"] == "mae"  # Not response_time_ms
    assert result["winner"] == "v1"  # Based on MAE, not latency


def test_significance_regression_no_errors_fallback_to_none():
    """Regression without residuals or response times → None."""
    stats = [
        _make_stats_regression("v1", 5, []),
        _make_stats_regression("v2", 5, []),
    ]
    result = compute_ab_significance(stats)
    assert result is None


def test_significance_regression_fields_complete():
    """Regression response contains all required fields."""
    stats = [
        _make_stats_regression("v1", 30, [0.5] * 30),
        _make_stats_regression("v2", 30, [2.0] * 30),
    ]
    result = compute_ab_significance(stats)
    assert result is not None
    required_keys = {
        "metric",
        "test",
        "p_value",
        "significant",
        "confidence_level",
        "winner",
        "min_samples_needed",
        "current_samples",
    }
    assert required_keys.issubset(result.keys())
