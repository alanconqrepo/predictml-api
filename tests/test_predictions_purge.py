"""
Tests for DELETE /predictions/purge
"""

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from src.db.models import ObservedResult, Prediction
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-purge-admin"
USER_TOKEN = "test-token-purge-user"
MODEL_A = "purge_model_alpha"
MODEL_B = "purge_model_beta"

NOW = datetime.now(timezone.utc).replace(tzinfo=None)
OLD_TS = NOW - timedelta(days=120)
RECENT_TS = NOW - timedelta(days=10)


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            admin = await DBService.create_user(
                db,
                username="purge_admin",
                email="purge_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        else:
            from sqlalchemy import select

            from src.db.models import User

            result = await db.execute(select(User).where(User.api_token == ADMIN_TOKEN))
            admin = result.scalar_one()

        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="purge_user",
                email="purge_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )

        # Create old predictions (120 days ago) on MODEL_A
        for i in range(3):
            pred = Prediction(
                user_id=admin.id,
                model_name=MODEL_A,
                model_version="1.0.0",
                input_features={"x": i},
                prediction_result=i,
                probabilities=None,
                response_time_ms=10.0,
                status="success",
                timestamp=OLD_TS,
                id_obs=f"obs-old-{i}",
            )
            db.add(pred)

        # Create a recent prediction on MODEL_A (should NOT be purged)
        pred_recent = Prediction(
            user_id=admin.id,
            model_name=MODEL_A,
            model_version="1.0.0",
            input_features={"x": 99},
            prediction_result=99,
            probabilities=None,
            response_time_ms=10.0,
            status="success",
            timestamp=RECENT_TS,
            id_obs="obs-recent-1",
        )
        db.add(pred_recent)

        # Create old predictions on MODEL_B
        pred_b = Prediction(
            user_id=admin.id,
            model_name=MODEL_B,
            model_version="2.0.0",
            input_features={"y": 1},
            prediction_result=1,
            probabilities=None,
            response_time_ms=5.0,
            status="success",
            timestamp=OLD_TS,
            id_obs="obs-b-1",
        )
        db.add(pred_b)

        await db.commit()

        # Create observed_results linked to one old prediction on MODEL_A
        # This triggers the warning in the purge response
        obs = ObservedResult(
            id_obs="obs-old-0",
            model_name=MODEL_A,
            observed_result=0,
            date_time=OLD_TS,
            user_id=admin.id,
        )
        db.add(obs)
        await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Auth checks
# ---------------------------------------------------------------------------


def test_purge_without_auth():
    response = client.delete("/predictions/purge?older_than_days=90")
    assert response.status_code in (401, 403)


def test_purge_non_admin_returns_403():
    response = client.delete(
        "/predictions/purge?older_than_days=90",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_purge_missing_older_than_days_returns_422():
    response = client.delete(
        "/predictions/purge",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 422


def test_purge_older_than_days_zero_is_valid():
    """older_than_days=0 is now allowed — purges everything (dry_run by default)."""
    response = client.delete(
        "/predictions/purge?older_than_days=0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    assert "deleted_observed_results_count" in data


# ---------------------------------------------------------------------------
# Dry-run (default)
# ---------------------------------------------------------------------------


def test_purge_dry_run_true_by_default():
    """dry_run=true by default — no deletion, coherent response."""
    response = client.delete(
        "/predictions/purge?older_than_days=90",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    # 3 old MODEL_A + 1 old MODEL_B = 4 should be counted
    assert data["deleted_count"] >= 4
    assert isinstance(data["models_affected"], list)
    assert MODEL_A in data["models_affected"]
    assert MODEL_B in data["models_affected"]


def test_purge_dry_run_does_not_delete():
    """After a dry_run, predictions must still exist."""
    client.delete(
        "/predictions/purge?older_than_days=90&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    # Count should still be the same
    response2 = client.delete(
        "/predictions/purge?older_than_days=90&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response2.status_code == 200
    assert response2.json()["deleted_count"] >= 4


def test_purge_response_schema():
    """The response contains all expected fields."""
    response = client.delete(
        "/predictions/purge?older_than_days=90",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    assert "dry_run" in data
    assert "deleted_count" in data
    assert "oldest_remaining" in data
    assert "models_affected" in data
    assert "linked_observed_results_count" in data


# ---------------------------------------------------------------------------
# oldest_remaining
# ---------------------------------------------------------------------------


def test_purge_oldest_remaining_points_to_recent_prediction():
    """oldest_remaining must point to the recent prediction (not deleted)."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    assert data["oldest_remaining"] is not None
    oldest_dt = datetime.fromisoformat(data["oldest_remaining"])
    # The recent prediction is at RECENT_TS (~10 days) — well after the cutoff
    assert oldest_dt > NOW - timedelta(days=90)


def test_purge_oldest_remaining_none_when_all_would_be_deleted():
    """If all predictions for the model would be deleted, oldest_remaining is None."""
    # MODEL_B has only one old prediction, none recent
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    assert data["oldest_remaining"] is None


# ---------------------------------------------------------------------------
# linked_observed_results_count (warning)
# ---------------------------------------------------------------------------


def test_purge_warns_linked_observed_results():
    """linked_observed_results_count > 0 if predictions linked to observed_results are deleted."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    # obs-old-0 on MODEL_A is linked to an ObservedResult
    assert data["linked_observed_results_count"] >= 1


def test_purge_no_linked_observed_results_for_model_b():
    """No observed_result linked to MODEL_B → linked_observed_results_count == 0."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    assert data["linked_observed_results_count"] == 0


# ---------------------------------------------------------------------------
# model_name filter
# ---------------------------------------------------------------------------


def test_purge_model_name_filter_limits_scope():
    """With model_name, only the targeted model appears in models_affected."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    assert data["models_affected"] == [MODEL_B]
    assert MODEL_A not in data["models_affected"]


def test_purge_model_name_filter_counts_only_target_model():
    """With model_name=MODEL_B, deleted_count corresponds only to that model."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    # MODEL_B has 1 old prediction
    assert data["deleted_count"] == 1


# ---------------------------------------------------------------------------
# Effective deletion (dry_run=false)
# ---------------------------------------------------------------------------


def test_purge_dry_run_false_deletes_predictions():
    """dry_run=false actually deletes predictions and returns the correct count."""
    # First verify with dry_run
    dry = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert dry.json()["deleted_count"] == 1

    # Actually delete
    real = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}&dry_run=false",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert real.status_code == 200
    assert real.json()["dry_run"] is False
    assert real.json()["deleted_count"] == 1

    # After deletion, nothing left to delete for MODEL_B
    after = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert after.json()["deleted_count"] == 0
    assert after.json()["models_affected"] == []


def test_purge_dry_run_false_preserves_recent_predictions():
    """dry_run=false does not delete recent predictions (< older_than_days)."""
    # Purge MODEL_A (3 old + 1 recent)
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}&dry_run=false",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    # Only the 3 old ones are deleted (MODEL_B was already deleted in the previous test)
    assert data["deleted_count"] == 3
    # oldest_remaining points to the recent prediction
    assert data["oldest_remaining"] is not None

    # No more old predictions on MODEL_A
    after = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert after.json()["deleted_count"] == 0


def test_purge_dry_run_false_returns_false_in_response():
    """The dry_run field in the response correctly reflects the passed parameter."""
    response = client.delete(
        f"/predictions/purge?older_than_days=1&model_name={MODEL_A}&dry_run=false",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["dry_run"] is False
