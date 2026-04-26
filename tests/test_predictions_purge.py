"""
Tests pour DELETE /predictions/purge
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


def test_purge_older_than_days_zero_returns_422():
    response = client.delete(
        "/predictions/purge?older_than_days=0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Dry-run (default)
# ---------------------------------------------------------------------------


def test_purge_dry_run_true_by_default():
    """dry_run=true par défaut — aucune suppression, réponse cohérente."""
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
    """Après un dry_run, les prédictions doivent toujours exister."""
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
    """La réponse contient tous les champs attendus."""
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
    """oldest_remaining doit pointer vers la prédiction récente (pas supprimée)."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    assert data["oldest_remaining"] is not None
    oldest_dt = datetime.fromisoformat(data["oldest_remaining"])
    # La prédiction récente est à RECENT_TS (~10 jours) — bien après la cutoff
    assert oldest_dt > NOW - timedelta(days=90)


def test_purge_oldest_remaining_none_when_all_would_be_deleted():
    """Si toutes les prédictions du modèle seraient supprimées, oldest_remaining est None."""
    # MODEL_B n'a qu'une prédiction ancienne, aucune récente
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
    """linked_observed_results_count > 0 si des prédictions liées à des observed_results sont supprimées."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    # obs-old-0 sur MODEL_A est lié à un ObservedResult
    assert data["linked_observed_results_count"] >= 1


def test_purge_no_linked_observed_results_for_model_b():
    """Aucun observed_result lié à MODEL_B → linked_observed_results_count == 0."""
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
    """Avec model_name, seul le modèle ciblé apparaît dans models_affected."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    assert data["models_affected"] == [MODEL_B]
    assert MODEL_A not in data["models_affected"]


def test_purge_model_name_filter_counts_only_target_model():
    """Avec model_name=MODEL_B, deleted_count correspond uniquement à ce modèle."""
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    # MODEL_B a 1 vieille prédiction
    assert data["deleted_count"] == 1


# ---------------------------------------------------------------------------
# Effective deletion (dry_run=false)
# ---------------------------------------------------------------------------


def test_purge_dry_run_false_deletes_predictions():
    """dry_run=false supprime réellement les prédictions et retourne le bon compte."""
    # Vérifier d'abord le dry_run
    dry = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert dry.json()["deleted_count"] == 1

    # Supprimer réellement
    real = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}&dry_run=false",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert real.status_code == 200
    assert real.json()["dry_run"] is False
    assert real.json()["deleted_count"] == 1

    # Après suppression, plus rien à supprimer pour MODEL_B
    after = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_B}&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert after.json()["deleted_count"] == 0
    assert after.json()["models_affected"] == []


def test_purge_dry_run_false_preserves_recent_predictions():
    """dry_run=false ne supprime pas les prédictions récentes (< older_than_days)."""
    # Purger MODEL_A (3 vieilles + 1 récente)
    response = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}&dry_run=false",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    data = response.json()
    # Seules les 3 vieilles sont supprimées (MODEL_B a déjà été supprimé dans le test précédent)
    assert data["deleted_count"] == 3
    # oldest_remaining pointe vers la prédiction récente
    assert data["oldest_remaining"] is not None

    # Plus de vieilles prédictions sur MODEL_A
    after = client.delete(
        f"/predictions/purge?older_than_days=90&model_name={MODEL_A}&dry_run=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert after.json()["deleted_count"] == 0


def test_purge_dry_run_false_returns_false_in_response():
    """Le champ dry_run dans la réponse reflète bien le paramètre passé."""
    response = client.delete(
        f"/predictions/purge?older_than_days=1&model_name={MODEL_A}&dry_run=false",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert response.status_code == 200
    assert response.json()["dry_run"] is False
