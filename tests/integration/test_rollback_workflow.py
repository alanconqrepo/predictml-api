"""
Integration tests — complete model rollback workflow.

Tested workflow:
  POST /models (v1.0.0, description="original")
  → PATCH /models/{name}/{version} (new description)
  → GET /models/{name}/{version}/history (verify snapshot entry)
  → POST /models/{name}/{version}/rollback/{history_id} (rollback)
  → GET /models/{name}/{version} (verify restoration)

Uses SQLite in-memory + FakeRedis + global MinIO mock.
Admin token: test-token-integ-rb-admin-ii99
"""

import asyncio
import io
import joblib

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-integ-rb-admin-ii99"
USER_TOKEN = "test-token-integ-rb-user-jj88"
RB_MODEL = "rb_rollback_integ_model"


def _make_pkl() -> bytes:
    """Create a serialized sklearn model."""
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    """Create test users."""
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="integ_rb_admin",
                email="integ_rb_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="integ_rb_user",
                email="integ_rb_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=1000,
            )
        await db.commit()


asyncio.run(_setup())

# Create the model once for the entire class (initial description = "original")
_r_create = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.joblib", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={
        "name": RB_MODEL,
        "version": "1.0.0",
        "description": "original description",
        "accuracy": "0.90",
    },
)
assert _r_create.status_code == 201, _r_create.text


class TestRollbackWorkflow:
    """Tests for the complete rollback workflow."""

    def test_01_model_created_with_history_entry(self):
        """After creation, GET /models/{name}/1.0.0/history contains a 'created' entry."""
        r = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        actions = [entry["action"] for entry in data["entries"]]
        assert "created" in actions

    def test_02_update_description_creates_updated_history(self):
        """PATCH description → an 'updated' entry appears in history."""
        r_patch = client.patch(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"description": "modified description"},
        )
        assert r_patch.status_code == 200

        r_hist = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_hist.status_code == 200
        data = r_hist.json()
        actions = [entry["action"] for entry in data["entries"]]
        assert "updated" in actions

    def test_03_rollback_restores_original_description(self):
        """Rollback to the 'created' entry → description restored to 'original description'."""
        # Get the id of the 'created' entry
        r_hist = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_hist.status_code == 200
        history = r_hist.json()["entries"]
        created_entry = next((e for e in history if e["action"] == "created"), None)
        assert created_entry is not None
        history_id = created_entry["id"]

        # Rollback
        r_rb = client.post(
            f"/models/{RB_MODEL}/1.0.0/rollback/{history_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_rb.status_code == 200
        data = r_rb.json()
        assert data["rolled_back_to_history_id"] == history_id

        # Verify that the description is restored
        r_get = client.get(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_get.status_code == 200
        assert r_get.json()["description"] == "original description"

    def test_04_rollback_creates_rollback_history_entry(self):
        """After a rollback, a 'rollback' entry appears in history."""
        r_hist = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_hist.status_code == 200
        data = r_hist.json()
        actions = [entry["action"] for entry in data["entries"]]
        assert "rollback" in actions

    def test_05_rollback_nonexistent_history_id_returns_404(self):
        """Rollback to a non-existent history id → 404."""
        r = client.post(
            f"/models/{RB_MODEL}/1.0.0/rollback/9999999",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_06_rollback_nonexistent_model_returns_404(self):
        """Rollback on a non-existent model → 404."""
        r = client.post(
            "/models/totally_nonexistent_model_rb/1.0.0/rollback/1",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_07_rollback_unauthorized_returns_403(self):
        """A standard user cannot perform a rollback → 403."""
        r = client.post(
            f"/models/{RB_MODEL}/1.0.0/rollback/1",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert r.status_code == 403

    def test_08_rollback_no_auth_returns_401_or_403(self):
        """Without token → 401 or 403."""
        r = client.post(f"/models/{RB_MODEL}/1.0.0/rollback/1")
        assert r.status_code in (401, 403)

    def test_09_history_total_increases_after_rollback(self):
        """After rollback, the total history count is greater than the initial total."""
        r = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        # At least: created + updated + rollback = 3 entries minimum
        assert r.json()["total"] >= 3

    def test_10_patch_accuracy_then_rollback_restores_accuracy(self):
        """PATCH accuracy → rollback → original accuracy restored."""
        # Read current accuracy
        r_get = client.get(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_get.status_code == 200
        original_accuracy = r_get.json().get("accuracy")

        # Modifier l'accuracy
        client.patch(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"accuracy": 0.55},
        )

        # Get the 'created' entry for rollback
        r_hist = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        history = r_hist.json()["entries"]
        created_entry = next((e for e in history if e["action"] == "created"), None)
        assert created_entry is not None

        # Rollback
        r_rb = client.post(
            f"/models/{RB_MODEL}/1.0.0/rollback/{created_entry['id']}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_rb.status_code == 200

        # Verify restoration
        r_after = client.get(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_after.status_code == 200
        assert r_after.json().get("accuracy") == original_accuracy
