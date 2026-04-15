"""
Tests d'intégration — workflow complet de rollback de modèle.

Workflow testé :
  POST /models (v1.0.0, description="original")
  → PATCH /models/{name}/{version} (nouvelle description)
  → GET /models/{name}/{version}/history (vérifier entrée snapshot)
  → POST /models/{name}/{version}/rollback/{history_id} (rollback)
  → GET /models/{name}/{version} (vérifier restauration)

Utilise SQLite in-memory + FakeRedis + MinIO mock global.
Token admin : test-token-integ-rb-admin-ii99
"""

import asyncio
import io
import pickle

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
    """Crée un modèle sklearn sérialisé."""
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


async def _setup():
    """Crée les utilisateurs de test."""
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

# Créer le modèle une fois pour toute la classe (description initiale = "original")
_r_create = client.post(
    "/models",
    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    files={"file": ("m.pkl", io.BytesIO(_make_pkl()), "application/octet-stream")},
    data={
        "name": RB_MODEL,
        "version": "1.0.0",
        "description": "description originale",
        "accuracy": "0.90",
    },
)
assert _r_create.status_code == 201, _r_create.text


class TestRollbackWorkflow:
    """Tests du workflow de rollback complet."""

    def test_01_model_created_with_history_entry(self):
        """Après création, GET /models/{name}/1.0.0/history contient une entrée 'created'."""
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
        """PATCH description → une entrée 'updated' apparaît dans l'historique."""
        r_patch = client.patch(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"description": "description modifiée"},
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
        """Rollback vers l'entrée 'created' → description revient à 'description originale'."""
        # Récupérer l'id de l'entrée 'created'
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

        # Vérifier que la description est restaurée
        r_get = client.get(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_get.status_code == 200
        assert r_get.json()["description"] == "description originale"

    def test_04_rollback_creates_rollback_history_entry(self):
        """Après un rollback, une entrée 'rollback' apparaît dans l'historique."""
        r_hist = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_hist.status_code == 200
        data = r_hist.json()
        actions = [entry["action"] for entry in data["entries"]]
        assert "rollback" in actions

    def test_05_rollback_nonexistent_history_id_returns_404(self):
        """Rollback vers un id d'historique inexistant → 404."""
        r = client.post(
            f"/models/{RB_MODEL}/1.0.0/rollback/9999999",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_06_rollback_nonexistent_model_returns_404(self):
        """Rollback sur un modèle inexistant → 404."""
        r = client.post(
            "/models/totally_nonexistent_model_rb/1.0.0/rollback/1",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 404

    def test_07_rollback_unauthorized_returns_403(self):
        """Un utilisateur standard ne peut pas faire de rollback → 403."""
        r = client.post(
            f"/models/{RB_MODEL}/1.0.0/rollback/1",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
        )
        assert r.status_code == 403

    def test_08_rollback_no_auth_returns_401_or_403(self):
        """Sans token → 401 ou 403."""
        r = client.post(f"/models/{RB_MODEL}/1.0.0/rollback/1")
        assert r.status_code in (401, 403)

    def test_09_history_total_increases_after_rollback(self):
        """Après rollback, le total de l'historique est supérieur au total initial."""
        r = client.get(
            f"/models/{RB_MODEL}/1.0.0/history",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r.status_code == 200
        # Au moins : created + updated + rollback = 3 entrées minimum
        assert r.json()["total"] >= 3

    def test_10_patch_accuracy_then_rollback_restores_accuracy(self):
        """PATCH accuracy → rollback → accuracy originale restaurée."""
        # Lire l'accuracy actuelle
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

        # Récupérer l'entrée 'created' pour rollback
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

        # Vérifier restauration
        r_after = client.get(
            f"/models/{RB_MODEL}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )
        assert r_after.status_code == 200
        assert r_after.json().get("accuracy") == original_accuracy
