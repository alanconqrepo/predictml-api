"""
Tests pour l'historique des changements de modèles et le rollback.

Couvre :
- Création automatique d'entrée "created" au POST /models
- Création d'entrée "updated" / "set_production" au PATCH
- Rétrogradation automatique de l'ancienne version en production
- Endpoints GET /models/{name}/history et GET /models/{name}/{version}/history
- Endpoint POST /models/{name}/{version}/rollback/{history_id}
- Permissions (auth requise, rollback admin-only)
- Pagination
- Contenu du snapshot (champs métadonnées, PAS les artifacts)
"""
import asyncio
import io
import pickle

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-history-admin-aa11"
USER_TOKEN = "test-token-history-user-bb22"

HIST_MODEL = "hist_test_model"


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    return pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="hist_admin",
                email="hist_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="hist_user",
                email="hist_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


def _create_model(name: str, version: str = "1.0.0", token: str = ADMIN_TOKEN, **extra) -> dict:
    data = {"name": name, "version": version, **extra}
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data=data,
    )
    assert r.status_code == 201, r.text
    return r.json()


def _get_history(name: str, version: str = None, token: str = ADMIN_TOKEN, **params) -> dict:
    if version:
        path = f"/models/{name}/{version}/history"
    else:
        path = f"/models/{name}/history"
    r = client.get(path, headers={"Authorization": f"Bearer {token}"}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Création automatique d'entrée d'historique
# ---------------------------------------------------------------------------


def test_history_created_on_model_post():
    """POST /models → entrée 'created' créée automatiquement dans l'historique."""
    name = f"{HIST_MODEL}_create"
    _create_model(name)

    data = _get_history(name, version="1.0.0")
    assert data["total"] >= 1
    entries = data["entries"]
    assert entries[0]["action"] == "created"
    assert entries[0]["model_name"] == name
    assert entries[0]["model_version"] == "1.0.0"
    assert entries[0]["changed_fields"] is None


def test_history_entry_on_patch():
    """PATCH description → entrée 'updated' avec changed_fields=['description']."""
    name = f"{HIST_MODEL}_patch"
    _create_model(name)

    client.patch(
        f"/models/{name}/1.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"description": "nouvelle desc"},
    )

    data = _get_history(name, version="1.0.0")
    latest = data["entries"][0]
    assert latest["action"] == "updated"
    assert "description" in latest["changed_fields"]
    assert latest["snapshot"]["description"] == "nouvelle desc"


def test_history_set_production_action():
    """PATCH is_production=True → action 'set_production'."""
    name = f"{HIST_MODEL}_setprod"
    _create_model(name)

    client.patch(
        f"/models/{name}/1.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_production": True},
    )

    data = _get_history(name, version="1.0.0")
    actions = [e["action"] for e in data["entries"]]
    assert "set_production" in actions


def test_history_set_production_demotes_other_version():
    """Quand v2 passe en production, l'historique de v1 doit montrer une rétrogradation."""
    name = f"{HIST_MODEL}_demote"
    _create_model(name, version="1.0.0")
    client.patch(
        f"/models/{name}/1.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_production": True},
    )

    _create_model(name, version="2.0.0")
    client.patch(
        f"/models/{name}/2.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"is_production": True},
    )

    data_v1 = _get_history(name, version="1.0.0")
    # La dernière entrée de v1 doit être set_production avec is_production=False
    entries_v1 = data_v1["entries"]
    demoted = [e for e in entries_v1 if e["action"] == "set_production"]
    assert len(demoted) >= 1
    # Le snapshot correspondant à la rétrogradation doit avoir is_production=False
    demotions_false = [e for e in demoted if e["snapshot"]["is_production"] is False]
    assert len(demotions_false) >= 1


# ---------------------------------------------------------------------------
# Endpoints GET historique
# ---------------------------------------------------------------------------


def test_get_history_all_versions():
    """GET /models/{name}/history retourne les entrées de toutes les versions."""
    name = f"{HIST_MODEL}_allv"
    _create_model(name, version="1.0.0")
    _create_model(name, version="2.0.0")

    data = _get_history(name)  # pas de version = toutes versions
    assert data["total"] >= 2
    versions = {e["model_version"] for e in data["entries"]}
    assert "1.0.0" in versions
    assert "2.0.0" in versions
    assert data["version"] is None


def test_get_history_specific_version():
    """GET /models/{name}/{version}/history retourne seulement les entrées de cette version."""
    name = f"{HIST_MODEL}_specificv"
    _create_model(name, version="1.0.0")
    _create_model(name, version="2.0.0")

    data = _get_history(name, version="1.0.0")
    for entry in data["entries"]:
        assert entry["model_version"] == "1.0.0"
    assert data["version"] == "1.0.0"


def test_history_requires_auth():
    """GET /models/{name}/history sans token → 401/403."""
    name = f"{HIST_MODEL}_noauth"
    _create_model(name)
    r = client.get(f"/models/{name}/history")
    assert r.status_code in [401, 403]


def test_history_accessible_to_non_admin():
    """Un utilisateur non-admin peut consulter l'historique."""
    name = f"{HIST_MODEL}_useraccess"
    _create_model(name)

    r = client.get(
        f"/models/{name}/history",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 200


def test_history_sorted_by_timestamp_desc():
    """Les entrées d'historique sont triées par timestamp DESC (les plus récentes en premier)."""
    name = f"{HIST_MODEL}_sorted"
    _create_model(name)
    client.patch(
        f"/models/{name}/1.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"description": "v1"},
    )
    client.patch(
        f"/models/{name}/1.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"description": "v2"},
    )

    data = _get_history(name, version="1.0.0")
    timestamps = [e["timestamp"] for e in data["entries"]]
    assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_admin_only():
    """POST /rollback avec token non-admin → 403."""
    name = f"{HIST_MODEL}_rollback_auth"
    _create_model(name)
    history = _get_history(name, version="1.0.0")
    history_id = history["entries"][0]["id"]

    r = client.post(
        f"/models/{name}/1.0.0/rollback/{history_id}",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


def test_rollback_requires_auth():
    """POST /rollback sans token → 401/403."""
    name = f"{HIST_MODEL}_rollback_noauth"
    _create_model(name)
    history = _get_history(name, version="1.0.0")
    history_id = history["entries"][0]["id"]

    r = client.post(f"/models/{name}/1.0.0/rollback/{history_id}")
    assert r.status_code in [401, 403]


def test_rollback_success():
    """Rollback restaure l'état antérieur des métadonnées."""
    name = f"{HIST_MODEL}_rollback_ok"
    _create_model(name, description="original")

    # Récupérer l'id de l'entrée "created" (état initial)
    history = _get_history(name, version="1.0.0")
    created_entry = next(e for e in history["entries"] if e["action"] == "created")
    created_id = created_entry["id"]

    # Modifier la description
    client.patch(
        f"/models/{name}/1.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"description": "modifiée"},
    )

    # Rollback vers l'état initial
    r = client.post(
        f"/models/{name}/1.0.0/rollback/{created_id}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 200
    result = r.json()
    assert result["rolled_back_to_history_id"] == created_id
    assert "description" in result["restored_fields"]

    # Vérifier que la description est bien restaurée
    model_r = client.get(f"/models/{name}/1.0.0")
    assert model_r.status_code == 200
    assert model_r.json()["description"] == "original"


def test_rollback_logs_new_history_entry():
    """Après un rollback, une nouvelle entrée 'rollback' est créée dans l'historique."""
    name = f"{HIST_MODEL}_rollback_log"
    _create_model(name)

    history_before = _get_history(name, version="1.0.0")
    total_before = history_before["total"]
    created_id = history_before["entries"][-1]["id"]  # le plus ancien (created)

    client.post(
        f"/models/{name}/1.0.0/rollback/{created_id}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )

    history_after = _get_history(name, version="1.0.0")
    assert history_after["total"] == total_before + 1
    assert history_after["entries"][0]["action"] == "rollback"


def test_rollback_nonexistent_history_id():
    """POST /rollback avec history_id inexistant → 404."""
    name = f"{HIST_MODEL}_rollback_404"
    _create_model(name)

    r = client.post(
        f"/models/{name}/1.0.0/rollback/999999",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 404


def test_rollback_wrong_model():
    """POST /rollback avec un history_id d'un autre modèle → 400."""
    name_a = f"{HIST_MODEL}_rollback_wrongA"
    name_b = f"{HIST_MODEL}_rollback_wrongB"
    _create_model(name_a)
    _create_model(name_b)

    # Récupérer un history_id appartenant à name_b
    history_b = _get_history(name_b, version="1.0.0")
    b_entry_id = history_b["entries"][0]["id"]

    # Tenter de l'utiliser sur name_a → 400
    r = client.post(
        f"/models/{name_a}/1.0.0/rollback/{b_entry_id}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 400


def test_rollback_nonexistent_model():
    """POST /rollback sur un modèle inexistant → 404."""
    r = client.post(
        "/models/inexistant_xyz/9.9.9/rollback/1",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Contenu du snapshot
# ---------------------------------------------------------------------------


def test_history_snapshot_excludes_artifact_fields():
    """Le snapshot ne doit pas contenir les champs d'artifacts (minio, mlflow)."""
    name = f"{HIST_MODEL}_snapshot_fields"
    _create_model(name)

    history = _get_history(name, version="1.0.0")
    snapshot = history["entries"][0]["snapshot"]

    artifact_fields = {"minio_bucket", "minio_object_key", "file_hash", "file_size_bytes", "mlflow_run_id"}
    for field in artifact_fields:
        assert field not in snapshot, f"Le champ artifact '{field}' ne devrait pas être dans le snapshot"


def test_history_snapshot_contains_metadata_fields():
    """Le snapshot contient les champs métadonnées clés."""
    name = f"{HIST_MODEL}_snapshot_meta"
    _create_model(name, description="test_desc", accuracy="0.95")

    history = _get_history(name, version="1.0.0")
    snapshot = history["entries"][0]["snapshot"]

    for field in ["description", "accuracy", "is_production", "is_active", "tags"]:
        assert field in snapshot, f"Le champ '{field}' devrait être dans le snapshot"

    assert snapshot["description"] == "test_desc"
    assert snapshot["accuracy"] == pytest.approx(0.95)


def test_history_snapshot_username_stored():
    """Le snapshot enregistre le username de l'auteur."""
    name = f"{HIST_MODEL}_username"
    _create_model(name)

    history = _get_history(name, version="1.0.0")
    entry = history["entries"][0]
    assert entry["changed_by_username"] == "hist_admin"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_history_pagination():
    """GET /history avec limit et offset fonctionnent correctement."""
    name = f"{HIST_MODEL}_pagination"
    _create_model(name)

    # Effectuer 4 modifications supplémentaires
    for i in range(4):
        client.patch(
            f"/models/{name}/1.0.0",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            json={"description": f"desc {i}"},
        )

    data_full = _get_history(name, version="1.0.0")
    total = data_full["total"]
    assert total >= 5  # 1 created + 4 updated

    # Première page
    data_p1 = _get_history(name, version="1.0.0", limit=3, offset=0)
    assert len(data_p1["entries"]) == 3
    assert data_p1["total"] == total

    # Deuxième page
    data_p2 = _get_history(name, version="1.0.0", limit=3, offset=3)
    assert len(data_p2["entries"]) >= 1

    # Les deux pages ne se chevauchent pas
    ids_p1 = {e["id"] for e in data_p1["entries"]}
    ids_p2 = {e["id"] for e in data_p2["entries"]}
    assert ids_p1.isdisjoint(ids_p2)
