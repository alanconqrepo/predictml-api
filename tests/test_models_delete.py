"""
Tests pour les endpoints DELETE /models/{name}/{version} et DELETE /models/{name}
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

ADMIN_TOKEN = "test-token-delete-models-admin"
USER_TOKEN = "test-token-delete-models-user"
MODEL_PREFIX = "delete_model"


def make_pkl_bytes() -> bytes:
    X, y = load_iris(return_X_y=True)
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="test_delete_models_admin",
                email="test_delete_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="test_delete_models_user",
                email="test_delete_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


def _create_model(name: str, version: str = "1.0.0") -> dict:
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("model.pkl", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data={"name": name, "version": version},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _create_mlflow_only_model(name: str, version: str = "1.0.0", run_id: str = "fakerunid123") -> dict:
    """Crée un modèle avec mlflow_run_id uniquement (pas de fichier → minio_object_key=None)."""
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        data={"name": name, "version": version, "mlflow_run_id": run_id},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _model_exists(name: str, version: str) -> bool:
    r = client.get("/models")
    return any(m["name"] == name and m["version"] == version for m in r.json())


# ---------------------------------------------------------------------------
# Auth — DELETE version
# ---------------------------------------------------------------------------

def test_delete_version_without_auth():
    """DELETE /models/{name}/{version} sans auth → 401/403"""
    _create_model(f"{MODEL_PREFIX}_noauth_v")
    r = client.delete(f"/models/{MODEL_PREFIX}_noauth_v/1.0.0")
    assert r.status_code in [401, 403]


def test_delete_version_with_invalid_token():
    """DELETE /models/{name}/{version} avec token invalide → 401"""
    _create_model(f"{MODEL_PREFIX}_badtoken_v")
    r = client.delete(
        f"/models/{MODEL_PREFIX}_badtoken_v/1.0.0",
        headers={"Authorization": "Bearer invalid"},
    )
    assert r.status_code == 401


def test_delete_version_non_admin_forbidden():
    """DELETE /models/{name}/{version} avec token non-admin → 403"""
    _create_model(f"{MODEL_PREFIX}_nonadmin_v")
    r = client.delete(
        f"/models/{MODEL_PREFIX}_nonadmin_v/1.0.0",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Auth — DELETE all versions
# ---------------------------------------------------------------------------

def test_delete_all_without_auth():
    """DELETE /models/{name} sans auth → 401/403"""
    _create_model(f"{MODEL_PREFIX}_noauth_all")
    r = client.delete(f"/models/{MODEL_PREFIX}_noauth_all")
    assert r.status_code in [401, 403]


def test_delete_all_with_invalid_token():
    """DELETE /models/{name} avec token invalide → 401"""
    _create_model(f"{MODEL_PREFIX}_badtoken_all")
    r = client.delete(
        f"/models/{MODEL_PREFIX}_badtoken_all",
        headers={"Authorization": "Bearer invalid"},
    )
    assert r.status_code == 401


def test_delete_all_non_admin_forbidden():
    """DELETE /models/{name} avec token non-admin → 403"""
    _create_model(f"{MODEL_PREFIX}_nonadmin_all")
    r = client.delete(
        f"/models/{MODEL_PREFIX}_nonadmin_all",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# DELETE version spécifique
# ---------------------------------------------------------------------------

def test_delete_specific_version_success():
    """DELETE version → 204, modèle absent de la liste"""
    name = f"{MODEL_PREFIX}_specific"
    _create_model(name, version="1.0.0")
    _create_model(name, version="2.0.0")

    r = client.delete(
        f"/models/{name}/1.0.0",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 204

    # v1 supprimée, v2 toujours présente
    assert not _model_exists(name, "1.0.0")
    assert _model_exists(name, "2.0.0")


def test_delete_specific_version_not_found():
    """DELETE version inexistante → 404"""
    r = client.delete(
        f"/models/inexistant_model/9.9.9",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 404


def test_delete_specific_version_twice():
    """DELETE la même version deux fois → 204 puis 404"""
    name = f"{MODEL_PREFIX}_twice"
    _create_model(name)

    r1 = client.delete(f"/models/{name}/1.0.0", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r1.status_code == 204

    r2 = client.delete(f"/models/{name}/1.0.0", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r2.status_code == 404


# ---------------------------------------------------------------------------
# DELETE toutes les versions
# ---------------------------------------------------------------------------

def test_delete_all_versions_success():
    """DELETE all → 200 avec résumé, toutes les versions supprimées"""
    name = f"{MODEL_PREFIX}_all"
    _create_model(name, version="1.0.0")
    _create_model(name, version="2.0.0")
    _create_model(name, version="3.0.0")

    r = client.delete(f"/models/{name}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 200

    data = r.json()
    assert data["name"] == name
    assert sorted(data["deleted_versions"]) == ["1.0.0", "2.0.0", "3.0.0"]
    assert isinstance(data["mlflow_runs_deleted"], list)
    assert isinstance(data["minio_objects_deleted"], list)

    assert not _model_exists(name, "1.0.0")
    assert not _model_exists(name, "2.0.0")
    assert not _model_exists(name, "3.0.0")


def test_delete_all_versions_not_found():
    """DELETE all sur nom inexistant → 404"""
    r = client.delete(
        "/models/inexistant_model_all",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert r.status_code == 404


def test_delete_all_versions_single_version():
    """DELETE all sur modèle avec une seule version → 200, 1 version supprimée"""
    name = f"{MODEL_PREFIX}_single"
    _create_model(name)

    r = client.delete(f"/models/{name}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 200
    assert r.json()["deleted_versions"] == ["1.0.0"]
    assert not _model_exists(name, "1.0.0")


# ---------------------------------------------------------------------------
# DELETE modèles mlflow-only (minio_object_key=None)
# ---------------------------------------------------------------------------

def test_delete_mlflow_only_version_success():
    """DELETE version d'un modèle mlflow-only → 204, pas d'appel MinIO delete"""
    from tests.conftest import _minio_mock

    name = f"{MODEL_PREFIX}_mlflow_only_v"
    _create_mlflow_only_model(name, run_id="mlflowrun001")

    _minio_mock.delete_model.reset_mock()

    r = client.delete(f"/models/{name}/1.0.0", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 204

    # minio_object_key est None → delete_model ne doit pas être appelé
    _minio_mock.delete_model.assert_not_called()
    assert not _model_exists(name, "1.0.0")


def test_delete_mlflow_only_all_versions():
    """DELETE all sur modèle mlflow-only → minio_objects_deleted vide"""
    name = f"{MODEL_PREFIX}_mlflow_only_all"
    _create_mlflow_only_model(name, version="1.0.0", run_id="mlflowrun002")
    _create_mlflow_only_model(name, version="2.0.0", run_id="mlflowrun003")

    r = client.delete(f"/models/{name}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert r.status_code == 200

    data = r.json()
    assert sorted(data["deleted_versions"]) == ["1.0.0", "2.0.0"]
    # Aucun objet MinIO à supprimer pour des modèles mlflow-only
    assert data["minio_objects_deleted"] == []
    assert not _model_exists(name, "1.0.0")
    assert not _model_exists(name, "2.0.0")
