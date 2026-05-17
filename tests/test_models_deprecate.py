"""
Tests pour PATCH /models/{name}/{version}/deprecate
"""

import asyncio
import io
import joblib
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sqlalchemy import select

from src.db.models import ModelMetadata, User
from src.main import app
from src.services.db_service import DBService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

client = TestClient(app)

ADMIN_TOKEN = "test-token-deprecate-admin"
USER_TOKEN = "test-token-deprecate-user"
MODEL_PREFIX = "dep_model"


def make_pkl_bytes() -> bytes:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = [0, 1, 0]
    _jbuf = io.BytesIO()
    joblib.dump(LogisticRegression(max_iter=200).fit(X, y), _jbuf)
    return _jbuf.getvalue()


async def _setup():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="deprecate_admin",
                email="deprecate_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="deprecate_user",
                email="deprecate_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=10000,
            )


asyncio.run(_setup())


IRIS_FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}


def _make_iris_lr() -> LogisticRegression:
    X = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 6.2, 4.9, 7.0],
            "sepal width (cm)": [3.5, 2.9, 3.1, 3.2],
            "petal length (cm)": [1.4, 4.3, 1.5, 4.7],
            "petal width (cm)": [0.2, 1.3, 0.1, 1.4],
        }
    )
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_model_cache(name: str, version: str) -> None:
    """Injecte un modèle iris dans le cache Redis pour éviter l'appel MinIO."""
    model = _make_iris_lr()
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=name,
            version=version,
            status="active",
            confidence_threshold=None,
            webhook_url=None,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{name}:{version}", _jbuf.getvalue()))


def _create_model(name: str, version: str = "1.0.0", is_production: bool = False) -> dict:
    data = {"name": name, "version": version}
    if is_production:
        data["is_production"] = "true"
    r = client.post(
        "/models",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        files={"file": ("model.joblib", io.BytesIO(make_pkl_bytes()), "application/octet-stream")},
        data=data,
    )
    assert r.status_code == 201, r.text
    return r.json()


def _deprecate(name: str, version: str, token: str = ADMIN_TOKEN):
    return client.patch(
        f"/models/{name}/{version}/deprecate",
        headers={"Authorization": f"Bearer {token}"},
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_deprecate_requires_auth():
    """PATCH /deprecate sans token → 401/403"""
    _create_model(f"{MODEL_PREFIX}_noauth")
    r = client.patch(f"/models/{MODEL_PREFIX}_noauth/1.0.0/deprecate")
    assert r.status_code in [401, 403]


def test_deprecate_requires_admin():
    """PATCH /deprecate avec token non-admin → 403"""
    _create_model(f"{MODEL_PREFIX}_nonadmin")
    r = _deprecate(f"{MODEL_PREFIX}_nonadmin", "1.0.0", token=USER_TOKEN)
    assert r.status_code == 403


def test_deprecate_invalid_token():
    """PATCH /deprecate avec token invalide → 401"""
    _create_model(f"{MODEL_PREFIX}_badtoken")
    r = client.patch(
        f"/models/{MODEL_PREFIX}_badtoken/1.0.0/deprecate",
        headers={"Authorization": "Bearer invalid"},
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Cas nominaux
# ---------------------------------------------------------------------------


def test_deprecate_sets_status_deprecated():
    """PATCH /deprecate → status=deprecated dans la réponse"""
    _create_model(f"{MODEL_PREFIX}_basic")
    r = _deprecate(f"{MODEL_PREFIX}_basic", "1.0.0")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "deprecated"
    assert data["name"] == f"{MODEL_PREFIX}_basic"
    assert data["version"] == "1.0.0"


def test_deprecate_sets_is_production_false():
    """PATCH /deprecate retire is_production même si le modèle était en production"""
    _create_model(f"{MODEL_PREFIX}_wasprod", is_production=True)
    r = _deprecate(f"{MODEL_PREFIX}_wasprod", "1.0.0")
    assert r.status_code == 200
    assert r.json()["is_production"] is False


def test_deprecate_sets_deprecated_at():
    """PATCH /deprecate → deprecated_at renseigné"""
    _create_model(f"{MODEL_PREFIX}_ts")
    r = _deprecate(f"{MODEL_PREFIX}_ts", "1.0.0")
    assert r.status_code == 200
    assert r.json()["deprecated_at"] is not None


def test_deprecate_response_includes_deprecated_by():
    """La réponse inclut deprecated_by avec le username admin"""
    _create_model(f"{MODEL_PREFIX}_by")
    r = _deprecate(f"{MODEL_PREFIX}_by", "1.0.0")
    assert r.status_code == 200
    assert r.json()["deprecated_by"] == "deprecate_admin"


# ---------------------------------------------------------------------------
# Erreurs
# ---------------------------------------------------------------------------


def test_deprecate_nonexistent_model():
    """PATCH /deprecate sur modèle inexistant → 404"""
    r = _deprecate("nonexistent_model_xyz", "9.9.9")
    assert r.status_code == 404


def test_deprecate_nonexistent_version():
    """PATCH /deprecate sur version inexistante → 404"""
    _create_model(f"{MODEL_PREFIX}_badver")
    r = _deprecate(f"{MODEL_PREFIX}_badver", "9.9.9")
    assert r.status_code == 404


def test_deprecate_already_deprecated():
    """PATCH /deprecate sur modèle déjà déprécié → 409"""
    _create_model(f"{MODEL_PREFIX}_twice")
    _deprecate(f"{MODEL_PREFIX}_twice", "1.0.0")
    r = _deprecate(f"{MODEL_PREFIX}_twice", "1.0.0")
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# GET /models — visibilité
# ---------------------------------------------------------------------------


def test_get_models_shows_deprecated():
    """GET /models liste les modèles dépréciés (ils ne sont pas archivés)"""
    name = f"{MODEL_PREFIX}_list_dep"
    _create_model(name)
    _deprecate(name, "1.0.0")
    r = client.get("/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert name in names


def test_get_models_deprecated_has_status_field():
    """GET /models → le modèle déprécié a bien status=deprecated"""
    name = f"{MODEL_PREFIX}_list_status"
    _create_model(name)
    _deprecate(name, "1.0.0")
    r = client.get("/models")
    assert r.status_code == 200
    match = next((m for m in r.json() if m["name"] == name), None)
    assert match is not None
    assert match["status"] == "deprecated"


def test_get_models_hides_archived():
    """GET /models n'affiche pas les modèles archivés par défaut"""
    name = f"{MODEL_PREFIX}_archived"
    _create_model(name)

    # Manually set status to "archived" in DB to simulate the archived state
    async def _set_archived():
        async with _TestSessionLocal() as db:
            result = await db.execute(
                select(ModelMetadata).where(
                    ModelMetadata.name == name,
                    ModelMetadata.version == "1.0.0",
                )
            )
            meta = result.scalar_one_or_none()
            if meta:
                meta.status = "archived"
                await db.commit()

    asyncio.run(_set_archived())

    r = client.get("/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert name not in names


# ---------------------------------------------------------------------------
# POST /predict — garde HTTP 410
# ---------------------------------------------------------------------------


def test_predict_deprecated_explicit_version_returns_410():
    """POST /predict avec version dépréciée explicite → 410 Gone"""
    name = f"{MODEL_PREFIX}_pred410"
    _create_model(name)
    _deprecate(name, "1.0.0")

    r = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
        json={"model_name": name, "model_version": "1.0.0", "features": IRIS_FEATURES},
    )
    assert r.status_code == 410


def test_predict_410_detail_contains_current_production():
    """Le body 410 mentionne la version production courante"""
    name = f"{MODEL_PREFIX}_hint"
    _create_model(name, version="1.0.0")
    _create_model(name, version="2.0.0", is_production=True)
    _deprecate(name, "1.0.0")

    r = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {USER_TOKEN}"},
        json={"model_name": name, "model_version": "1.0.0", "features": IRIS_FEATURES},
    )
    assert r.status_code == 410
    detail = r.json()["detail"]
    assert "Current production:" in detail
    assert f"{name}/2.0.0" in detail


def test_predict_active_version_still_works_after_another_deprecation():
    """POST /predict sur version active fonctionne même si une autre version est dépréciée"""
    name = f"{MODEL_PREFIX}_active_ok"
    _create_model(name, version="1.0.0")
    _create_model(name, version="2.0.0", is_production=True)
    _deprecate(name, "1.0.0")
    _inject_model_cache(name, "2.0.0")

    try:
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"model_name": name, "model_version": "2.0.0", "features": IRIS_FEATURES},
        )
        assert r.status_code == 200
    finally:
        asyncio.run(model_service._redis.delete(f"model:{name}:2.0.0"))


def test_predict_implicit_routing_skips_deprecated():
    """POST /predict sans version explicite skippe les versions dépréciées"""
    name = f"{MODEL_PREFIX}_routing"
    _create_model(name, version="1.0.0")
    _create_model(name, version="2.0.0", is_production=True)
    _deprecate(name, "1.0.0")
    _inject_model_cache(name, "2.0.0")

    try:
        r = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            json={"model_name": name, "features": IRIS_FEATURES},
        )
        # Should route to v2.0.0 (active production), not 410
        assert r.status_code == 200
        assert r.json()["model_version"] == "2.0.0"
    finally:
        asyncio.run(model_service._redis.delete(f"model:{name}:2.0.0"))
