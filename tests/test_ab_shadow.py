"""
Tests A/B Testing & Shadow Deployment.

Stratégie :
  - SQLite in-memory (conftest) + FakeRedis (model_service._redis)
  - Modèles sklearn créés à la volée, injectés dans le cache
  - Le shadow background task est patché pour utiliser _TestSessionLocal
    (AsyncSessionLocal de production pointe sur PostgreSQL — indisponible en tests)
  - Chaque test nettoie le cache avec try/finally
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

# Tokens et noms uniques à ce fichier
TEST_TOKEN = "test-token-ab-shadow-xr9k"
AB_MODEL = "ab_test_model"
SHADOW_MODEL = "shadow_test_model"
V1 = "1.0.0"
V2 = "2.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lr_model() -> LogisticRegression:
    """Crée un LogisticRegression minimal avec feature_names_in_."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]})
    y = [0, 1, 0, 1]
    return LogisticRegression(max_iter=1000).fit(X, y)


def _inject_cache(model_name: str, version: str, model) -> str:
    """Injecte un modèle sklearn dans le cache FakeRedis."""
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
# Setup : utilisateur + ModelMetadata
# ---------------------------------------------------------------------------


async def _setup():
    async with _TestSessionLocal() as db:
        # Utilisateur de test
        if not await DBService.get_user_by_token(db, TEST_TOKEN):
            await DBService.create_user(
                db,
                username="test_ab_shadow_user",
                email="ab_shadow@test.com",
                api_token=TEST_TOKEN,
                role="admin",
                rate_limit=99999,
            )

        # Versions A/B pour AB_MODEL
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
                    minio_object_key=f"{AB_MODEL}/v{ver}.pkl",
                    is_active=True,
                    is_production=is_prod,
                    deployment_mode=mode,
                    traffic_weight=weight,
                )

        # Versions production + shadow pour SHADOW_MODEL
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
                    minio_object_key=f"{SHADOW_MODEL}/v{ver}.pkl",
                    is_active=True,
                    is_production=is_prod,
                    deployment_mode=mode,
                    traffic_weight=weight,
                )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Test 1 — A/B routing : distribution pondérée du trafic
# ---------------------------------------------------------------------------


def test_ab_routing_distributes_traffic():
    """
    Avec v1 (80%) et v2 (20%), sur N appels sans version explicite,
    les deux versions doivent être sélectionnées avec le ratio attendu.
    Tolérance ±20 points de pourcentage (probabiliste).
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

        # Les deux versions doivent être sélectionnées
        assert V1 in version_counts, "v1 n'a jamais été sélectionnée"
        assert V2 in version_counts, "v2 n'a jamais été sélectionnée"

        # Ratio approximatif (±25 pp de tolérance pour N=100)
        v1_ratio = version_counts[V1] / n
        assert 0.55 <= v1_ratio <= 1.0, f"v1 ratio trop loin de 80% : {v1_ratio:.0%}"
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 2 — A/B routing : selected_version est présent dans la réponse
# ---------------------------------------------------------------------------


def test_ab_routing_returns_selected_version():
    """Lors d'un routage A/B, selected_version est renseigné dans la réponse."""
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
        # selected_version doit être présent (non-None) lors d'un routage automatique
        assert data.get("selected_version") is not None
        assert data["selected_version"] in [V1, V2]
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 3 — Version explicite contourne le routage A/B
# ---------------------------------------------------------------------------


def test_explicit_version_bypasses_ab_routing():
    """POST /predict avec model_version explicite → toujours la version demandée."""
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
            # selected_version est None quand la version est explicite
            assert r.json()["selected_version"] is None
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 4 — Shadow : la réponse vient du modèle production
# ---------------------------------------------------------------------------


def test_shadow_primary_response_is_production():
    """
    Avec v1 (production) et v2 (shadow), POST /predict sans version
    → la réponse vient de v1 (pas du shadow).
    """
    model = _make_lr_model()
    key_v1 = _inject_cache(SHADOW_MODEL, V1, model)
    key_v2 = _inject_cache(SHADOW_MODEL, V2, model)
    try:
        # Patcher AsyncSessionLocal pour que le shadow task utilise la DB de test
        with patch("src.api.predict.AsyncSessionLocal", _TestSessionLocal):
            r = client.post(
                "/predict",
                headers=_headers(),
                json={"model_name": SHADOW_MODEL, "features": {"f1": 1.0, "f2": 2.0}},
            )
        assert r.status_code == 200
        data = r.json()
        # La réponse client doit venir de v1 (production / is_production=True)
        assert data["model_version"] == V1
    finally:
        asyncio.run(model_service.clear_cache(key_v1))
        asyncio.run(model_service.clear_cache(key_v2))


# ---------------------------------------------------------------------------
# Test 5 — Shadow : prédiction shadow enregistrée avec is_shadow=True
# ---------------------------------------------------------------------------


def test_shadow_prediction_logged_with_is_shadow_flag():
    """
    Après un appel avec shadow actif, une prédiction is_shadow=True est en DB pour v2.
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

    # Vérifier en DB : il doit exister une prédiction shadow pour v2
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

    assert len(prod_rows) >= 1, "Prédiction production absente de la DB"
    assert len(shadow_rows) >= 1, "Prédiction shadow absente de la DB"
    assert prod_rows[0].model_version == V1
    assert shadow_rows[0].model_version == V2


# ---------------------------------------------------------------------------
# Test 6 — Validation des poids : somme > 1.0 → 422
# ---------------------------------------------------------------------------


def test_patch_ab_weight_sum_exceeds_one_returns_422():
    """
    PATCH d'une version avec un traffic_weight qui ferait dépasser 1.0 → 422.
    v1 est déjà à 0.8 → tenter de mettre v2 à 0.9 doit échouer.
    """
    r = client.patch(
        f"/models/{AB_MODEL}/{V2}",
        headers=_headers(),
        json={"deployment_mode": "ab_test", "traffic_weight": 0.9},
    )
    assert r.status_code == 422
    assert "1.0" in r.json()["detail"] or "dépasse" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Test 7 — PATCH valide : mise à jour du deployment_mode
# ---------------------------------------------------------------------------


def test_patch_deployment_mode_valid():
    """PATCH avec traffic_weight ≤ 1.0 − 0.8 = 0.2 → 200."""
    # v1 pèse 0.8, on met v2 à 0.2 → somme = 1.0 : acceptable
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
# Test 8 — GET /models/{name}/ab-compare : structure de réponse
# ---------------------------------------------------------------------------


def test_get_ab_compare_structure():
    """GET /models/{name}/ab-compare → ABCompareResponse avec champs attendus."""
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
    # Chaque version doit avoir les champs attendus
    for vs in data["versions"]:
        assert "version" in vs
        assert "total_predictions" in vs
        assert "shadow_predictions" in vs
        assert "error_rate" in vs
        assert "prediction_distribution" in vs
    # Le champ ab_significance doit être présent (None ou objet valide)
    assert "ab_significance" in data
    sig = data["ab_significance"]
    if sig is not None:
        assert "metric" in sig
        assert "test" in sig
        assert "p_value" in sig
        assert "significant" in sig
        assert "current_samples" in sig


# ---------------------------------------------------------------------------
# Test 9 — GET /models/{name}/ab-compare : modèle inexistant → 404
# ---------------------------------------------------------------------------


def test_get_ab_compare_unknown_model_returns_404():
    """GET /models/{name}/ab-compare pour un modèle inexistant → 404."""
    r = client.get(
        "/models/nonexistent_model_ab_xyz/ab-compare",
        headers=_headers(),
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 10 — Fallback legacy : aucun mode configuré → is_production utilisé
# ---------------------------------------------------------------------------


async def _create_legacy_model():
    """Crée un modèle avec comportement legacy (is_production, pas de deployment_mode)."""
    async with _TestSessionLocal() as db:
        name = "legacy_model_ab_test"
        if not await DBService.get_model_metadata(db, name, V1):
            await DBService.create_model_metadata(
                db,
                name=name,
                version=V1,
                minio_bucket="models",
                minio_object_key=f"{name}/v{V1}.pkl",
                is_active=True,
                is_production=True,
                deployment_mode=None,
                traffic_weight=None,
            )
        return name


def test_legacy_fallback_uses_production_version():
    """
    Sans deployment_mode configuré, le routage fallback utilise is_production=True.
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
# Test 11 — GET /models/{name}/shadow-compare : structure de réponse
# ---------------------------------------------------------------------------


def test_get_shadow_compare_structure():
    """GET /models/{name}/shadow-compare → ShadowCompareResponse avec champs attendus."""
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
# Test 12 — GET /models/{name}/shadow-compare : modèle inexistant → 404
# ---------------------------------------------------------------------------


def test_get_shadow_compare_unknown_model_returns_404():
    """GET /models/{name}/shadow-compare pour un modèle inexistant → 404."""
    r = client.get(
        "/models/nonexistent_shadow_model_xyz/shadow-compare",
        headers=_headers(),
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Test 13 — GET /models/{name}/shadow-compare : sans version shadow → n_comparable=0
# ---------------------------------------------------------------------------


def test_get_shadow_compare_no_shadow_version():
    """Modèle sans version shadow → shadow_version=None, n_comparable=0, insufficient_data."""
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
# Test 14 — GET /models/{name}/shadow-compare : avec paires en DB → métriques calculées
# ---------------------------------------------------------------------------


SHADOW_PAIRS_MODEL = "shadow_pairs_test_model"


async def _setup_shadow_pairs_model():
    """Crée un modèle dédié avec version shadow + production pour le test de paires."""
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
                    minio_object_key=f"{SHADOW_PAIRS_MODEL}/v{ver}.pkl",
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
    """Avec 15 paires en DB → n_comparable=15, agreement_rate calculé, recommendation."""
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
    # Les deux prédisent 0 → accord = 100 %
    assert data["agreement_rate"] == 1.0
    # shadow_confidence_delta = 0.90 - 0.75 = +0.15
    assert data["shadow_confidence_delta"] is not None
    assert abs(data["shadow_confidence_delta"] - 0.15) < 0.01
    # shadow_latency_delta_ms = 8 - 10 = -2.0
    assert data["shadow_latency_delta_ms"] is not None
    assert abs(data["shadow_latency_delta_ms"] - (-2.0)) < 0.5
    # Pas d'observed_results → accuracy_available=False
    assert data["accuracy_available"] is False
    assert data["shadow_accuracy"] is None
    assert data["production_accuracy"] is None
    # recommendation : confidence_delta = 0.15 > 0.05 → shadow_better
    assert data["recommendation"] == "shadow_better"
