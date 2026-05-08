"""
Tests pour GET /models/{name}/confidence-distribution.

Stratégie :
  - 404 sur modèle inconnu
  - 401 sans token
  - Aucune prédiction → sample_count=0, histogram vide
  - Calcul basique (sample_count, mean_confidence, bins non vides)
  - pct_high_confidence quand toutes les prédictions très confiantes
  - pct_uncertain quand toutes les prédictions peu confiantes
  - Seuils personnalisés via ?high_threshold et ?uncertain_threshold
  - Filtrage par version
  - Filtrage par fenêtre temporelle (days)
  - Probabilités stockées en dict → max extrait correctement
"""

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from src.db.models import Prediction
from src.main import app
from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

client = TestClient(app)

TOKEN = "test-token-conf-dist-x7q2"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
MODEL = "conf_dist_model"
MODEL_V = "conf_dist_versioned"
MODEL_OLD = "conf_dist_old"
MODEL_DICT = "conf_dist_dict"
VERSION_A = "1.0.0"
VERSION_B = "2.0.0"

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def _pred(uid, model_name, version, probs, ts=None):
    return Prediction(
        user_id=uid,
        model_name=model_name,
        model_version=version,
        input_features={"x": 1},
        prediction_result=1,
        probabilities=probs,
        response_time_ms=5.0,
        status="success",
        timestamp=ts or NOW,
    )


async def _setup():
    async with _TestSessionLocal() as db:
        user = await DBService.get_user_by_token(db, TOKEN)
        if not user:
            user = await DBService.create_user(
                db,
                username="conf_dist_user",
                email="conf_dist_user@test.com",
                api_token=TOKEN,
                role="admin",
                rate_limit=100000,
            )

        uid = user.id

        # MODEL — mixed confidences for basic tests
        for conf in [0.55, 0.65, 0.75, 0.85, 0.95]:
            db.add(_pred(uid, MODEL, VERSION_A, [1 - conf, conf]))

        # MODEL_V — two versions, version A has 3 preds, version B has 2
        for _ in range(3):
            db.add(_pred(uid, MODEL_V, VERSION_A, [0.2, 0.8]))
        for _ in range(2):
            db.add(_pred(uid, MODEL_V, VERSION_B, [0.1, 0.9]))

        # MODEL_OLD — one recent prediction + one 30-day-old prediction
        db.add(_pred(uid, MODEL_OLD, VERSION_A, [0.3, 0.7]))
        db.add(_pred(uid, MODEL_OLD, VERSION_A, [0.4, 0.6], ts=NOW - timedelta(days=30)))

        # MODEL_DICT — probabilities stored as dict
        db.add(_pred(uid, MODEL_DICT, VERSION_A, {"class_0": 0.1, "class_1": 0.9}))

        await db.commit()


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_confidence_distribution_model_not_found():
    r = client.get("/models/nonexistent_conf_model/confidence-distribution", headers=AUTH)
    assert r.status_code == 404


def test_confidence_distribution_no_auth():
    r = client.get(f"/models/{MODEL}/confidence-distribution")
    assert r.status_code in (401, 403)


def test_confidence_distribution_no_predictions():
    """Modèle sans aucune prédiction → sample_count=0, histogram vide."""
    # Create model metadata via POST so 404 check passes
    import io
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pkl = pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))

    client.post(
        "/models",
        data={"name": "conf_dist_empty", "version": "1.0.0"},
        files={"file": ("m.pkl", io.BytesIO(pkl), "application/octet-stream")},
        headers=AUTH,
    )
    r = client.get("/models/conf_dist_empty/confidence-distribution", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["sample_count"] == 0
    assert data["histogram"] == []
    assert data["mean_confidence"] == 0.0
    assert data["pct_high_confidence"] == 0.0
    assert data["pct_uncertain"] == 0.0


def test_confidence_distribution_basic():
    """Retourne sample_count correct et histogram non vide."""
    import io, pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pkl = pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))
    client.post(
        "/models",
        data={"name": MODEL, "version": VERSION_A},
        files={"file": ("m.pkl", io.BytesIO(pkl), "application/octet-stream")},
        headers=AUTH,
    )

    r = client.get(f"/models/{MODEL}/confidence-distribution", headers=AUTH, params={"days": 7})
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == MODEL
    assert data["period_days"] == 7
    assert data["sample_count"] == 5
    assert 0.0 < data["mean_confidence"] < 1.0
    assert len(data["histogram"]) == 10
    # bins should cover [0.5, 1.0]
    assert data["histogram"][0]["bin_min"] == 0.5
    assert data["histogram"][-1]["bin_max"] == 1.0
    # percentages must sum to ~1
    total_pct = sum(b["pct"] for b in data["histogram"])
    assert abs(total_pct - 1.0) < 0.01


def test_confidence_distribution_pct_high_confidence():
    """Toutes les prédictions > 0.9 → pct_high_confidence=1.0."""
    import io, pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pkl = pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))

    async def _add():
        async with _TestSessionLocal() as db:
            user = await DBService.get_user_by_token(db, TOKEN)
            for _ in range(5):
                db.add(_pred(user.id, "conf_dist_high", "1.0.0", [0.05, 0.95]))
            await db.commit()

    asyncio.run(_add())

    client.post(
        "/models",
        data={"name": "conf_dist_high", "version": "1.0.0"},
        files={"file": ("m.pkl", io.BytesIO(pkl), "application/octet-stream")},
        headers=AUTH,
    )
    r = client.get(
        "/models/conf_dist_high/confidence-distribution",
        headers=AUTH,
        params={"days": 7, "high_threshold": 0.80},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["pct_high_confidence"] == 1.0
    assert data["pct_uncertain"] == 0.0


def test_confidence_distribution_pct_uncertain():
    """Toutes les prédictions à ~0.55 → pct_uncertain=1.0 avec seuil par défaut 0.60."""
    import io, pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pkl = pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))

    async def _add():
        async with _TestSessionLocal() as db:
            user = await DBService.get_user_by_token(db, TOKEN)
            for _ in range(5):
                db.add(_pred(user.id, "conf_dist_unc", "1.0.0", [0.45, 0.55]))
            await db.commit()

    asyncio.run(_add())

    client.post(
        "/models",
        data={"name": "conf_dist_unc", "version": "1.0.0"},
        files={"file": ("m.pkl", io.BytesIO(pkl), "application/octet-stream")},
        headers=AUTH,
    )
    r = client.get(
        "/models/conf_dist_unc/confidence-distribution",
        headers=AUTH,
        params={"days": 7},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["pct_uncertain"] == 1.0
    assert data["pct_high_confidence"] == 0.0


def test_confidence_distribution_custom_thresholds():
    """?high_threshold=0.9&uncertain_threshold=0.7 modifie les pourcentages."""
    r_default = client.get(
        f"/models/{MODEL}/confidence-distribution",
        headers=AUTH,
        params={"days": 7},
    )
    r_strict = client.get(
        f"/models/{MODEL}/confidence-distribution",
        headers=AUTH,
        params={"days": 7, "high_threshold": 0.9, "uncertain_threshold": 0.7},
    )
    assert r_default.status_code == 200
    assert r_strict.status_code == 200
    # Stricter thresholds → fewer "high" predictions, more "uncertain"
    assert r_strict.json()["pct_high_confidence"] <= r_default.json()["pct_high_confidence"]
    assert r_strict.json()["pct_uncertain"] >= r_default.json()["pct_uncertain"]


def test_confidence_distribution_version_filter():
    """?version=X ne compte que les prédictions de cette version."""
    import io, pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pkl = pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))
    for v in (VERSION_A, VERSION_B):
        client.post(
            "/models",
            data={"name": MODEL_V, "version": v},
            files={"file": ("m.pkl", io.BytesIO(pkl), "application/octet-stream")},
            headers=AUTH,
        )

    r_a = client.get(
        f"/models/{MODEL_V}/confidence-distribution",
        headers=AUTH,
        params={"days": 7, "version": VERSION_A},
    )
    r_b = client.get(
        f"/models/{MODEL_V}/confidence-distribution",
        headers=AUTH,
        params={"days": 7, "version": VERSION_B},
    )
    assert r_a.status_code == 200
    assert r_b.status_code == 200
    assert r_a.json()["sample_count"] == 3
    assert r_b.json()["sample_count"] == 2
    assert r_a.json()["version"] == VERSION_A
    assert r_b.json()["version"] == VERSION_B


def test_confidence_distribution_days_filter():
    """Une prédiction vieille de 30 jours est exclue avec days=7."""
    import io, pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pkl = pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))
    client.post(
        "/models",
        data={"name": MODEL_OLD, "version": VERSION_A},
        files={"file": ("m.pkl", io.BytesIO(pkl), "application/octet-stream")},
        headers=AUTH,
    )

    r_narrow = client.get(
        f"/models/{MODEL_OLD}/confidence-distribution",
        headers=AUTH,
        params={"days": 7},
    )
    r_wide = client.get(
        f"/models/{MODEL_OLD}/confidence-distribution",
        headers=AUTH,
        params={"days": 60},
    )
    assert r_narrow.status_code == 200
    assert r_wide.status_code == 200
    assert r_narrow.json()["sample_count"] == 1
    assert r_wide.json()["sample_count"] == 2


def test_confidence_distribution_dict_probabilities():
    """Probabilités stockées en dict → max extrait correctement."""
    import io, pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pkl = pickle.dumps(LogisticRegression(max_iter=200).fit(X, y))
    client.post(
        "/models",
        data={"name": MODEL_DICT, "version": VERSION_A},
        files={"file": ("m.pkl", io.BytesIO(pkl), "application/octet-stream")},
        headers=AUTH,
    )

    r = client.get(
        f"/models/{MODEL_DICT}/confidence-distribution",
        headers=AUTH,
        params={"days": 7},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["sample_count"] == 1
    # max({"class_0": 0.1, "class_1": 0.9}) = 0.9 → falls in [0.8, 0.9) or [0.9, 1.0] bin
    assert data["mean_confidence"] == 0.9
    assert data["pct_high_confidence"] == 1.0
