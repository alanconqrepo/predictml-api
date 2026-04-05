"""
Smoke tests pour l'API predictml — tests contre l'API live Docker.

PRÉREQUIS :
    docker-compose up -d
    Attendre que l'API soit prête sur http://localhost:8000

DIFFÉRENCE AVEC pytest tests/ :
    - Ce script  -> frappe l'API live (Docker requis, vraie DB, vrai MinIO)
    - pytest tests/ -> utilise TestClient FastAPI en mémoire (aucun Docker, MinIO mocké)

EXÉCUTION :
    python smoke-tests/test_multimodel_api.py

    Variables d'environnement optionnelles :
        API_BASE_URL=http://localhost:8000   (défaut)
        API_TOKEN=<admin_token>              (défaut : token admin du docker-compose)
"""
import os
import sys
import json
import uuid
from datetime import datetime, timedelta

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("API_TOKEN", "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA")

AUTH = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
PUBLIC = {"Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Résultats globaux
# ---------------------------------------------------------------------------
results = {"passed": 0, "failed": 0, "skipped": 0}


def _ok(label: str):
    results["passed"] += 1
    print(f"  PASSED  {label}")


def _fail(label: str, detail: str = ""):
    results["failed"] += 1
    msg = f"  FAILED  {label}"
    if detail:
        msg += f"\n          -> {detail}"
    print(msg)


def _skip(label: str, reason: str = ""):
    results["skipped"] += 1
    msg = f"  SKIP    {label}"
    if reason:
        msg += f"  ({reason})"
    print(msg)


def _section(title: str):
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def test_root():
    label = "GET /"
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        if r.status_code == 200 and "message" in r.json():
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_health():
    label = "GET /health"
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_models_list() -> list:
    """Retourne la liste des modèles disponibles (ou [] si échec)."""
    label = "GET /models"
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        if r.status_code == 200 and isinstance(r.json(), list):
            models = r.json()
            _ok(f"{label}  ({len(models)} modèles)")
            return models
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return []


def test_models_cached():
    label = "GET /models/cached"
    try:
        r = requests.get(f"{BASE_URL}/models/cached", timeout=5)
        if r.status_code == 200 and "cached_models" in r.json():
            data = r.json()
            _ok(f"{label}  ({data['count']} en cache)")
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_model_get(name: str, version: str) -> dict | None:
    """Retourne les métadonnées du modèle, ou None si introuvable."""
    label = f"GET /models/{name}/{version}"
    try:
        r = requests.get(f"{BASE_URL}/models/{name}/{version}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            loaded = data.get("model_loaded", False)
            feature_names = data.get("feature_names")
            info = f"model_loaded={loaded}"
            if feature_names:
                info += f", feature_names={feature_names}"
            _ok(f"{label}  ({info})")
            return data
        if r.status_code == 404:
            _skip(label, "modèle absent de la base")
            return None
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return None


def test_model_get_not_found():
    label = "GET /models/ghost_model/9.9.9  -> 404 attendu"
    try:
        r = requests.get(f"{BASE_URL}/models/ghost_model/9.9.9", timeout=5)
        if r.status_code == 404:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} (attendu 404)")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_model_patch(name: str, version: str) -> bool:
    label = f"PATCH /models/{name}/{version}  (description)"
    try:
        payload = {"description": "smoke-test patch OK"}
        r = requests.patch(
            f"{BASE_URL}/models/{name}/{version}",
            json=payload,
            headers=AUTH,
            timeout=5,
        )
        if r.status_code == 200:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def test_predict_with_model(models: list) -> bool:
    """
    Cherche le premier modèle chargé avec feature_names_in_ et teste POST /predict.
    Skip gracieusement si aucun modèle compatible n'est disponible.
    """
    label = "POST /predict  (dict features)"

    candidate = None
    feature_names = None

    for m in models:
        name = m.get("name")
        version = m.get("version")
        if not name or not version:
            continue
        try:
            r = requests.get(f"{BASE_URL}/models/{name}/{version}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("model_loaded") and data.get("feature_names"):
                    candidate = (name, version)
                    feature_names = data["feature_names"]
                    break
        except Exception:
            continue

    if not candidate:
        _skip(
            label,
            "aucun modèle chargé avec feature_names_in_ disponible "
            "(entraîner un modèle avec DataFrame pandas, ex: create_multiple_advanced_models.py)"
        )
        return None  # type: ignore

    name, version = candidate
    # Construire le dict de features avec des valeurs réalistes (0.5 par défaut)
    features = {fn: 0.5 for fn in feature_names}

    payload = {
        "model_name": name,
        "model_version": version,
        "id_obs": f"smoke-{uuid.uuid4().hex[:8]}",
        "features": features,
    }

    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, headers=AUTH, timeout=10)
        if r.status_code == 200:
            _ok(f"{label}  (modèle={name} v{version})")
            return r.json().get("id_obs")  # retourner l'id_obs pour observed-results
        _fail(label, f"status={r.status_code} body={r.text[:300]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_predict_invalid_model():
    label = "POST /predict  modèle inexistant -> 404 attendu"
    payload = {
        "model_name": "ghost_model",
        "features": {"x": 1.0},
    }
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, headers=AUTH, timeout=5)
        if r.status_code == 404:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} (attendu 404)")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_predict_features_as_list():
    """Vérifie que l'API rejette les features au format liste (attendu 422)."""
    label = "POST /predict  features=liste -> 422 attendu"
    payload = {
        "model_name": "iris_model",
        "features": [5.1, 3.5, 1.4, 0.2],
    }
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, headers=AUTH, timeout=5)
        if r.status_code == 422:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} (attendu 422 — features liste non autorisée)")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_predict_no_auth():
    label = "POST /predict  sans token -> 401 attendu"
    payload = {"model_name": "iris_model", "features": {"x": 1.0}}
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        if r.status_code == 401:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} (attendu 401)")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_predictions_history(model_name: str | None):
    label = "GET /predictions  (historique)"
    if not model_name:
        _skip(label, "aucun modèle disponible pour filtrer")
        return None  # type: ignore

    start = (datetime.utcnow() - timedelta(days=30)).isoformat()
    end = datetime.utcnow().isoformat()

    try:
        r = requests.get(
            f"{BASE_URL}/predictions",
            params={"name": model_name, "start": start, "end": end, "limit": 10},
            headers=AUTH,
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            _ok(f"{label}  (total={data.get('total', '?')})")
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_predictions_missing_params():
    label = "GET /predictions  sans params -> 422 attendu"
    try:
        r = requests.get(f"{BASE_URL}/predictions", headers=AUTH, timeout=5)
        if r.status_code == 422:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} (attendu 422)")
    except Exception as e:
        _fail(label, str(e))
    return False


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def test_users_create() -> tuple[int | None, str | None]:
    """Crée un utilisateur temporaire. Retourne (user_id, token) ou (None, None)."""
    label = "POST /users  (créer user temporaire)"
    unique = uuid.uuid4().hex[:8]
    payload = {
        "username": f"smoke_{unique}",
        "email": f"smoke_{unique}@example.com",
        "role": "user",
        "rate_limit": 100,
    }
    try:
        r = requests.post(f"{BASE_URL}/users", json=payload, headers=AUTH, timeout=5)
        if r.status_code == 201:
            data = r.json()
            _ok(f"{label}  (id={data['id']}, username={data['username']})")
            return data["id"], data["api_token"]
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return None, None


def test_users_list():
    label = "GET /users  (liste admin)"
    try:
        r = requests.get(f"{BASE_URL}/users", headers=AUTH, timeout=5)
        if r.status_code == 200 and isinstance(r.json(), list):
            _ok(f"{label}  ({len(r.json())} utilisateurs)")
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_users_get(user_id: int | None):
    label = f"GET /users/{user_id}"
    if user_id is None:
        _skip(label, "user_id non disponible")
        return None  # type: ignore
    try:
        r = requests.get(f"{BASE_URL}/users/{user_id}", headers=AUTH, timeout=5)
        if r.status_code == 200 and r.json().get("id") == user_id:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_users_delete(user_id: int | None):
    label = f"DELETE /users/{user_id}  (cleanup)"
    if user_id is None:
        _skip(label, "user_id non disponible")
        return None  # type: ignore
    try:
        r = requests.delete(f"{BASE_URL}/users/{user_id}", headers=AUTH, timeout=5)
        if r.status_code == 204:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_users_no_auth():
    label = "GET /users  sans token -> 401 attendu"
    try:
        r = requests.get(f"{BASE_URL}/users", timeout=5)
        if r.status_code == 401:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} (attendu 401)")
    except Exception as e:
        _fail(label, str(e))
    return False


# ---------------------------------------------------------------------------
# Observed Results
# ---------------------------------------------------------------------------

def test_observed_results_post(model_name: str | None, id_obs: str | None):
    label = "POST /observed-results"
    if not model_name:
        _skip(label, "aucun modèle disponible")
        return None  # type: ignore

    obs_id = id_obs if id_obs else f"smoke-obs-{uuid.uuid4().hex[:8]}"
    payload = {
        "data": [
            {
                "id_obs": obs_id,
                "model_name": model_name,
                "date_time": datetime.utcnow().isoformat(),
                "observed_result": 1,
            }
        ]
    }
    try:
        r = requests.post(
            f"{BASE_URL}/observed-results", json=payload, headers=AUTH, timeout=5
        )
        if r.status_code == 200 and "upserted" in r.json():
            _ok(f"{label}  (upserted={r.json()['upserted']})")
            return obs_id
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_observed_results_get(model_name: str | None):
    label = "GET /observed-results"
    if not model_name:
        _skip(label, "aucun modèle disponible")
        return None  # type: ignore
    try:
        r = requests.get(
            f"{BASE_URL}/observed-results",
            params={"model_name": model_name, "limit": 10},
            headers=AUTH,
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            _ok(f"{label}  (total={data.get('total', '?')})")
            return True
        _fail(label, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        _fail(label, str(e))
    return False


def test_observed_results_no_auth():
    label = "GET /observed-results  sans token -> 401 attendu"
    try:
        r = requests.get(f"{BASE_URL}/observed-results", timeout=5)
        if r.status_code == 401:
            _ok(label)
            return True
        _fail(label, f"status={r.status_code} (attendu 401)")
    except Exception as e:
        _fail(label, str(e))
    return False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Smoke Tests — predictml-api")
    print(f"  URL    : {BASE_URL}")
    print(f"  Token  : {API_TOKEN[:12]}...")
    print()
    print("  PRÉREQUIS : docker-compose up -d")
    print()
    print("  DIFFERENCE AVEC pytest tests/ :")
    print("    - Smoke tests  : API live Docker (vraie DB, vrai MinIO)")
    print("    - pytest tests/: TestClient en memoire, sans Docker")
    print("=" * 60)

    # ── Infrastructure ───────────────────────────────────────────────────────
    _section("Infrastructure")
    root_ok = test_root()
    if not root_ok:
        print("\n  L'API ne répond pas. Vérifiez : docker-compose up -d")
        sys.exit(1)
    test_health()

    # ── Models ───────────────────────────────────────────────────────────────
    _section("Models")
    available_models = test_models_list()
    test_models_cached()
    test_model_get_not_found()

    # Tenter GET /models/{name}/{version} sur le premier modèle disponible
    first_model_name = None
    first_model_version = None
    if available_models:
        first = available_models[0]
        first_model_name = first.get("name")
        first_model_version = first.get("version")

    if first_model_name and first_model_version:
        meta = test_model_get(first_model_name, first_model_version)
        test_model_patch(first_model_name, first_model_version)
    else:
        _skip(f"GET /models/{{name}}/{{version}}", "aucun modèle en base")
        _skip(f"PATCH /models/{{name}}/{{version}}", "aucun modèle en base")

    # ── Predictions ──────────────────────────────────────────────────────────
    _section("Predictions")
    predict_id_obs = test_predict_with_model(available_models)
    test_predict_invalid_model()
    test_predict_features_as_list()
    test_predict_no_auth()
    test_predictions_history(first_model_name)
    test_predictions_missing_params()

    # ── Users ────────────────────────────────────────────────────────────────
    _section("Users")
    test_users_no_auth()
    test_users_list()
    created_user_id, _ = test_users_create()
    test_users_get(created_user_id)
    test_users_delete(created_user_id)

    # ── Observed Results ─────────────────────────────────────────────────────
    _section("Observed Results")
    id_obs_str = predict_id_obs if isinstance(predict_id_obs, str) else None
    test_observed_results_post(first_model_name, id_obs_str)
    test_observed_results_get(first_model_name)
    test_observed_results_no_auth()

    # ── Résumé ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    total = results["passed"] + results["failed"] + results["skipped"]
    print(f"  Résultat : {results['passed']} PASSED / {results['failed']} FAILED / {results['skipped']} SKIPPED  (total: {total})")
    print("=" * 60)

    sys.exit(0 if results["failed"] == 0 else 1)
