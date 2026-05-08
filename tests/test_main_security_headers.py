"""
Tests pour SecurityHeadersMiddleware de src/main.py.

Vérifie que chaque réponse HTTP contient les en-têtes de sécurité
ajoutés par SecurityHeadersMiddleware, quel que soit l'endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# En-têtes attendus et leurs valeurs
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store",
}


# ---------------------------------------------------------------------------
# Tests individuels sur GET /
# ---------------------------------------------------------------------------


def test_x_content_type_options_header():
    """X-Content-Type-Options: nosniff doit être présent."""
    r = client.get("/")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"


def test_x_frame_options_header():
    """X-Frame-Options: DENY doit être présent."""
    r = client.get("/")
    assert r.headers.get("X-Frame-Options") == "DENY"


def test_x_xss_protection_header():
    """X-XSS-Protection: 1; mode=block doit être présent."""
    r = client.get("/")
    assert r.headers.get("X-XSS-Protection") == "1; mode=block"


def test_referrer_policy_header():
    """Referrer-Policy: strict-origin-when-cross-origin doit être présent."""
    r = client.get("/")
    assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_cache_control_header():
    """Cache-Control: no-store doit être présent."""
    r = client.get("/")
    assert r.headers.get("Cache-Control") == "no-store"


# ---------------------------------------------------------------------------
# Vérification sur plusieurs endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/health", "/models"])
def test_security_headers_present_on_multiple_endpoints(path):
    """Tous les en-têtes de sécurité sont présents sur chaque endpoint public."""
    r = client.get(path)
    for header, expected_value in _SECURITY_HEADERS.items():
        assert r.headers.get(header) == expected_value, (
            f"En-tête '{header}' manquant ou incorrect sur {path} "
            f"(obtenu: {r.headers.get(header)!r}, attendu: {expected_value!r})"
        )


def test_security_headers_present_on_404():
    """Les en-têtes de sécurité sont ajoutés même sur les réponses 4xx."""
    r = client.get("/endpoint-inexistant-xyz-999")
    for header, expected_value in _SECURITY_HEADERS.items():
        assert r.headers.get(header) == expected_value, (
            f"En-tête '{header}' manquant sur réponse 404"
        )


def test_security_headers_present_on_401():
    """Les en-têtes de sécurité sont ajoutés sur les réponses 401."""
    r = client.get("/health/dependencies")  # requiert admin
    assert r.status_code in (401, 403)
    for header, expected_value in _SECURITY_HEADERS.items():
        assert r.headers.get(header) == expected_value, (
            f"En-tête '{header}' manquant sur réponse {r.status_code}"
        )
