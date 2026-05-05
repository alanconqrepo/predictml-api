"""
Tests pour l'endpoint GET /metrics (Prometheus).

Couvre :
- Accès libre sans METRICS_TOKEN configuré
- Accès avec METRICS_TOKEN valide → 200
- Accès avec METRICS_TOKEN invalide → 401
- Content-type Prometheus text/plain
- Branche PROMETHEUS_MULTIPROC_DIR
"""

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestMetricsEndpointOpen:
    """Endpoint /metrics sans METRICS_TOKEN configuré (accès libre)."""

    def test_metrics_returns_200_without_token_configured(self):
        """Sans METRICS_TOKEN → 200 quel que soit le header."""
        with patch("src.main.settings") as mock_settings:
            mock_settings.METRICS_TOKEN = ""
            mock_settings.ENABLE_OTEL = False
            resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_is_prometheus(self):
        """Le content-type doit contenir text/plain (format Prometheus)."""
        with patch("src.main.settings") as mock_settings:
            mock_settings.METRICS_TOKEN = ""
            mock_settings.ENABLE_OTEL = False
            resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_body_not_empty(self):
        """La réponse doit contenir au moins du contenu Prometheus."""
        with patch("src.main.settings") as mock_settings:
            mock_settings.METRICS_TOKEN = ""
            mock_settings.ENABLE_OTEL = False
            resp = client.get("/metrics")
        assert len(resp.content) > 0


class TestMetricsEndpointAuth:
    """Endpoint /metrics avec METRICS_TOKEN configuré."""

    def test_metrics_valid_token_returns_200(self):
        """Bearer token valide → 200."""
        with patch("src.main.settings") as mock_settings:
            mock_settings.METRICS_TOKEN = "secret-metrics-token"
            mock_settings.ENABLE_OTEL = False
            resp = client.get(
                "/metrics",
                headers={"Authorization": "Bearer secret-metrics-token"},
            )
        assert resp.status_code == 200

    def test_metrics_invalid_token_returns_401(self):
        """Bearer token invalide → 401."""
        with patch("src.main.settings") as mock_settings:
            mock_settings.METRICS_TOKEN = "secret-metrics-token"
            mock_settings.ENABLE_OTEL = False
            resp = client.get(
                "/metrics",
                headers={"Authorization": "Bearer wrong-token"},
            )
        assert resp.status_code == 401

    def test_metrics_missing_header_returns_401(self):
        """Aucun header Authorization quand METRICS_TOKEN défini → 401."""
        with patch("src.main.settings") as mock_settings:
            mock_settings.METRICS_TOKEN = "secret-metrics-token"
            mock_settings.ENABLE_OTEL = False
            resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_metrics_wrong_scheme_returns_401(self):
        """Header Authorization sans 'Bearer ' → 401."""
        with patch("src.main.settings") as mock_settings:
            mock_settings.METRICS_TOKEN = "secret-metrics-token"
            mock_settings.ENABLE_OTEL = False
            resp = client.get(
                "/metrics",
                headers={"Authorization": "secret-metrics-token"},
            )
        assert resp.status_code == 401


class TestMetricsMultiprocess:
    """Branche PROMETHEUS_MULTIPROC_DIR."""

    def test_metrics_multiprocess_dir_uses_collector(self, tmp_path):
        """Quand PROMETHEUS_MULTIPROC_DIR est défini → CollectorRegistry utilisé."""
        with (
            patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}),
            patch("src.main.settings") as mock_settings,
            patch("src.main.prom_multiprocess.MultiProcessCollector") as mock_collector,
            patch("src.main.generate_latest", return_value=b"# metrics\n"),
        ):
            mock_settings.METRICS_TOKEN = ""
            mock_settings.ENABLE_OTEL = False
            resp = client.get("/metrics")
        assert resp.status_code == 200
        mock_collector.assert_called_once()
