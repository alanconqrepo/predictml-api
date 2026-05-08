"""
Tests unitaires — classe Settings (src/core/config.py).

Note technique :
  Settings utilise des attributs de classe (class-level variables) évalués à
  l'import du module. Pour tester les overrides d'env vars, on recharge le module
  avec importlib.reload() après avoir modifié os.environ.

Couvre :
- Valeurs par défaut de tous les champs critiques
- Parsing booléen (true/false, case-insensitive)
- Parsing entier et float
- Parsing liste CSV (ALERT_EMAIL_TO)
- Override via variables d'environnement
"""

import importlib
import os
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _env_patch(env_vars: dict):
    """Context manager qui set des env vars sans recharger Settings."""
    with patch.dict(os.environ, env_vars):
        yield


@contextmanager
def _with_env(**env_vars):
    """Context manager qui set des env vars, recharge Settings, puis restaure."""
    original = {}
    for k, v in env_vars.items():
        original[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        import src.core.config as cfg

        importlib.reload(cfg)
        yield cfg.Settings()
    finally:
        for k, old_v in original.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v
        # Recharger pour restaurer les valeurs originales
        importlib.reload(cfg)


class TestSettingsDefaults:
    """Valeurs par défaut de Settings via le singleton."""

    def test_api_title_contains_predictml(self):
        """API_TITLE contient 'PredictML'."""
        from src.core.config import settings

        assert "PredictML" in settings.API_TITLE

    def test_api_version_non_empty(self):
        """API_VERSION est définie et non vide."""
        from src.core.config import settings

        assert settings.API_VERSION != ""

    def test_host_default_is_0_0_0_0(self):
        """HOST vaut '0.0.0.0' par défaut."""
        from src.core.config import settings

        assert settings.HOST == "0.0.0.0"

    def test_settings_singleton_is_not_none(self):
        """Le singleton 'settings' est importable et non None."""
        from src.core.config import settings

        assert settings is not None

    def test_settings_has_database_url(self):
        """settings.DATABASE_URL est défini."""
        from src.core.config import settings

        assert hasattr(settings, "DATABASE_URL")
        assert settings.DATABASE_URL != ""

    def test_settings_has_minio_endpoint(self):
        """settings.MINIO_ENDPOINT est défini."""
        from src.core.config import settings

        assert hasattr(settings, "MINIO_ENDPOINT")

    def test_settings_has_redis_url(self):
        """settings.REDIS_URL est défini."""
        from src.core.config import settings

        assert hasattr(settings, "REDIS_URL")


class TestBooleanParsingLogic:
    """Teste la logique de parsing booléen utilisée par Settings."""

    def test_true_lowercase_parses_true(self):
        """La logique '.lower() == \"true\"' retourne True pour 'true'."""
        assert ("true".lower() == "true") is True

    def test_true_uppercase_parses_true(self):
        """La logique '.lower() == \"true\"' retourne True pour 'TRUE'."""
        assert ("TRUE".lower() == "true") is True

    def test_yes_parses_false(self):
        """La logique '.lower() == \"true\"' retourne False pour 'yes'."""
        assert ("yes".lower() == "true") is False

    def test_1_parses_false(self):
        """La logique '.lower() == \"true\"' retourne False pour '1'."""
        assert ("1".lower() == "true") is False

    def test_false_lowercase_parses_false(self):
        """La logique '.lower() == \"true\"' retourne False pour 'false'."""
        assert ("false".lower() == "true") is False

    def test_empty_string_parses_false(self):
        """La logique '.lower() == \"true\"' retourne False pour ''."""
        assert ("".lower() == "true") is False


class TestListParsingLogic:
    """Teste la logique de parsing CSV utilisée pour ALERT_EMAIL_TO."""

    def test_empty_string_gives_empty_list(self):
        """Une chaîne vide → liste vide."""
        val = ""
        result = [e.strip() for e in val.split(",") if e.strip()]
        assert result == []

    def test_single_email_gives_list_of_one(self):
        """Une adresse email → liste d'un élément."""
        val = "a@test.com"
        result = [e.strip() for e in val.split(",") if e.strip()]
        assert result == ["a@test.com"]

    def test_two_emails_gives_list_of_two(self):
        """Deux adresses → liste de deux éléments."""
        val = "a@test.com,b@test.com"
        result = [e.strip() for e in val.split(",") if e.strip()]
        assert len(result) == 2
        assert "a@test.com" in result
        assert "b@test.com" in result

    def test_emails_with_spaces_stripped(self):
        """Les espaces autour des adresses sont supprimés."""
        val = " a@test.com , b@test.com "
        result = [e.strip() for e in val.split(",") if e.strip()]
        assert "a@test.com" in result
        assert "b@test.com" in result


class TestSettingsWithEnvOverride:
    """Teste les overrides via variables d'environnement (reload requis)."""

    def test_debug_true_when_env_set(self):
        """DEBUG=true → Settings.DEBUG is True après reload."""
        with _with_env(DEBUG="true") as s:
            assert s.DEBUG is True

    def test_debug_false_when_env_not_set(self):
        """Sans DEBUG env var → Settings.DEBUG is False."""
        with _with_env(DEBUG="false") as s:
            assert s.DEBUG is False

    def test_port_overridable_via_api_port(self):
        """API_PORT=9090 → Settings.PORT == 9090."""
        with _with_env(API_PORT="9090") as s:
            assert s.PORT == 9090

    def test_minio_secure_true_when_set(self):
        """MINIO_SECURE=true → Settings.MINIO_SECURE is True."""
        with _with_env(MINIO_SECURE="true") as s:
            assert s.MINIO_SECURE is True

    def test_redis_cache_ttl_overridable(self):
        """REDIS_CACHE_TTL=7200 → Settings.REDIS_CACHE_TTL == 7200."""
        with _with_env(REDIS_CACHE_TTL="7200") as s:
            assert s.REDIS_CACHE_TTL == 7200

    def test_alert_email_to_parsed_from_csv(self):
        """ALERT_EMAIL_TO=a@x.com,b@x.com → liste de 2 éléments."""
        with _with_env(ALERT_EMAIL_TO="a@x.com,b@x.com") as s:
            assert len(s.ALERT_EMAIL_TO) == 2

    def test_alert_email_to_empty_when_unset(self):
        """ALERT_EMAIL_TO vide → liste vide."""
        with _with_env(ALERT_EMAIL_TO="") as s:
            assert s.ALERT_EMAIL_TO == []

    def test_performance_drift_threshold_overridable(self):
        """PERFORMANCE_DRIFT_ALERT_THRESHOLD=0.20 → 0.20."""
        with _with_env(PERFORMANCE_DRIFT_ALERT_THRESHOLD="0.20") as s:
            assert s.PERFORMANCE_DRIFT_ALERT_THRESHOLD == 0.20

    def test_smtp_starttls_false_when_set(self):
        """SMTP_STARTTLS=false → Settings.SMTP_STARTTLS is False."""
        with _with_env(SMTP_STARTTLS="false") as s:
            assert s.SMTP_STARTTLS is False

    def test_enable_email_alerts_true_when_set(self):
        """ENABLE_EMAIL_ALERTS=true → Settings.ENABLE_EMAIL_ALERTS is True."""
        with _with_env(ENABLE_EMAIL_ALERTS="true") as s:
            assert s.ENABLE_EMAIL_ALERTS is True

    def test_secret_key_overridable(self):
        """SECRET_KEY=my-key → Settings.SECRET_KEY == 'my-key'."""
        with _with_env(SECRET_KEY="my-super-secret-key") as s:
            assert s.SECRET_KEY == "my-super-secret-key"


class TestRequireEnvFunction:
    """Tests unitaires pour la fonction _require_env (lignes non couvertes)."""

    def test_missing_required_var_raises_environment_error(self):
        """Variable obligatoire absente (pas de default) → EnvironmentError."""
        import pytest
        from src.core.config import _require_env

        key = "PYTEST_MISSING_VAR_COVERAGE_XYZ"
        os.environ.pop(key, None)
        with pytest.raises(EnvironmentError, match=key):
            _require_env(key)  # aucun default → _MISSING

    def test_insecure_value_with_debug_true_warns_not_raises(self):
        """Valeur non sécurisée + DEBUG=true → UserWarning émis, pas d'exception."""
        import warnings
        from src.core.config import _require_env

        key = "PYTEST_INSECURE_VAR_DEBUG_XYZ"
        with _env_patch({key: "bad_val", "DEBUG": "true"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = _require_env(key, "bad_val", {"bad_val"})
        assert result == "bad_val"
        security_warns = [x for x in w if "SECURITY" in str(x.message)]
        assert len(security_warns) >= 1

    def test_insecure_value_with_debug_false_raises(self):
        """Valeur non sécurisée + DEBUG=false → EnvironmentError."""
        import pytest
        from src.core.config import _require_env

        key = "PYTEST_INSECURE_PROD_VAR_XYZ"
        with _env_patch({key: "bad_val", "DEBUG": "false"}):
            with pytest.raises(EnvironmentError, match="non sécurisée"):
                _require_env(key, "bad_val", {"bad_val"})

    def test_missing_var_with_default_returns_default(self):
        """Variable absente mais default fourni → default retourné sans exception."""
        from src.core.config import _require_env

        key = "PYTEST_MISSING_VAR_WITH_DEFAULT_XYZ"
        os.environ.pop(key, None)
        result = _require_env(key, "fallback_value")
        assert result == "fallback_value"
