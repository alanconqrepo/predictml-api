"""
Tests unitaires — utilitaires et logging (src/core/utils.py, src/core/logging.py).

Couvre :
- _utcnow() : retourne un datetime naive UTC récent
- setup_logging() : configure structlog en mode debug ou production
"""

import logging
from datetime import datetime


class TestUtcNow:
    """Tests pour _utcnow()."""

    def test_utcnow_returns_datetime(self):
        """_utcnow() retourne un objet datetime."""
        from src.core.utils import _utcnow

        result = _utcnow()
        assert isinstance(result, datetime)

    def test_utcnow_is_naive(self):
        """_utcnow() retourne un datetime sans tzinfo (naive)."""
        from src.core.utils import _utcnow

        result = _utcnow()
        assert result.tzinfo is None

    def test_utcnow_is_recent(self):
        """_utcnow() retourne une valeur proche de datetime.utcnow()."""
        from src.core.utils import _utcnow

        before = datetime.utcnow()
        result = _utcnow()
        after = datetime.utcnow()
        assert before <= result <= after

    def test_utcnow_two_calls_are_ordered(self):
        """Deux appels successifs à _utcnow() : le second >= au premier."""
        from src.core.utils import _utcnow

        first = _utcnow()
        second = _utcnow()
        assert second >= first


class TestSetupLogging:
    """Tests pour setup_logging(debug)."""

    def _capture_state(self):
        """Sauvegarde l'état courant du logging pour restauration."""
        root = logging.getLogger()
        return {
            "handlers": list(root.handlers),
            "level": root.level,
            "uvicorn_level": logging.getLogger("uvicorn.access").level,
            "sqlalchemy_level": logging.getLogger("sqlalchemy.engine").level,
        }

    def _restore_state(self, state):
        """Restaure l'état du logging après un test."""
        root = logging.getLogger()
        root.handlers = state["handlers"]
        root.setLevel(state["level"])
        logging.getLogger("uvicorn.access").setLevel(state["uvicorn_level"])
        logging.getLogger("sqlalchemy.engine").setLevel(state["sqlalchemy_level"])

    def test_setup_logging_prod_sets_info_level(self):
        """setup_logging(debug=False) → root logger niveau INFO."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            setup_logging(debug=False)
            assert logging.getLogger().level == logging.INFO
        finally:
            self._restore_state(state)

    def test_setup_logging_debug_sets_debug_level(self):
        """setup_logging(debug=True) → root logger niveau DEBUG."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            setup_logging(debug=True)
            assert logging.getLogger().level == logging.DEBUG
        finally:
            self._restore_state(state)

    def test_setup_logging_prod_silences_uvicorn_access(self):
        """setup_logging(debug=False) → uvicorn.access niveau >= WARNING."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            setup_logging(debug=False)
            assert logging.getLogger("uvicorn.access").level >= logging.WARNING
        finally:
            self._restore_state(state)

    def test_setup_logging_prod_silences_sqlalchemy(self):
        """setup_logging(debug=False) → sqlalchemy.engine niveau >= WARNING."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            setup_logging(debug=False)
            assert logging.getLogger("sqlalchemy.engine").level >= logging.WARNING
        finally:
            self._restore_state(state)

    def test_setup_logging_attaches_stream_handler(self):
        """setup_logging() attache un StreamHandler au root logger."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            setup_logging(debug=False)
            handlers = logging.getLogger().handlers
            assert any(isinstance(h, logging.StreamHandler) for h in handlers)
        finally:
            self._restore_state(state)

    def test_setup_logging_no_exception_in_debug_mode(self):
        """setup_logging(debug=True) s'exécute sans lever d'exception."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            setup_logging(debug=True)
        except Exception as exc:
            raise AssertionError(f"setup_logging(debug=True) a levé une exception : {exc}")
        finally:
            self._restore_state(state)

    def test_setup_logging_no_exception_in_prod_mode(self):
        """setup_logging(debug=False) s'exécute sans lever d'exception."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            setup_logging(debug=False)
        except Exception as exc:
            raise AssertionError(f"setup_logging(debug=False) a levé une exception : {exc}")
        finally:
            self._restore_state(state)

    def test_setup_logging_debug_does_not_silence_sqlalchemy(self):
        """setup_logging(debug=True) → sqlalchemy.engine non forcé à WARNING."""
        from src.core.logging import setup_logging

        state = self._capture_state()
        try:
            # Réinitialiser explicitement avant le test pour repartir d'un état connu
            logging.getLogger("sqlalchemy.engine").setLevel(logging.NOTSET)
            setup_logging(debug=True)
            # En mode debug, sqlalchemy.engine ne doit pas être forcé à WARNING
            assert logging.getLogger("sqlalchemy.engine").level != logging.WARNING
        finally:
            self._restore_state(state)
