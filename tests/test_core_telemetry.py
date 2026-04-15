"""
Tests unitaires — setup_telemetry() (src/core/telemetry.py).

Stratégie :
  Toute la pile OpenTelemetry est mockée via unittest.mock.patch.
  Les imports OTEL ont lieu au chargement du module ; on patche les symboles
  directement dans le module src.core.telemetry.

Couvre :
- Initialisation TracerProvider / MeterProvider / LoggerProvider
- Appel des setters OTEL globaux (set_tracer_provider, set_meter_provider, etc.)
- Instrumentation FastAPI / SQLAlchemy conditionnelle
- Utilisation des settings (service.name, endpoint)
"""

from unittest.mock import MagicMock, call, patch


# -------------------------------------------------------------------------
# Utilitaire : liste des patches OTEL à appliquer en bloc
# -------------------------------------------------------------------------

PATCHES = [
    "src.core.telemetry.Resource",
    "src.core.telemetry.TracerProvider",
    "src.core.telemetry.BatchSpanProcessor",
    "src.core.telemetry.OTLPSpanExporter",
    "src.core.telemetry.trace",
    "src.core.telemetry.PeriodicExportingMetricReader",
    "src.core.telemetry.OTLPMetricExporter",
    "src.core.telemetry.MeterProvider",
    "src.core.telemetry.metrics",
    "src.core.telemetry.LoggerProvider",
    "src.core.telemetry.BatchLogRecordProcessor",
    "src.core.telemetry.OTLPLogExporter",
    "src.core.telemetry.set_logger_provider",
    "src.core.telemetry.LoggingHandler",
    "src.core.telemetry.LoggingInstrumentor",
    "src.core.telemetry.FastAPIInstrumentor",
    "src.core.telemetry.SQLAlchemyInstrumentor",
]


def _all_mocks(**extras):
    """Context manager retournant un dict nom→mock pour tous les patches OTEL."""
    import contextlib
    import logging

    @contextlib.contextmanager
    def _ctx():
        # Sauvegarder les handlers du root logger pour les restaurer après
        root_logger = logging.getLogger()
        saved_handlers = list(root_logger.handlers)
        try:
            with contextlib.ExitStack() as stack:
                mocks = {}
                for target in PATCHES:
                    name = target.split(".")[-1]
                    mocks[name] = stack.enter_context(patch(target))
                # Injecter des retours sensibles
                mocks["TracerProvider"].return_value = MagicMock()
                mocks["MeterProvider"].return_value = MagicMock()
                mocks["LoggerProvider"].return_value = MagicMock()
                mocks["trace"].get_tracer.return_value = MagicMock()
                # Fixer le level du handler OTEL mocké pour éviter TypeError
                # (logging.callHandlers fait `record.levelno >= hdlr.level`)
                mocks["LoggingHandler"].return_value.level = logging.NOTSET
                mocks.update(extras)
                yield mocks
        finally:
            # Restaurer les handlers du root logger (évite la fuite de mock handlers)
            root_logger.handlers = saved_handlers

    return _ctx()


class TestSetupTelemetryBasic:
    """Tests de base pour setup_telemetry()."""

    def test_setup_telemetry_runs_without_exception(self):
        """setup_telemetry(app) s'exécute sans lever d'exception."""
        from src.core.telemetry import setup_telemetry

        mock_app = MagicMock()
        with _all_mocks():
            setup_telemetry(mock_app)

    def test_setup_telemetry_returns_tracer(self):
        """setup_telemetry() retourne un objet (le tracer)."""
        from src.core.telemetry import setup_telemetry

        mock_app = MagicMock()
        with _all_mocks() as mocks:
            result = setup_telemetry(mock_app)
            assert result is mocks["trace"].get_tracer.return_value


class TestTracerProvider:
    """Tests relatifs à la configuration des traces."""

    def test_tracer_provider_created(self):
        """TracerProvider est instancié une fois."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            mocks["TracerProvider"].assert_called_once()

    def test_tracer_provider_receives_resource(self):
        """TracerProvider est créé avec un argument resource."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            call_kwargs = mocks["TracerProvider"].call_args[1]
            assert "resource" in call_kwargs

    def test_batch_span_processor_receives_otlp_exporter(self):
        """BatchSpanProcessor est appelé avec un OTLPSpanExporter."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            mocks["BatchSpanProcessor"].assert_called_once()
            args = mocks["BatchSpanProcessor"].call_args[0]
            # Premier argument = l'exporter
            assert args[0] is mocks["OTLPSpanExporter"].return_value

    def test_set_tracer_provider_called_once(self):
        """trace.set_tracer_provider est appelé exactement une fois."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            mocks["trace"].set_tracer_provider.assert_called_once()


class TestMeterProvider:
    """Tests relatifs à la configuration des métriques."""

    def test_meter_provider_created_with_metric_readers(self):
        """MeterProvider est instancié avec metric_readers."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            call_kwargs = mocks["MeterProvider"].call_args[1]
            assert "metric_readers" in call_kwargs

    def test_set_meter_provider_called_once(self):
        """metrics.set_meter_provider est appelé exactement une fois."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            mocks["metrics"].set_meter_provider.assert_called_once()


class TestLoggerProvider:
    """Tests relatifs à la configuration des logs OTEL."""

    def test_logger_provider_created(self):
        """LoggerProvider est instancié une fois."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            mocks["LoggerProvider"].assert_called_once()

    def test_set_logger_provider_called(self):
        """set_logger_provider est appelé avec le LoggerProvider."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            mocks["set_logger_provider"].assert_called_once()


class TestInstrumentation:
    """Tests relatifs à l'auto-instrumentation."""

    def test_fastapi_instrumentor_called(self):
        """FastAPIInstrumentor.instrument_app est appelé avec l'app."""
        from src.core.telemetry import setup_telemetry

        mock_app = MagicMock()
        with _all_mocks() as mocks:
            setup_telemetry(mock_app)
            mocks["FastAPIInstrumentor"].instrument_app.assert_called_once_with(mock_app)

    def test_sqlalchemy_not_called_when_engine_none(self):
        """SQLAlchemyInstrumentor.instrument() n'est PAS appelé si engine=None."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock(), engine=None)
            mocks["SQLAlchemyInstrumentor"].return_value.instrument.assert_not_called()

    def test_sqlalchemy_called_with_sync_engine(self):
        """SQLAlchemyInstrumentor().instrument() est appelé avec engine.sync_engine."""
        from src.core.telemetry import setup_telemetry

        mock_engine = MagicMock()
        mock_engine.sync_engine = MagicMock()
        with _all_mocks() as mocks:
            setup_telemetry(MagicMock(), engine=mock_engine)
            mocks["SQLAlchemyInstrumentor"].return_value.instrument.assert_called_once_with(
                engine=mock_engine.sync_engine
            )


class TestResourceConfiguration:
    """Tests relatifs à la configuration de la Resource OTEL."""

    def test_resource_create_uses_service_name(self):
        """Resource.create() reçoit 'service.name' depuis les settings."""
        from src.core.config import settings
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            call_args = mocks["Resource"].create.call_args
            resource_dict = call_args[0][0]
            assert "service.name" in resource_dict
            assert resource_dict["service.name"] == settings.OTEL_SERVICE_NAME

    def test_resource_create_uses_api_version(self):
        """Resource.create() inclut 'service.version'."""
        from src.core.telemetry import setup_telemetry

        with _all_mocks() as mocks:
            setup_telemetry(MagicMock())
            call_args = mocks["Resource"].create.call_args
            resource_dict = call_args[0][0]
            assert "service.version" in resource_dict
