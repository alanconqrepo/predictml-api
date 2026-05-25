"""
Structured (JSON) logging configuration for the application.

In production (DEBUG=False): JSON output suitable for ELK/Datadog/CloudWatch.
In development (DEBUG=True): coloured, human-readable terminal output.

Architecture:
  The structlog handler is attached to both the "src" logger (propagate=False)
  AND the root logger. This way, even if uvicorn reconfigures the root logger
  via dictConfig inside its workers, "src.*" loggers keep their own direct
  handler and continue writing to stderr — visible in docker logs.
"""

import logging
import sys

import structlog


def setup_logging(debug: bool = False) -> None:
    """Configure structlog with JSON rendering in production and pretty-print in debug."""
    level = logging.DEBUG if debug else logging.INFO

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Root logger — for third-party libs and fallback
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # "src" logger with a direct handler + propagate=False.
    # Uvicorn applies its own dictConfig in each worker after our setup_logging().
    # With propagate=False, "src.*" logs no longer go through the root (potentially
    # overwritten) — they have their own guaranteed handler.
    src_logger = logging.getLogger("src")
    src_logger.handlers.clear()
    src_logger.addHandler(handler)
    src_logger.setLevel(level)
    src_logger.propagate = False

    # Reduce noise from third-party libs in production
    if not debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
