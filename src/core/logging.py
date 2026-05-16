"""
Configuration du logging structuré (JSON) pour l'application.

En production (DEBUG=False) : sortie JSON exploitable par ELK/Datadog/CloudWatch.
En développement (DEBUG=True) : sortie colorée lisible dans le terminal.

Architecture :
  On attache le handler structlog au logger "src" (propagate=False) en PLUS
  du root logger. Ainsi, même si uvicorn reconfigure le root logger via
  dictConfig dans ses workers, le logger "src.*" conserve son handler direct
  et continue d'écrire sur stderr — visible dans docker logs.
"""

import logging
import sys

import structlog


def setup_logging(debug: bool = False) -> None:
    """Configure structlog avec rendu JSON en production et pretty-print en debug."""
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

    # Root logger — pour les libs tierces et le fallback
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Logger "src" avec handler direct + propagate=False.
    # Uvicorn applique son propre dictConfig dans chaque worker après notre
    # setup_logging(). Avec propagate=False, les logs "src.*" ne passent plus
    # par le root (potentiellement écrasé) — ils ont leur propre handler garanti.
    src_logger = logging.getLogger("src")
    src_logger.handlers.clear()
    src_logger.addHandler(handler)
    src_logger.setLevel(level)
    src_logger.propagate = False

    # Réduire le bruit des libs tierces en production
    if not debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
