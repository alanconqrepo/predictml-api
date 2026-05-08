"""
Point d'entrée principal de l'application FastAPI
"""

import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict

import httpx
import structlog
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client import multiprocess as prom_multiprocess
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from src.api import models, monitoring, observed_results, predict, users
from src.core.config import settings
from src.core.logging import setup_logging
from src.core.rate_limit import limiter
from src.core.security import require_admin
from src.db.database import close_db, engine, get_db, init_db
from src.schemas.health import DependencyDetail, DependencyHealthResponse
from src.services.minio_service import minio_service
from src.services.model_service import model_service

setup_logging(debug=settings.DEBUG)
logger = structlog.get_logger(__name__)


async def run_migrations() -> None:
    """Applique les migrations Alembic en attente (alembic upgrade head)."""

    def _run() -> None:
        cfg = AlembicConfig("alembic.ini")
        alembic_command.upgrade(cfg, "head")

    await asyncio.get_event_loop().run_in_executor(None, _run)


def _check_metrics_config() -> None:
    """Lève RuntimeError en production si METRICS_TOKEN n'est pas défini."""
    if not settings.METRICS_TOKEN:
        logger.warning("METRICS_TOKEN non défini — endpoint /metrics accessible publiquement")
        if not settings.DEBUG:
            raise RuntimeError(
                "METRICS_TOKEN doit être défini en production. "
                'Générez-en un avec : python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info(
        "Démarrage de l'application",
        title=settings.API_TITLE,
        version=settings.API_VERSION,
    )

    _check_metrics_config()

    try:
        await run_migrations()
        logger.info("Migrations Alembic appliquées")
    except Exception as e:
        logger.warning("Avertissement migrations", error=str(e))

    try:
        await init_db()
        logger.info("Base de données connectée")
    except Exception as e:
        logger.warning("Avertissement DB", error=str(e))

    logger.info(
        "Application prête",
        postgresql=(
            settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "configured"
        ),
        minio_endpoint=settings.MINIO_ENDPOINT,
        minio_bucket=settings.MINIO_BUCKET,
    )

    # Démarrer le scheduler de supervision (alertes + webhooks sur événements modèle)
    try:
        from src.tasks.supervision_reporter import start_scheduler

        start_scheduler()
        logger.info("Scheduler de supervision démarré")
    except Exception as e:
        logger.warning("Impossible de démarrer le scheduler", error=str(e))

    # Démarrer le scheduler de ré-entraînement automatique
    try:
        from src.tasks.retrain_scheduler import start_retrain_scheduler

        await start_retrain_scheduler()
        logger.info("Scheduler de ré-entraînement démarré")
    except Exception as e:
        logger.warning("Impossible de démarrer le scheduler de ré-entraînement", error=str(e))

    yield

    # Shutdown
    logger.info("Fermeture de l'application")
    try:
        from src.tasks.supervision_reporter import stop_scheduler

        stop_scheduler()
    except Exception:
        pass

    try:
        from src.tasks.retrain_scheduler import stop_retrain_scheduler

        stop_retrain_scheduler()
    except Exception:
        pass
    try:
        await close_db()
        logger.info("Connexions DB fermées")
    except Exception as e:
        logger.warning("Erreur fermeture DB", error=str(e))
    try:
        await model_service.close()
    except Exception as e:
        logger.warning("Erreur fermeture Redis", error=str(e))
    logger.info("Application arrêtée")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Ajoute les en-têtes de sécurité HTTP à chaque réponse."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response


# Créer l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="API REST pour faire des prédictions avec plusieurs modèles scikit-learn",
    lifespan=lifespan,
)

# Rate limiting par IP (slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — n'autoriser que les origines explicitement configurées
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.add_middleware(SecurityHeadersMiddleware)

# Instrumenter l'app sans exposer — l'endpoint /metrics est défini plus bas
Instrumentator().instrument(app)

# Activer OpenTelemetry si demandé
if settings.ENABLE_OTEL:
    from src.core.telemetry import setup_telemetry

    setup_telemetry(app, engine=engine)

# Inclure les routers
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(users.router)
app.include_router(observed_results.router)
app.include_router(monitoring.router)


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    if settings.METRICS_TOKEN:
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {settings.METRICS_TOKEN}"
        if not hmac.compare_digest(auth.encode(), expected.encode()):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        prom_multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """
    Endpoint racine - Informations sur l'API
    """
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint

    Vérifie que l'API fonctionne correctement
    """
    try:
        available = await model_service.get_available_models(db)
        cached = await model_service.get_cached_models()

        return {
            "status": "healthy",
            "database": "connected",
            "models_available": len(available),
            "models_cached": len(cached),
        }
    except Exception as e:
        logger.warning("Health check dégradé", error=str(e))
        return {
            "status": "degraded",
            "database": "error",
        }


async def _check_db(db: AsyncSession) -> DependencyDetail:
    start = time.monotonic()
    try:
        await db.execute(text("SELECT 1"))
        return DependencyDetail(status="ok", latency_ms=round((time.monotonic() - start) * 1000, 1))
    except Exception as exc:
        logger.warning("Health check DB failed", error=str(exc))
        return DependencyDetail(status="error", latency_ms=None, detail="connexion échouée")


async def _check_redis() -> DependencyDetail:
    start = time.monotonic()
    try:
        redis = await model_service._get_redis()
        await redis.ping()
        return DependencyDetail(status="ok", latency_ms=round((time.monotonic() - start) * 1000, 1))
    except Exception as exc:
        logger.warning("Health check Redis failed", error=str(exc))
        return DependencyDetail(status="error", latency_ms=None, detail="connexion échouée")


async def _check_minio() -> DependencyDetail:
    start = time.monotonic()
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, minio_service.client.bucket_exists, minio_service.bucket)
        return DependencyDetail(status="ok", latency_ms=round((time.monotonic() - start) * 1000, 1))
    except Exception as exc:
        logger.warning("Health check MinIO failed", error=str(exc))
        return DependencyDetail(status="error", latency_ms=None, detail="connexion échouée")


async def _check_mlflow() -> DependencyDetail:
    start = time.monotonic()
    mlflow_uri = settings.MLFLOW_TRACKING_URI
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{mlflow_uri}/api/2.0/mlflow/experiments/list?max_results=1")
            resp.raise_for_status()
        return DependencyDetail(status="ok", latency_ms=round((time.monotonic() - start) * 1000, 1))
    except Exception as exc:
        logger.warning("Health check MLflow failed", error=str(exc))
        return DependencyDetail(status="error", latency_ms=None, detail="connexion échouée")


@app.get("/health/dependencies", response_model=DependencyHealthResponse)
async def health_dependencies(
    db: AsyncSession = Depends(get_db), _: object = Depends(require_admin)
):
    """
    Health check détaillé — vérifie DB, Redis, MinIO et MLflow en parallèle.

    - **ok** : toutes les dépendances répondent
    - **degraded** : ≥ 1 dépendance non-critique en erreur
    - **critical** : la base de données est inaccessible
    """
    db_result, redis_result, minio_result, mlflow_result = await asyncio.gather(
        _check_db(db),
        _check_redis(),
        _check_minio(),
        _check_mlflow(),
    )

    deps: Dict[str, DependencyDetail] = {
        "database": db_result,
        "redis": redis_result,
        "minio": minio_result,
        "mlflow": mlflow_result,
    }

    if db_result.status == "error":
        global_status = "critical"
    elif any(d.status == "error" for d in deps.values()):
        global_status = "degraded"
    else:
        global_status = "ok"

    return DependencyHealthResponse(
        status=global_status,
        checked_at=datetime.now(timezone.utc),
        dependencies=deps,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
