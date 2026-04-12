"""
Point d'entrée principal de l'application FastAPI
"""

import asyncio
from contextlib import asynccontextmanager

import structlog
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from src.api import models, observed_results, predict, users
from src.core.config import settings
from src.core.logging import setup_logging
from src.db.database import close_db, engine, get_db, init_db
from src.services.model_service import model_service

setup_logging(debug=settings.DEBUG)
logger = structlog.get_logger(__name__)


async def run_migrations() -> None:
    """Applique les migrations Alembic en attente (alembic upgrade head)."""

    def _run() -> None:
        cfg = AlembicConfig("alembic.ini")
        alembic_command.upgrade(cfg, "head")

    await asyncio.get_event_loop().run_in_executor(None, _run)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info(
        "Démarrage de l'application",
        title=settings.API_TITLE,
        version=settings.API_VERSION,
    )

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

    yield

    # Shutdown
    logger.info("Fermeture de l'application")
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


# Créer l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="API REST pour faire des prédictions avec plusieurs modèles scikit-learn",
    lifespan=lifespan,
)

# Activer OpenTelemetry si demandé
if settings.ENABLE_OTEL:
    from src.core.telemetry import setup_telemetry

    setup_telemetry(app, engine=engine)

# Inclure les routers
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(users.router)
app.include_router(observed_results.router)


@app.get("/")
async def root(db: AsyncSession = Depends(get_db)):
    """
    Endpoint racine - Informations sur l'API

    Retourne les informations générales et la liste des modèles disponibles
    """
    try:
        available = await model_service.get_available_models(db)
        cached = await model_service.get_cached_models()

        return {
            "message": "API de prédiction sklearn - Multi Models v2.0",
            "status": "active",
            "models_available": [m["name"] for m in available],
            "models_count": len(available),
            "models_cached_count": len(cached),
        }
    except Exception as e:
        return {
            "message": "API de prédiction sklearn - Multi Models v2.0",
            "status": "active",
            "error": str(e),
            "note": "Exécutez init_db.py pour initialiser la base de données",
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
        return {
            "status": "degraded",
            "database": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
