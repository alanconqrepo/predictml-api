"""
Point d'entrée principal de l'application FastAPI
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.database import get_db, init_db, close_db
from src.services.model_service import model_service
from src.api import predict, models, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    print("=" * 60)
    print(f"🚀 {settings.API_TITLE} v{settings.API_VERSION}")
    print("=" * 60)

    try:
        await init_db()
        print("✅ Base de données initialisée")
    except Exception as e:
        print(f"⚠️  Avertissement DB: {e}")

    print(f"📊 PostgreSQL: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'configured'}")
    print(f"💾 MinIO: {settings.MINIO_ENDPOINT} / Bucket: {settings.MINIO_BUCKET}")
    print("=" * 60)
    print("✅ Application prête!")
    print("=" * 60)

    yield

    # Shutdown
    print("\n👋 Fermeture de l'application...")
    try:
        await close_db()
        print("✅ Connexions DB fermées")
    except Exception as e:
        print(f"⚠️  Erreur lors de la fermeture DB: {e}")
    print("👋 Au revoir!")


# Créer l'application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="API REST pour faire des prédictions avec plusieurs modèles scikit-learn",
    lifespan=lifespan
)

# Inclure les routers
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(users.router)


@app.get("/")
async def root(db: AsyncSession = Depends(get_db)):
    """
    Endpoint racine - Informations sur l'API

    Retourne les informations générales et la liste des modèles disponibles
    """
    try:
        available = await model_service.get_available_models(db)
        cached = model_service.get_cached_models()

        return {
            "message": "API de prédiction sklearn - Multi Models v2.0",
            "status": "active",
            "models_available": [m["name"] for m in available],
            "models_count": len(available),
            "models_cached_count": len(cached)
        }
    except Exception as e:
        return {
            "message": "API de prédiction sklearn - Multi Models v2.0",
            "status": "active",
            "error": str(e),
            "note": "Exécutez init_db.py pour initialiser la base de données"
        }


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint

    Vérifie que l'API fonctionne correctement
    """
    try:
        available = await model_service.get_available_models(db)
        cached = model_service.get_cached_models()

        return {
            "status": "healthy",
            "database": "connected",
            "models_available": len(available),
            "models_cached": len(cached)
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
