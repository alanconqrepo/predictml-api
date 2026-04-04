"""
Service pour les opérations de base de données
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import User, Prediction, ModelMetadata


class DBService:
    """Service pour les opérations CRUD de la base de données"""

    # === Users ===

    @staticmethod
    async def get_user_by_token(db: AsyncSession, api_token: str) -> Optional[User]:
        """Récupère un utilisateur par son token"""
        result = await db.execute(
            select(User).where(and_(User.api_token == api_token, User.is_active == True))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """Récupère un utilisateur par son ID"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def create_user(
        db: AsyncSession,
        username: str,
        email: str,
        api_token: str,
        role: str = "user",
        rate_limit: int = 1000
    ) -> User:
        """Crée un nouvel utilisateur"""
        user = User(
            username=username,
            email=email,
            api_token=api_token,
            role=role,
            rate_limit_per_day=rate_limit
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def update_user_last_login(db: AsyncSession, user_id: int):
        """Met à jour la dernière connexion d'un utilisateur"""
        user = await DBService.get_user_by_id(db, user_id)
        if user:
            user.last_login = datetime.utcnow()
            await db.commit()

    @staticmethod
    async def get_user_prediction_count_today(db: AsyncSession, user_id: int) -> int:
        """Compte le nombre de prédictions d'un utilisateur aujourd'hui"""
        today = datetime.utcnow().date()
        result = await db.execute(
            select(func.count(Prediction.id)).where(
                and_(
                    Prediction.user_id == user_id,
                    func.date(Prediction.timestamp) == today
                )
            )
        )
        return result.scalar() or 0

    # === Predictions ===

    @staticmethod
    async def create_prediction(
        db: AsyncSession,
        user_id: int,
        model_name: str,
        model_version: Optional[str],
        input_features: list,
        prediction_result: any,
        probabilities: Optional[list],
        response_time_ms: float,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        id_obs: Optional[str] = None
    ) -> Prediction:
        """Enregistre une prédiction"""
        prediction = Prediction(
            user_id=user_id,
            model_name=model_name,
            model_version=model_version,
            id_obs=id_obs,
            input_features=input_features,
            prediction_result=prediction_result,
            probabilities=probabilities,
            response_time_ms=response_time_ms,
            client_ip=client_ip,
            user_agent=user_agent,
            status=status,
            error_message=error_message
        )
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)
        return prediction

    @staticmethod
    async def get_predictions(
        db: AsyncSession,
        user_id: Optional[int] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Prediction]:
        """Récupère l'historique des prédictions avec filtres"""
        query = select(Prediction).order_by(Prediction.timestamp.desc())

        if user_id:
            query = query.where(Prediction.user_id == user_id)
        if model_name:
            query = query.where(Prediction.model_name == model_name)

        query = query.limit(limit).offset(offset)

        result = await db.execute(query)
        return result.scalars().all()

    # === Model Metadata ===

    @staticmethod
    async def create_model_metadata(
        db: AsyncSession,
        name: str,
        version: str,
        minio_bucket: str,
        minio_object_key: str,
        **kwargs
    ) -> ModelMetadata:
        """Crée les métadonnées d'un modèle"""
        metadata = ModelMetadata(
            name=name,
            version=version,
            minio_bucket=minio_bucket,
            minio_object_key=minio_object_key,
            **kwargs
        )
        db.add(metadata)
        await db.commit()
        await db.refresh(metadata)
        return metadata

    @staticmethod
    async def get_model_metadata(
        db: AsyncSession,
        name: str,
        version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Récupère les métadonnées d'un modèle"""
        query = select(ModelMetadata).where(
            and_(
                ModelMetadata.name == name,
                ModelMetadata.is_active == True
            )
        )

        if version:
            query = query.where(ModelMetadata.version == version)
        else:
            # Récupérer la version la plus récente si non spécifiée
            query = query.order_by(ModelMetadata.created_at.desc())

        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_active_models(db: AsyncSession) -> List[ModelMetadata]:
        """Récupère tous les modèles actifs"""
        result = await db.execute(
            select(ModelMetadata).where(ModelMetadata.is_active == True)
        )
        return result.scalars().all()

    @staticmethod
    async def deactivate_model(db: AsyncSession, name: str, version: str) -> bool:
        """Désactive un modèle"""
        metadata = await DBService.get_model_metadata(db, name, version)
        if metadata:
            metadata.is_active = False
            metadata.deprecated_at = datetime.utcnow()
            await db.commit()
            return True
        return False


# Instance globale (optionnel)
db_service = DBService()
