"""
Service pour les opérations de base de données
"""
import secrets
from datetime import datetime, timezone

def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)
from typing import Optional, List
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.db.models import User, Prediction, ModelMetadata, ObservedResult


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
    async def get_all_users(db: AsyncSession) -> List[User]:
        """Récupère tous les utilisateurs"""
        result = await db.execute(select(User).order_by(User.created_at.desc()))
        return result.scalars().all()

    @staticmethod
    async def delete_user(db: AsyncSession, user_id: int) -> bool:
        """Supprime un utilisateur (et ses prédictions via cascade)"""
        user = await DBService.get_user_by_id(db, user_id)
        if not user:
            return False
        await db.delete(user)
        await db.commit()
        return True

    @staticmethod
    async def update_user(db: AsyncSession, user_id: int, **kwargs) -> Optional["User"]:
        """Met à jour les propriétés d'un utilisateur. Génère un nouveau token si regenerate_token=True."""
        user = await DBService.get_user_by_id(db, user_id)
        if not user:
            return None
        regenerate = kwargs.pop("regenerate_token", False)
        for field, value in kwargs.items():
            if value is not None and hasattr(user, field):
                setattr(user, field, value)
        if regenerate:
            user.api_token = secrets.token_urlsafe(32)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def update_user_last_login(db: AsyncSession, user_id: int):
        """Met à jour la dernière connexion d'un utilisateur"""
        user = await DBService.get_user_by_id(db, user_id)
        if user:
            user.last_login = _utcnow()
            await db.commit()

    @staticmethod
    async def get_user_prediction_count_today(db: AsyncSession, user_id: int) -> int:
        """Compte le nombre de prédictions d'un utilisateur aujourd'hui"""
        today = _utcnow().date()
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
        input_features: dict,
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
        model_name: str,
        start: datetime,
        end: datetime,
        model_version: Optional[str] = None,
        username: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[Prediction], int]:
        """
        Récupère l'historique des prédictions avec filtres.

        Returns:
            Tuple (liste de prédictions, total sans pagination)
        """
        filters = [
            Prediction.model_name == model_name,
            Prediction.timestamp >= start,
            Prediction.timestamp <= end,
        ]
        if model_version:
            filters.append(Prediction.model_version == model_version)
        if username:
            filters.append(User.username == username)

        base_query = (
            select(Prediction)
            .join(User, Prediction.user_id == User.id)
            .where(and_(*filters))
        )

        total_result = await db.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total = total_result.scalar() or 0

        result = await db.execute(
            base_query.order_by(Prediction.timestamp.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all(), total

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
            # Sans version explicite : priorité à is_production=True, sinon la plus récente
            query = query.order_by(
                ModelMetadata.is_production.desc(),
                ModelMetadata.created_at.desc()
            )

        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_active_models(db: AsyncSession) -> List[ModelMetadata]:
        """Récupère tous les modèles actifs avec leur créateur"""
        result = await db.execute(
            select(ModelMetadata)
            .options(selectinload(ModelMetadata.creator))
            .where(ModelMetadata.is_active == True)
        )
        return result.scalars().all()

    @staticmethod
    async def deactivate_model(db: AsyncSession, name: str, version: str) -> bool:
        """Désactive un modèle"""
        metadata = await DBService.get_model_metadata(db, name, version)
        if metadata:
            metadata.is_active = False
            metadata.deprecated_at = _utcnow()
            await db.commit()
            return True
        return False


    # === Observed Results ===

    @staticmethod
    async def upsert_observed_results(
        db: AsyncSession,
        records: list[dict],
    ) -> int:
        """
        Insère ou écrase des résultats observés.

        La clé d'unicité est (id_obs, model_name) — si la paire existe déjà,
        observed_result, date_time et user_id sont mis à jour.

        Args:
            records: liste de dicts avec les clés id_obs, model_name,
                     observed_result, date_time, user_id

        Returns:
            Nombre de lignes affectées
        """
        stmt = pg_insert(ObservedResult).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_observed_result_obs_model",
            set_={
                "observed_result": stmt.excluded.observed_result,
                "date_time": stmt.excluded.date_time,
                "user_id": stmt.excluded.user_id,
            },
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount

    @staticmethod
    async def get_observed_results(
        db: AsyncSession,
        model_name: Optional[str] = None,
        id_obs: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """
        Récupère les résultats observés avec filtres optionnels.

        Returns:
            Tuple (liste d'ObservedResult avec relation user chargée, total sans pagination)
        """
        filters = []
        if model_name:
            filters.append(ObservedResult.model_name == model_name)
        if id_obs:
            filters.append(ObservedResult.id_obs == id_obs)
        if start:
            filters.append(ObservedResult.date_time >= start)
        if end:
            filters.append(ObservedResult.date_time <= end)

        base_query = (
            select(ObservedResult)
            .options(selectinload(ObservedResult.user))
            .where(and_(*filters) if filters else True)
        )

        total_result = await db.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total = total_result.scalar() or 0

        result = await db.execute(
            base_query.order_by(ObservedResult.date_time.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all(), total


# Instance globale (optionnel)
db_service = DBService()
