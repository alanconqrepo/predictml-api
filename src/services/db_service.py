"""
Service pour les opérations de base de données
"""

import secrets
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, List, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.utils import _utcnow
from src.db.models import (
    HistoryActionType,
    ModelHistory,
    ModelMetadata,
    ObservedResult,
    Prediction,
    User,
)


class DBService:
    """Service pour les opérations CRUD de la base de données"""

    # === Users ===

    @staticmethod
    async def get_user_by_token(db: AsyncSession, api_token: str) -> Optional[User]:
        """Récupère un utilisateur par son token"""
        result = await db.execute(
            select(User).where(and_(User.api_token == api_token, User.is_active.is_(True)))
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
        rate_limit: int = 1000,
    ) -> User:
        """Crée un nouvel utilisateur"""
        user = User(
            username=username,
            email=email,
            api_token=api_token,
            role=role,
            rate_limit_per_day=rate_limit,
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
    async def update_user_last_login(db: AsyncSession, user_id: int) -> None:
        """Met à jour la dernière connexion d'un utilisateur."""
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
                and_(Prediction.user_id == user_id, func.date(Prediction.timestamp) == today)
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
        prediction_result: Any,
        probabilities: Optional[list],
        response_time_ms: float,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        id_obs: Optional[str] = None,
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
            error_message=error_message,
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
        id_obs: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[int] = None,
    ) -> tuple[List[Prediction], int]:
        """
        Récupère l'historique des prédictions avec filtres (pagination par curseur).

        Le curseur est l'id de la dernière prédiction vue. La prochaine page retourne
        les prédictions avec un id strictement inférieur au curseur.

        Returns:
            Tuple (liste de prédictions — limit+1 pour détecter la page suivante, total)
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
        if id_obs:
            filters.append(Prediction.id_obs == id_obs)
        if cursor is not None:
            filters.append(Prediction.id < cursor)

        base_query = (
            select(Prediction).join(User, Prediction.user_id == User.id).where(and_(*filters))
        )

        total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
        total = total_result.scalar() or 0

        result = await db.execute(base_query.order_by(Prediction.id.desc()).limit(limit + 1))
        return result.scalars().all(), total

    @staticmethod
    async def count_predictions(
        db: AsyncSession,
        model_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        model_version: Optional[str] = None,
    ) -> int:
        """Compte les prédictions réussies pour un modèle sur une période."""
        filters = [
            Prediction.model_name == model_name,
            Prediction.status == "success",
        ]
        if start:
            filters.append(Prediction.timestamp >= start)
        if end:
            filters.append(Prediction.timestamp <= end)
        if model_version:
            filters.append(Prediction.model_version == model_version)

        result = await db.execute(select(func.count(Prediction.id)).where(and_(*filters)))
        return result.scalar() or 0

    @staticmethod
    async def get_performance_pairs(
        db: AsyncSession,
        model_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        model_version: Optional[str] = None,
    ) -> list:
        """
        Retourne les paires (prediction_result, observed_result, probabilities, timestamp)
        pour les prédictions ayant un observed_result associé.
        JOIN sur (id_obs, model_name), uniquement les prédictions status='success'.
        """
        filters = [
            Prediction.model_name == model_name,
            Prediction.status == "success",
            Prediction.id_obs.isnot(None),
        ]
        if start:
            filters.append(Prediction.timestamp >= start)
        if end:
            filters.append(Prediction.timestamp <= end)
        if model_version:
            filters.append(Prediction.model_version == model_version)

        stmt = (
            select(
                Prediction.prediction_result,
                ObservedResult.observed_result,
                Prediction.probabilities,
                Prediction.timestamp,
            )
            .join(
                ObservedResult,
                and_(
                    Prediction.id_obs == ObservedResult.id_obs,
                    Prediction.model_name == ObservedResult.model_name,
                ),
            )
            .where(and_(*filters))
            .order_by(Prediction.timestamp)
        )

        result = await db.execute(stmt)
        return result.all()

    @staticmethod
    async def get_accuracy_drift(
        db: AsyncSession,
        model_name: str,
        start: datetime,
        end: datetime,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Retourne l'accuracy journalière pour le suivi de drift de performance.
        Requête SQL avec fenêtre temporelle, agrégat par jour.
        Plus légère que get_performance_pairs() : ne sélectionne pas les probabilities.
        Retourne une liste de dicts {"date": "YYYY-MM-DD", "matched_count": int, "accuracy": float}.
        """
        filters = [
            Prediction.model_name == model_name,
            Prediction.status == "success",
            Prediction.id_obs.isnot(None),
            Prediction.timestamp >= start,
            Prediction.timestamp <= end,
        ]
        if model_version:
            filters.append(Prediction.model_version == model_version)

        stmt = (
            select(
                func.date(Prediction.timestamp).label("day"),
                Prediction.prediction_result,
                ObservedResult.observed_result,
            )
            .join(
                ObservedResult,
                and_(
                    Prediction.id_obs == ObservedResult.id_obs,
                    Prediction.model_name == ObservedResult.model_name,
                ),
            )
            .where(and_(*filters))
            .order_by(func.date(Prediction.timestamp))
        )

        result = await db.execute(stmt)
        rows = result.all()

        daily: dict[str, list] = {}
        for row in rows:
            day = str(row.day)
            daily.setdefault(day, []).append((str(row.prediction_result), str(row.observed_result)))

        return [
            {
                "date": day,
                "matched_count": len(items),
                "accuracy": round(sum(p == o for p, o in items) / len(items), 4),
            }
            for day, items in sorted(daily.items())
        ]

    @staticmethod
    async def get_feature_production_stats(
        db: AsyncSession,
        model_name: str,
        model_version: Optional[str],
        days: int = 7,
    ) -> dict:
        """
        Calcule les statistiques des features de production sur une fenêtre glissante.

        Retourne un dict {feature: {mean, std, min, max, count, values}} où
        `values` contient les valeurs brutes numériques (pour le calcul PSI).
        Seules les features numériques (int/float) sont incluses.
        """
        from datetime import timedelta

        import numpy as np

        cutoff = _utcnow() - timedelta(days=days)

        filters = [
            Prediction.model_name == model_name,
            Prediction.status == "success",
            Prediction.timestamp >= cutoff,
        ]
        if model_version:
            filters.append(Prediction.model_version == model_version)

        stmt = select(Prediction.input_features).where(and_(*filters))
        result = await db.execute(stmt)
        rows = result.scalars().all()

        feature_values: dict[str, list] = {}
        for input_features in rows:
            if not isinstance(input_features, dict):
                continue
            for feature, value in input_features.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    feature_values.setdefault(feature, []).append(float(value))

        stats: dict = {}
        for feature, values in feature_values.items():
            arr = np.array(values, dtype=float)
            stats[feature] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(arr),
                "values": values,
            }

        return stats

    @staticmethod
    async def get_prediction_stats(
        db: AsyncSession,
        days: int = 30,
        model_name: Optional[str] = None,
    ) -> List[dict]:
        """Retourne les statistiques agrégées des prédictions par modèle sur une fenêtre glissante.

        Calcul Python-side (compatible SQLite + PostgreSQL) :
        - total, erreurs, taux d'erreur
        - temps de réponse moyen, p50, p95 (uniquement pour les prédictions réussies)
        """
        cutoff = _utcnow() - timedelta(days=days)
        filters = [Prediction.timestamp >= cutoff]
        if model_name:
            filters.append(Prediction.model_name == model_name)

        stmt = select(
            Prediction.model_name,
            Prediction.status,
            Prediction.response_time_ms,
        ).where(and_(*filters))

        result = await db.execute(stmt)
        rows = result.all()

        grouped: dict = defaultdict(lambda: {"times": [], "errors": 0, "total": 0})
        for row in rows:
            g = grouped[row.model_name]
            g["total"] += 1
            if row.status != "success":
                g["errors"] += 1
            elif row.response_time_ms is not None:
                g["times"].append(row.response_time_ms)

        def _percentile(data: list, p: float) -> Optional[float]:
            if not data:
                return None
            idx = max(0, int(len(data) * p / 100) - 1)
            return round(data[idx], 2)

        stats = []
        for name, g in sorted(grouped.items()):
            total = g["total"]
            errors = g["errors"]
            times = sorted(g["times"])
            n = len(times)
            stats.append(
                {
                    "model_name": name,
                    "total_predictions": total,
                    "error_count": errors,
                    "error_rate": round(errors / total, 4) if total > 0 else 0.0,
                    "avg_response_time_ms": round(sum(times) / n, 2) if n > 0 else None,
                    "p50_response_time_ms": _percentile(times, 50),
                    "p95_response_time_ms": _percentile(times, 95),
                }
            )
        return stats

    # === Model Metadata ===

    @staticmethod
    async def create_model_metadata(
        db: AsyncSession,
        name: str,
        version: str,
        minio_bucket: str,
        minio_object_key: str,
        **kwargs,
    ) -> ModelMetadata:
        """Crée les métadonnées d'un modèle"""
        metadata = ModelMetadata(
            name=name,
            version=version,
            minio_bucket=minio_bucket,
            minio_object_key=minio_object_key,
            **kwargs,
        )
        db.add(metadata)
        await db.commit()
        await db.refresh(metadata)
        return metadata

    @staticmethod
    async def get_model_metadata(
        db: AsyncSession, name: str, version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Récupère les métadonnées d'un modèle"""
        query = select(ModelMetadata).where(
            and_(ModelMetadata.name == name, ModelMetadata.is_active.is_(True))
        )

        if version:
            query = query.where(ModelMetadata.version == version)
        else:
            # Sans version explicite : priorité à is_production=True, sinon la plus récente
            query = query.order_by(
                ModelMetadata.is_production.desc(), ModelMetadata.created_at.desc()
            )

        result = await db.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_active_models(db: AsyncSession) -> List[ModelMetadata]:
        """Récupère tous les modèles actifs avec leur créateur"""
        result = await db.execute(
            select(ModelMetadata)
            .options(selectinload(ModelMetadata.creator))
            .where(ModelMetadata.is_active.is_(True))
        )
        return result.scalars().all()

    @staticmethod
    async def get_models_last_seen(db: AsyncSession) -> dict[str, datetime]:
        """Retourne la date de dernière prédiction réussie par nom de modèle."""
        result = await db.execute(
            select(Prediction.model_name, func.max(Prediction.timestamp).label("last_seen"))
            .where(Prediction.status == "success")
            .group_by(Prediction.model_name)
        )
        return {row.model_name: row.last_seen for row in result.all()}

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

        total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
        total = total_result.scalar() or 0

        result = await db.execute(
            base_query.order_by(ObservedResult.date_time.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all(), total

    # === Model History ===

    @staticmethod
    async def log_model_history(
        db: AsyncSession,
        model: ModelMetadata,
        action: HistoryActionType,
        user_id: Optional[int],
        username: Optional[str],
        changed_fields: Optional[List[str]] = None,
    ) -> ModelHistory:
        """Crée une entrée d'historique avec un snapshot complet de l'état actuel du modèle.
        L'appelant est responsable du commit."""
        entry = ModelHistory(
            model_name=model.name,
            model_version=model.version,
            changed_by_user_id=user_id,
            changed_by_username=username,
            action=action,
            snapshot=_build_snapshot(model),
            changed_fields=changed_fields,
        )
        db.add(entry)
        return entry

    @staticmethod
    async def get_model_history(
        db: AsyncSession,
        model_name: str,
        model_version: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple:
        """Récupère l'historique d'un modèle (toutes versions ou une version spécifique).

        Returns:
            Tuple (liste d'entrées triées par timestamp DESC, total sans pagination)
        """
        filters = [ModelHistory.model_name == model_name]
        if model_version:
            filters.append(ModelHistory.model_version == model_version)

        base_query = select(ModelHistory).where(and_(*filters))

        total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
        total = total_result.scalar() or 0

        result = await db.execute(
            base_query.order_by(ModelHistory.timestamp.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all()), total

    @staticmethod
    async def get_history_entry_by_id(
        db: AsyncSession,
        history_id: int,
    ) -> Optional[ModelHistory]:
        """Récupère une entrée d'historique par son id."""
        result = await db.execute(select(ModelHistory).where(ModelHistory.id == history_id))
        return result.scalar_one_or_none()


# ===========================================================================
# Helpers snapshot (module-level, utilisables aussi depuis api/models.py)
# ===========================================================================

_SNAPSHOT_FIELDS = [
    "description",
    "algorithm",
    "features_count",
    "classes",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "training_metrics",
    "confidence_threshold",
    "trained_by",
    "training_date",
    "training_dataset",
    "training_params",
    "feature_baseline",
    "tags",
    "webhook_url",
    "is_production",
    "is_active",
    "deprecated_at",
]

# Champs restaurables lors d'un rollback (is_active et deprecated_at exclus :
# gérés explicitement par les opérations métier)
_ROLLBACK_FIELDS = [
    "description",
    "algorithm",
    "features_count",
    "classes",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "training_metrics",
    "confidence_threshold",
    "trained_by",
    "training_date",
    "training_dataset",
    "training_params",
    "feature_baseline",
    "tags",
    "webhook_url",
    "is_production",
]


def _build_snapshot(model: ModelMetadata) -> dict:
    """Construit un dict JSON-sérialisable des champs mutables d'un modèle."""
    result: dict = {}
    for field in _SNAPSHOT_FIELDS:
        value = getattr(model, field, None)
        if isinstance(value, datetime):
            value = value.isoformat()
        result[field] = value
    return result


# Instance globale (optionnel)
db_service = DBService()
