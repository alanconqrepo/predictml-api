"""
Service pour les opérations de base de données
"""

import secrets
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, List, Optional

from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.metrics import f1_score as sklearn_f1_score
from sqlalchemy import and_, delete, func, or_, select
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
    async def get_all_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
        """Récupère les utilisateurs avec pagination offset/limit"""
        result = await db.execute(
            select(User).order_by(User.created_at.desc()).offset(skip).limit(limit)
        )
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

    @staticmethod
    async def get_user_usage(db: AsyncSession, user_id: int, days: int = 30) -> dict:
        """Retourne les statistiques d'usage d'un utilisateur sur les N derniers jours.

        Agrège les prédictions par modèle (calls, errors, avg_latency_ms)
        et par jour (calls), calculées côté Python pour compatibilité SQLite/PostgreSQL.
        """
        cutoff = _utcnow() - timedelta(days=days)
        stmt = select(
            Prediction.model_name,
            Prediction.status,
            Prediction.response_time_ms,
            Prediction.timestamp,
        ).where(and_(Prediction.user_id == user_id, Prediction.timestamp >= cutoff))

        result = await db.execute(stmt)
        rows = result.all()

        by_model: dict = defaultdict(lambda: {"calls": 0, "errors": 0, "latencies": []})
        by_day: dict = defaultdict(int)

        for row in rows:
            by_model[row.model_name]["calls"] += 1
            if row.status != "success":
                by_model[row.model_name]["errors"] += 1
            elif row.response_time_ms is not None:
                by_model[row.model_name]["latencies"].append(row.response_time_ms)
            if row.timestamp is not None:
                day_key = row.timestamp.date()
                by_day[day_key] += 1

        model_stats = []
        for model_name, g in sorted(by_model.items()):
            lats = g["latencies"]
            model_stats.append(
                {
                    "model_name": model_name,
                    "calls": g["calls"],
                    "errors": g["errors"],
                    "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else None,
                }
            )

        day_stats = [{"date": d, "calls": c} for d, c in sorted(by_day.items())]

        return {
            "total_calls": len(rows),
            "by_model": model_stats,
            "by_day": day_stats,
        }

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
        is_shadow: bool = False,
        max_confidence: Optional[float] = None,
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
            is_shadow=is_shadow,
            max_confidence=max_confidence,
        )
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)
        return prediction

    @staticmethod
    async def get_prediction_by_id(
        db: AsyncSession,
        prediction_id: int,
    ) -> Optional[Prediction]:
        """Récupère une prédiction par son id, avec la relation user chargée."""
        result = await db.execute(
            select(Prediction)
            .options(selectinload(Prediction.user))
            .where(Prediction.id == prediction_id)
        )
        return result.scalar_one_or_none()

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
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
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
        if min_confidence is not None:
            filters.append(Prediction.max_confidence >= min_confidence)
        if max_confidence is not None:
            filters.append(Prediction.max_confidence <= max_confidence)

        base_query = (
            select(Prediction).join(User, Prediction.user_id == User.id).where(and_(*filters))
        )

        total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
        total = total_result.scalar() or 0

        result = await db.execute(base_query.order_by(Prediction.id.desc()).limit(limit + 1))
        return result.scalars().all(), total

    @staticmethod
    async def get_predictions_for_export(
        db: AsyncSession,
        model_name: Optional[str],
        start: datetime,
        end: datetime,
        status_filter: Optional[str] = None,
        limit: int = 500,
        cursor: Optional[int] = None,
    ) -> List[Prediction]:
        """
        Récupère une page de prédictions pour l'export streaming (cursor keyset DESC).
        Contrairement à get_predictions, model_name est optionnel et le user est eager-loaded.
        """
        filters = [
            Prediction.timestamp >= start,
            Prediction.timestamp <= end,
        ]
        if model_name:
            filters.append(Prediction.model_name == model_name)
        if status_filter:
            filters.append(Prediction.status == status_filter)
        if cursor is not None:
            filters.append(Prediction.id < cursor)

        query = (
            select(Prediction)
            .where(and_(*filters))
            .options(selectinload(Prediction.user))
            .order_by(Prediction.id.desc())
            .limit(limit)
        )

        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_predictions_with_features(
        db: AsyncSession,
        model_name: str,
        days: int = 7,
        limit: int = 200,
    ) -> List[Prediction]:
        """
        Récupère les N dernières prédictions réussies d'un modèle sur une fenêtre glissante,
        avec leurs input_features (pour le calcul de z-scores par feature).
        """
        cutoff = _utcnow() - timedelta(days=days)

        stmt = (
            select(Prediction)
            .where(
                and_(
                    Prediction.model_name == model_name,
                    Prediction.status == "success",
                    Prediction.timestamp >= cutoff,
                    Prediction.input_features.isnot(None),
                )
            )
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

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
    async def get_performance_timeline(
        db: AsyncSession,
        model_name: str,
    ) -> List[dict]:
        """
        Retourne l'évolution chronologique des métriques par version pour un modèle.
        Pour chaque version active, calcule accuracy/F1 (classification) ou MAE (régression)
        via les paires (prediction, observed_result). Ordonnées par created_at ASC.
        """
        stmt = (
            select(ModelMetadata)
            .where(and_(ModelMetadata.name == model_name, ModelMetadata.is_active.is_(True)))
            .order_by(ModelMetadata.created_at.asc())
        )
        result = await db.execute(stmt)
        versions = result.scalars().all()

        timeline = []
        for meta in versions:
            pairs = await DBService.get_performance_pairs(
                db, model_name, model_version=meta.version
            )
            sample_count = len(pairs)

            accuracy = None
            mae = None
            f1 = None

            if sample_count > 0:
                y_true = [row.observed_result for row in pairs]
                y_pred = [row.prediction_result for row in pairs]

                is_classification = bool(meta.classes)
                if not is_classification:
                    if any(row.probabilities for row in pairs):
                        is_classification = True
                    elif y_pred and all(
                        isinstance(v, (int, str, bool)) for v in y_pred if v is not None
                    ):
                        is_classification = True

                if is_classification:
                    y_true_s = [str(v) for v in y_true]
                    y_pred_s = [str(v) for v in y_pred]
                    accuracy = round(accuracy_score(y_true_s, y_pred_s), 4)
                    f1 = round(
                        float(
                            sklearn_f1_score(
                                y_true_s, y_pred_s, average="weighted", zero_division=0
                            )
                        ),
                        4,
                    )
                else:
                    y_true_f = [float(v) for v in y_true]
                    y_pred_f = [float(v) for v in y_pred]
                    mae = round(float(mean_absolute_error(y_true_f, y_pred_f)), 4)

            training_stats = meta.training_stats or {}
            trained_at = None
            raw_trained_at = training_stats.get("trained_at")
            if raw_trained_at:
                try:
                    trained_at = datetime.fromisoformat(raw_trained_at)
                except (ValueError, TypeError):
                    trained_at = None

            timeline.append(
                {
                    "version": meta.version,
                    "deployed_at": meta.created_at,
                    "accuracy": accuracy,
                    "mae": mae,
                    "f1_score": f1,
                    "sample_count": sample_count,
                    "trained_at": trained_at,
                    "n_rows_trained": training_stats.get("n_rows"),
                }
            )

        return timeline

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

        def _is_float_val(v: str) -> bool:
            try:
                f = float(v)
                return f != int(f)
            except (ValueError, TypeError):
                return False

        result = []
        for day, items in sorted(daily.items()):
            is_regression = any(_is_float_val(p) or _is_float_val(o) for p, o in items)
            entry: dict = {
                "date": day,
                "matched_count": len(items),
                "accuracy": round(sum(p == o for p, o in items) / len(items), 4),
            }
            if is_regression:
                try:
                    entry["mae"] = round(
                        sum(abs(float(p) - float(o)) for p, o in items) / len(items), 4
                    )
                except (ValueError, TypeError):
                    entry["mae"] = None
            else:
                entry["mae"] = None
            result.append(entry)
        return result

    @staticmethod
    async def get_confidence_trend(
        db: AsyncSession,
        model_name: str,
        version: Optional[str],
        days: int,
        confidence_threshold: float = 0.5,
    ) -> dict:
        """
        Agrège la confiance (max des probabilités) par jour sur une fenêtre glissante.
        Retourne overall stats + liste journalière triée.
        Compatible SQLite (tests) et PostgreSQL (production).
        """
        import numpy as np

        cutoff = _utcnow() - timedelta(days=days)

        filters = [
            Prediction.model_name == model_name,
            Prediction.status == "success",
            Prediction.is_shadow.is_(False),
            Prediction.probabilities.isnot(None),
            Prediction.timestamp >= cutoff,
        ]
        if version:
            filters.append(Prediction.model_version == version)

        stmt = select(
            func.date(Prediction.timestamp).label("day"),
            Prediction.probabilities,
        ).where(and_(*filters))

        result = await db.execute(stmt)
        rows = result.all()

        daily: dict[str, list[float]] = {}
        for row in rows:
            probs = row.probabilities
            if isinstance(probs, dict):
                confidence = max(probs.values()) if probs else None
            elif isinstance(probs, list):
                confidence = max(probs) if probs else None
            else:
                continue
            if confidence is not None:
                daily.setdefault(str(row.day), []).append(float(confidence))

        all_confidences: list[float] = []
        trend = []
        for day in sorted(daily):
            values = daily[day]
            arr = np.array(values)
            all_confidences.extend(values)
            trend.append(
                {
                    "date": day,
                    "mean_confidence": round(float(np.mean(arr)), 4),
                    "p25": round(float(np.percentile(arr, 25)), 4),
                    "p75": round(float(np.percentile(arr, 75)), 4),
                    "predictions": len(values),
                    "low_confidence_count": int(np.sum(arr < confidence_threshold)),
                }
            )

        if not all_confidences:
            return {"has_data": False, "overall": None, "trend": []}

        all_arr = np.array(all_confidences)
        low_count = int(np.sum(all_arr < confidence_threshold))
        overall = {
            "mean_confidence": round(float(np.mean(all_arr)), 4),
            "p25_confidence": round(float(np.percentile(all_arr, 25)), 4),
            "p75_confidence": round(float(np.percentile(all_arr, 75)), 4),
            "low_confidence_rate": round(low_count / len(all_confidences), 4),
        }
        return {"has_data": True, "overall": overall, "trend": trend}

    @staticmethod
    async def get_confidence_distribution(
        db: AsyncSession,
        model_name: str,
        version: Optional[str],
        days: int,
        high_threshold: float = 0.80,
        uncertain_threshold: float = 0.60,
    ) -> dict:
        import numpy as np

        cutoff = _utcnow() - timedelta(days=days)

        filters = [
            Prediction.model_name == model_name,
            Prediction.status == "success",
            Prediction.is_shadow.is_(False),
            Prediction.probabilities.isnot(None),
            Prediction.timestamp >= cutoff,
        ]
        if version:
            filters.append(Prediction.model_version == version)

        stmt = select(Prediction.probabilities).where(and_(*filters))
        result = await db.execute(stmt)
        rows = result.scalars().all()

        confidences = []
        for probs in rows:
            if isinstance(probs, dict):
                c = max(probs.values()) if probs else None
            elif isinstance(probs, list):
                c = max(probs) if probs else None
            else:
                continue
            if c is not None:
                confidences.append(float(c))

        if not confidences:
            return {"has_data": False}

        arr = np.array(confidences)
        total = len(confidences)
        counts, edges = np.histogram(arr, bins=10, range=(0.5, 1.0))

        histogram = [
            {
                "bin_min": round(float(edges[i]), 2),
                "bin_max": round(float(edges[i + 1]), 2),
                "count": int(counts[i]),
                "pct": round(float(counts[i]) / total, 4),
            }
            for i in range(len(counts))
        ]

        return {
            "has_data": True,
            "sample_count": total,
            "mean_confidence": round(float(np.mean(arr)), 4),
            "pct_high_confidence": round(float(np.sum(arr > high_threshold)) / total, 4),
            "pct_uncertain": round(float(np.sum(arr < uncertain_threshold)) / total, 4),
            "histogram": histogram,
        }

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

        total_rows = 0
        feature_values: dict[str, list] = {}
        for input_features in rows:
            if not isinstance(input_features, dict):
                continue
            total_rows += 1
            for feature, value in input_features.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    feature_values.setdefault(feature, []).append(float(value))

        stats: dict = {}
        for feature, values in feature_values.items():
            arr = np.array(values, dtype=float)
            count = len(arr)
            null_rate = round(1.0 - count / total_rows, 6) if total_rows > 0 else 0.0
            stats[feature] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": count,
                "values": values,
                "null_rate": null_rate,
            }

        return stats

    @staticmethod
    async def get_prediction_label_distribution(
        db: AsyncSession,
        model_name: str,
        model_version: Optional[str] = None,
        days: int = 7,
    ) -> tuple[dict[str, int], int]:
        """Returns ({label: count}, total) for successful predictions in the last N days."""
        cutoff = _utcnow() - timedelta(days=days)
        filters = [
            Prediction.model_name == model_name,
            Prediction.status == "success",
            Prediction.timestamp >= cutoff,
            Prediction.prediction_result.is_not(None),
        ]
        if model_version:
            filters.append(Prediction.model_version == model_version)

        stmt = select(Prediction.prediction_result).where(and_(*filters))
        result = await db.execute(stmt)
        rows = result.scalars().all()

        counts: dict[str, int] = {}
        for val in rows:
            key = str(val)
            counts[key] = counts.get(key, 0) + 1

        return counts, len(rows)

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

    @staticmethod
    async def purge_predictions(
        db: AsyncSession,
        older_than_days: int,
        model_name: Optional[str] = None,
        dry_run: bool = True,
    ) -> dict:
        """
        Purge les prédictions plus anciennes que N jours (rétention RGPD).

        En mode dry_run=True : comptage sans suppression.
        En mode dry_run=False : suppression effective et commit.

        Returns:
            dict avec dry_run, deleted_count, oldest_remaining,
            models_affected et linked_observed_results_count.
        """
        cutoff = _utcnow() - timedelta(days=older_than_days)

        filters = [Prediction.timestamp < cutoff]
        if model_name:
            filters.append(Prediction.model_name == model_name)

        # Count predictions to purge
        count_result = await db.execute(select(func.count(Prediction.id)).where(and_(*filters)))
        deleted_count = count_result.scalar() or 0

        # Distinct model names affected
        models_result = await db.execute(
            select(Prediction.model_name).where(and_(*filters)).distinct()
        )
        models_affected = sorted(row[0] for row in models_result.all())

        # Count predictions linked to observed_results (performance data loss warning)
        linked_result = await db.execute(
            select(func.count(Prediction.id))
            .join(
                ObservedResult,
                and_(
                    Prediction.id_obs == ObservedResult.id_obs,
                    Prediction.model_name == ObservedResult.model_name,
                ),
            )
            .where(and_(*filters))
        )
        linked_count = linked_result.scalar() or 0

        # Oldest prediction that will remain after the purge
        remaining_filters: list = [Prediction.timestamp >= cutoff]
        if model_name:
            remaining_filters.append(Prediction.model_name == model_name)

        oldest_result = await db.execute(
            select(func.min(Prediction.timestamp)).where(and_(*remaining_filters))
        )
        oldest_remaining = oldest_result.scalar()

        if not dry_run and deleted_count > 0:
            await db.execute(delete(Prediction).where(and_(*filters)))
            await db.commit()

        return {
            "dry_run": dry_run,
            "deleted_count": deleted_count,
            "oldest_remaining": oldest_remaining,
            "models_affected": models_affected,
            "linked_observed_results_count": linked_count,
        }

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
            result = await db.execute(query)
            return result.scalar_one_or_none()
        else:
            # Sans version explicite : priorité à is_production=True, sinon la plus récente
            # Exclure les versions dépréciées pour éviter de les sélectionner en routage
            query = query.where(ModelMetadata.status != "deprecated").order_by(
                ModelMetadata.is_production.desc(), ModelMetadata.created_at.desc()
            )
            result = await db.execute(query)
            return result.scalars().first()

    @staticmethod
    async def get_all_active_models(
        db: AsyncSession,
        is_production: Optional[bool] = None,
        algorithm: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        deployment_mode: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """Récupère tous les modèles actifs avec leur créateur"""
        query = (
            select(ModelMetadata)
            .options(selectinload(ModelMetadata.creator))
            .where(ModelMetadata.is_active.is_(True))
            .where(ModelMetadata.status != "archived")
        )
        if is_production is not None:
            query = query.where(ModelMetadata.is_production == is_production)
        if algorithm:
            query = query.where(ModelMetadata.algorithm == algorithm)
        if min_accuracy is not None:
            query = query.where(ModelMetadata.accuracy >= min_accuracy)
        if deployment_mode:
            query = query.where(ModelMetadata.deployment_mode == deployment_mode)
        if search:
            pattern = f"%{search}%"
            query = query.where(
                or_(
                    ModelMetadata.name.ilike(pattern),
                    ModelMetadata.description.ilike(pattern),
                )
            )
        result = await db.execute(query)
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

    @staticmethod
    async def deprecate_model(db: AsyncSession, name: str, version: str) -> Optional[ModelMetadata]:
        """Marque un modèle comme déprécié (status=deprecated, is_production=False)."""
        result = await db.execute(
            select(ModelMetadata).where(
                and_(
                    ModelMetadata.name == name,
                    ModelMetadata.version == version,
                    ModelMetadata.is_active.is_(True),
                )
            )
        )
        metadata = result.scalar_one_or_none()
        if not metadata:
            return None
        metadata.status = "deprecated"
        metadata.is_production = False
        metadata.deprecated_at = _utcnow()
        await db.commit()
        await db.refresh(metadata)
        return metadata

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

    @staticmethod
    async def get_observed_results_for_export(
        db: AsyncSession,
        model_name: Optional[str],
        start: datetime,
        end: datetime,
        limit: int = 500,
        cursor: Optional[int] = None,
    ) -> List[ObservedResult]:
        """
        Récupère une page de résultats observés pour l'export streaming (cursor keyset DESC).
        """
        filters = [
            ObservedResult.date_time >= start,
            ObservedResult.date_time <= end,
        ]
        if model_name:
            filters.append(ObservedResult.model_name == model_name)
        if cursor is not None:
            filters.append(ObservedResult.id < cursor)

        query = (
            select(ObservedResult)
            .where(and_(*filters))
            .options(selectinload(ObservedResult.user))
            .order_by(ObservedResult.id.desc())
            .limit(limit)
        )

        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_observed_results_stats(
        db: AsyncSession,
        model_name: Optional[str] = None,
    ) -> dict:
        """Taux de couverture du ground truth : combien de prédictions ont un résultat observé.

        Compatible SQLite (tests) et PostgreSQL (production).
        Retourne total_predictions, labeled_count, coverage_rate, oldest/newest label,
        et un breakdown by_version (si model_name fourni) ou by_model (si global).
        """
        pred_filters = []
        if model_name:
            pred_filters.append(Prediction.model_name == model_name)

        where_pred = and_(*pred_filters) if pred_filters else True

        total_predictions = (
            await db.execute(select(func.count(Prediction.id)).where(where_pred))
        ).scalar() or 0

        # Predictions joined with observed_results on (id_obs, model_name)
        labeled_base = (
            select(func.count(Prediction.id))
            .join(
                ObservedResult,
                and_(
                    Prediction.id_obs == ObservedResult.id_obs,
                    Prediction.model_name == ObservedResult.model_name,
                ),
            )
            .where(where_pred)
        )
        labeled_count = (await db.execute(labeled_base)).scalar() or 0

        obs_filters = []
        if model_name:
            obs_filters.append(ObservedResult.model_name == model_name)
        where_obs = and_(*obs_filters) if obs_filters else True

        dates_row = (
            await db.execute(
                select(
                    func.min(ObservedResult.date_time),
                    func.max(ObservedResult.date_time),
                ).where(where_obs)
            )
        ).one()
        oldest_label = dates_row[0]
        newest_label = dates_row[1]

        coverage_rate = (
            round(labeled_count / total_predictions, 3) if total_predictions > 0 else 0.0
        )

        if model_name:
            pred_by_version = (
                await db.execute(
                    select(Prediction.model_version, func.count(Prediction.id).label("cnt"))
                    .where(Prediction.model_name == model_name)
                    .group_by(Prediction.model_version)
                )
            ).all()

            labeled_by_version_rows = (
                await db.execute(
                    select(Prediction.model_version, func.count(Prediction.id).label("cnt"))
                    .join(
                        ObservedResult,
                        and_(
                            Prediction.id_obs == ObservedResult.id_obs,
                            Prediction.model_name == ObservedResult.model_name,
                        ),
                    )
                    .where(Prediction.model_name == model_name)
                    .group_by(Prediction.model_version)
                )
            ).all()
            labeled_map = {r.model_version: r.cnt for r in labeled_by_version_rows}

            by_version = sorted(
                [
                    {
                        "version": r.model_version or "unknown",
                        "predictions": r.cnt,
                        "labeled": labeled_map.get(r.model_version, 0),
                        "coverage": (
                            round(labeled_map.get(r.model_version, 0) / r.cnt, 3)
                            if r.cnt > 0
                            else 0.0
                        ),
                    }
                    for r in pred_by_version
                ],
                key=lambda x: x["version"],
                reverse=True,
            )

            return {
                "model_name": model_name,
                "total_predictions": total_predictions,
                "labeled_count": labeled_count,
                "coverage_rate": coverage_rate,
                "oldest_label": oldest_label,
                "newest_label": newest_label,
                "by_version": by_version,
                "by_model": None,
            }
        else:
            pred_by_model = (
                await db.execute(
                    select(Prediction.model_name, func.count(Prediction.id).label("cnt")).group_by(
                        Prediction.model_name
                    )
                )
            ).all()

            labeled_by_model_rows = (
                await db.execute(
                    select(Prediction.model_name, func.count(Prediction.id).label("cnt"))
                    .join(
                        ObservedResult,
                        and_(
                            Prediction.id_obs == ObservedResult.id_obs,
                            Prediction.model_name == ObservedResult.model_name,
                        ),
                    )
                    .group_by(Prediction.model_name)
                )
            ).all()
            labeled_map_model = {r.model_name: r.cnt for r in labeled_by_model_rows}

            by_model = sorted(
                [
                    {
                        "model_name": r.model_name,
                        "predictions": r.cnt,
                        "labeled": labeled_map_model.get(r.model_name, 0),
                        "coverage": (
                            round(labeled_map_model.get(r.model_name, 0) / r.cnt, 3)
                            if r.cnt > 0
                            else 0.0
                        ),
                    }
                    for r in pred_by_model
                ],
                key=lambda x: x["model_name"] or "",
            )

            return {
                "model_name": None,
                "total_predictions": total_predictions,
                "labeled_count": labeled_count,
                "coverage_rate": coverage_rate,
                "oldest_label": oldest_label,
                "newest_label": newest_label,
                "by_version": None,
                "by_model": by_model,
            }

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
    async def get_retrain_history(
        db: AsyncSession,
        model_name: str,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple:
        """Retourne l'historique des ré-entraînements d'un modèle.

        Une entrée correspond à toute version dont parent_version IS NOT NULL,
        triée par created_at DESC (retrain le plus récent en premier).

        Returns:
            Tuple (liste de ModelMetadata, total sans pagination)
        """
        filters = [
            ModelMetadata.name == model_name,
            ModelMetadata.parent_version.isnot(None),
        ]

        base_query = select(ModelMetadata).where(and_(*filters))

        total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
        total = total_result.scalar() or 0

        result = await db.execute(
            base_query.order_by(ModelMetadata.created_at.desc()).limit(limit).offset(offset)
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

    # === A/B Testing & Shadow Deployment ===

    @staticmethod
    async def get_ab_comparison_stats(
        db: AsyncSession,
        model_name: str,
        days: int = 30,
    ) -> list[dict]:
        """
        Retourne les statistiques par version pour une comparaison A/B.

        Group by (model_version, is_shadow) — agrégation Python-side pour compatibilité SQLite.
        Retourne une liste de dicts, un par version active dans la fenêtre.
        """
        cutoff = _utcnow() - timedelta(days=days)

        stmt = select(
            Prediction.model_version,
            Prediction.is_shadow,
            Prediction.status,
            Prediction.response_time_ms,
            Prediction.prediction_result,
        ).where(
            and_(
                Prediction.model_name == model_name,
                Prediction.timestamp >= cutoff,
            )
        )

        result = await db.execute(stmt)
        rows = result.all()

        # group: version -> {shadow: bool -> {times, errors, total, dist}}
        grouped: dict = defaultdict(
            lambda: {
                True: {"times": [], "errors": 0, "total": 0, "dist": defaultdict(int)},
                False: {"times": [], "errors": 0, "total": 0, "dist": defaultdict(int)},
            }
        )

        for row in rows:
            ver = row.model_version or "unknown"
            shadow = bool(row.is_shadow)
            g = grouped[ver][shadow]
            g["total"] += 1
            if row.status != "success":
                g["errors"] += 1
            else:
                if row.response_time_ms is not None:
                    g["times"].append(row.response_time_ms)
                if not shadow and row.prediction_result is not None:
                    g["dist"][str(row.prediction_result)] += 1

        def _p95(data: list) -> Optional[float]:
            if not data:
                return None
            s = sorted(data)
            idx = max(0, int(len(s) * 0.95) - 1)
            return round(s[idx], 2)

        stats = []
        for ver, shadow_map in sorted(grouped.items()):
            prod = shadow_map[False]
            shad = shadow_map[True]
            total_prod = prod["total"]
            times = prod["times"]
            n = len(times)
            stats.append(
                {
                    "version": ver,
                    "total_predictions": total_prod,
                    "shadow_predictions": shad["total"],
                    "error_rate": round(prod["errors"] / total_prod, 4) if total_prod > 0 else 0.0,
                    "avg_response_time_ms": round(sum(times) / n, 2) if n > 0 else None,
                    "p95_response_time_ms": _p95(times),
                    "prediction_distribution": dict(prod["dist"]),
                    "response_times": times,
                    "error_count": prod["errors"],
                }
            )
        return stats

    @staticmethod
    async def get_shadow_agreement_rate(
        db: AsyncSession,
        model_name: str,
        days: int = 30,
    ) -> dict[str, float]:
        """
        Pour chaque version shadow, calcule le taux de concordance des prédictions
        shadow vs production sur les mêmes id_obs.

        Retourne {shadow_version: agreement_rate} — dict vide si aucun id_obs renseigné.
        """
        from sqlalchemy.orm import aliased

        cutoff = _utcnow() - timedelta(days=days)

        prod_p = aliased(Prediction)
        shadow_p = aliased(Prediction)

        stmt = (
            select(
                shadow_p.model_version.label("shadow_version"),
                shadow_p.prediction_result.label("shadow_pred"),
                prod_p.prediction_result.label("prod_pred"),
            )
            .join(
                prod_p,
                and_(
                    shadow_p.id_obs == prod_p.id_obs,
                    shadow_p.model_name == prod_p.model_name,
                    prod_p.is_shadow.is_(False),
                    prod_p.status == "success",
                ),
            )
            .where(
                and_(
                    shadow_p.model_name == model_name,
                    shadow_p.is_shadow.is_(True),
                    shadow_p.status == "success",
                    shadow_p.timestamp >= cutoff,
                    shadow_p.id_obs.isnot(None),
                )
            )
        )

        result = await db.execute(stmt)
        rows = result.all()

        per_version: dict[str, list] = defaultdict(list)
        for row in rows:
            match = str(row.shadow_pred) == str(row.prod_pred)
            per_version[row.shadow_version].append(match)

        return {
            ver: round(sum(matches) / len(matches), 4)
            for ver, matches in per_version.items()
            if matches
        }

    @staticmethod
    async def get_shadow_comparison_stats(
        db: AsyncSession,
        model_name: str,
        period_days: int = 30,
    ) -> dict:
        """
        Calcule les métriques comparatives entre la version shadow et la version production
        sur les paires id_obs communs dans la fenêtre glissante.

        Retourne un dict avec : shadow_version, production_version, n_comparable,
        agreement_rate, shadow_confidence_delta, shadow_latency_delta_ms,
        shadow_accuracy, production_accuracy, accuracy_available.
        """
        from sqlalchemy.orm import aliased

        cutoff = _utcnow() - timedelta(days=period_days)

        meta_result = await db.execute(
            select(ModelMetadata).where(
                and_(ModelMetadata.name == model_name, ModelMetadata.is_active.is_(True))
            )
        )
        metas = meta_result.scalars().all()

        shadow_meta = next((m for m in metas if m.deployment_mode == "shadow"), None)
        prod_meta = next((m for m in metas if m.is_production), None)

        base = {
            "shadow_version": shadow_meta.version if shadow_meta else None,
            "production_version": prod_meta.version if prod_meta else None,
            "n_comparable": 0,
            "agreement_rate": None,
            "shadow_confidence_delta": None,
            "shadow_latency_delta_ms": None,
            "shadow_accuracy": None,
            "production_accuracy": None,
            "accuracy_available": False,
        }

        if not shadow_meta or not prod_meta:
            return base

        prod_p = aliased(Prediction)
        shadow_p = aliased(Prediction)

        stmt = (
            select(
                shadow_p.prediction_result.label("shadow_pred"),
                prod_p.prediction_result.label("prod_pred"),
                shadow_p.max_confidence.label("shadow_conf"),
                prod_p.max_confidence.label("prod_conf"),
                shadow_p.response_time_ms.label("shadow_rt"),
                prod_p.response_time_ms.label("prod_rt"),
                shadow_p.id_obs.label("id_obs"),
            )
            .join(
                prod_p,
                and_(
                    shadow_p.id_obs == prod_p.id_obs,
                    shadow_p.model_name == prod_p.model_name,
                    prod_p.is_shadow.is_(False),
                    prod_p.status == "success",
                    prod_p.model_version == prod_meta.version,
                ),
            )
            .where(
                and_(
                    shadow_p.model_name == model_name,
                    shadow_p.is_shadow.is_(True),
                    shadow_p.status == "success",
                    shadow_p.model_version == shadow_meta.version,
                    shadow_p.timestamp >= cutoff,
                    shadow_p.id_obs.isnot(None),
                )
            )
        )

        result = await db.execute(stmt)
        rows = result.all()
        n_comparable = len(rows)
        base["n_comparable"] = n_comparable

        if n_comparable == 0:
            return base

        matches = [str(r.shadow_pred) == str(r.prod_pred) for r in rows]
        base["agreement_rate"] = round(sum(matches) / n_comparable, 4)

        conf_pairs = [
            (r.shadow_conf, r.prod_conf)
            for r in rows
            if r.shadow_conf is not None and r.prod_conf is not None
        ]
        if conf_pairs:
            base["shadow_confidence_delta"] = round(
                sum(s - p for s, p in conf_pairs) / len(conf_pairs), 4
            )

        rt_pairs = [
            (r.shadow_rt, r.prod_rt)
            for r in rows
            if r.shadow_rt is not None and r.prod_rt is not None
        ]
        if rt_pairs:
            base["shadow_latency_delta_ms"] = round(
                sum(s - p for s, p in rt_pairs) / len(rt_pairs), 2
            )

        observed_ids = list({r.id_obs for r in rows})
        obs_result = await db.execute(
            select(ObservedResult.id_obs, ObservedResult.observed_result).where(
                and_(
                    ObservedResult.model_name == model_name,
                    ObservedResult.id_obs.in_(observed_ids),
                )
            )
        )
        obs_by_id = {r.id_obs: r.observed_result for r in obs_result.all()}

        base["accuracy_available"] = bool(obs_by_id)

        if obs_by_id:
            shadow_correct = prod_correct = count_with_obs = 0
            for r in rows:
                obs = obs_by_id.get(r.id_obs)
                if obs is None:
                    continue
                count_with_obs += 1
                if str(r.shadow_pred) == str(obs):
                    shadow_correct += 1
                if str(r.prod_pred) == str(obs):
                    prod_correct += 1
            if count_with_obs > 0:
                base["shadow_accuracy"] = round(shadow_correct / count_with_obs, 4)
                base["production_accuracy"] = round(prod_correct / count_with_obs, 4)

        return base

    # === Monitoring / Supervision Dashboard ===

    @staticmethod
    async def get_global_monitoring_stats(
        db: AsyncSession,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Retourne les statistiques agrégées par model_name sur une plage calendaire.

        Aggrégation Python-side (compatible SQLite + PostgreSQL, percentiles inclus).
        Retourne une liste de dicts, un par model_name distinct dans la période.
        """
        stmt = select(
            Prediction.model_name,
            Prediction.model_version,
            Prediction.is_shadow,
            Prediction.status,
            Prediction.response_time_ms,
            Prediction.timestamp,
        ).where(
            and_(
                Prediction.timestamp >= start,
                Prediction.timestamp <= end,
            )
        )

        result = await db.execute(stmt)
        rows = result.all()

        # groupe: model_name -> {rows}
        grouped: dict[str, dict] = defaultdict(
            lambda: {
                "versions": set(),
                "total": 0,
                "shadow": 0,
                "errors": 0,
                "times": [],
                "last_ts": None,
            }
        )

        for row in rows:
            g = grouped[row.model_name]
            if row.model_version:
                g["versions"].add(row.model_version)
            if row.is_shadow:
                g["shadow"] += 1
            else:
                g["total"] += 1
                if row.status != "success":
                    g["errors"] += 1
                elif row.response_time_ms is not None:
                    g["times"].append(row.response_time_ms)
            if g["last_ts"] is None or row.timestamp > g["last_ts"]:
                g["last_ts"] = row.timestamp

        def _percentile(data: list, p: float) -> Optional[float]:
            if not data:
                return None
            s = sorted(data)
            idx = max(0, int(len(s) * p) - 1)
            return round(s[idx], 2)

        stats = []
        for model_name, g in sorted(grouped.items()):
            total = g["total"]
            times = g["times"]
            n = len(times)
            stats.append(
                {
                    "model_name": model_name,
                    "versions": sorted(g["versions"]),
                    "total_predictions": total,
                    "shadow_predictions": g["shadow"],
                    "error_count": g["errors"],
                    "error_rate": round(g["errors"] / total, 4) if total > 0 else 0.0,
                    "avg_latency_ms": round(sum(times) / n, 2) if n > 0 else None,
                    "p50_latency_ms": _percentile(times, 0.50),
                    "p95_latency_ms": _percentile(times, 0.95),
                    "last_prediction": g["last_ts"],
                }
            )
        return stats

    @staticmethod
    async def get_model_predictions_timeseries(
        db: AsyncSession,
        model_name: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Retourne l'évolution quotidienne des prédictions (non-shadow) pour un modèle.

        Retourne une liste de dicts triée par date asc :
        [{date, total_predictions, error_count, error_rate, avg/p50/p95_latency_ms}]
        """
        stmt = select(
            Prediction.status,
            Prediction.response_time_ms,
            Prediction.timestamp,
        ).where(
            and_(
                Prediction.model_name == model_name,
                Prediction.is_shadow.is_(False),
                Prediction.timestamp >= start,
                Prediction.timestamp <= end,
            )
        )

        result = await db.execute(stmt)
        rows = result.all()

        # group by date string "YYYY-MM-DD"
        daily: dict[str, dict] = defaultdict(lambda: {"total": 0, "errors": 0, "times": []})
        for row in rows:
            day = row.timestamp.strftime("%Y-%m-%d")
            g = daily[day]
            g["total"] += 1
            if row.status != "success":
                g["errors"] += 1
            elif row.response_time_ms is not None:
                g["times"].append(row.response_time_ms)

        def _percentile(data: list, p: float) -> Optional[float]:
            if not data:
                return None
            s = sorted(data)
            idx = max(0, int(len(s) * p) - 1)
            return round(s[idx], 2)

        return [
            {
                "date": day,
                "total_predictions": g["total"],
                "error_count": g["errors"],
                "error_rate": round(g["errors"] / g["total"], 4) if g["total"] > 0 else 0.0,
                "avg_latency_ms": (
                    round(sum(g["times"]) / len(g["times"]), 2) if g["times"] else None
                ),
                "p50_latency_ms": _percentile(g["times"], 0.50),
                "p95_latency_ms": _percentile(g["times"], 0.95),
            }
            for day, g in sorted(daily.items())
        ]

    @staticmethod
    async def get_model_version_stats_range(
        db: AsyncSession,
        model_name: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Retourne les statistiques par version (et shadow/non-shadow) sur une plage calendaire.

        Retourne une liste de dicts : un par version (non-shadow + shadow séparément).
        """
        stmt = select(
            Prediction.model_version,
            Prediction.is_shadow,
            Prediction.status,
            Prediction.response_time_ms,
        ).where(
            and_(
                Prediction.model_name == model_name,
                Prediction.timestamp >= start,
                Prediction.timestamp <= end,
            )
        )

        result = await db.execute(stmt)
        rows = result.all()

        # group: version -> {shadow: {total, errors, times}}
        grouped: dict[str, dict] = defaultdict(
            lambda: {
                False: {"total": 0, "errors": 0, "times": []},
                True: {"total": 0, "errors": 0, "times": []},
            }
        )

        for row in rows:
            ver = row.model_version or "unknown"
            shadow = bool(row.is_shadow)
            g = grouped[ver][shadow]
            g["total"] += 1
            if row.status != "success":
                g["errors"] += 1
            elif row.response_time_ms is not None:
                g["times"].append(row.response_time_ms)

        def _percentile(data: list, p: float) -> Optional[float]:
            if not data:
                return None
            s = sorted(data)
            idx = max(0, int(len(s) * p) - 1)
            return round(s[idx], 2)

        stats = []
        for ver, shadow_map in sorted(grouped.items()):
            prod = shadow_map[False]
            shad = shadow_map[True]
            total = prod["total"]
            times = prod["times"]
            n = len(times)
            stats.append(
                {
                    "version": ver,
                    "total_predictions": total,
                    "shadow_predictions": shad["total"],
                    "error_count": prod["errors"],
                    "error_rate": round(prod["errors"] / total, 4) if total > 0 else 0.0,
                    "avg_latency_ms": round(sum(times) / n, 2) if n > 0 else None,
                    "p50_latency_ms": _percentile(times, 0.50),
                    "p95_latency_ms": _percentile(times, 0.95),
                }
            )
        return stats

    @staticmethod
    async def get_model_recent_errors(
        db: AsyncSession,
        model_name: str,
        start: datetime,
        end: datetime,
        limit: int = 5,
    ) -> list[str]:
        """
        Retourne les derniers messages d'erreur distincts pour un modèle sur la période.
        """
        stmt = (
            select(Prediction.error_message)
            .where(
                and_(
                    Prediction.model_name == model_name,
                    Prediction.status == "error",
                    Prediction.error_message.isnot(None),
                    Prediction.timestamp >= start,
                    Prediction.timestamp <= end,
                )
            )
            .order_by(Prediction.timestamp.desc())
            .limit(50)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        # déduplique en conservant l'ordre
        seen: list[str] = []
        for msg in rows:
            if msg not in seen:
                seen.append(msg)
            if len(seen) >= limit:
                break
        return seen


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
    "status",
    "deprecated_at",
    "traffic_weight",
    "deployment_mode",
    "parent_version",
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
    "status",
    "traffic_weight",
    "deployment_mode",
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
