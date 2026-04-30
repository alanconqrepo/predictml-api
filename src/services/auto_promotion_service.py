"""
Service d'auto-promotion post-retrain et circuit breaker d'auto-demotion.

Évalue si un modèle récemment entraîné satisfait la politique de promotion
configurée. L'évaluation repose sur les paires (prédiction, résultat observé)
historiques du modèle et sur les temps de réponse enregistrés en production.

Le circuit breaker d'auto-demotion évalue périodiquement si un modèle en
production doit être retiré sur la base du drift ou d'une chute d'accuracy.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple

import structlog
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.db.models import Prediction
from src.db.models.model_history import HistoryActionType, ModelHistory
from src.db.models.model_metadata import ModelMetadata
from src.services import drift_service
from src.services.db_service import DBService
from src.services.email_service import email_service

logger = structlog.get_logger(__name__)


def _is_regression_pairs(pairs: list) -> bool:
    """Retourne True si les paires semblent provenir d'un modèle de régression.

    Heuristique : au moins une valeur (prédiction ou observé) est un float non-entier.
    """
    for pred, obs, _, _ in pairs:
        for val in (pred, obs):
            try:
                f = float(val)
                if f != int(f):
                    return True
            except (ValueError, TypeError):
                pass
    return False


async def evaluate_auto_promotion(
    db: AsyncSession,
    model_name: str,
    policy: dict,
    version: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Évalue si le modèle ``model_name`` doit être promu automatiquement
    en production selon ``policy``.

    Retourne ``(should_promote, reason)``.

    Logique :
    1. Récupère toutes les paires (prediction, observed_result) pour le modèle.
    2. Si le nombre de paires < ``min_sample_validation`` → non promu.
    3. Détecte automatiquement le type (classification / régression) :
       - Régression : si ``max_mae`` est défini, vérifie la MAE sur les N dernières paires.
       - Classification : si ``min_accuracy`` est défini, vérifie l'accuracy (égalité exacte).
    4. Si ``max_latency_p95_ms`` est défini : calcule le P95 des temps de
       réponse des prédictions réussies → non promu si P95 > seuil.
    """
    min_accuracy: Optional[float] = policy.get("min_accuracy")
    max_mae: Optional[float] = policy.get("max_mae")
    max_latency_p95_ms: Optional[float] = policy.get("max_latency_p95_ms")
    min_sample_validation: int = int(policy.get("min_sample_validation", 10))

    # --- Paires de validation (toutes versions confondues, triées par timestamp) ---
    pairs = await DBService.get_performance_pairs(db, model_name)
    n_samples = len(pairs)

    if n_samples < min_sample_validation:
        return (
            False,
            (
                f"Échantillons insuffisants pour la validation : "
                f"{n_samples}/{min_sample_validation} requis."
            ),
        )

    recent = pairs[-min_sample_validation:]
    is_regression = _is_regression_pairs(recent)

    # --- Vérification de la métrique de performance ---
    if is_regression:
        if max_mae is not None:
            try:
                mae = sum(abs(float(p) - float(o)) for p, o, _, _ in recent) / len(recent)
            except (ValueError, TypeError):
                mae = None
            if mae is not None and mae > max_mae:
                return (
                    False,
                    (f"MAE trop élevée : {mae:.4f} > {max_mae:.4f} requis."),
                )
    else:
        if min_accuracy is not None:
            correct = sum(1 for pred, obs, _, _ in recent if str(pred) == str(obs))
            accuracy = correct / len(recent)
            if accuracy < min_accuracy:
                return (
                    False,
                    (f"Précision insuffisante : {accuracy:.4f} " f"< {min_accuracy:.4f} requis."),
                )

    # --- Vérification de la latence P95 ---
    if max_latency_p95_ms is not None:
        stmt = select(Prediction.response_time_ms).where(
            and_(
                Prediction.model_name == model_name,
                Prediction.status == "success",
                Prediction.response_time_ms.isnot(None),
            )
        )
        result = await db.execute(stmt)
        times = sorted(r[0] for r in result.all())
        if times:
            idx = max(0, int(len(times) * 0.95) - 1)
            p95 = times[idx]
            if p95 > max_latency_p95_ms:
                return (
                    False,
                    (
                        f"Latence P95 trop élevée : {p95:.1f}ms "
                        f"> {max_latency_p95_ms:.1f}ms max."
                    ),
                )

    # --- Vérification des golden tests ---
    min_golden_test_pass_rate: Optional[float] = policy.get("min_golden_test_pass_rate")
    if min_golden_test_pass_rate is not None and version is not None:
        from src.services.golden_test_service import GoldenTestService

        result = await GoldenTestService.run_tests(db, model_name, version)
        if result.total_tests > 0 and result.pass_rate < min_golden_test_pass_rate:
            return (
                False,
                (
                    f"Tests de régression insuffisants : {result.pass_rate:.2%} "
                    f"< {min_golden_test_pass_rate:.2%} requis "
                    f"({result.failed}/{result.total_tests} échecs)."
                ),
            )

    return True, "Tous les critères de promotion sont satisfaits."


# ---------------------------------------------------------------------------
# Circuit breaker — auto-demotion
# ---------------------------------------------------------------------------

_DRIFT_ORDER = {"ok": 0, "warning": 1, "critical": 2, "insufficient_data": -1, "no_baseline": -1}


def _utcnow() -> datetime:
    return datetime.utcnow()


async def evaluate_auto_demotion(
    db: AsyncSession,
    model_name: str,
    policy: dict,
) -> Tuple[bool, str]:
    """
    Évalue si le modèle ``model_name`` en production doit être automatiquement
    retiré selon la politique ``policy``.

    Garde-fous :
    - N'agit que si au moins une autre version active (non-production) existe.
    - Respecte le cooldown pour éviter les oscillations.

    Retourne ``(demoted, reason)``.
    """
    from src.services.webhook_service import send_webhook

    auto_demote: bool = policy.get("auto_demote", False)
    if not auto_demote:
        return False, "Auto-demotion désactivée."

    demote_on_drift: str = policy.get("demote_on_drift", "critical")
    demote_on_accuracy_below: Optional[float] = policy.get("demote_on_accuracy_below")
    demote_cooldown_hours: int = int(policy.get("demote_cooldown_hours", 24))
    min_sample_validation: int = int(policy.get("min_sample_validation", 10))

    # --- Trouver le modèle en production ---
    prod_result = await db.execute(
        select(ModelMetadata).where(
            and_(
                ModelMetadata.name == model_name,
                ModelMetadata.is_production.is_(True),
                ModelMetadata.is_active.is_(True),
            )
        )
    )
    prod_meta = prod_result.scalars().first()
    if prod_meta is None:
        return False, "Aucune version en production."

    # --- Garde-fou : version de fallback ---
    fallback_result = await db.execute(
        select(ModelMetadata).where(
            and_(
                ModelMetadata.name == model_name,
                ModelMetadata.is_production.is_(False),
                ModelMetadata.is_active.is_(True),
                ModelMetadata.status != "archived",
            )
        )
    )
    fallback_versions = fallback_result.scalars().all()
    if not fallback_versions:
        logger.critical(
            "Auto-demotion impossible : aucun fallback disponible",
            model=model_name,
            version=prod_meta.version,
        )
        if settings.ENABLE_EMAIL_ALERTS:
            email_service.send_auto_demotion_alert(
                model_name, prod_meta.version, "Drift ou dégradation détectée", no_fallback=True
            )
        return False, "Aucune version de fallback disponible — demotion annulée."

    # --- Garde-fou : cooldown ---
    if demote_cooldown_hours > 0:
        cooldown_since = _utcnow() - timedelta(hours=demote_cooldown_hours)
        recent_demote_result = await db.execute(
            select(ModelHistory)
            .where(
                and_(
                    ModelHistory.model_name == model_name,
                    ModelHistory.action == HistoryActionType.AUTO_DEMOTE,
                    ModelHistory.timestamp >= cooldown_since,
                )
            )
            .order_by(desc(ModelHistory.timestamp))
            .limit(1)
        )
        recent_demote = recent_demote_result.scalars().first()
        if recent_demote is not None:
            until = recent_demote.timestamp + timedelta(hours=demote_cooldown_hours)
            return (
                False,
                f"Cooldown actif jusqu'à {until.strftime('%Y-%m-%dT%H:%M')} UTC.",
            )

    # --- Collecter les raisons de demotion ---
    reasons: list[str] = []

    # Vérification du drift (features + output)
    if prod_meta.feature_baseline:
        production_stats = await DBService.get_feature_production_stats(
            db, model_name, prod_meta.version, days=1
        )
        features = drift_service.compute_feature_drift(
            prod_meta.feature_baseline, production_stats, min_count=10
        )
        feature_summary = drift_service.summarize_drift(features, baseline_available=True)
        output_report = await drift_service.compute_output_drift(
            model_name=model_name,
            period_days=1,
            db=db,
            model_version=prod_meta.version,
            min_predictions=10,
        )
        output_status = output_report.status

        trigger_level = _DRIFT_ORDER.get(demote_on_drift, 2)
        feature_level = _DRIFT_ORDER.get(feature_summary, -1)
        output_level = _DRIFT_ORDER.get(output_status, -1)

        if feature_level >= trigger_level > 0:
            reasons.append(f"Drift features {feature_summary} détecté.")
        if output_level >= trigger_level > 0:
            reasons.append(f"Drift de sortie {output_status} détecté.")

    # Vérification de l'accuracy
    if demote_on_accuracy_below is not None:
        pairs = await DBService.get_performance_pairs(db, model_name)
        if len(pairs) >= min_sample_validation:
            recent = pairs[-min_sample_validation:]
            is_regression = _is_regression_pairs(recent)
            if not is_regression:
                correct = sum(1 for pred, obs, _, _ in recent if str(pred) == str(obs))
                accuracy = correct / len(recent)
                if accuracy < demote_on_accuracy_below:
                    reasons.append(
                        f"Accuracy insuffisante : {accuracy:.4f} < {demote_on_accuracy_below:.4f}."
                    )

    if not reasons:
        return False, "Aucun critère de demotion déclenché."

    # --- Demotion ---
    combined_reason = " ".join(reasons)
    prod_meta.is_production = False

    entry = await DBService.log_model_history(
        db,
        prod_meta,
        HistoryActionType.AUTO_DEMOTE,
        user_id=None,
        username="scheduler",
        changed_fields=["is_production"],
    )
    entry.snapshot["auto_demote_reason"] = combined_reason
    await db.commit()

    logger.warning(
        "Modèle auto-démis",
        model=model_name,
        version=prod_meta.version,
        reason=combined_reason,
    )

    if settings.ENABLE_EMAIL_ALERTS:
        email_service.send_auto_demotion_alert(model_name, prod_meta.version, combined_reason)

    if prod_meta.webhook_url:
        asyncio.create_task(
            send_webhook(
                prod_meta.webhook_url,
                {
                    "model_name": model_name,
                    "version": prod_meta.version,
                    "timestamp": _utcnow().isoformat() + "Z",
                    "details": {"reason": combined_reason},
                },
                event_type="auto_demote",
            )
        )

    return True, combined_reason
