"""
Post-retrain auto-promotion service and auto-demotion circuit breaker.

Evaluates whether a recently trained model satisfies the configured promotion
policy. Evaluation relies on historical (prediction, observed_result) pairs
and on response times recorded in production.

The auto-demotion circuit breaker periodically evaluates whether a production
model should be demoted based on drift or an accuracy drop.
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
from src.services.metrics_service import compute_auc

logger = structlog.get_logger(__name__)


def _is_regression_pairs(pairs: list) -> bool:
    """Return True if the pairs appear to come from a regression model.

    Heuristic: at least one value (prediction or observed) is a non-integer float.
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
    Evaluate whether model ``model_name`` should be automatically promoted
    to production according to ``policy``.

    Returns ``(should_promote, reason)``.

    Logic:
    1. Retrieve all (prediction, observed_result) pairs for the model.
    2. If the number of pairs < ``min_sample_validation`` → not promoted.
    3. Automatically detect type (classification / regression):
       - Regression: if ``max_mae`` is set, check MAE on the last N pairs.
       - Classification: if ``min_accuracy`` is set, check accuracy (exact equality).
    4. If ``max_latency_p95_ms`` is set: compute P95 of response times for
       successful predictions → not promoted if P95 > threshold.
    """
    min_accuracy: Optional[float] = policy.get("min_accuracy")
    min_auc: Optional[float] = policy.get("min_auc")
    max_mae: Optional[float] = policy.get("max_mae")
    max_latency_p95_ms: Optional[float] = policy.get("max_latency_p95_ms")
    min_sample_validation: int = int(policy.get("min_sample_validation", 10))

    # --- Validation pairs (all versions combined, sorted by timestamp) ---
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

    # --- Performance metric check ---
    if is_regression:
        if max_mae is not None:
            try:
                mae = sum(abs(float(p) - float(o)) for p, o, _, _ in recent) / len(recent)
            except (ValueError, TypeError):
                mae = None
            if mae is not None and mae > max_mae:
                return (
                    False,
                    (f"MAE trop élevé : {mae:.4f} > {max_mae:.4f} (maximum autorisé)."),
                )
    else:
        if min_accuracy is not None:
            correct = sum(1 for pred, obs, _, _ in recent if str(pred) == str(obs))
            accuracy = correct / len(recent)
            if accuracy < min_accuracy:
                return (
                    False,
                    (f"Accuracy insuffisante : {accuracy:.4f} < {min_accuracy:.4f} (minimum requis)."),
                )

        if min_auc is not None:
            y_true = [obs for _, obs, _, _ in recent]
            y_prob = [prob for _, _, prob, _ in recent]
            auc = compute_auc(y_true, y_prob)
            if auc is not None and auc < min_auc:
                return (
                    False,
                    f"AUC insuffisant : {auc:.4f} < {min_auc:.4f} (minimum requis).",
                )

    # --- P95 latency check ---
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
                    (f"Latence P95 trop élevée : {p95:.1f} ms > {max_latency_p95_ms:.1f} ms (maximum autorisé)."),
                )

    # --- Golden tests check ---
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
    Evaluate whether the production model ``model_name`` should be automatically
    demoted according to ``policy``.

    Guardrails:
    - Only acts if at least one other active (non-production) version exists.
    - Respects the cooldown to avoid oscillations.

    Returns ``(demoted, reason)``.
    """
    from src.services.webhook_service import send_webhook

    auto_demote: bool = policy.get("auto_demote", False)
    if not auto_demote:
        return False, "Auto-demotion disabled."

    demote_on_drift: str = policy.get("demote_on_drift", "critical")
    demote_on_accuracy_below: Optional[float] = policy.get("demote_on_accuracy_below")
    demote_cooldown_hours: int = int(policy.get("demote_cooldown_hours", 24))
    min_sample_validation: int = int(policy.get("min_sample_validation", 10))

    # --- Find the production model ---
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
        return False, "No version in production."

    # --- Guardrail: fallback version ---
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
            "Auto-demotion impossible: no fallback available",
            model=model_name,
            version=prod_meta.version,
        )
        if settings.ENABLE_EMAIL_ALERTS:
            email_service.send_auto_demotion_alert(
                model_name,
                prod_meta.version,
                "Dérive ou dégradation de performance détectée — aucune version de secours disponible.",
                no_fallback=True,
            )
        return False, "Aucune version de secours disponible — rétrogradation annulée."

    # --- Guardrail: cooldown ---
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
                f"Cooldown active until {until.strftime('%Y-%m-%dT%H:%M')} UTC.",
            )

    # --- Collect demotion reasons ---
    reasons: list[str] = []

    # Drift check (features + output)
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
            reasons.append(f"Feature drift {feature_summary} detected.")
        if output_level >= trigger_level > 0:
            reasons.append(f"Output drift {output_status} detected.")

    # Accuracy check
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
                        f"Insufficient accuracy: {accuracy:.4f} < {demote_on_accuracy_below:.4f}."
                    )

    if not reasons:
        return False, "No demotion criterion triggered."

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
        "Model auto-demoted",
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
