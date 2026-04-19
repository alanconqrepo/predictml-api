"""
Service d'auto-promotion post-retrain.

Évalue si un modèle récemment entraîné satisfait la politique de promotion
configurée. L'évaluation repose sur les paires (prédiction, résultat observé)
historiques du modèle et sur les temps de réponse enregistrés en production.
"""

from typing import Optional, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Prediction
from src.services.db_service import DBService


async def evaluate_auto_promotion(
    db: AsyncSession,
    model_name: str,
    policy: dict,
) -> Tuple[bool, str]:
    """
    Évalue si le modèle ``model_name`` doit être promu automatiquement
    en production selon ``policy``.

    Retourne ``(should_promote, reason)``.

    Logique :
    1. Récupère toutes les paires (prediction, observed_result) pour le modèle.
    2. Si le nombre de paires < ``min_sample_validation`` → non promu.
    3. Si ``min_accuracy`` est défini : calcule l'accuracy sur les
       ``min_sample_validation`` paires les plus récentes → non promu si
       accuracy < seuil.
    4. Si ``max_latency_p95_ms`` est défini : calcule le P95 des temps de
       réponse des prédictions réussies → non promu si P95 > seuil.
    """
    min_accuracy: Optional[float] = policy.get("min_accuracy")
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

    # --- Vérification de l'accuracy ---
    if min_accuracy is not None:
        recent = pairs[-min_sample_validation:]
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

    return True, "Tous les critères de promotion sont satisfaits."
