"""
Service de détection de data drift.

Calcule trois métriques par feature numérique :
- Z-score : écart normalisé entre la moyenne de production et la baseline
- PSI (Population Stability Index) : divergence de distribution via bins normaux
- Null rate : taux de valeurs nulles/manquantes en production vs baseline

Seuils de statut :
  Z-score    : ok < 2 | warning 2–3 | critical ≥ 3
  PSI        : ok < 0.1 | warning 0.1–0.2 | critical ≥ 0.2
  Null rate  : ok si écart absolu < 5 pts | warning 5–15 pts | critical > 15 pts ou prod > 30 %

Statut final = pire des trois statuts.
"""

import math
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from scipy import stats

from src.schemas.model import FeatureDriftResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.model import OutputDriftResponse

_N_BINS = 10
_EPS = 1e-6  # évite log(0)


def _status_from_z(z: float) -> str:
    if z < 2.0:
        return "ok"
    if z < 3.0:
        return "warning"
    return "critical"


def _status_from_psi(psi: float) -> str:
    if psi < 0.1:
        return "ok"
    if psi < 0.2:
        return "warning"
    return "critical"


def _status_from_null_rate(prod_null_rate: float, baseline_null_rate: float) -> str:
    if prod_null_rate > 0.30:
        return "critical"
    diff = prod_null_rate - baseline_null_rate
    if diff > 0.15:
        return "critical"
    if diff >= 0.05:
        return "warning"
    return "ok"


def _worst_status(*statuses: str) -> str:
    order = {"ok": 0, "warning": 1, "critical": 2}
    return max(statuses, key=lambda s: order.get(s, -1))


def _compute_psi(
    prod_values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    baseline_min: float,
    baseline_max: float,
) -> float:
    """
    Calcule le PSI en comparant la distribution de production à la distribution
    N(baseline_mean, baseline_std) découpée en _N_BINS bins égaux.
    """
    if baseline_std <= 0:
        return 0.0

    # Bornes des bins : légèrement élargies au-delà de [min, max]
    margin = 0.5 * baseline_std
    low = baseline_min - margin
    high = baseline_max + margin
    if high <= low:
        return 0.0

    bin_edges = np.linspace(low, high, _N_BINS + 1)

    # Proportions attendues (distribution normale baseline)
    expected = np.diff(stats.norm.cdf(bin_edges, loc=baseline_mean, scale=baseline_std))
    expected = np.clip(expected, _EPS, None)
    expected /= expected.sum()

    # Proportions observées en production
    counts, _ = np.histogram(prod_values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    actual = counts / total
    actual = np.clip(actual, _EPS, None)

    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return round(psi, 6)


def compute_feature_drift(
    baseline: Dict[str, dict],
    production_stats: Dict[str, dict],
    min_count: int = 30,
) -> Dict[str, FeatureDriftResult]:
    """
    Compare les stats de production au baseline par feature.

    Args:
        baseline: {feature: {mean, std, min, max, null_rate?}} — issu de model_metadata.feature_baseline
        production_stats: {feature: {mean, std, min, max, count, values, null_rate?}} — calculé depuis les prédictions
        min_count: nombre minimum de prédictions pour calculer le drift

    Returns:
        Dict[feature, FeatureDriftResult]
    """
    results: Dict[str, FeatureDriftResult] = {}

    all_features = set(baseline.keys()) | set(production_stats.keys())

    for feature in sorted(all_features):
        bl = baseline.get(feature)
        ps = production_stats.get(feature, {})

        prod_count = int(ps.get("count", 0))
        prod_mean: Optional[float] = ps.get("mean")
        prod_std: Optional[float] = ps.get("std")
        null_rate_prod: Optional[float] = ps.get("null_rate")

        # Pas de baseline pour cette feature
        if bl is None:
            results[feature] = FeatureDriftResult(
                production_mean=round(prod_mean, 6) if prod_mean is not None else None,
                production_std=round(prod_std, 6) if prod_std is not None else None,
                production_count=prod_count,
                null_rate_production=(
                    round(null_rate_prod, 6) if null_rate_prod is not None else None
                ),
                drift_status="no_baseline",
            )
            continue

        bl_mean = float(bl.get("mean", 0))
        bl_std = float(bl.get("std", 0))
        bl_min = float(bl.get("min", 0))
        bl_max = float(bl.get("max", 0))
        null_rate_base: Optional[float] = bl.get("null_rate")

        # Null rate status (computed independently of distribution metrics)
        null_rate_status: Optional[str] = None
        if null_rate_prod is not None and null_rate_base is not None:
            null_rate_status = _status_from_null_rate(null_rate_prod, null_rate_base)

        # Données de production insuffisantes
        if prod_count < min_count or prod_mean is None:
            results[feature] = FeatureDriftResult(
                baseline_mean=round(bl_mean, 6),
                baseline_std=round(bl_std, 6),
                baseline_min=round(bl_min, 6),
                baseline_max=round(bl_max, 6),
                production_mean=round(prod_mean, 6) if prod_mean is not None else None,
                production_std=round(prod_std, 6) if prod_std is not None else None,
                production_count=prod_count,
                null_rate_production=(
                    round(null_rate_prod, 6) if null_rate_prod is not None else None
                ),
                null_rate_baseline=round(null_rate_base, 6) if null_rate_base is not None else None,
                null_rate_status=null_rate_status,
                drift_status="insufficient_data",
            )
            continue

        # Z-score
        z_score: Optional[float] = None
        z_status = "ok"
        if bl_std > 0:
            z_score = abs(prod_mean - bl_mean) / bl_std
            z_score = round(z_score, 4)
            z_status = _status_from_z(z_score)

        # PSI (nécessite les valeurs brutes)
        psi: Optional[float] = None
        psi_status = "ok"
        raw_values = ps.get("values")
        if raw_values is not None and len(raw_values) >= min_count:
            arr = np.array(raw_values, dtype=float)
            psi = _compute_psi(arr, bl_mean, bl_std, bl_min, bl_max)
            psi_status = _status_from_psi(psi)

        # Statut final : pire des trois dimensions
        candidate_statuses = [z_status, psi_status]
        if null_rate_status is not None:
            candidate_statuses.append(null_rate_status)

        results[feature] = FeatureDriftResult(
            baseline_mean=round(bl_mean, 6),
            baseline_std=round(bl_std, 6),
            baseline_min=round(bl_min, 6),
            baseline_max=round(bl_max, 6),
            production_mean=round(prod_mean, 6),
            production_std=round(prod_std, 6) if prod_std is not None else None,
            production_count=prod_count,
            z_score=z_score,
            psi=round(psi, 6) if psi is not None else None,
            null_rate_production=round(null_rate_prod, 6) if null_rate_prod is not None else None,
            null_rate_baseline=round(null_rate_base, 6) if null_rate_base is not None else None,
            null_rate_status=null_rate_status,
            drift_status=_worst_status(*candidate_statuses),
        )

    return results


def summarize_drift(features: Dict[str, FeatureDriftResult], baseline_available: bool) -> str:
    """Retourne le statut global le plus défavorable parmi toutes les features.

    Prend en compte drift_status (Z-score + PSI) et null_rate_status comme quatrième dimension.
    """
    if not baseline_available:
        return "no_baseline"

    statuses = [f.drift_status for f in features.values() if f.drift_status not in ("no_baseline",)]
    if not statuses:
        return "insufficient_data"

    ranked = [s for s in statuses if s in ("ok", "warning", "critical")]
    # null_rate_status des features à données insuffisantes contribue aussi au résumé
    null_ranked = [
        f.null_rate_status
        for f in features.values()
        if f.null_rate_status in ("ok", "warning", "critical")
    ]
    all_ranked = ranked + null_ranked

    if not all_ranked:
        return "insufficient_data"

    order = {"ok": 0, "warning": 1, "critical": 2}
    return max(all_ranked, key=lambda s: order[s])


def is_nan_safe(value: float) -> bool:
    """Vérifie si une valeur float est NaN ou infinie."""
    return math.isnan(value) or math.isinf(value)


_OUTPUT_DRIFT_EPS = 1e-6


async def compute_output_drift(
    model_name: str,
    period_days: int,
    db: "AsyncSession",
    model_version: Optional[str] = None,
    min_predictions: int = 30,
) -> "OutputDriftResponse":
    """
    Calcule le drift de distribution des sorties (label shift).

    Compare la distribution récente de prediction_result à la distribution
    d'entraînement stockée dans training_stats.label_distribution.

    PSI < 0.1 → ok | 0.1–0.2 → warning | ≥ 0.2 → critical
    Retourne status="no_baseline" si label_distribution absent.
    """
    from src.schemas.model import OutputDriftClassResult, OutputDriftResponse
    from src.services.db_service import DBService

    metadata = await DBService.get_model_metadata(db, model_name, model_version)
    if not metadata:
        return OutputDriftResponse(
            model_name=model_name,
            model_version=model_version,
            period_days=period_days,
            predictions_analyzed=0,
            status="no_baseline",
        )

    training_stats = metadata.training_stats or {}
    baseline_distribution = training_stats.get("label_distribution")

    if not baseline_distribution:
        return OutputDriftResponse(
            model_name=model_name,
            model_version=metadata.version,
            period_days=period_days,
            predictions_analyzed=0,
            status="no_baseline",
        )

    label_counts, total = await DBService.get_prediction_label_distribution(
        db, model_name, metadata.version, days=period_days
    )

    if total < min_predictions:
        return OutputDriftResponse(
            model_name=model_name,
            model_version=metadata.version,
            period_days=period_days,
            predictions_analyzed=total,
            status="insufficient_data",
        )

    current_distribution = {label: count / total for label, count in label_counts.items()}

    all_labels = sorted(set(baseline_distribution.keys()) | set(current_distribution.keys()))

    psi = 0.0
    for label in all_labels:
        bl = max(float(baseline_distribution.get(label, 0.0)), _OUTPUT_DRIFT_EPS)
        cur = max(current_distribution.get(label, 0.0), _OUTPUT_DRIFT_EPS)
        psi += (cur - bl) * math.log(cur / bl)
    psi = round(psi, 6)

    drift_status = _status_from_psi(psi)

    by_class = [
        OutputDriftClassResult(
            label=label,
            baseline_ratio=round(float(baseline_distribution.get(label, 0.0)), 4),
            current_ratio=round(current_distribution.get(label, 0.0), 4),
            delta=round(
                current_distribution.get(label, 0.0) - float(baseline_distribution.get(label, 0.0)),
                4,
            ),
        )
        for label in all_labels
    ]

    return OutputDriftResponse(
        model_name=model_name,
        model_version=metadata.version,
        period_days=period_days,
        predictions_analyzed=total,
        status=drift_status,
        psi=psi,
        baseline_distribution={k: round(float(v), 4) for k, v in baseline_distribution.items()},
        current_distribution={k: round(v, 4) for k, v in current_distribution.items()},
        by_class=by_class,
    )
