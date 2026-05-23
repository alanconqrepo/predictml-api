"""
Service de calcul de significativité statistique pour les comparaisons A/B.
"""

import math
from typing import Optional

from scipy import stats
from scipy.stats import norm


def _cohen_h(p1: float, p2: float) -> float:
    """Effet de taille de Cohen h pour comparer deux proportions."""
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def _min_samples_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Calcule le nombre minimal d'observations par groupe pour détecter la différence observée.

    Formule basée sur l'effet de taille de Cohen h (comparaison de proportions).
    Puissance cible : 80 % (z_beta = 0.84).
    """
    h = abs(_cohen_h(p1, p2))
    if h < 1e-10:
        return 0
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return math.ceil((z_alpha + z_beta) ** 2 / h**2)


def _cohen_d(mean1: float, mean2: float, std1: float, std2: float, n1: int, n2: int) -> float:
    """Cohen d (pooled) pour comparer deux distributions continues."""
    if n1 + n2 <= 2:
        return 0.0
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    if pooled_std == 0:
        return 0.0
    return abs(mean1 - mean2) / pooled_std


def _min_samples_continuous(d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Calcule le nombre minimal d'observations par groupe pour détecter la différence observée.

    Formule basée sur Cohen d (distributions continues).
    Puissance cible : 80 % (z_beta = 0.84).
    """
    if d < 1e-10:
        return 0
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return math.ceil(2 * (z_alpha + z_beta) ** 2 / d**2)


def _run_chi2_error_rate(
    va: dict, vb: dict, alpha: float, confidence_level: float
) -> Optional[dict]:
    """Chi-² sur le taux d'erreur API (métrique catégorielle)."""
    n_a = va["total_predictions"]
    n_b = vb["total_predictions"]
    err_a = va.get("error_count", 0)
    err_b = vb.get("error_count", 0)
    if n_a < 2 or n_b < 2 or (err_a + err_b) == 0:
        return None
    contingency = [[err_a, n_a - err_a], [err_b, n_b - err_b]]
    _, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
    p1, p2 = err_a / n_a, err_b / n_b
    winner = va["version"] if p1 < p2 else (vb["version"] if p2 < p1 else None)
    return {
        "metric": "error_rate",
        "test": "chi2",
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "confidence_level": confidence_level,
        "winner": winner,
        "min_samples_needed": _min_samples_proportions(p1, p2, alpha=alpha),
        "current_samples": {va["version"]: n_a, vb["version"]: n_b},
    }


def _run_mann_whitney_mae(
    va: dict, vb: dict, alpha: float, confidence_level: float
) -> Optional[dict]:
    """Mann-Whitney U sur les résidus de prédiction (MAE — régression)."""
    errors_a: list[float] = va.get("prediction_errors", [])
    errors_b: list[float] = vb.get("prediction_errors", [])
    if len(errors_a) < 2 or len(errors_b) < 2:
        return None
    _, p_value = stats.mannwhitneyu(errors_a, errors_b, alternative="two-sided")
    mean_a, mean_b = sum(errors_a) / len(errors_a), sum(errors_b) / len(errors_b)
    std_a = math.sqrt(sum((e - mean_a) ** 2 for e in errors_a) / len(errors_a))
    std_b = math.sqrt(sum((e - mean_b) ** 2 for e in errors_b) / len(errors_b))
    d = _cohen_d(mean_a, mean_b, std_a, std_b, len(errors_a), len(errors_b))
    winner = va["version"] if mean_a < mean_b else (vb["version"] if mean_b < mean_a else None)
    return {
        "metric": "mae",
        "test": "mann_whitney_u",
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "confidence_level": confidence_level,
        "winner": winner,
        "min_samples_needed": _min_samples_continuous(d, alpha=alpha),
        "current_samples": {va["version"]: len(errors_a), vb["version"]: len(errors_b)},
    }


def _run_mann_whitney_latency(
    va: dict, vb: dict, alpha: float, confidence_level: float
) -> Optional[dict]:
    """Mann-Whitney U sur les temps de réponse (latence)."""
    times_a: list[float] = va.get("response_times", [])
    times_b: list[float] = vb.get("response_times", [])
    if len(times_a) < 2 or len(times_b) < 2:
        return None
    _, p_value = stats.mannwhitneyu(times_a, times_b, alternative="two-sided")
    mean_a, mean_b = sum(times_a) / len(times_a), sum(times_b) / len(times_b)
    std_a = math.sqrt(sum((t - mean_a) ** 2 for t in times_a) / len(times_a))
    std_b = math.sqrt(sum((t - mean_b) ** 2 for t in times_b) / len(times_b))
    d = _cohen_d(mean_a, mean_b, std_a, std_b, len(times_a), len(times_b))
    winner = va["version"] if mean_a < mean_b else (vb["version"] if mean_b < mean_a else None)
    return {
        "metric": "response_time_ms",
        "test": "mann_whitney_u",
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "confidence_level": confidence_level,
        "winner": winner,
        "min_samples_needed": _min_samples_continuous(d, alpha=alpha),
        "current_samples": {va["version"]: len(times_a), vb["version"]: len(times_b)},
    }


_METRIC_RUNNERS = {
    "error_rate": _run_chi2_error_rate,
    "mae": _run_mann_whitney_mae,
    "response_time_ms": _run_mann_whitney_latency,
}

_AUTO_ORDER = ["error_rate", "mae", "response_time_ms"]


def compute_ab_significance(
    version_stats: list[dict],
    confidence_level: float = 0.95,
    metric: Optional[str] = None,
) -> Optional[dict]:
    """
    Calcule la significativité statistique entre les deux versions A/B les plus actives.

    - ``metric=None`` (défaut) : sélection automatique par priorité (error_rate → mae → response_time_ms).
    - ``metric="error_rate"`` | ``"mae"`` | ``"response_time_ms"`` : force la métrique choisie.

    Retourne None si moins de 2 versions disponibles ou données insuffisantes pour la métrique demandée.
    """
    active = [s for s in version_stats if s.get("total_predictions", 0) >= 1]
    if len(active) < 2:
        return None

    va, vb = sorted(active, key=lambda s: s["total_predictions"], reverse=True)[:2]
    alpha = 1.0 - confidence_level

    if metric and metric in _METRIC_RUNNERS:
        return _METRIC_RUNNERS[metric](va, vb, alpha, confidence_level)

    # Mode auto : essaie dans l'ordre jusqu'à trouver des données suffisantes
    for m in _AUTO_ORDER:
        result = _METRIC_RUNNERS[m](va, vb, alpha, confidence_level)
        if result is not None:
            return result

    return None
