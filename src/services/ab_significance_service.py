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


def compute_ab_significance(
    version_stats: list[dict],
    confidence_level: float = 0.95,
) -> Optional[dict]:
    """
    Calcule la significativité statistique entre les deux versions A/B les plus actives.

    Sélectionne les 2 versions avec le plus de prédictions sur la fenêtre d'analyse.
    Priorité au Chi-² sur le taux d'erreur (si au moins une erreur est observée),
    sinon fallback Mann-Whitney U sur les temps de réponse.

    Retourne None si moins de 2 versions disponibles ou données insuffisantes.
    """
    active = [s for s in version_stats if s.get("total_predictions", 0) >= 1]
    if len(active) < 2:
        return None

    sorted_by_volume = sorted(active, key=lambda s: s["total_predictions"], reverse=True)
    va, vb = sorted_by_volume[0], sorted_by_volume[1]

    alpha = 1.0 - confidence_level
    n_a = va["total_predictions"]
    n_b = vb["total_predictions"]

    # --- Chi-² sur le taux d'erreur (métrique catégorielle) ---
    err_a = va.get("error_count", 0)
    err_b = vb.get("error_count", 0)

    if n_a >= 2 and n_b >= 2 and (err_a + err_b) > 0:
        ok_a = n_a - err_a
        ok_b = n_b - err_b
        contingency = [[err_a, ok_a], [err_b, ok_b]]
        chi2, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
        significant = bool(p_value < alpha)

        p1 = err_a / n_a
        p2 = err_b / n_b
        min_samples = _min_samples_proportions(p1, p2, alpha=alpha)

        if p1 < p2:
            winner = va["version"]
        elif p2 < p1:
            winner = vb["version"]
        else:
            winner = None

        return {
            "metric": "error_rate",
            "test": "chi2",
            "p_value": round(p_value, 6),
            "significant": significant,
            "confidence_level": confidence_level,
            "winner": winner,
            "min_samples_needed": min_samples,
            "current_samples": {va["version"]: n_a, vb["version"]: n_b},
        }

    # --- Mann-Whitney U sur les résidus de prédiction (régression) ---
    errors_a: list[float] = va.get("prediction_errors", [])
    errors_b: list[float] = vb.get("prediction_errors", [])

    if len(errors_a) >= 2 and len(errors_b) >= 2:
        _, p_value = stats.mannwhitneyu(errors_a, errors_b, alternative="two-sided")
        significant = bool(p_value < alpha)

        mean_a = sum(errors_a) / len(errors_a)
        mean_b = sum(errors_b) / len(errors_b)
        std_a = math.sqrt(sum((e - mean_a) ** 2 for e in errors_a) / len(errors_a))
        std_b = math.sqrt(sum((e - mean_b) ** 2 for e in errors_b) / len(errors_b))

        d = _cohen_d(mean_a, mean_b, std_a, std_b, len(errors_a), len(errors_b))
        min_samples = _min_samples_continuous(d, alpha=alpha)

        # Vainqueur = version avec MAE plus faible
        if mean_a < mean_b:
            winner = va["version"]
        elif mean_b < mean_a:
            winner = vb["version"]
        else:
            winner = None

        return {
            "metric": "mae",
            "test": "mann_whitney_u",
            "p_value": round(p_value, 6),
            "significant": significant,
            "confidence_level": confidence_level,
            "winner": winner,
            "min_samples_needed": min_samples,
            "current_samples": {va["version"]: len(errors_a), vb["version"]: len(errors_b)},
        }

    # --- Mann-Whitney U sur les temps de réponse (métrique continue) ---
    times_a: list[float] = va.get("response_times", [])
    times_b: list[float] = vb.get("response_times", [])

    if len(times_a) >= 2 and len(times_b) >= 2:
        _, p_value = stats.mannwhitneyu(times_a, times_b, alternative="two-sided")
        significant = bool(p_value < alpha)

        mean_a = sum(times_a) / len(times_a)
        mean_b = sum(times_b) / len(times_b)
        std_a = math.sqrt(sum((t - mean_a) ** 2 for t in times_a) / len(times_a))
        std_b = math.sqrt(sum((t - mean_b) ** 2 for t in times_b) / len(times_b))

        d = _cohen_d(mean_a, mean_b, std_a, std_b, len(times_a), len(times_b))
        min_samples = _min_samples_continuous(d, alpha=alpha)

        if mean_a < mean_b:
            winner = va["version"]
        elif mean_b < mean_a:
            winner = vb["version"]
        else:
            winner = None

        return {
            "metric": "response_time_ms",
            "test": "mann_whitney_u",
            "p_value": round(p_value, 6),
            "significant": significant,
            "confidence_level": confidence_level,
            "winner": winner,
            "min_samples_needed": min_samples,
            "current_samples": {va["version"]: len(times_a), vb["version"]: len(times_b)},
        }

    return None
