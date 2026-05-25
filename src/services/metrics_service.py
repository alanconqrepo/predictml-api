"""
ML metric computation functions shared between the API and services.
"""

from __future__ import annotations

from typing import Optional

from sklearn.metrics import roc_auc_score


def compute_auc(
    y_true: list,
    y_prob: list,
    classes: Optional[list] = None,
) -> Optional[float]:
    """Compute AUC-ROC from predicted probabilities.

    - Binary: uses the positive class probability (index 1).
    - Multiclass: weighted OvR (weighted average).
    - Returns ``None`` if probabilities are absent or if computation fails
      (e.g. only one class in ``y_true``, malformed probabilities...).

    Parameters
    ----------
    y_true : list
        Observed values (labels).
    y_prob : list
        List of per-class probability lists (one entry per prediction).
        May contain ``None`` if the prediction had no probabilities.
    classes : list | None
        Known model classes (optional, used to disambiguate).

    Returns
    -------
    float | None
        AUC rounded to 4 decimal places, or ``None``.
    """
    if not y_prob or not any(p is not None for p in y_prob):
        return None
    try:
        y_true_s = [str(v) for v in y_true]
        unique_labels = sorted(set(y_true_s))
        n_classes = len(unique_labels)
        if n_classes < 2:
            return None

        if n_classes == 2:
            # Binary classification: probability of the 2nd class (index 1, alphabetical order)
            probs = [p[1] if isinstance(p, (list, tuple)) and len(p) >= 2 else None for p in y_prob]
            if any(v is None for v in probs):
                return None
            return round(float(roc_auc_score(y_true_s, probs)), 4)
        else:
            # Multiclass classification: weighted OvR
            if not all(isinstance(p, (list, tuple)) for p in y_prob):
                return None
            n_cols = len(y_prob[0])
            if n_cols != n_classes:
                # Inconsistent dimensions → abort
                return None
            return round(
                float(
                    roc_auc_score(
                        y_true_s,
                        y_prob,
                        multi_class="ovr",
                        average="weighted",
                        labels=unique_labels,
                    )
                ),
                4,
            )
    except Exception:
        return None


def compute_roc_curve(
    y_true: list,
    y_prob: list,
) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """Compute (FPR, TPR) points of the ROC curve for binary classification.

    Returns ``(None, None)`` if probabilities are absent, if it is not a
    binary problem, or if computation fails.

    Parameters
    ----------
    y_true : list
        Observed values.
    y_prob : list
        List of probability lists (one per prediction).

    Returns
    -------
    tuple[list[float] | None, list[float] | None]
        (fpr, tpr) rounded to 4 decimal places, or (None, None).
    """
    if not y_prob or not any(p is not None for p in y_prob):
        return None, None
    try:
        from sklearn.metrics import roc_curve as sklearn_roc_curve

        y_true_s = [str(v) for v in y_true]
        unique_labels = sorted(set(y_true_s))
        if len(unique_labels) != 2:
            return None, None

        probs = [p[1] if isinstance(p, (list, tuple)) and len(p) >= 2 else None for p in y_prob]
        if any(v is None for v in probs):
            return None, None

        fpr, tpr, _ = sklearn_roc_curve(y_true_s, probs)
        return (
            [round(float(v), 4) for v in fpr],
            [round(float(v), 4) for v in tpr],
        )
    except Exception:
        return None, None
