"""
Fonctions de calcul de métriques ML partagées entre l'API et les services.
"""

from __future__ import annotations

from typing import Optional

from sklearn.metrics import roc_auc_score


def compute_auc(
    y_true: list,
    y_prob: list,
    classes: Optional[list] = None,
) -> Optional[float]:
    """Calcule l'AUC-ROC à partir des probabilités prédites.

    - Binaire  : utilise la probabilité de la classe positive (index 1).
    - Multiclasse : OvR pondéré (weighted average).
    - Retourne ``None`` si les probabilités sont absentes ou si le calcul échoue
      (e.g. une seule classe dans ``y_true``, probas malformées…).

    Parameters
    ----------
    y_true : list
        Valeurs observées (labels).
    y_prob : list
        Liste de listes de probabilités par classe (une entrée par prédiction).
        Peut contenir des ``None`` si la prédiction n'avait pas de probabilités.
    classes : list | None
        Classes connues du modèle (optionnel, utilisé pour lever l'ambiguïté).

    Returns
    -------
    float | None
        AUC arrondie à 4 décimales, ou ``None``.
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
            # Classification binaire : prob de la 2e classe (index 1, ordre alphabétique)
            probs = [p[1] if isinstance(p, (list, tuple)) and len(p) >= 2 else None for p in y_prob]
            if any(v is None for v in probs):
                return None
            return round(float(roc_auc_score(y_true_s, probs)), 4)
        else:
            # Classification multiclasse : OvR pondéré
            if not all(isinstance(p, (list, tuple)) for p in y_prob):
                return None
            n_cols = len(y_prob[0])
            if n_cols != n_classes:
                # Dimensions incohérentes → abandon
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
    """Calcule les points (FPR, TPR) de la courbe ROC pour la classification binaire.

    Retourne ``(None, None)`` si les probabilités sont absentes, si ce n'est pas
    un problème binaire, ou si le calcul échoue.

    Parameters
    ----------
    y_true : list
        Valeurs observées.
    y_prob : list
        Liste de listes de probabilités (une par prédiction).

    Returns
    -------
    tuple[list[float] | None, list[float] | None]
        (fpr, tpr) arrondis à 4 décimales, ou (None, None).
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
