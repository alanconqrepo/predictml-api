"""
Service pour l'explication SHAP des prédictions (feature importances locales).

Supporte :
  - TreeExplainer : RandomForest, GradientBoosting, DecisionTree, ExtraTrees, HistGradientBoosting
  - LinearExplainer : LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGD
"""

import numpy as np
import shap
from fastapi import HTTPException, status

# Types sklearn compatibles avec shap.TreeExplainer (pas de données background requises)
_TREE_TYPES = frozenset(
    [
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
    ]
)

# Types sklearn compatibles avec shap.LinearExplainer (nécessite des données background)
_LINEAR_TYPES = frozenset(
    [
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
        "SGDRegressor",
        "LinearSVC",
        "LinearSVR",
    ]
)


def compute_shap_explanation(
    model,
    feature_names: list,
    x: np.ndarray,
    prediction_result,
    feature_baseline: dict | None,
) -> dict:
    """
    Calcule les valeurs SHAP locales pour une observation.

    Paramètres
    ----------
    model : objet sklearn
    feature_names : liste ordonnée des noms de features
    x : array numpy de shape (1, n_features)
    prediction_result : résultat de model.predict(x)[0] (pour résoudre l'index de classe)
    feature_baseline : dict {feature: {mean, std, min, max}} issu de model_metadata.feature_baseline

    Retourne
    --------
    dict avec les clés :
      - shap_values : dict {feature_name: float}
      - base_value  : float
      - model_type  : "tree" | "linear"
    """
    model_class = type(model).__name__

    if model_class in _TREE_TYPES:
        return _explain_tree(model, feature_names, x, prediction_result)

    if model_class in _LINEAR_TYPES:
        return _explain_linear(model, feature_names, x, prediction_result, feature_baseline)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            f"Type de modèle '{model_class}' non supporté pour l'explication SHAP. "
            "Types supportés — arbres : RandomForest, GradientBoosting, DecisionTree, ExtraTrees,"
            " HistGradientBoosting. Linéaires : LogisticRegression, LinearRegression, Ridge,"
            " Lasso, ElasticNet, SGD."
        ),
    )


def _resolve_class_index(model, prediction_result) -> int:
    """Retourne l'index de la classe prédite dans model.classes_, ou 0 par défaut."""
    if hasattr(model, "classes_"):
        classes = model.classes_.tolist()
        if prediction_result in classes:
            return classes.index(prediction_result)
    return 0


def _extract_vals_and_base(shap_vals, base_vals, class_idx: int):
    """
    Extrait les valeurs SHAP et la base value pour une classe donnée.

    Compatible avec les différents formats de sortie selon la version de SHAP :
      - list[ndarray]         → classificateur multi-classe (une array par classe)
      - ndarray 3D            → (n_samples, n_features, n_classes) — format SHAP récent
      - ndarray 2D            → (n_samples, n_features) — régresseur ou binaire compressé
    base_vals peut être un scalaire, un array 0D, ou un array 1D (une valeur par classe).
    """
    b = np.asarray(base_vals)

    if isinstance(shap_vals, list):
        # Format liste : list[array(n_samples, n_features)], un par classe
        vals = shap_vals[class_idx][0]
        base = float(b[class_idx]) if b.ndim > 0 and len(b) > 1 else float(b.ravel()[0])
    elif shap_vals.ndim == 3:
        # Format 3D : (n_samples, n_features, n_classes)
        vals = shap_vals[0, :, class_idx]
        base = float(b[class_idx]) if b.ndim > 0 and len(b) > class_idx else float(b.ravel()[0])
    else:
        # Format 2D : (n_samples, n_features) — régresseur ou classificateur binaire
        vals = shap_vals[0]
        if b.ndim > 0 and len(b) > 1:
            base = float(b[class_idx]) if class_idx < len(b) else float(b[0])
        else:
            base = float(b.ravel()[0])

    return vals, base


def _explain_tree(model, feature_names: list, x: np.ndarray, prediction_result) -> dict:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x)
    base_vals = explainer.expected_value

    class_idx = _resolve_class_index(model, prediction_result)
    vals, base = _extract_vals_and_base(shap_vals, base_vals, class_idx)

    return {
        "shap_values": {name: float(v) for name, v in zip(feature_names, vals)},
        "base_value": base,
        "model_type": "tree",
    }


def _explain_linear(
    model, feature_names: list, x: np.ndarray, prediction_result, feature_baseline: dict | None
) -> dict:
    # Construire une donnée de background : moyenne des features de training, ou zéros
    if feature_baseline:
        background = np.array(
            [[feature_baseline.get(f, {}).get("mean", 0.0) for f in feature_names]],
            dtype=float,
        )
    else:
        background = np.zeros((1, len(feature_names)), dtype=float)

    masker = shap.maskers.Independent(background)
    explainer = shap.LinearExplainer(model, masker=masker)
    shap_vals = explainer.shap_values(x)
    base_vals = explainer.expected_value

    class_idx = _resolve_class_index(model, prediction_result)
    vals, base = _extract_vals_and_base(shap_vals, base_vals, class_idx)

    return {
        "shap_values": {name: float(v) for name, v in zip(feature_names, vals)},
        "base_value": base,
        "model_type": "linear",
    }
