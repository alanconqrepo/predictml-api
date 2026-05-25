"""
SHAP explanation service for predictions (local feature importances).

Supports:
  - TreeExplainer: RandomForest, GradientBoosting, DecisionTree, ExtraTrees, HistGradientBoosting
  - LinearExplainer: LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGD
"""

import numpy as np
import shap
from fastapi import HTTPException, status

# sklearn types compatible with shap.TreeExplainer (no background data required)
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

# sklearn types compatible with shap.LinearExplainer (requires background data)
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
    Compute local SHAP values for one observation.

    Parameters
    ----------
    model : sklearn object
    feature_names : ordered list of feature names
    x : numpy array of shape (1, n_features)
    prediction_result : result of model.predict(x)[0] (to resolve class index)
    feature_baseline : dict {feature: {mean, std, min, max}} from model_metadata.feature_baseline

    Returns
    -------
    dict with keys:
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
            f"Model type '{model_class}' not supported for SHAP explanation. "
            "Supported types — trees: RandomForest, GradientBoosting, DecisionTree, ExtraTrees,"
            " HistGradientBoosting. Linear: LogisticRegression, LinearRegression, Ridge,"
            " Lasso, ElasticNet, SGD."
        ),
    )


def _resolve_class_index(model, prediction_result) -> int:
    """Return the index of the predicted class in model.classes_, or 0 by default."""
    if hasattr(model, "classes_"):
        classes = model.classes_.tolist()
        if prediction_result in classes:
            return classes.index(prediction_result)
    return 0


def _extract_vals_and_base(shap_vals, base_vals, class_idx: int):
    """
    Extract SHAP values and base value for a given class.

    Compatible with the different output formats depending on the SHAP version:
      - list[ndarray]         → multi-class classifier (one array per class)
      - ndarray 3D            → (n_samples, n_features, n_classes) — recent SHAP format
      - ndarray 2D            → (n_samples, n_features) — regressor or compressed binary
    base_vals can be a scalar, a 0D array, or a 1D array (one value per class).
    """
    b = np.asarray(base_vals)

    if isinstance(shap_vals, list):
        # List format: list[array(n_samples, n_features)], one per class
        vals = shap_vals[class_idx][0]
        base = float(b[class_idx]) if b.ndim > 0 and len(b) > 1 else float(b.ravel()[0])
    elif shap_vals.ndim == 3:
        # 3D format: (n_samples, n_features, n_classes)
        vals = shap_vals[0, :, class_idx]
        base = float(b[class_idx]) if b.ndim > 0 and len(b) > class_idx else float(b.ravel()[0])
    else:
        # 2D format: (n_samples, n_features) — regressor or binary classifier
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
    # Build background data: mean of training features, or zeros
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
