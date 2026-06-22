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
    x,
    prediction_result,
    feature_baseline: dict | None,
) -> dict:
    """
    Compute local SHAP values for one observation.

    Parameters
    ----------
    model : sklearn object or Pipeline
    feature_names : ordered list of input feature names
    x : numpy array of shape (1, n_features) or pandas DataFrame
    prediction_result : result of model.predict(x)[0] (to resolve class index)
    feature_baseline : dict {feature: {mean, std, min, max}} from model_metadata.feature_baseline

    Returns
    -------
    dict with keys:
      - shap_values : dict {feature_name: float}
      - base_value  : float
      - model_type  : "tree" | "linear" | "pipeline"
    """
    from sklearn.pipeline import Pipeline as _Pipeline

    if isinstance(model, _Pipeline):
        return _explain_pipeline(model, feature_names, x, prediction_result, feature_baseline)

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
            " Lasso, ElasticNet, SGD. Pipelines wrapping any of these are also supported."
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


def _explain_pipeline(
    model, feature_names: list, x, prediction_result, feature_baseline: dict | None
) -> dict:
    """Handle sklearn Pipeline: transform X through preprocessing steps then compute SHAP
    on the final estimator. OHE-expanded contributions are summed back to original features."""
    import pandas as pd

    final_estimator = model.steps[-1][1]
    final_name = type(final_estimator).__name__

    if final_name not in _TREE_TYPES and final_name not in _LINEAR_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Pipeline's final estimator '{final_name}' is not supported for SHAP. "
                "Supported estimators: trees (RandomForest, GradientBoosting…) and linear models."
            ),
        )

    # Ensure x is a DataFrame so the preprocessor handles mixed types correctly
    if not isinstance(x, pd.DataFrame):
        x_df = pd.DataFrame(
            (
                x
                if hasattr(x, "__len__") and len(np.array(x).shape) == 2
                else np.array(x).reshape(1, -1)
            ),
            columns=feature_names,
        )
    else:
        x_df = x

    # Transform through all steps except the last (the estimator)
    preprocessor = model[:-1]
    x_transformed = preprocessor.transform(x_df)

    # Recover the expanded feature names after transformation
    try:
        transformed_names = list(preprocessor.get_feature_names_out())
    except Exception:
        transformed_names = [f"f{i}" for i in range(x_transformed.shape[1])]

    # Compute SHAP on the final estimator with transformed features
    if final_name in _TREE_TYPES:
        result = _explain_tree(final_estimator, transformed_names, x_transformed, prediction_result)
    else:
        result = _explain_linear(
            final_estimator,
            transformed_names,
            x_transformed,
            prediction_result,
            feature_baseline=None,
        )

    # Aggregate OHE-expanded SHAP values back to the original input feature names
    result["shap_values"] = _aggregate_pipeline_shap(result["shap_values"], feature_names)
    result["model_type"] = "pipeline"
    return result


def _aggregate_pipeline_shap(shap_dict: dict, original_names: list) -> dict:
    """Sum SHAP contributions for one-hot encoded features back to their source feature.

    sklearn ColumnTransformer produces names like ``num__age`` or ``cat__pclass_1st``.
    The part after ``__`` is matched against original feature names:
    exact match for numericals, prefix match for OHE categoricals.
    """
    # Sort originals longest-first to avoid partial collisions (e.g. "class" vs "pclass")
    sorted_orig = sorted(original_names, key=len, reverse=True)
    aggregated: dict = {}

    for trans_feat, shap_val in shap_dict.items():
        # Strip transformer prefix: "num__age" → "age", "cat__pclass_1st" → "pclass_1st"
        core = trans_feat.split("__", 1)[-1] if "__" in trans_feat else trans_feat

        matched = None
        if core in original_names:
            matched = core
        else:
            for orig in sorted_orig:
                if core.startswith(orig + "_") or core == orig:
                    matched = orig
                    break

        key = matched if matched else trans_feat
        aggregated[key] = aggregated.get(key, 0.0) + shap_val

    return aggregated
