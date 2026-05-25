"""
Input schema validation service for ML models.

Used by:
- POST /models/{name}/{version}/validate-input
- POST /predict with ?strict_validation=true
"""

from typing import Any, Dict, List, Optional, Tuple

from src.schemas.model import InputValidationError, InputValidationWarning


def validate_input_features(
    input_features: Dict[str, Any],
    expected_features: List[str],
) -> Tuple[List[InputValidationError], List[InputValidationWarning]]:
    """
    Validate input features against the list of expected features.

    Returns (errors, warnings):
    - errors  : missing or unexpected features — blocking in strict mode
    - warnings: string values coercible to float — informational
    """
    errors: List[InputValidationError] = []
    warnings: List[InputValidationWarning] = []

    input_keys = set(input_features.keys())
    expected_set = set(expected_features)

    for feature in sorted(expected_set - input_keys):
        errors.append(InputValidationError(type="missing_feature", feature=feature))

    for feature in sorted(input_keys - expected_set):
        errors.append(InputValidationError(type="unexpected_feature", feature=feature))

    for feature, value in input_features.items():
        if isinstance(value, str):
            try:
                float(value)
                warnings.append(
                    InputValidationWarning(
                        type="type_coercion",
                        feature=feature,
                        from_type="string",
                        to_type="float",
                    )
                )
            except ValueError:
                pass

    return errors, warnings


def resolve_expected_features(
    loaded_model: Any,
    feature_baseline: Optional[Dict[str, Any]],
) -> Optional[List[str]]:
    """
    Resolve the list of expected features from the model or the baseline.

    Priority:
    1. feature_names_in_ from the sklearn model (trained on a pandas DataFrame)
    2. Keys from feature_baseline stored in the DB
    3. None if no source is available
    """
    if loaded_model is not None and hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)
    if feature_baseline:
        return sorted(feature_baseline.keys())
    return None
