"""
Service de validation du schéma d'entrée des modèles ML.

Utilisé par :
- POST /models/{name}/{version}/validate-input
- POST /predict avec ?strict_validation=true
"""

from typing import Any, Dict, List, Optional, Tuple

from src.schemas.model import InputValidationError, InputValidationWarning


def validate_input_features(
    input_features: Dict[str, Any],
    expected_features: List[str],
) -> Tuple[List[InputValidationError], List[InputValidationWarning]]:
    """
    Valide les features d'entrée contre la liste des features attendues.

    Retourne (errors, warnings) :
    - errors  : features manquantes ou inattendues — bloquant en mode strict
    - warnings : valeurs string coercibles en float — informatif
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
    Résout la liste des features attendues depuis le modèle ou le baseline.

    Priorité :
    1. feature_names_in_ du modèle sklearn (entraîné sur DataFrame pandas)
    2. Clés de feature_baseline stockées en DB
    3. None si aucune source disponible
    """
    if loaded_model is not None and hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)
    if feature_baseline:
        return sorted(feature_baseline.keys())
    return None
