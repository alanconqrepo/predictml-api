"""
Tests de validation des schémas Pydantic.
Vérifie que PredictionInput accepte uniquement le format dict {feature_name: value}
et rejette les listes.
"""
import pytest
from pydantic import ValidationError
from src.schemas.prediction import PredictionInput, PredictionOutput


# ── Format dict (seul format accepté) ────────────────────────────────────────

def test_prediction_input_accepts_dict_features():
    """Le format dict {nom_feature: valeur} est accepté"""
    obj = PredictionInput(
        model_name="iris_model",
        features={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    )
    assert isinstance(obj.features, dict)
    assert obj.features["sepal_length"] == 5.1


def test_prediction_input_accepts_dict_with_floats():
    """Dict avec valeurs float accepté"""
    obj = PredictionInput(
        model_name="iris_model",
        features={"f1": 5.1, "f2": 3.5}
    )
    assert obj.features["f1"] == 5.1


def test_prediction_input_accepts_dict_with_ints():
    """Dict avec valeurs entières accepté"""
    obj = PredictionInput(
        model_name="iris_model",
        features={"f1": 5, "f2": 3}
    )
    assert obj.features["f1"] == 5


def test_prediction_input_accepts_dict_with_strings():
    """Dict avec valeurs string accepté (features catégorielles)"""
    obj = PredictionInput(
        model_name="loan_model",
        features={"status": "employed", "education": "bachelor"}
    )
    assert obj.features["status"] == "employed"


def test_prediction_input_accepts_dict_with_mixed_types():
    """Dict avec valeurs mixtes (float + str) accepté"""
    obj = PredictionInput(
        model_name="loan_model",
        features={"age": 35, "income": 65000.0, "status": "employed"}
    )
    assert obj.features["status"] == "employed"
    assert obj.features["age"] == 35


def test_prediction_input_rejects_dict_with_non_scalar_values():
    """Dict avec valeurs non scalaires (ex: liste) doit lever une ValidationError"""
    with pytest.raises(ValidationError):
        PredictionInput(model_name="test", features={"a": [1, 2, 3]})


# ── Rejet du format liste ─────────────────────────────────────────────────────

def test_prediction_input_rejects_list_features():
    """Une liste de features doit lever une ValidationError"""
    with pytest.raises(ValidationError):
        PredictionInput(model_name="iris_model", features=[5.1, 3.5, 1.4, 0.2])


def test_prediction_input_rejects_list_of_ints():
    """Une liste d'entiers doit lever une ValidationError"""
    with pytest.raises(ValidationError):
        PredictionInput(model_name="iris_model", features=[5, 3, 1, 0])


def test_prediction_input_rejects_list_of_strings():
    """Une liste de strings doit lever une ValidationError"""
    with pytest.raises(ValidationError):
        PredictionInput(model_name="loan_model", features=["employed", "bachelor", "good"])


def test_prediction_input_rejects_mixed_list():
    """Une liste mixte (floats + strings) doit lever une ValidationError"""
    with pytest.raises(ValidationError):
        PredictionInput(
            model_name="loan_model",
            features=[35, 65000, 15000, "employed", "bachelor", "good"]
        )


# ── id_obs ────────────────────────────────────────────────────────────────────

def test_prediction_input_id_obs_optional():
    """id_obs est optionnel (None par défaut)"""
    obj = PredictionInput(model_name="iris_model", features={"f1": 5.1})
    assert obj.id_obs is None


def test_prediction_input_id_obs_accepted():
    """id_obs est accepté quand fourni"""
    obj = PredictionInput(
        model_name="iris_model",
        id_obs="obs-001",
        features={"sepal_length": 5.1, "sepal_width": 3.5}
    )
    assert obj.id_obs == "obs-001"


def test_prediction_input_id_obs_with_dict_features():
    """id_obs fonctionne avec le format dict"""
    obj = PredictionInput(
        model_name="iris_model",
        id_obs="patient-42",
        features={"sepal_length": 5.1, "sepal_width": 3.5}
    )
    assert obj.id_obs == "patient-42"
    assert isinstance(obj.features, dict)


# ── PredictionOutput ──────────────────────────────────────────────────────────

def test_prediction_output_includes_id_obs():
    """PredictionOutput expose id_obs"""
    out = PredictionOutput(model_name="iris_model", model_version="1.0.0", id_obs="obs-001", prediction=1)
    assert out.id_obs == "obs-001"


def test_prediction_output_id_obs_nullable():
    """id_obs peut être None dans la réponse"""
    out = PredictionOutput(model_name="iris_model", model_version="1.0.0", id_obs=None, prediction=1)
    assert out.id_obs is None


def test_prediction_output_includes_model_version():
    """PredictionOutput expose model_version"""
    out = PredictionOutput(model_name="iris_model", model_version="2.0.0", prediction=1)
    assert out.model_version == "2.0.0"


def test_prediction_input_model_version_optional():
    """model_version est optionnel dans PredictionInput"""
    inp = PredictionInput(model_name="iris_model", features={"f1": 1.0, "f2": 2.0})
    assert inp.model_version is None


def test_prediction_input_model_version_accepted():
    """model_version fourni est bien conservé"""
    inp = PredictionInput(model_name="iris_model", model_version="2.0.0", features={"f1": 1.0})
    assert inp.model_version == "2.0.0"


# ── Champs requis ─────────────────────────────────────────────────────────────

def test_prediction_input_rejects_missing_features():
    """features est requis"""
    with pytest.raises(ValidationError):
        PredictionInput(model_name="test")


def test_prediction_input_rejects_missing_model_name():
    """model_name est requis"""
    with pytest.raises(ValidationError):
        PredictionInput(features={"f1": 1.0})
