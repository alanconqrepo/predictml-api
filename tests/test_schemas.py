"""
Tests de validation des schémas Pydantic.
Vérifie que PredictionInput accepte float, int, str et les types mixtes,
ainsi que le nouveau format dict {feature_name: value} avec id_obs optionnel.
"""
import pytest
from pydantic import ValidationError
from src.schemas.prediction import PredictionInput, PredictionOutput


# ── Format liste (ancien format) ──────────────────────────────────────────────

def test_prediction_input_accepts_floats():
    """Cas nominal : liste de floats (modèles numériques)"""
    obj = PredictionInput(model_name="iris_model", features=[5.1, 3.5, 1.4, 0.2])
    assert len(obj.features) == 4
    assert obj.features[0] == 5.1


def test_prediction_input_accepts_ints():
    """Entiers acceptés et conservés comme valeurs numériques"""
    obj = PredictionInput(model_name="iris_model", features=[5, 3, 1, 0])
    assert len(obj.features) == 4


def test_prediction_input_accepts_strings():
    """Strings acceptées (features catégorielles, ex: loan_model)"""
    obj = PredictionInput(
        model_name="loan_model",
        features=["employed", "bachelor", "good"]
    )
    assert obj.features[0] == "employed"


def test_prediction_input_accepts_mixed_features():
    """Cas réel loan_model : floats + strings dans la même liste"""
    obj = PredictionInput(
        model_name="loan_model",
        features=[35, 65000, 15000, "employed", "bachelor", "good"]
    )
    assert len(obj.features) == 6
    assert obj.features[3] == "employed"
    assert obj.features[0] == 35


def test_prediction_input_list_element_dict_rejected():
    """Un dict en tant qu'élément d'une liste de features doit lever une ValidationError"""
    with pytest.raises(ValidationError):
        PredictionInput(model_name="test", features=[{"key": "value"}])


# ── Format dict (nouveau format) ──────────────────────────────────────────────

def test_prediction_input_accepts_dict_features():
    """Le format dict {nom_feature: valeur} est accepté"""
    obj = PredictionInput(
        model_name="iris_model",
        features={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    )
    assert isinstance(obj.features, dict)
    assert obj.features["sepal_length"] == 5.1


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


# ── id_obs ────────────────────────────────────────────────────────────────────

def test_prediction_input_id_obs_optional():
    """id_obs est optionnel (None par défaut)"""
    obj = PredictionInput(model_name="iris_model", features=[5.1, 3.5, 1.4, 0.2])
    assert obj.id_obs is None


def test_prediction_input_id_obs_accepted():
    """id_obs est accepté quand fourni"""
    obj = PredictionInput(
        model_name="iris_model",
        id_obs="obs-001",
        features=[5.1, 3.5, 1.4, 0.2]
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
    """id_obs peut être None dans la réponse (ancien format)"""
    out = PredictionOutput(model_name="iris_model", model_version="1.0.0", id_obs=None, prediction=1)
    assert out.id_obs is None


def test_prediction_output_includes_model_version():
    """PredictionOutput expose model_version"""
    out = PredictionOutput(model_name="iris_model", model_version="2.0.0", prediction=1)
    assert out.model_version == "2.0.0"


def test_prediction_input_model_version_optional():
    """model_version est optionnel dans PredictionInput"""
    inp = PredictionInput(model_name="iris_model", features=[1.0, 2.0])
    assert inp.model_version is None


def test_prediction_input_model_version_accepted():
    """model_version fourni est bien conservé"""
    inp = PredictionInput(model_name="iris_model", model_version="2.0.0", features=[1.0, 2.0])
    assert inp.model_version == "2.0.0"


# ── Champs requis ─────────────────────────────────────────────────────────────

def test_prediction_input_rejects_missing_features():
    """features est requis"""
    with pytest.raises(ValidationError):
        PredictionInput(model_name="test")


def test_prediction_input_rejects_missing_model_name():
    """model_name est requis"""
    with pytest.raises(ValidationError):
        PredictionInput(features=[1.0, 2.0])
