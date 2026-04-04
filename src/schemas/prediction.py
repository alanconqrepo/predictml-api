"""
Schémas Pydantic pour les prédictions
"""
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    """Données d'entrée pour une prédiction"""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Format liste (ancien)",
                    "value": {
                        "model_name": "iris_model",
                        "features": [5.1, 3.5, 1.4, 0.2]
                    }
                },
                {
                    "summary": "Format dict avec id_obs (nouveau)",
                    "value": {
                        "model_name": "iris_model",
                        "id_obs": "obs-001",
                        "features": {
                            "sepal_length": 5.1,
                            "sepal_width": 3.5,
                            "petal_length": 1.4,
                            "petal_width": 0.2
                        }
                    }
                }
            ]
        }
    )

    model_name: str = Field(
        ...,
        description="Nom du modèle à utiliser (sans extension .pkl)",
        json_schema_extra={"example": "iris_model"}
    )
    id_obs: Optional[str] = Field(
        None,
        description="Identifiant de l'observation (stocké dans la table predictions)",
        json_schema_extra={"example": "obs-001"}
    )
    features: Union[
        List[Union[float, int, str]],
        Dict[str, Union[float, int, str]]
    ] = Field(
        ...,
        description=(
            "Features pour la prédiction. "
            "Deux formats acceptés : "
            "(1) liste ordonnée [5.1, 3.5, 1.4, 0.2] — l'ordre doit correspondre au modèle ; "
            "(2) dict nommé {\"sepal_length\": 5.1, ...} — le modèle doit exposer feature_names_in_."
        )
    )


class PredictionOutput(BaseModel):
    """Résultat d'une prédiction"""

    model_name: str = Field(..., description="Nom du modèle utilisé")
    id_obs: Optional[str] = Field(None, description="Identifiant de l'observation (si fourni)")
    prediction: float | int | str = Field(..., description="Prédiction du modèle")
    probability: Optional[List[float]] = Field(
        None,
        description="Probabilités par classe (si disponible)"
    )


class ModelsListResponse(BaseModel):
    """Liste des modèles disponibles"""

    models: List[str] = Field(..., description="Liste des noms de modèles")
    count: int = Field(..., description="Nombre de modèles disponibles")
    cached: List[str] = Field(..., description="Modèles actuellement en cache")


class HealthResponse(BaseModel):
    """Réponse du health check"""

    status: str = Field(..., description="Statut de l'API")
    models_available: int = Field(..., description="Nombre de modèles disponibles")
    models_cached: int = Field(..., description="Nombre de modèles en cache")


class RootResponse(BaseModel):
    """Réponse de l'endpoint racine"""

    message: str
    status: str
    models_available: List[str]
    models_count: int
    models_cached: List[str]
