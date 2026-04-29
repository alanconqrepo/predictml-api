"""
Schémas Pydantic pour les prédictions
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    """Données d'entrée pour une prédiction"""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Format dict",
                    "value": {
                        "model_name": "iris_model",
                        "features": {
                            "sepal_length": 5.1,
                            "sepal_width": 3.5,
                            "petal_length": 1.4,
                            "petal_width": 0.2,
                        },
                    },
                },
                {
                    "summary": "Format dict avec id_obs",
                    "value": {
                        "model_name": "iris_model",
                        "id_obs": "obs-001",
                        "features": {
                            "sepal_length": 5.1,
                            "sepal_width": 3.5,
                            "petal_length": 1.4,
                            "petal_width": 0.2,
                        },
                    },
                },
            ]
        }
    )

    model_name: str = Field(
        ...,
        description="Nom du modèle à utiliser (sans extension .pkl)",
        json_schema_extra={"example": "iris_model"},
    )
    model_version: Optional[str] = Field(
        None,
        description=(
            "Version du modèle (ex: '1.0.0'). "
            "Si absent, utilise la version is_production=True ; "
            "à défaut, la version la plus récente."
        ),
        json_schema_extra={"example": "1.0.0"},
    )
    id_obs: Optional[str] = Field(
        None,
        description="Identifiant de l'observation (stocké dans la table predictions)",
        json_schema_extra={"example": "obs-001"},
    )
    features: Dict[str, Union[float, int, str]] = Field(
        ...,
        description=(
            "Features pour la prédiction sous forme de dict nommé "
            '{"feature1": valeur, "feature2": valeur, ...}. '
            "Le modèle doit exposer feature_names_in_ (entraîné avec un DataFrame pandas)."
        ),
    )


class PredictionOutput(BaseModel):
    """Résultat d'une prédiction"""

    model_name: str = Field(..., description="Nom du modèle utilisé")
    model_version: str = Field(..., description="Version du modèle utilisée")
    id_obs: Optional[str] = Field(None, description="Identifiant de l'observation (si fourni)")
    prediction: float | int | str = Field(..., description="Prédiction du modèle")
    probability: Optional[List[float]] = Field(
        None, description="Probabilités par classe (si disponible)"
    )
    low_confidence: Optional[bool] = Field(
        None,
        description=(
            "True si la probabilité max est en dessous du seuil de confiance du modèle. "
            "None si le modèle n'a pas de seuil configuré ou ne supporte pas predict_proba."
        ),
    )
    selected_version: Optional[str] = Field(
        None,
        description=(
            "Version sélectionnée par le routage A/B (uniquement si model_version n'était pas "
            "spécifié dans la requête et qu'un test A/B est actif)."
        ),
    )
    shap_values: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Valeurs SHAP par feature (uniquement si ?explain=true). "
            "None si le paramètre n'est pas activé ou si le type de modèle n'est pas supporté."
        ),
    )
    shap_base_value: Optional[float] = Field(
        None,
        description="Valeur de base SHAP E[f(X)] (uniquement si ?explain=true et modèle supporté).",
    )


class PredictionResponse(BaseModel):
    """Une prédiction loggée"""

    id: int
    model_name: str
    model_version: Optional[str]
    id_obs: Optional[str]
    input_features: Any
    prediction_result: Any
    probabilities: Optional[List[float]]
    response_time_ms: float
    timestamp: datetime
    status: str
    error_message: Optional[str]
    username: Optional[str]  # depuis la relation User
    is_shadow: bool = False  # True si prédiction shadow (non retournée au client)

    model_config = ConfigDict(from_attributes=True)


class PredictionsListResponse(BaseModel):
    """Résultat paginé de la liste des prédictions (curseur basé sur l'id)"""

    total: int
    limit: int
    next_cursor: Optional[int]
    predictions: List[PredictionResponse]


class BatchPredictionItem(BaseModel):
    """Un item d'entrée pour une prédiction batch"""

    features: Dict[str, Union[float, int, str]] = Field(
        ..., description="Features sous forme de dict nommé"
    )
    id_obs: Optional[str] = Field(None, description="Identifiant de l'observation (optionnel)")


class BatchPredictionInput(BaseModel):
    """Données d'entrée pour une prédiction batch"""

    model_name: str = Field(..., description="Nom du modèle à utiliser (sans extension .pkl)")
    model_version: Optional[str] = Field(
        None, description="Version du modèle (ex: '1.0.0'). Si absent, utilise is_production=True."
    )
    inputs: List[BatchPredictionItem] = Field(
        ..., min_length=1, description="Liste d'observations à scorer"
    )


class BatchPredictionResultItem(BaseModel):
    """Résultat d'une prédiction individuelle dans un batch"""

    id_obs: Optional[str] = Field(None, description="Identifiant de l'observation (si fourni)")
    prediction: Union[float, int, str] = Field(..., description="Prédiction du modèle")
    probability: Optional[List[float]] = Field(
        None, description="Probabilités par classe (si disponible)"
    )
    low_confidence: Optional[bool] = Field(
        None,
        description=(
            "True si la probabilité max est en dessous du seuil de confiance du modèle. "
            "None si le modèle n'a pas de seuil configuré ou ne supporte pas predict_proba."
        ),
    )


class BatchPredictionOutput(BaseModel):
    """Résultat d'une prédiction batch"""

    model_name: str = Field(..., description="Nom du modèle utilisé")
    model_version: str = Field(..., description="Version du modèle utilisée")
    predictions: List[BatchPredictionResultItem] = Field(
        ..., description="Liste des résultats dans le même ordre que les inputs"
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


class ExplainInput(BaseModel):
    """Données d'entrée pour une explication SHAP"""

    model_name: str = Field(
        ...,
        description="Nom du modèle à utiliser (sans extension .pkl)",
        json_schema_extra={"example": "iris_model"},
    )
    model_version: Optional[str] = Field(
        None,
        description="Version du modèle (ex: '1.0.0'). Si absent, utilise is_production=True.",
        json_schema_extra={"example": "1.0.0"},
    )
    features: Dict[str, Union[float, int, str]] = Field(
        ...,
        description="Features pour l'explication sous forme de dict nommé.",
    )


class ExplainOutput(BaseModel):
    """Résultat d'une explication SHAP locale"""

    model_name: str = Field(..., description="Nom du modèle utilisé")
    model_version: str = Field(..., description="Version du modèle utilisée")
    prediction: Union[float, int, str] = Field(
        ..., description="Prédiction du modèle pour ces features"
    )
    shap_values: Dict[str, float] = Field(
        ...,
        description=(
            "Valeurs SHAP par feature — contribution de chaque feature à la prédiction. "
            "Valeur positive = pousse vers la classe prédite, négative = pousse à l'opposé."
        ),
    )
    base_value: float = Field(
        ...,
        description="Valeur de base du modèle E[f(X)] — prédiction moyenne sur les données d'entraînement.",
    )
    model_type: str = Field(..., description="Type d'explainer SHAP utilisé : 'tree' ou 'linear'.")


class RootResponse(BaseModel):
    """Réponse de l'endpoint racine"""

    message: str
    status: str
    models_available: List[str]
    models_count: int
    models_cached: List[str]


class PredictionStatsItem(BaseModel):
    """Statistiques agrégées des prédictions pour un modèle"""

    model_name: str
    total_predictions: int
    error_count: int
    error_rate: float
    avg_response_time_ms: Optional[float] = None
    p50_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None


class PredictionStatsResponse(BaseModel):
    """Réponse de GET /predictions/stats"""

    days: int
    model_name: Optional[str] = None
    stats: List[PredictionStatsItem]


class PurgeResponse(BaseModel):
    """Résultat d'une purge de prédictions (rétention RGPD)"""

    dry_run: bool = Field(..., description="True si simulation sans suppression réelle")
    deleted_count: int = Field(
        ..., description="Nombre de prédictions supprimées (ou à supprimer en dry_run)"
    )
    oldest_remaining: Optional[datetime] = Field(
        None,
        description="Timestamp de la prédiction la plus ancienne restante après la purge",
    )
    models_affected: List[str] = Field(..., description="Liste des modèles concernés par la purge")
    linked_observed_results_count: int = Field(
        ...,
        description=(
            "Nombre de prédictions supprimées liées à des observed_results. "
            "Avertissement si > 0 : des données de performance historiques seront perdues."
        ),
    )


class AnomalyFeatureDetail(BaseModel):
    """Détail d'une feature aberrante dans une prédiction"""

    value: float = Field(..., description="Valeur observée de la feature")
    z_score: float = Field(..., description="Z-score : |value - baseline_mean| / baseline_std")
    baseline_mean: float = Field(..., description="Moyenne de la baseline")
    baseline_std: float = Field(..., description="Écart-type de la baseline")


class AnomalyPredictionEntry(BaseModel):
    """Une prédiction contenant au moins une feature aberrante"""

    prediction_id: int = Field(..., description="Identifiant de la prédiction")
    timestamp: datetime = Field(..., description="Horodatage de la prédiction")
    prediction_result: Any = Field(..., description="Résultat de la prédiction")
    max_confidence: Optional[float] = Field(None, description="Probabilité max (si disponible)")
    anomalous_features: Dict[str, AnomalyFeatureDetail] = Field(
        ..., description="Features dont le z-score dépasse le seuil"
    )


class AnomaliesResponse(BaseModel):
    """Réponse de GET /predictions/anomalies"""

    model_name: str = Field(..., description="Nom du modèle analysé")
    period_days: int = Field(..., description="Fenêtre temporelle analysée (jours)")
    z_threshold: float = Field(..., description="Seuil z-score utilisé")
    total_checked: int = Field(..., description="Nombre de prédictions analysées")
    anomalous_count: int = Field(..., description="Nombre de prédictions avec features aberrantes")
    anomaly_rate: float = Field(..., description="Taux de prédictions anomales (anomalous / total)")
    predictions: List[AnomalyPredictionEntry] = Field(
        ..., description="Prédictions avec features aberrantes"
    )
    error: Optional[str] = Field(
        None,
        description="Code d'erreur si l'analyse est impossible (ex : 'no_baseline')",
    )
