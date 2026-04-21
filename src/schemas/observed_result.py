"""
Schémas Pydantic pour les résultats observés
"""

from datetime import datetime
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ObservedResultInput(BaseModel):
    """Un résultat observé à soumettre"""

    id_obs: str = Field(..., description="Identifiant de l'observation")
    model_name: str = Field(..., description="Nom du modèle concerné")
    date_time: datetime = Field(..., description="Horodatage de l'observation (ISO 8601)")
    observed_result: Union[float, int, str] = Field(
        ..., description="Résultat réellement observé (même type que prediction_result)"
    )


class ObservedResultsUpsertRequest(BaseModel):
    """Corps de la requête POST /observed-results"""

    data: List[ObservedResultInput] = Field(
        ...,
        description="Liste des résultats observés à insérer ou écraser",
        min_length=1,
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [
                    {
                        "id_obs": "obs-001",
                        "model_name": "iris_model",
                        "date_time": "2024-06-01T12:00:00",
                        "observed_result": 0,
                    },
                    {
                        "id_obs": "obs-002",
                        "model_name": "iris_model",
                        "date_time": "2024-06-01T12:05:00",
                        "observed_result": 2,
                    },
                ]
            }
        }
    )


class ObservedResultResponse(BaseModel):
    """Un résultat observé retourné par l'API"""

    id: int
    id_obs: str
    model_name: str
    observed_result: Any
    date_time: datetime
    username: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class ObservedResultsUpsertResponse(BaseModel):
    """Réponse après upsert"""

    upserted: int = Field(..., description="Nombre de lignes insérées ou mises à jour")


class ObservedResultsListResponse(BaseModel):
    """Résultat paginé de la liste des résultats observés"""

    total: int
    limit: int
    offset: int
    results: List[ObservedResultResponse]


class ObservedResultsStatsVersionItem(BaseModel):
    version: str
    predictions: int
    labeled: int
    coverage: float


class ObservedResultsStatsModelItem(BaseModel):
    model_name: str
    predictions: int
    labeled: int
    coverage: float


class ObservedResultsStatsResponse(BaseModel):
    model_name: Optional[str] = Field(None, description="Nom du modèle, null si global")
    total_predictions: int
    labeled_count: int
    coverage_rate: float
    oldest_label: Optional[datetime] = None
    newest_label: Optional[datetime] = None
    by_version: Optional[List[ObservedResultsStatsVersionItem]] = None
    by_model: Optional[List[ObservedResultsStatsModelItem]] = None


class CSVParseError(BaseModel):
    row: int = Field(..., description="Numéro de ligne dans le CSV (en-tête = 1)")
    reason: str = Field(..., description="Raison du rejet")


class CSVUploadResponse(BaseModel):
    upserted: int = Field(..., description="Nombre de lignes importées avec succès")
    skipped_rows: int = Field(..., description="Nombre de lignes ignorées pour erreur")
    parse_errors: List[CSVParseError] = Field(default_factory=list)
    filename: str = Field(..., description="Nom du fichier uploadé")
