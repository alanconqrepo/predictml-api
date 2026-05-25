"""
Pydantic schemas for observed results
"""

from datetime import datetime
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ObservedResultInput(BaseModel):
    """An observed result to submit"""

    id_obs: str = Field(..., description="Observation identifier")
    model_name: str = Field(..., description="Name of the model concerned")
    date_time: datetime = Field(..., description="Observation timestamp (ISO 8601)")
    observed_result: Union[float, int, str] = Field(
        ..., description="Actually observed result (same type as prediction_result)"
    )


class ObservedResultsUpsertRequest(BaseModel):
    """Request body for POST /observed-results"""

    data: List[ObservedResultInput] = Field(
        ...,
        description="List of observed results to insert or overwrite",
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
    """An observed result returned by the API"""

    id: int
    id_obs: str
    model_name: str
    observed_result: Any
    date_time: datetime
    username: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class ObservedResultsUpsertResponse(BaseModel):
    """Response after upsert"""

    upserted: int = Field(..., description="Number of rows inserted or updated")


class ObservedResultsListResponse(BaseModel):
    """Paginated result of the observed results list"""

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
    model_name: Optional[str] = Field(None, description="Model name, null if global")
    total_predictions: int
    labeled_count: int
    coverage_rate: float
    oldest_label: Optional[datetime] = None
    newest_label: Optional[datetime] = None
    by_version: Optional[List[ObservedResultsStatsVersionItem]] = None
    by_model: Optional[List[ObservedResultsStatsModelItem]] = None


class CSVParseError(BaseModel):
    row: int = Field(..., description="Row number in the CSV (header = 1)")
    reason: str = Field(..., description="Reason for rejection")


class CSVUploadResponse(BaseModel):
    upserted: int = Field(..., description="Number of rows successfully imported")
    skipped_rows: int = Field(..., description="Number of rows skipped due to errors")
    parse_errors: List[CSVParseError] = Field(default_factory=list)
    filename: str = Field(..., description="Name of the uploaded file")
