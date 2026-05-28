"""
Pydantic schemas for predictions
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")


class PredictionInput(BaseModel):
    """Input data for a prediction"""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Dict format",
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
                    "summary": "Dict format with id_obs",
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
        description="Name of the model to use (without .joblib extension)",
        json_schema_extra={"example": "iris_model"},
    )
    model_version: Optional[str] = Field(
        None,
        description=(
            "Model version (e.g. '1.0.0'). "
            "If absent, uses the is_production=True version; "
            "otherwise, falls back to the most recent version."
        ),
        json_schema_extra={"example": "1.0.0"},
    )
    id_obs: Optional[str] = Field(
        None,
        description="Observation identifier (stored in the predictions table)",
        json_schema_extra={"example": "obs-001"},
    )
    timestamp: Optional[datetime] = Field(
        None,
        description=(
            "Forced prediction timestamp (UTC). "
            "If absent, the server uses the current time. "
            "Allows injecting historical or future predictions."
        ),
        json_schema_extra={"example": "2025-01-15T08:32:00"},
    )
    features: Dict[str, Union[float, int, str]] = Field(
        ...,
        description=(
            "Prediction features as a named dict "
            '{"feature1": value, "feature2": value, ...}. '
            "The model must expose feature_names_in_ (trained with a pandas DataFrame)."
        ),
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError("Invalid model name (allowed characters: a-z A-Z 0-9 _ -)")
        return v

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _VERSION_RE.match(v):
            raise ValueError("Invalid version (expected format: X.Y or X.Y.Z)")
        return v


class PredictionOutput(BaseModel):
    """Result of a prediction"""

    id: Optional[int] = Field(
        None, description="Database ID of the logged prediction (None if store=false)"
    )
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    id_obs: Optional[str] = Field(None, description="Observation identifier (if provided)")
    prediction: float | int | str = Field(..., description="Model prediction")
    probability: Optional[List[float]] = Field(
        None, description="Per-class probabilities (if available)"
    )
    low_confidence: Optional[bool] = Field(
        None,
        description=(
            "True if the max probability is below the model's confidence threshold. "
            "None if the model has no threshold configured or does not support predict_proba."
        ),
    )
    selected_version: Optional[str] = Field(
        None,
        description=(
            "Version selected by A/B routing (only if model_version was not "
            "specified in the request and an A/B test is active)."
        ),
    )
    shap_values: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "SHAP values per feature (only if ?explain=true). "
            "None if the parameter is not enabled or the model type is not supported."
        ),
    )
    shap_base_value: Optional[float] = Field(
        None,
        description="SHAP base value E[f(X)] (only if ?explain=true and model is supported).",
    )


class PredictionResponse(BaseModel):
    """A logged prediction"""

    id: int
    model_name: str
    model_version: Optional[str]
    id_obs: Optional[str]
    input_features: Any
    prediction_result: Any
    probabilities: Optional[List[float]]
    max_confidence: Optional[float] = None
    response_time_ms: float
    timestamp: datetime
    status: str
    error_message: Optional[str]
    username: Optional[str]  # from the User relation
    is_shadow: bool = False  # True if shadow prediction (not returned to the client)

    model_config = ConfigDict(from_attributes=True)


class PredictionsListResponse(BaseModel):
    """Paginated result of the predictions list (id-based cursor)"""

    total: int
    limit: int
    next_cursor: Optional[int]
    predictions: List[PredictionResponse]


class BatchPredictionItem(BaseModel):
    """An input item for a batch prediction"""

    features: Dict[str, Union[float, int, str]] = Field(..., description="Features as a named dict")
    id_obs: Optional[str] = Field(None, description="Observation identifier (optional)")
    timestamp: Optional[datetime] = Field(
        None,
        description=(
            "Forced timestamp for this item (UTC). " "If absent, the server uses the current time."
        ),
    )


class BatchPredictionInput(BaseModel):
    """Input data for a batch prediction"""

    model_name: str = Field(..., description="Name of the model to use (without .joblib extension)")
    model_version: Optional[str] = Field(
        None, description="Model version (e.g. '1.0.0'). If absent, uses is_production=True."
    )
    inputs: List[BatchPredictionItem] = Field(
        ..., min_length=1, description="List of observations to score"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError("Invalid model name (allowed characters: a-z A-Z 0-9 _ -)")
        return v

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _VERSION_RE.match(v):
            raise ValueError("Invalid version (expected format: X.Y or X.Y.Z)")
        return v


class BatchPredictionResultItem(BaseModel):
    """Result of an individual prediction in a batch"""

    id_obs: Optional[str] = Field(None, description="Observation identifier (if provided)")
    prediction: Union[float, int, str] = Field(..., description="Model prediction")
    probability: Optional[List[float]] = Field(
        None, description="Per-class probabilities (if available)"
    )
    low_confidence: Optional[bool] = Field(
        None,
        description=(
            "True if the max probability is below the model's confidence threshold. "
            "None if the model has no threshold configured or does not support predict_proba."
        ),
    )


class BatchPredictionOutput(BaseModel):
    """Result of a batch prediction"""

    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    predictions: List[BatchPredictionResultItem] = Field(
        ..., description="List of results in the same order as the inputs"
    )


class ModelsListResponse(BaseModel):
    """List of available models"""

    models: List[str] = Field(..., description="List of model names")
    count: int = Field(..., description="Number of available models")
    cached: List[str] = Field(..., description="Models currently in cache")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="API status")
    models_available: int = Field(..., description="Number of available models")
    models_cached: int = Field(..., description="Number of cached models")


class ExplainInput(BaseModel):
    """Input data for a SHAP explanation"""

    model_name: str = Field(
        ...,
        description="Name of the model to use (without .joblib extension)",
        json_schema_extra={"example": "iris_model"},
    )
    model_version: Optional[str] = Field(
        None,
        description="Model version (e.g. '1.0.0'). If absent, uses is_production=True.",
        json_schema_extra={"example": "1.0.0"},
    )
    features: Dict[str, Union[float, int, str]] = Field(
        ...,
        description="Features for the explanation as a named dict.",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError("Invalid model name (allowed characters: a-z A-Z 0-9 _ -)")
        return v

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _VERSION_RE.match(v):
            raise ValueError("Invalid version (expected format: X.Y or X.Y.Z)")
        return v


class ExplainOutput(BaseModel):
    """Result of a local SHAP explanation"""

    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    prediction: Union[float, int, str] = Field(
        ..., description="Model prediction for these features"
    )
    shap_values: Dict[str, float] = Field(
        ...,
        description=(
            "SHAP values per feature — contribution of each feature to the prediction. "
            "Positive value = pushes toward predicted class, negative = pushes away."
        ),
    )
    base_value: float = Field(
        ...,
        description="Model base value E[f(X)] — average prediction over the training data.",
    )
    model_type: str = Field(..., description="SHAP explainer type used: 'tree' or 'linear'.")


class RootResponse(BaseModel):
    """Response from the root endpoint"""

    message: str
    status: str
    models_available: List[str]
    models_count: int
    models_cached: List[str]


class PredictionStatsItem(BaseModel):
    """Aggregated prediction statistics for a model"""

    model_name: str
    total_predictions: int
    error_count: int
    error_rate: float
    avg_response_time_ms: Optional[float] = None
    p50_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None


class PredictionStatsResponse(BaseModel):
    """Response for GET /predictions/stats"""

    days: int
    model_name: Optional[str] = None
    stats: List[PredictionStatsItem]


class PurgeResponse(BaseModel):
    """Result of a prediction purge (GDPR data retention)"""

    dry_run: bool = Field(..., description="True if simulation without actual deletion")
    deleted_count: int = Field(
        ..., description="Number of predictions deleted (or to be deleted in dry_run)"
    )
    deleted_observed_results_count: int = Field(
        0,
        description="Number of observed_results deleted in cascade (0 in dry_run)",
    )
    oldest_remaining: Optional[datetime] = Field(
        None,
        description="Timestamp of the oldest remaining prediction after the purge",
    )
    models_affected: List[str] = Field(..., description="List of models affected by the purge")
    linked_observed_results_count: int = Field(
        ...,
        description=(
            "Number of observed_results linked to predictions to be purged "
            "(in dry_run: estimate; after deletion: matches deleted_observed_results_count)."
        ),
    )


class UnlabeledPredictionItem(BaseModel):
    """A successfully scored prediction that has no associated observed result."""

    id: int = Field(..., description="Prediction database ID")
    id_obs: Optional[str] = Field(None, description="Business observation identifier")
    model_name: str = Field(..., description="Model name")
    model_version: Optional[str] = Field(None, description="Model version")
    prediction_result: Any = Field(..., description="Model prediction")
    max_confidence: Optional[float] = Field(
        None, description="Max probability (None for regression models)"
    )
    timestamp: datetime = Field(..., description="Prediction timestamp (UTC)")


class UnlabeledPredictionsResponse(BaseModel):
    """Response for GET /predictions/unlabeled"""

    total_unlabeled: int = Field(
        ..., description="Total number of predictions without an observed result"
    )
    returned: int = Field(..., description="Number of predictions returned in this response")
    strategy: str = Field(..., description="Sampling strategy: uncertainty, recent, random")
    model_name: Optional[str] = Field(None, description="Model name filter applied (None = all)")
    model_version: Optional[str] = Field(None, description="Version filter applied")
    predictions: List[UnlabeledPredictionItem] = Field(
        ..., description="Predictions to annotate, ordered by the selected strategy"
    )


class AnomalyFeatureDetail(BaseModel):
    """Detail of an anomalous feature in a prediction"""

    value: float = Field(..., description="Observed value of the feature")
    z_score: float = Field(..., description="Z-score: |value - baseline_mean| / baseline_std")
    baseline_mean: float = Field(..., description="Baseline mean")
    baseline_std: float = Field(..., description="Baseline standard deviation")


class AnomalyPredictionEntry(BaseModel):
    """A prediction containing at least one anomalous feature"""

    prediction_id: int = Field(..., description="Prediction identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    prediction_result: Any = Field(..., description="Prediction result")
    max_confidence: Optional[float] = Field(None, description="Max probability (if available)")
    id_obs: Optional[str] = Field(
        None,
        description="Business observation identifier (join key with observed_results)",
    )
    ground_truth: Optional[Any] = Field(
        None, description="Actually observed result (if available via observed_results)"
    )
    anomalous_features: Dict[str, AnomalyFeatureDetail] = Field(
        ..., description="Features whose z-score exceeds the threshold"
    )


class AnomaliesResponse(BaseModel):
    """Response for GET /predictions/anomalies"""

    model_name: str = Field(..., description="Name of the analyzed model")
    period_days: int = Field(..., description="Analyzed time window (days)")
    z_threshold: float = Field(..., description="Z-score threshold used")
    total_checked: int = Field(..., description="Number of predictions analyzed")
    anomalous_count: int = Field(..., description="Number of predictions with anomalous features")
    anomaly_rate: float = Field(
        ..., description="Rate of anomalous predictions (anomalous / total)"
    )
    predictions: List[AnomalyPredictionEntry] = Field(
        ..., description="Predictions with anomalous features"
    )
    error: Optional[str] = Field(
        None,
        description="Error code if analysis is impossible (e.g. 'no_baseline')",
    )
