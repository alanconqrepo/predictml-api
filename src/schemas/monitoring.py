"""
Pydantic schemas for the model supervision dashboard
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MonitoringPeriod(BaseModel):
    start: datetime
    end: datetime


class ModelHealthSummary(BaseModel):
    model_name: str
    versions: list[str]
    deployment_modes: dict[str, Optional[str]]  # version → mode
    total_predictions: int  # non-shadow only
    shadow_predictions: int
    error_count: int
    error_rate: float
    avg_latency_ms: Optional[float]
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    feature_drift_status: str  # ok / warning / critical / no_baseline / no_data
    performance_drift_status: str  # ok / warning / critical / no_data
    output_drift_status: str  # ok / warning / critical / no_baseline / no_data
    last_prediction: Optional[datetime]
    health_status: str  # worst-of: ok / warning / critical


class GlobalStats(BaseModel):
    total_predictions: int
    total_shadow: int
    total_errors: int
    error_rate: float
    avg_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    active_models: int
    models_critical: int
    models_warning: int
    models_ok: int


class GlobalDashboard(BaseModel):
    period: MonitoringPeriod
    global_stats: GlobalStats
    models: list[ModelHealthSummary]


class TimeseriesPoint(BaseModel):
    date: str  # "YYYY-MM-DD"
    total_predictions: int
    error_count: int
    error_rate: float
    avg_latency_ms: Optional[float]
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]


class VersionStats(BaseModel):
    version: str
    deployment_mode: Optional[str]
    traffic_weight: Optional[float]
    total_predictions: int
    shadow_predictions: int
    error_count: int
    error_rate: float
    avg_latency_ms: Optional[float]
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]


class ModelDetailDashboard(BaseModel):
    model_name: str
    period: MonitoringPeriod
    per_version_stats: list[VersionStats]
    timeseries: list[TimeseriesPoint]
    performance_by_day: list[dict]  # [{date, accuracy/mae, matched_count}]
    feature_drift: dict  # raw response from drift service
    ab_comparison: Optional[dict]  # raw ab-compare response
    recent_errors: list[str]  # last distinct error messages (max 5)
