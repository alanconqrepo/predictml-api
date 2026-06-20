"""
Pydantic schemas for task_runs (ARQ jobs).
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel


class TaskRunEnqueued(BaseModel):
    """Immediate 202 response after a retrain is enqueued."""

    job_id: UUID
    status: str  # "queued"
    model_name: str
    model_version: str
    new_version: str  # version that will be created on success
    triggered_by: str
    enqueued_at: datetime


class TaskRunStatus(BaseModel):
    """Complete job status — GET /jobs/{job_id}."""

    job_id: UUID
    task_type: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    new_version: Optional[str] = None
    triggered_by: Optional[str] = None
    status: str  # queued | running | success | failed | cancelled
    enqueued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[str] = None

    class Config:
        from_attributes = True


class TaskRunList(BaseModel):
    """Liste paginée de jobs — GET /jobs."""

    items: list[TaskRunStatus]
    total: int
    limit: int
    offset: int
