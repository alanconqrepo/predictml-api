"""
Schémas Pydantic pour l'endpoint /health/dependencies
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DependencyDetail(BaseModel):
    status: str  # "ok" | "error"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class DependencyHealthResponse(BaseModel):
    status: str  # "ok" | "degraded" | "critical"
    checked_at: datetime
    dependencies: dict[str, DependencyDetail]
