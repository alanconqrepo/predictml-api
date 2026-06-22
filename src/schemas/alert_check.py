"""
Pydantic schemas for alert_check_logs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AlertCheckLogRead(BaseModel):
    id: int
    checked_at: datetime
    check_type: str
    model_name: str
    model_version: Optional[str] = None
    result: str
    alert_sent: bool
    webhook_sent: bool
    new_predictions_count: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class AlertCheckLogList(BaseModel):
    items: List[AlertCheckLogRead]
    total: int
    limit: int
    offset: int
