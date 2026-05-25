"""Pydantic schemas for the Golden Test Set"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class GoldenTestCreate(BaseModel):
    """Payload for creating a golden test case"""

    input_features: Dict[str, Union[float, int, str]]
    expected_output: str = Field(..., max_length=500)
    description: Optional[str] = Field(None, max_length=500)


class GoldenTestResponse(BaseModel):
    """Representation of a golden test case"""

    id: int
    model_name: str
    input_features: Dict[str, Union[float, int, str]]
    expected_output: str
    description: Optional[str]
    created_at: datetime
    created_by_user_id: Optional[int]

    model_config = {"from_attributes": True}


class GoldenTestRunDetail(BaseModel):
    """Detailed result of a test case during an execution"""

    test_id: int
    description: Optional[str]
    input: Dict[str, Union[float, int, str]]
    expected: str
    actual: str
    passed: bool


class GoldenTestRunResponse(BaseModel):
    """Complete result of a golden test set execution"""

    model_name: str
    version: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    details: List[GoldenTestRunDetail]
