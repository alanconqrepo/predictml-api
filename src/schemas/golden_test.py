"""Schémas Pydantic pour le Golden Test Set"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class GoldenTestCreate(BaseModel):
    """Payload de création d'un cas de test golden"""

    input_features: Dict[str, Union[float, int, str]]
    expected_output: str = Field(..., max_length=500)
    description: Optional[str] = Field(None, max_length=500)


class GoldenTestResponse(BaseModel):
    """Représentation d'un cas de test golden"""

    id: int
    model_name: str
    input_features: Dict[str, Union[float, int, str]]
    expected_output: str
    description: Optional[str]
    created_at: datetime
    created_by_user_id: Optional[int]

    model_config = {"from_attributes": True}


class GoldenTestRunDetail(BaseModel):
    """Résultat détaillé d'un cas de test lors d'une exécution"""

    test_id: int
    description: Optional[str]
    input: Dict[str, Union[float, int, str]]
    expected: str
    actual: str
    passed: bool


class GoldenTestRunResponse(BaseModel):
    """Résultat complet d'une exécution du golden test set"""

    model_name: str
    version: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    details: List[GoldenTestRunDetail]
