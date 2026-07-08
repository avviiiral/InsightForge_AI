"""Shared Pydantic models used by the API layer and (optionally) agents.

These are intentionally decoupled from pandas — they describe the *shape*
of data moving across the API boundary, not the in-memory analytics
representation used internally by agents (which mostly pass DataFrames and
plain dicts around for speed and simplicity).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int
    n_columns: int
    columns: list[str]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: str
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[Any] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class DatasetHealth(BaseModel):
    health_score: float
    total_rows: int
    total_columns: int
    duplicate_rows: int
    total_missing_cells: int
    missing_pct: float
    columns_with_missing: list[str]
    warnings: list[str]


class ChartSpec(BaseModel):
    chart_type: str
    title: str
    x: Optional[str] = None
    y: Optional[str] = None
    reason: str
    figure_json: Optional[str] = None


class KPI(BaseModel):
    name: str
    value: Any
    formatted_value: str
    trend: Optional[str] = None
    delta: Optional[float] = None


class Insight(BaseModel):
    category: str
    title: str
    description: str
    severity: str = "info"  # info | warning | critical


class ForecastPoint(BaseModel):
    period: str
    value: float
    is_forecast: bool


class QueryRequest(BaseModel):
    dataset_id: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    data_preview: Optional[list[dict[str, Any]]] = None
    used_llm: bool = False


class ExportRequest(BaseModel):
    dataset_id: str
    export_format: str  # pdf | excel | markdown | pptx | csv
