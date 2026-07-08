"""SchemaDetectionAgent — infers data types AND semantic meaning per column.

This is the cornerstone of the "no predefined schema" requirement: nothing
downstream depends on hard-coded column names. Every other agent reads the
`semantic_type` produced here (numeric, categorical, datetime, email,
phone, currency, id, boolean, location, text) to decide how to treat a
column, regardless of what it happens to be called.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app.agents.base_agent import BaseAgent

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^[\+\(]?[0-9][0-9\-\.\s\(\)]{6,17}[0-9]$")
_CURRENCY_RE = re.compile(r"^[\$\€\£\¥]\s?-?[\d,]+(\.\d+)?$|^-?[\d,]+(\.\d+)?\s?(USD|EUR|GBP|INR|JPY)$")
_ID_NAME_HINTS = ("id", "_id", "uuid", "guid", "code", "key")
_LOCATION_NAME_HINTS = (
    "city", "state", "country", "region", "address", "zip", "postal", "location",
    "latitude", "longitude", "lat", "lng", "province", "continent",
)
_CURRENCY_NAME_HINTS = ("price", "amount", "revenue", "cost", "salary", "income", "fee", "total", "budget")


class SchemaDetectionAgent(BaseAgent):
    name = "schema_detection_agent"
    description = "Detects dtypes and semantic column types without relying on fixed column names."

    def _execute(self, dataframe: pd.DataFrame, sample_size: int = 500) -> dict[str, Any]:
        profiles = {}
        for col in dataframe.columns:
            profiles[col] = self._profile_column(dataframe[col], col, sample_size)
        summary = self._summarize(profiles)
        return {"column_types": profiles, "summary": summary}

    # -- internals -----------------------------------------------------

    def _profile_column(self, series: pd.Series, name: str, sample_size: int) -> dict[str, Any]:
        non_null = series.dropna()
        sample = non_null.head(sample_size)
        pandas_dtype = str(series.dtype)
        semantic = self._infer_semantic_type(series, sample, name)
        return {
            "pandas_dtype": pandas_dtype,
            "semantic_type": semantic,
            "is_nullable": bool(series.isna().any()),
        }

    def _infer_semantic_type(self, series: pd.Series, sample: pd.Series, name: str) -> str:
        name_lower = name.lower()
        n = max(len(sample), 1)

        # Boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        unique_vals = set(str(v).strip().lower() for v in sample.unique()[:10])
        if unique_vals and unique_vals.issubset({"true", "false", "yes", "no", "0", "1", "y", "n"}):
            if series.nunique(dropna=True) <= 2:
                return "boolean"

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if series.dtype == object and self._looks_like_date(sample):
            return "datetime"

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            if any(h in name_lower for h in _CURRENCY_NAME_HINTS):
                return "currency"
            if any(h in name_lower for h in _ID_NAME_HINTS) and self._is_id_like(series):
                return "identifier"
            return "numeric"

        # Object/string-based semantic checks
        if series.dtype == object:
            str_sample = sample.astype(str).str.strip()
            if len(str_sample) and (str_sample.map(lambda v: bool(_EMAIL_RE.match(v))).mean() > 0.7):
                return "email"
            if any(h in name_lower for h in _LOCATION_NAME_HINTS):
                return "location"
            if len(str_sample) and (str_sample.map(lambda v: bool(_CURRENCY_RE.match(v))).mean() > 0.6):
                return "currency"
            if any(h in name_lower for h in _ID_NAME_HINTS) and self._is_id_like(series):
                return "identifier"
            if len(str_sample) and (str_sample.map(lambda v: bool(_PHONE_RE.match(v))).mean() > 0.6):
                return "phone"

            avg_len = str_sample.map(len).mean() if len(str_sample) else 0
            n_unique = series.nunique(dropna=True)
            n_total = max(len(series.dropna()), 1)
            if n_unique / n_total < 0.5 and n_unique <= max(50, int(n_total * 0.2)):
                return "categorical"
            if avg_len > 30:
                return "text"
            return "categorical"

        return "unknown"

    @staticmethod
    def _is_id_like(series: pd.Series) -> bool:
        n = series.dropna().shape[0]
        if n == 0:
            return False
        return series.nunique(dropna=True) / n > 0.9

    @staticmethod
    def _looks_like_date(sample: pd.Series) -> bool:
        if sample.empty:
            return False
        str_sample = sample.astype(str).head(50)
        try:
            parsed = pd.to_datetime(str_sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(str_sample, errors="coerce")
        return parsed.notna().mean() > 0.7

    @staticmethod
    def _summarize(profiles: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        summary: dict[str, list[str]] = {}
        for col, info in profiles.items():
            summary.setdefault(info["semantic_type"], []).append(col)
        return summary
