"""ProfilingAgent — builds a detailed per-column profile for the dashboard."""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.agents.base_agent import BaseAgent


class ProfilingAgent(BaseAgent):
    name = "profiling_agent"
    description = "Generates per-column profiles (counts, top values, numeric summaries)."

    def _execute(
        self, dataframe: pd.DataFrame, column_types: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        column_types = column_types or {}
        profiles = {}
        for col in dataframe.columns:
            series = dataframe[col]
            semantic = column_types.get(col, {}).get("semantic_type", "unknown")
            profiles[col] = self._profile_one(series, semantic)
        return {"profiles": profiles}

    def _profile_one(self, series: pd.Series, semantic: str) -> dict[str, Any]:
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        base = {
            "count": n_total,
            "missing": n_missing,
            "missing_pct": round(100 * n_missing / max(n_total, 1), 2),
            "unique": n_unique,
            "sample_values": [self._safe(v) for v in series.dropna().unique()[:5]],
        }

        if semantic in ("numeric", "currency") or pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            base["stats"] = {
                "mean": self._safe(desc.get("mean")),
                "std": self._safe(desc.get("std")),
                "min": self._safe(desc.get("min")),
                "p25": self._safe(desc.get("25%")),
                "median": self._safe(desc.get("50%")),
                "p75": self._safe(desc.get("75%")),
                "max": self._safe(desc.get("max")),
                "skew": self._safe(series.skew()) if n_unique > 2 else None,
            }
        elif semantic == "datetime":
            parsed = pd.to_datetime(series, errors="coerce")
            base["stats"] = {
                "min_date": str(parsed.min()) if parsed.notna().any() else None,
                "max_date": str(parsed.max()) if parsed.notna().any() else None,
                "span_days": (
                    int((parsed.max() - parsed.min()).days) if parsed.notna().sum() > 1 else None
                ),
            }
        else:
            top_values = series.value_counts(dropna=True).head(5)
            base["stats"] = {
                "top_values": {str(k): int(v) for k, v in top_values.items()},
                "avg_length": (
                    round(series.dropna().astype(str).map(len).mean(), 1) if n_total > n_missing else 0
                ),
            }
        return base

    @staticmethod
    def _safe(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (int, float)):
            return round(float(value), 4)
        return str(value)
