"""CorrelationAgent — correlation matrix + ranked list of strongest relationships."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent


class CorrelationAgent(BaseAgent):
    name = "correlation_agent"
    description = "Computes a numeric correlation matrix and ranks the strongest pairwise relationships."

    def _execute(
        self, dataframe: pd.DataFrame, numeric_columns: list[str] | None = None, method: str = "pearson"
    ) -> dict[str, Any]:
        numeric_columns = numeric_columns or list(dataframe.select_dtypes(include=np.number).columns)
        numeric_columns = [c for c in numeric_columns if dataframe[c].nunique(dropna=True) > 1]

        if len(numeric_columns) < 2:
            return {"matrix": {}, "top_pairs": [], "note": "Fewer than 2 usable numeric columns."}

        corr = dataframe[numeric_columns].corr(method=method).round(4)
        pairs = []
        for col_a, col_b in combinations(numeric_columns, 2):
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": float(value),
                    "strength": self._classify_strength(value),
                }
            )
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

        return {
            "matrix": corr.to_dict(),
            "top_pairs": pairs[:15],
            "method": method,
            "columns_analyzed": numeric_columns,
        }

    @staticmethod
    def _classify_strength(value: float) -> str:
        abs_v = abs(value)
        direction = "positive" if value > 0 else "negative"
        if abs_v >= 0.8:
            return f"very strong {direction}"
        if abs_v >= 0.6:
            return f"strong {direction}"
        if abs_v >= 0.4:
            return f"moderate {direction}"
        if abs_v >= 0.2:
            return f"weak {direction}"
        return "negligible"
