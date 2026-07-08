"""StatisticsAgent — dataset-wide descriptive statistics and distribution shape."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.agents.base_agent import BaseAgent


class StatisticsAgent(BaseAgent):
    name = "statistics_agent"
    description = "Computes descriptive statistics and distribution characteristics for numeric columns."

    def _execute(self, dataframe: pd.DataFrame, numeric_columns: list[str] | None = None) -> dict[str, Any]:
        numeric_columns = numeric_columns or list(dataframe.select_dtypes(include=np.number).columns)
        results = {}
        for col in numeric_columns:
            series = dataframe[col].dropna()
            if series.empty or series.nunique() < 2:
                continue
            results[col] = self._describe_column(series)
        return {"numeric_summary": results, "columns_analyzed": list(results.keys())}

    def _describe_column(self, series: pd.Series) -> dict[str, Any]:
        skewness = float(series.skew())
        kurtosis = float(series.kurt())
        shape = self._classify_shape(skewness, kurtosis)

        normal_test_p = None
        if 8 <= len(series) <= 5000:
            try:
                _, normal_test_p = stats.shapiro(series.sample(min(len(series), 500), random_state=42))
            except Exception:  # noqa: BLE001
                normal_test_p = None

        return {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "variance": round(float(series.var()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "range": round(float(series.max() - series.min()), 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "distribution_shape": shape,
            "normality_p_value": round(normal_test_p, 4) if normal_test_p is not None else None,
            "likely_normal": bool(normal_test_p and normal_test_p > 0.05),
        }

    @staticmethod
    def _classify_shape(skewness: float, kurtosis: float) -> str:
        if skewness > 1:
            return "right-skewed"
        if skewness < -1:
            return "left-skewed"
        if kurtosis > 3:
            return "heavy-tailed"
        if kurtosis < -1:
            return "flat (platykurtic)"
        return "approximately symmetric"
