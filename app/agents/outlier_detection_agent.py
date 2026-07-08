"""OutlierDetectionAgent — classical statistical outlier detection (IQR + Z-score)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent


class OutlierDetectionAgent(BaseAgent):
    name = "outlier_detection_agent"
    description = "Flags outliers per numeric column using the IQR and Z-score methods."

    def _execute(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
    ) -> dict[str, Any]:
        numeric_columns = numeric_columns or list(dataframe.select_dtypes(include=np.number).columns)
        results = {}
        for col in numeric_columns:
            series = dataframe[col].dropna()
            if series.nunique() < 3:
                continue
            results[col] = self._analyze_column(series, iqr_multiplier, z_threshold)
        total_outliers = sum(r["iqr_outlier_count"] for r in results.values())
        return {"columns": results, "total_iqr_outliers": int(total_outliers)}

    def _analyze_column(self, series: pd.Series, iqr_multiplier: float, z_threshold: float) -> dict[str, Any]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

        mean, std = series.mean(), series.std()
        z_scores = (series - mean) / std if std > 0 else pd.Series(0, index=series.index)
        z_outliers = series[z_scores.abs() > z_threshold]

        return {
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "iqr_outlier_count": int(len(iqr_outliers)),
            "iqr_outlier_pct": round(100 * len(iqr_outliers) / len(series), 2),
            "z_outlier_count": int(len(z_outliers)),
            "sample_outlier_values": [round(float(v), 4) for v in iqr_outliers.head(10).tolist()],
        }
