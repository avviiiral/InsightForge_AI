"""DataQualityAgent — computes a dataset health score and flags issues."""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.agents.base_agent import BaseAgent


class DataQualityAgent(BaseAgent):
    name = "data_quality_agent"
    description = "Computes missing-value stats, duplicate rows, and an overall health score."

    def _execute(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        n_rows, n_cols = dataframe.shape
        total_cells = max(n_rows * n_cols, 1)

        missing_per_col = dataframe.isna().sum()
        total_missing = int(missing_per_col.sum())
        missing_pct = round(100 * total_missing / total_cells, 2)
        columns_with_missing = missing_per_col[missing_per_col > 0].index.tolist()

        duplicate_rows = int(dataframe.duplicated().sum())
        duplicate_pct = round(100 * duplicate_rows / max(n_rows, 1), 2)

        constant_cols = [c for c in dataframe.columns if dataframe[c].nunique(dropna=True) <= 1]
        high_cardinality_cols = [
            c for c in dataframe.select_dtypes(include="object").columns
            if dataframe[c].nunique(dropna=True) > 0.9 * n_rows and n_rows > 20
        ]

        warnings: list[str] = []
        if missing_pct > 20:
            warnings.append(f"High overall missing-data rate ({missing_pct}%).")
        if duplicate_pct > 5:
            warnings.append(f"{duplicate_rows} duplicate rows detected ({duplicate_pct}%).")
        if constant_cols:
            warnings.append(f"{len(constant_cols)} column(s) have a single constant value: {constant_cols[:5]}")
        if high_cardinality_cols:
            warnings.append(
                f"{len(high_cardinality_cols)} column(s) look like unique identifiers: {high_cardinality_cols[:5]}"
            )
        if n_rows < 30:
            warnings.append("Very small dataset (< 30 rows) — statistical results may be unreliable.")

        health_score = self._compute_health_score(missing_pct, duplicate_pct, constant_cols, n_cols)

        return {
            "health_score": health_score,
            "total_rows": int(n_rows),
            "total_columns": int(n_cols),
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": duplicate_pct,
            "total_missing_cells": total_missing,
            "missing_pct": missing_pct,
            "missing_by_column": {k: int(v) for k, v in missing_per_col.items() if v > 0},
            "columns_with_missing": columns_with_missing,
            "constant_columns": constant_cols,
            "high_cardinality_columns": high_cardinality_cols,
            "warnings": warnings,
        }

    @staticmethod
    def _compute_health_score(
        missing_pct: float, duplicate_pct: float, constant_cols: list[str], n_cols: int
    ) -> float:
        score = 100.0
        score -= min(missing_pct * 0.8, 40)
        score -= min(duplicate_pct * 1.0, 20)
        if n_cols:
            score -= min(len(constant_cols) / n_cols * 100 * 0.2, 15)
        return round(max(score, 0.0), 1)
