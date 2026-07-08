"""FeatureEngineeringAgent — suggests and generates simple derived features.

Kept deliberately conservative and dataset-agnostic: date-part extraction,
numeric binning, and one-hot-encoding suggestions for low-cardinality
categoricals. It never invents domain knowledge it can't infer from the
data itself.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.agents.base_agent import BaseAgent


class FeatureEngineeringAgent(BaseAgent):
    name = "feature_engineering_agent"
    description = "Suggests and (optionally) generates derived features from existing columns."

    def _execute(
        self,
        dataframe: pd.DataFrame,
        column_types: dict[str, dict[str, Any]] | None = None,
        apply_transformations: bool = False,
    ) -> dict[str, Any]:
        column_types = column_types or {}
        suggestions: list[dict[str, str]] = []
        df = dataframe.copy()

        for col, info in column_types.items():
            semantic = info.get("semantic_type")
            if semantic == "datetime":
                suggestions.append(
                    {"column": col, "suggestion": f"Extract year/month/day-of-week/quarter from '{col}'."}
                )
                if apply_transformations:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = parsed.dt.year
                    df[f"{col}_month"] = parsed.dt.month
                    df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                    df[f"{col}_quarter"] = parsed.dt.quarter

            elif semantic in ("numeric", "currency"):
                if df[col].nunique(dropna=True) > 10:
                    suggestions.append(
                        {"column": col, "suggestion": f"Bin '{col}' into quartile buckets (Low/Med/High/Very High)."}
                    )
                    if apply_transformations:
                        try:
                            df[f"{col}_bucket"] = pd.qcut(
                                df[col], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop"
                            )
                        except ValueError:
                            pass

            elif semantic == "categorical":
                n_unique = df[col].nunique(dropna=True)
                if 1 < n_unique <= 15:
                    suggestions.append(
                        {"column": col, "suggestion": f"One-hot encode '{col}' ({n_unique} categories) for ML use."}
                    )

        result: dict[str, Any] = {"suggestions": suggestions, "n_suggestions": len(suggestions)}
        if apply_transformations:
            result["engineered_dataframe"] = df
            result["new_columns"] = [c for c in df.columns if c not in dataframe.columns]
        return result
