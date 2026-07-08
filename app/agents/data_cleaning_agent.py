"""DataCleaningAgent — produces an analysis-ready copy of the dataset.

The original DataFrame is never mutated in place; this agent always
returns a *new* cleaned DataFrame plus a human-readable log of every
transformation it applied, so users can audit exactly what changed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent


class DataCleaningAgent(BaseAgent):
    name = "data_cleaning_agent"
    description = "Cleans a dataset: trims strings, drops exact duplicates, imputes missing values."

    def _execute(
        self,
        dataframe: pd.DataFrame,
        column_types: dict[str, dict[str, Any]] | None = None,
        drop_duplicates: bool = True,
        impute_missing: bool = True,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
    ) -> dict[str, Any]:
        df = dataframe.copy()
        actions: list[str] = []

        # 1. Trim whitespace on string/object columns.
        object_cols = df.select_dtypes(include="object").columns
        for col in object_cols:
            before_na = df[col].isna().sum()
            df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
            after_na = df[col].isna().sum()
            if after_na > before_na:
                actions.append(f"Column '{col}': normalized empty/placeholder strings to missing values.")

        # 2. Drop exact duplicate rows.
        if drop_duplicates:
            n_before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            n_removed = n_before - len(df)
            if n_removed:
                actions.append(f"Removed {n_removed} exact duplicate row(s).")

        # 3. Impute missing values per semantic type.
        if impute_missing:
            column_types = column_types or {}
            for col in df.columns:
                if df[col].isna().sum() == 0:
                    continue
                semantic = column_types.get(col, {}).get("semantic_type")
                if pd.api.types.is_numeric_dtype(df[col]) or semantic in ("numeric", "currency"):
                    fill_value = (
                        df[col].median() if numeric_strategy == "median" else df[col].mean()
                    )
                    df[col] = df[col].fillna(fill_value)
                    actions.append(f"Column '{col}': imputed missing numeric values with {numeric_strategy} ({fill_value:.2f}).")
                elif semantic == "datetime" or pd.api.types.is_datetime64_any_dtype(df[col]):
                    actions.append(f"Column '{col}': left datetime gaps as-is (no safe default).")
                else:
                    mode_vals = df[col].mode(dropna=True)
                    if not mode_vals.empty:
                        fill_value = mode_vals.iloc[0]
                        df[col] = df[col].fillna(fill_value)
                        actions.append(f"Column '{col}': imputed missing categorical values with mode ('{fill_value}').")

        if not actions:
            actions.append("Dataset was already clean — no transformations were necessary.")

        return {
            "cleaned_dataframe": df,
            "actions_taken": actions,
            "rows_before": int(len(dataframe)),
            "rows_after": int(len(df)),
        }
