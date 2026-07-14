"""KPIAgent — surfaces headline metrics for the top numeric columns.

For each candidate numeric column it reports the total, average, and (when
a datetime column is available) a period-over-period growth rate — the
kind of headline numbers an executive dashboard leads with.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.agents.base_agent import BaseAgent


class KPIAgent(BaseAgent):
    name = "kpi_agent"
    description = "Computes headline KPI cards (totals, averages, growth) for key numeric columns."

    def _execute(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        datetime_column: str | None = None,
        max_kpis: int = 6,
    ) -> dict[str, Any]:
        numeric_columns = numeric_columns or list(dataframe.select_dtypes(include="number").columns)
        numeric_columns = numeric_columns[:max_kpis]
        kpis = []

        for col in numeric_columns:
            series = dataframe[col].dropna()
            if series.empty:
                continue
            total = float(series.sum())
            mean = float(series.mean())
            growth = self._growth_rate(dataframe, col, datetime_column) if datetime_column else None
            kpis.append(
                {
                    "name": f"Total {col}",
                    "column": col,
                    "aggregation": "SUM",
                    "value": round(total, 2),
                    "formatted_value": self._format_number(total),
                    "trend": self._trend_label(growth),
                    "delta": growth,
                }
            )
            kpis.append(
                {
                    "name": f"Average {col}",
                    "column": col,
                    "aggregation": "AVG",
                    "value": round(mean, 2),
                    "formatted_value": self._format_number(mean),
                    "trend": None,
                    "delta": None,
                }
            )

        kpis.append(
            {
                "name": "Total Records",
                "column": "All Rows",
                "aggregation": "COUNT",
                "value": int(len(dataframe)),
                "formatted_value": f"{len(dataframe):,}",
                "trend": None,
                "delta": None,
            }
        )

        return {"kpis": kpis[: max_kpis + 1]}

    @staticmethod
    def _growth_rate(df: pd.DataFrame, col: str, date_col: str) -> float | None:
        try:
            temp = df[[date_col, col]].dropna().copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col]).sort_values(date_col)
            if len(temp) < 4:
                return None
            midpoint = len(temp) // 2
            first_half = temp[col].iloc[:midpoint].sum()
            second_half = temp[col].iloc[midpoint:].sum()
            if first_half == 0:
                return None
            return round(100 * (second_half - first_half) / abs(first_half), 2)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _trend_label(growth: float | None) -> str | None:
        if growth is None:
            return None
        if growth > 1:
            return "up"
        if growth < -1:
            return "down"
        return "flat"

    @staticmethod
    def _format_number(value: float) -> str:
        abs_v = abs(value)
        if abs_v >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if abs_v >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs_v >= 1_000:
            return f"{value / 1_000:.2f}K"
        return f"{value:,.2f}"
