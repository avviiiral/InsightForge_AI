"""DashboardBuilderAgent — arranges KPIs and chart recommendations into a grid layout.

Produces a plain layout spec (rows of KPI cards, rows of charts) that the
Streamlit frontend renders directly — keeping all "what goes where"
decisions out of the UI code.
"""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent


class DashboardBuilderAgent(BaseAgent):
    name = "dashboard_builder_agent"
    description = "Arranges KPIs and recommended charts into a dashboard layout specification."

    def _execute(
        self,
        kpis: list[dict[str, Any]],
        chart_recommendations: list[dict[str, Any]],
        kpi_row_size: int = 4,
        chart_row_size: int = 2,
    ) -> dict[str, Any]:
        kpi_rows = [kpis[i: i + kpi_row_size] for i in range(0, len(kpis), kpi_row_size)]
        chart_rows = [
            chart_recommendations[i: i + chart_row_size]
            for i in range(0, len(chart_recommendations), chart_row_size)
        ]
        return {
            "layout": {
                "kpi_rows": kpi_rows,
                "chart_rows": chart_rows,
            },
            "total_kpis": len(kpis),
            "total_charts": len(chart_recommendations),
        }
