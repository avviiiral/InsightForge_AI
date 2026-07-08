"""ReportAgent — assembles every prior agent's output into one report structure.

This agent performs no analysis itself; it is purely a composition step
that `ExportAgent` (PDF/Excel/Markdown/PPTX) and the Streamlit "Report"
tab both consume, so the report layout only has to be defined once.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base_agent import BaseAgent


class ReportAgent(BaseAgent):
    name = "report_agent"
    description = "Assembles all agent outputs into a single structured report."

    def _execute(
        self,
        dataset_name: str,
        quality: dict[str, Any],
        profiles: dict[str, Any],
        statistics: dict[str, Any],
        correlation: dict[str, Any],
        outliers: dict[str, Any],
        kpis: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        executive_summary: dict[str, Any],
        forecast: dict[str, Any] | None = None,
        business_narrative: str | None = None,
    ) -> dict[str, Any]:
        report = {
            "title": f"InsightForge-AI Analytics Report — {dataset_name}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset_name,
            "executive_summary": executive_summary,
            "business_narrative": business_narrative,
            "data_quality": quality,
            "kpis": kpis,
            "column_profiles": profiles,
            "statistics": statistics,
            "correlation": correlation,
            "outliers": outliers,
            "forecast": forecast,
            "insights": insights,
            "recommendations": recommendations,
        }
        return {"report": report}
