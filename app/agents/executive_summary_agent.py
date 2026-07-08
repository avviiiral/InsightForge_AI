"""ExecutiveSummaryAgent — a 3-5 bullet, C-suite-ready summary of the analysis."""
from __future__ import annotations

from typing import Any, Optional

from app.agents.base_agent import BaseAgent
from app.llm.llm_factory import LLMRouter


class ExecutiveSummaryAgent(BaseAgent):
    name = "executive_summary_agent"
    description = "Produces a short bullet-point executive summary of the full analysis."

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        super().__init__()
        self.llm_router = llm_router

    def _execute(
        self,
        quality: dict[str, Any],
        kpis: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        dataset_name: str = "the dataset",
    ) -> dict[str, Any]:
        bullets = self._deterministic_bullets(quality, kpis, insights, recommendations)

        ai_summary = None
        if self.llm_router and self.llm_router.is_enabled():
            ai_summary = self.llm_router.generate(
                system_prompt=(
                    "You are writing an executive summary for a data analytics report. "
                    "Produce exactly 4-5 crisp bullet points (start each with '- '), each under 25 words. "
                    "Use only the facts provided."
                ),
                user_prompt=(
                    f"Dataset: {dataset_name}\nHealth score: {quality.get('health_score')}\n"
                    f"KPIs: {kpis}\nInsights: {insights}\nRecommendations: {recommendations}"
                ),
                max_tokens=250,
            )

        return {
            "bullets": bullets,
            "ai_bullets": ai_summary.split("\n") if ai_summary else None,
            "used_llm": ai_summary is not None,
        }

    @staticmethod
    def _deterministic_bullets(
        quality: dict[str, Any], kpis: list[dict[str, Any]], insights: list[dict[str, Any]], recommendations: list[dict[str, Any]]
    ) -> list[str]:
        bullets = []
        bullets.append(
            f"Dataset health score: {quality.get('health_score', 'N/A')}/100 "
            f"across {quality.get('total_rows', 0):,} rows and {quality.get('total_columns', 0)} columns."
        )
        top_kpi = next((k for k in kpis if k["name"] != "Total Records"), None)
        if top_kpi:
            bullets.append(f"{top_kpi['name']}: {top_kpi['formatted_value']}.")
        critical = [i for i in insights if i.get("severity") == "critical"]
        bullets.append(
            f"{len(critical)} critical issue(s) identified." if critical else "No critical data issues identified."
        )
        if recommendations:
            bullets.append(f"Top recommendation: {recommendations[0]['action']}")
        bullets.append(f"{len(insights)} total insight(s) generated across quality, statistics, and correlation analysis.")
        return bullets
