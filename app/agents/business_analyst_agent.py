"""BusinessAnalystAgent — synthesizes KPIs, insights, and forecasts into a narrative.

Deterministic template by default; LLM-enhanced narrative when a provider
is configured. This agent is what powers the "Business Analyst" view in
the Streamlit dashboard — a readable paragraph rather than raw numbers.
"""
from __future__ import annotations

from typing import Any, Optional

from app.agents.base_agent import BaseAgent
from app.llm.llm_factory import LLMRouter


class BusinessAnalystAgent(BaseAgent):
    name = "business_analyst_agent"
    description = "Produces a business-readable narrative synthesizing KPIs, insights, and forecasts."

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        super().__init__()
        self.llm_router = llm_router

    def _execute(
        self,
        kpis: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        forecast: dict[str, Any] | None = None,
        dataset_name: str = "the uploaded dataset",
    ) -> dict[str, Any]:
        if self.llm_router and self.llm_router.is_enabled():
            narrative = self._llm_narrative(kpis, insights, forecast, dataset_name)
            if narrative:
                return {"narrative": narrative, "used_llm": True}

        return {"narrative": self._template_narrative(kpis, insights, forecast, dataset_name), "used_llm": False}

    @staticmethod
    def _template_narrative(
        kpis: list[dict[str, Any]], insights: list[dict[str, Any]], forecast: dict[str, Any] | None, dataset_name: str
    ) -> str:
        lines = [f"Analysis of {dataset_name}:"]
        record_kpi = next((k for k in kpis if k["name"] == "Total Records"), None)
        if record_kpi:
            lines.append(f"The dataset contains {record_kpi['formatted_value']} records.")

        top_kpis = [k for k in kpis if k["name"] != "Total Records"][:3]
        if top_kpis:
            kpi_text = "; ".join(f"{k['name']} = {k['formatted_value']}" for k in top_kpis)
            lines.append(f"Headline metrics: {kpi_text}.")

        critical = [i for i in insights if i.get("severity") == "critical"]
        warnings = [i for i in insights if i.get("severity") == "warning"]
        if critical:
            lines.append(f"⚠ {len(critical)} critical issue(s) require attention, including: {critical[0]['title']}.")
        elif warnings:
            lines.append(f"{len(warnings)} item(s) are worth reviewing, including: {warnings[0]['title']}.")
        else:
            lines.append("No critical data quality issues were detected.")

        if forecast and forecast.get("forecast"):
            last_hist = forecast["history"][-1]["value"] if forecast.get("history") else None
            last_fc = forecast["forecast"][-1]["value"]
            if last_hist is not None:
                direction = "increase" if last_fc > last_hist else "decrease"
                lines.append(
                    f"The forecast ({forecast.get('method')}) projects a {direction} to approximately "
                    f"{last_fc:.2f} by {forecast['forecast'][-1]['period']}."
                )

        return " ".join(lines)

    def _llm_narrative(self, kpis, insights, forecast, dataset_name) -> Optional[str]:
        system_prompt = (
            "You are a business analyst writing an executive-ready narrative (max 150 words). "
            "Use ONLY the numbers given — never fabricate figures. Be direct and actionable."
        )
        user_prompt = (
            f"Dataset: {dataset_name}\nKPIs: {kpis}\nInsights: {insights}\nForecast: {forecast}\n"
            "Write the narrative now."
        )
        return self.llm_router.generate(system_prompt, user_prompt, max_tokens=300)
