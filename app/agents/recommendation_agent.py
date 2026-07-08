"""RecommendationAgent — converts insights into actionable recommendations."""
from __future__ import annotations

from typing import Any, Optional

from app.agents.base_agent import BaseAgent
from app.llm.llm_factory import LLMRouter

_SEVERITY_ACTIONS = {
    "critical": "Immediate action recommended",
    "warning": "Should be reviewed soon",
    "info": "Good to know",
}


class RecommendationAgent(BaseAgent):
    name = "recommendation_agent"
    description = "Generates prioritized, actionable recommendations from prior agent insights."

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        super().__init__()
        self.llm_router = llm_router

    def _execute(self, insights: list[dict[str, Any]], dataset_name: str = "this dataset") -> dict[str, Any]:
        recommendations = []
        for insight in insights:
            rec = self._rule_based_recommendation(insight)
            if rec:
                recommendations.append(rec)

        recommendations.sort(key=lambda r: {"critical": 0, "warning": 1, "info": 2}.get(r["priority"], 3))

        ai_recommendation = None
        if self.llm_router and self.llm_router.is_enabled():
            ai_recommendation = self._llm_recommendation(insights, dataset_name)
            if ai_recommendation:
                recommendations.insert(0, {
                    "title": "AI Strategic Recommendation",
                    "action": ai_recommendation,
                    "priority": "info",
                    "source": "llm",
                })

        return {"recommendations": recommendations[:15], "used_llm": ai_recommendation is not None}

    @staticmethod
    def _rule_based_recommendation(insight: dict[str, Any]) -> dict[str, Any] | None:
        category = insight.get("category")
        severity = insight.get("severity", "info")
        title = insight.get("title", "")
        description = insight.get("description", "")

        templates = {
            "quality": f"{_SEVERITY_ACTIONS.get(severity)}: {description} Consider re-running the Data Cleaning Agent.",
            "outliers": f"{_SEVERITY_ACTIONS.get(severity)}: {description} Investigate whether these are data errors or genuine extreme events.",
            "correlation": f"Explore causality behind: {description} This relationship may be useful for forecasting or segmentation.",
            "distribution": f"{description} Prefer median-based metrics over the mean for skewed columns.",
        }
        action = templates.get(category)
        if not action:
            return None
        return {"title": title, "action": action, "priority": severity, "source": "rule_based"}

    def _llm_recommendation(self, insights: list[dict[str, Any]], dataset_name: str) -> str | None:
        system_prompt = (
            "You are a business analyst. Based on the list of data insights provided, write ONE "
            "concise, prioritized strategic recommendation (max 80 words) that a business "
            "stakeholder could act on this week."
        )
        user_prompt = f"Dataset: {dataset_name}\nInsights: {insights}\nWrite the recommendation now."
        return self.llm_router.generate(system_prompt, user_prompt, max_tokens=200)
