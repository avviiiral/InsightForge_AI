"""InsightAgent — turns statistical results into plain-English insights.

Deterministic by default (rule-based, always available); if an LLM is
configured, its narrative is layered on top as an additional "AI Insight"
entry rather than replacing the deterministic ones — this way results
stay reproducible and auditable even when an LLM is in the loop.
"""
from __future__ import annotations

from typing import Any, Optional

from app.agents.base_agent import BaseAgent
from app.llm.llm_factory import LLMRouter


class InsightAgent(BaseAgent):
    name = "insight_agent"
    description = "Generates plain-English insights from quality, statistics, and correlation results."

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        super().__init__()
        self.llm_router = llm_router

    def _execute(
        self,
        quality: dict[str, Any] | None = None,
        statistics: dict[str, Any] | None = None,
        correlation: dict[str, Any] | None = None,
        outliers: dict[str, Any] | None = None,
        kpis: dict[str, Any] | None = None,
        dataset_name: str = "this dataset",
    ) -> dict[str, Any]:
        insights: list[dict[str, str]] = []

        if quality:
            insights.extend(self._quality_insights(quality))
        if statistics:
            insights.extend(self._statistics_insights(statistics))
        if correlation:
            insights.extend(self._correlation_insights(correlation))
        if outliers:
            insights.extend(self._outlier_insights(outliers))

        ai_insight = None
        if self.llm_router and self.llm_router.is_enabled():
            ai_insight = self._generate_llm_insight(quality, statistics, correlation, kpis, dataset_name)
            if ai_insight:
                insights.append({"category": "ai_narrative", "title": "AI-Generated Insight",
                                  "description": ai_insight, "severity": "info"})

        return {"insights": insights, "used_llm": ai_insight is not None, "count": len(insights)}

    # -- deterministic rules --------------------------------------------

    @staticmethod
    def _quality_insights(quality: dict[str, Any]) -> list[dict[str, str]]:
        out = []
        score = quality.get("health_score", 100)
        if score >= 90:
            out.append({"category": "quality", "title": "High data quality",
                        "description": f"Overall health score is {score}/100 — the dataset is clean and analysis-ready.",
                        "severity": "info"})
        elif score >= 70:
            out.append({"category": "quality", "title": "Moderate data quality",
                        "description": f"Health score is {score}/100. Minor cleanup could improve reliability of downstream analysis.",
                        "severity": "warning"})
        else:
            out.append({"category": "quality", "title": "Data quality needs attention",
                        "description": f"Health score is only {score}/100 — review missing values and duplicates before trusting results.",
                        "severity": "critical"})
        for w in quality.get("warnings", []):
            out.append({"category": "quality", "title": "Data quality warning", "description": w, "severity": "warning"})
        return out

    @staticmethod
    def _statistics_insights(statistics: dict[str, Any]) -> list[dict[str, str]]:
        out = []
        for col, stats in statistics.get("numeric_summary", {}).items():
            shape = stats.get("distribution_shape")
            if shape and shape != "approximately symmetric":
                out.append({
                    "category": "distribution",
                    "title": f"'{col}' distribution is {shape}",
                    "description": (
                        f"'{col}' has a mean of {stats['mean']} and median of {stats['median']}, "
                        f"consistent with a {shape} distribution. Consider this when choosing aggregate metrics."
                    ),
                    "severity": "info",
                })
        return out

    @staticmethod
    def _correlation_insights(correlation: dict[str, Any]) -> list[dict[str, str]]:
        out = []
        for pair in correlation.get("top_pairs", [])[:5]:
            if abs(pair["correlation"]) >= 0.6:
                out.append({
                    "category": "correlation",
                    "title": f"{pair['column_a']} ↔ {pair['column_b']} are correlated",
                    "description": (
                        f"'{pair['column_a']}' and '{pair['column_b']}' show a {pair['strength']} "
                        f"relationship (r = {pair['correlation']})."
                    ),
                    "severity": "info",
                })
        return out

    @staticmethod
    def _outlier_insights(outliers: dict[str, Any]) -> list[dict[str, str]]:
        out = []
        for col, info in outliers.get("columns", {}).items():
            if info["iqr_outlier_pct"] > 3:
                out.append({
                    "category": "outliers",
                    "title": f"Outliers detected in '{col}'",
                    "description": (
                        f"{info['iqr_outlier_count']} value(s) ({info['iqr_outlier_pct']}%) in '{col}' "
                        f"fall outside the expected range [{info['lower_bound']}, {info['upper_bound']}]."
                    ),
                    "severity": "warning",
                })
        return out

    # -- optional LLM enrichment -----------------------------------------

    def _generate_llm_insight(self, quality, statistics, correlation, kpis, dataset_name) -> Optional[str]:
        system_prompt = (
            "You are a senior data analyst. Given structured summary statistics about a dataset, "
            "write a concise, business-friendly paragraph (max 120 words) highlighting the most "
            "important patterns. Do not invent numbers that are not present in the data provided."
        )
        user_prompt = (
            f"Dataset: {dataset_name}\n"
            f"Quality summary: {quality}\n"
            f"Statistics summary: {statistics}\n"
            f"Correlation summary: {correlation}\n"
            f"KPIs: {kpis}\n"
            "Write the analyst paragraph now."
        )
        return self.llm_router.generate(system_prompt, user_prompt, max_tokens=300)
