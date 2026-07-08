"""Orchestrator — runs the full agent pipeline end-to-end.

Each pipeline step is a small, independent function that takes the
running `context` dict and returns the fields it adds to it. This
"list of nodes over a shared state dict" shape is deliberate: it is
directly portable to LangGraph (`app/agents/langgraph_adapter.py` shows
how) without rewriting any agent logic — each step here maps 1:1 onto a
LangGraph node operating on a shared state object.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from app.agents.anomaly_detection_agent import AnomalyDetectionAgent
from app.agents.business_analyst_agent import BusinessAnalystAgent
from app.agents.chart_recommendation_agent import ChartRecommendationAgent
from app.agents.correlation_agent import CorrelationAgent
from app.agents.dashboard_builder_agent import DashboardBuilderAgent
from app.agents.data_cleaning_agent import DataCleaningAgent
from app.agents.data_quality_agent import DataQualityAgent
from app.agents.executive_summary_agent import ExecutiveSummaryAgent
from app.agents.insight_agent import InsightAgent
from app.agents.kpi_agent import KPIAgent
from app.agents.outlier_detection_agent import OutlierDetectionAgent
from app.agents.profiling_agent import ProfilingAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.report_agent import ReportAgent
from app.agents.schema_detection_agent import SchemaDetectionAgent
from app.agents.statistics_agent import StatisticsAgent
from app.llm.llm_factory import LLMRouter, get_llm_router
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Runs the full data-to-insight pipeline and collects every agent's execution log."""

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        self.llm_router = llm_router or get_llm_router()
        self.schema_agent = SchemaDetectionAgent()
        self.quality_agent = DataQualityAgent()
        self.cleaning_agent = DataCleaningAgent()
        self.profiling_agent = ProfilingAgent()
        self.statistics_agent = StatisticsAgent()
        self.correlation_agent = CorrelationAgent()
        self.outlier_agent = OutlierDetectionAgent()
        self.anomaly_agent = AnomalyDetectionAgent()
        self.chart_agent = ChartRecommendationAgent()
        self.kpi_agent = KPIAgent()
        self.insight_agent = InsightAgent(llm_router=self.llm_router)
        self.recommendation_agent = RecommendationAgent(llm_router=self.llm_router)
        self.business_analyst_agent = BusinessAnalystAgent(llm_router=self.llm_router)
        self.executive_summary_agent = ExecutiveSummaryAgent(llm_router=self.llm_router)
        self.dashboard_agent = DashboardBuilderAgent()
        self.report_agent = ReportAgent()

        self.execution_log: list[dict[str, Any]] = []

    def run_full_pipeline(
        self, dataframe: pd.DataFrame, dataset_name: str = "dataset", clean_data: bool = True
    ) -> dict[str, Any]:
        """Run every agent in sequence and return the full analysis context."""
        self.execution_log = []
        context: dict[str, Any] = {"dataframe": dataframe, "dataset_name": dataset_name}

        self._step("schema_detection", lambda ctx: self.schema_agent.run(dataframe=ctx["dataframe"]), context,
                   assign=lambda d: {"column_types": d["column_types"], "schema_summary": d["summary"]})

        self._step("data_quality", lambda ctx: self.quality_agent.run(dataframe=ctx["dataframe"]), context,
                   assign=lambda d: {"quality": d})

        if clean_data:
            self._step(
                "data_cleaning",
                lambda ctx: self.cleaning_agent.run(dataframe=ctx["dataframe"], column_types=ctx["column_types"]),
                context,
                assign=lambda d: {"dataframe": d["cleaned_dataframe"], "cleaning_actions": d["actions_taken"]},
            )

        numeric_cols = [c for c, i in context["column_types"].items() if i["semantic_type"] in ("numeric", "currency")]
        datetime_cols = [c for c, i in context["column_types"].items() if i["semantic_type"] == "datetime"]
        context["numeric_columns"] = numeric_cols
        context["datetime_columns"] = datetime_cols

        self._step("profiling", lambda ctx: self.profiling_agent.run(
            dataframe=ctx["dataframe"], column_types=ctx["column_types"]), context,
            assign=lambda d: {"profiles": d["profiles"]})

        self._step("statistics", lambda ctx: self.statistics_agent.run(
            dataframe=ctx["dataframe"], numeric_columns=ctx["numeric_columns"]), context,
            assign=lambda d: {"statistics": d})

        self._step("correlation", lambda ctx: self.correlation_agent.run(
            dataframe=ctx["dataframe"], numeric_columns=ctx["numeric_columns"]), context,
            assign=lambda d: {"correlation": d})

        self._step("outlier_detection", lambda ctx: self.outlier_agent.run(
            dataframe=ctx["dataframe"], numeric_columns=ctx["numeric_columns"]), context,
            assign=lambda d: {"outliers": d})

        self._step("anomaly_detection", lambda ctx: self.anomaly_agent.run(
            dataframe=ctx["dataframe"], numeric_columns=ctx["numeric_columns"]), context,
            assign=lambda d: {"anomalies": d})

        self._step("chart_recommendation", lambda ctx: self.chart_agent.run(
            dataframe=ctx["dataframe"], column_types=ctx["column_types"]), context,
            assign=lambda d: {"chart_recommendations": d["recommendations"]})

        self._step("kpi_generation", lambda ctx: self.kpi_agent.run(
            dataframe=ctx["dataframe"], numeric_columns=ctx["numeric_columns"],
            datetime_column=(ctx["datetime_columns"][0] if ctx["datetime_columns"] else None)), context,
            assign=lambda d: {"kpis": d["kpis"]})

        self._step("insight_generation", lambda ctx: self.insight_agent.run(
            quality=ctx["quality"], statistics=ctx["statistics"], correlation=ctx["correlation"],
            outliers=ctx["outliers"], kpis=ctx["kpis"], dataset_name=ctx["dataset_name"]), context,
            assign=lambda d: {"insights": d["insights"]})

        self._step("recommendation_generation", lambda ctx: self.recommendation_agent.run(
            insights=ctx["insights"], dataset_name=ctx["dataset_name"]), context,
            assign=lambda d: {"recommendations": d["recommendations"]})

        self._step("business_analysis", lambda ctx: self.business_analyst_agent.run(
            kpis=ctx["kpis"], insights=ctx["insights"], forecast=None, dataset_name=ctx["dataset_name"]), context,
            assign=lambda d: {"business_narrative": d["narrative"]})

        self._step("executive_summary", lambda ctx: self.executive_summary_agent.run(
            quality=ctx["quality"], kpis=ctx["kpis"], insights=ctx["insights"],
            recommendations=ctx["recommendations"], dataset_name=ctx["dataset_name"]), context,
            assign=lambda d: {"executive_summary": d})

        self._step("dashboard_layout", lambda ctx: self.dashboard_agent.run(
            kpis=ctx["kpis"], chart_recommendations=ctx["chart_recommendations"]), context,
            assign=lambda d: {"dashboard_layout": d["layout"]})

        self._step("report_assembly", lambda ctx: self.report_agent.run(
            dataset_name=ctx["dataset_name"], quality=ctx["quality"], profiles=ctx["profiles"],
            statistics=ctx["statistics"], correlation=ctx["correlation"], outliers=ctx["outliers"],
            kpis=ctx["kpis"], insights=ctx["insights"], recommendations=ctx["recommendations"],
            executive_summary=ctx["executive_summary"], forecast=None,
            business_narrative=ctx["business_narrative"]), context,
            assign=lambda d: {"report": d["report"]})

        context["execution_log"] = self.execution_log
        context["llm_provider"] = self.llm_router.provider_name
        context["llm_enabled"] = self.llm_router.is_enabled()
        return context

    def _step(
        self,
        step_name: str,
        runner: Callable[[dict[str, Any]], Any],
        context: dict[str, Any],
        assign: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        result = runner(context)
        self.execution_log.append(self._sanitize_log_entry(result.to_dict()))
        if result.success:
            context.update(assign(result.data))
        else:
            logger.warning(f"Pipeline step '{step_name}' failed: {result.error}")

    @staticmethod
    def _sanitize_log_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """Strip heavy/non-JSON-serializable payloads (e.g. DataFrames) from a log entry.

        The execution log is meant for observability (which agent ran, how long it
        took, whether it succeeded) — not for carrying full result payloads across
        the API boundary, so we keep only lightweight, JSON-safe data here.
        """
        import pandas as pd

        def _clean(value: Any) -> Any:
            if isinstance(value, pd.DataFrame):
                return f"<DataFrame: {value.shape[0]} rows x {value.shape[1]} cols>"
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_clean(v) for v in value]
            return value

        entry["data"] = _clean(entry.get("data", {}))
        return entry
