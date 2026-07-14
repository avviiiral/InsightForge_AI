"""Optional LangGraph adapter.

InsightForge-AI's `Orchestrator` already runs its agents as a sequence of
pure functions over a shared `context` dict — the exact shape LangGraph
expects from graph nodes. This module is NOT imported by default (LangGraph
is not a hard dependency) but shows how to lift the same agents into a
`langgraph.graph.StateGraph` if you want branching, retries, or
human-in-the-loop checkpoints instead of the default linear pipeline.

Install with: pip install langgraph
"""
from __future__ import annotations

from typing import Any, TypedDict

from app.agents.correlation_agent import CorrelationAgent
from app.agents.data_quality_agent import DataQualityAgent
from app.agents.insight_agent import InsightAgent
from app.agents.kpi_agent import KPIAgent
from app.agents.schema_detection_agent import SchemaDetectionAgent
from app.agents.statistics_agent import StatisticsAgent
from app.llm.llm_factory import get_llm_router


class PipelineState(TypedDict, total=False):
    dataframe: Any
    column_types: dict
    quality: dict
    statistics: dict
    correlation: dict
    kpis: list
    insights: list


def build_langgraph_pipeline():
    """Build and compile a LangGraph graph mirroring `Orchestrator.run_full_pipeline`.

    Raises ImportError with a clear message if `langgraph` isn't installed —
    this keeps the core application fully functional without the extra
    dependency while still being "LangGraph-ready" for teams that want it.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "LangGraph is not installed. Run `pip install langgraph` to use this optional adapter; "
            "the standard `Orchestrator` works without it."
        ) from exc

    schema_agent = SchemaDetectionAgent()
    quality_agent = DataQualityAgent()
    statistics_agent = StatisticsAgent()
    correlation_agent = CorrelationAgent()
    kpi_agent = KPIAgent()
    insight_agent = InsightAgent(llm_router=get_llm_router())

    def node_schema(state: PipelineState) -> PipelineState:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise ValueError("PipelineState missing 'dataframe'")

        result = schema_agent.run(dataframe=dataframe)
        return {"column_types": result.data.get("column_types", {})}

    def node_quality(state: PipelineState) -> PipelineState:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise ValueError("PipelineState missing 'dataframe'")

        result = quality_agent.run(dataframe=dataframe)
        return {"quality": result.data}

    def node_statistics(state: PipelineState) -> PipelineState:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise ValueError("PipelineState missing 'dataframe'")

        column_types = state.get("column_types", {})

        numeric_cols = [
            c
            for c, info in column_types.items()
            if info["semantic_type"] in ("numeric", "currency")
        ]

        result = statistics_agent.run(
            dataframe=dataframe,
            numeric_columns=numeric_cols,
        )

        return {"statistics": result.data}

    def node_correlation(state: PipelineState) -> PipelineState:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise ValueError("PipelineState missing 'dataframe'")

        column_types = state.get("column_types", {})

        numeric_cols = [
            c
            for c, info in column_types.items()
            if info["semantic_type"] in ("numeric", "currency")
        ]

        result = correlation_agent.run(
            dataframe=dataframe,
            numeric_columns=numeric_cols,
        )

        return {"correlation": result.data}

    def node_kpis(state: PipelineState) -> PipelineState:
        dataframe = state.get("dataframe")
        if dataframe is None:
            raise ValueError("PipelineState missing 'dataframe'")

        column_types = state.get("column_types", {})

        numeric_cols = [
            c
            for c, info in column_types.items()
            if info["semantic_type"] in ("numeric", "currency")
        ]

        result = kpi_agent.run(
            dataframe=dataframe,
            numeric_columns=numeric_cols,
        )

        return {"kpis": result.data.get("kpis", [])}

    def node_insights(state: PipelineState) -> PipelineState:
        result = insight_agent.run(
            quality=state.get("quality"),
            statistics=state.get("statistics"),
            correlation=state.get("correlation"),
            kpis=state.get("kpis"),
        )

        return {"insights": result.data.get("insights", [])}

    graph = StateGraph(PipelineState)

    graph.add_node("schema_detection", node_schema)
    graph.add_node("data_quality", node_quality)
    graph.add_node("statistics", node_statistics)
    graph.add_node("correlation", node_correlation)
    graph.add_node("kpis", node_kpis)
    graph.add_node("insights", node_insights)

    graph.set_entry_point("schema_detection")

    graph.add_edge("schema_detection", "data_quality")
    graph.add_edge("data_quality", "statistics")
    graph.add_edge("statistics", "correlation")
    graph.add_edge("correlation", "kpis")
    graph.add_edge("kpis", "insights")
    graph.add_edge("insights", END)

    return graph.compile()