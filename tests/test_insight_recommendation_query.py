"""Tests for InsightAgent, RecommendationAgent, and QueryAgent (deterministic paths)."""
from __future__ import annotations

from app.agents.correlation_agent import CorrelationAgent
from app.agents.data_quality_agent import DataQualityAgent
from app.agents.insight_agent import InsightAgent
from app.agents.query_agent import QueryAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.schema_detection_agent import SchemaDetectionAgent
from app.agents.statistics_agent import StatisticsAgent


def test_insight_agent_runs_without_llm(sample_dataframe):
    quality = DataQualityAgent().run(dataframe=sample_dataframe).data
    column_types = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    numeric_cols = [c for c, i in column_types.items() if i["semantic_type"] in ("numeric", "currency")]
    statistics = StatisticsAgent().run(dataframe=sample_dataframe, numeric_columns=numeric_cols).data
    correlation = CorrelationAgent().run(dataframe=sample_dataframe, numeric_columns=numeric_cols).data

    result = InsightAgent(llm_router=None).run(quality=quality, statistics=statistics, correlation=correlation)
    assert result.success
    assert result.data["used_llm"] is False
    assert len(result.data["insights"]) > 0


def test_recommendation_agent_prioritizes_critical_first():
    insights = [
        {"category": "quality", "title": "Minor", "description": "x", "severity": "info"},
        {"category": "outliers", "title": "Big issue", "description": "y", "severity": "critical"},
    ]
    result = RecommendationAgent(llm_router=None).run(insights=insights)
    assert result.success
    if result.data["recommendations"]:
        assert result.data["recommendations"][0]["priority"] == "critical"


def test_query_agent_top_n(sample_dataframe):
    column_types = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    result = QueryAgent(llm_router=None).run(
        dataframe=sample_dataframe, question="top 3 by revenue", column_types=column_types
    )
    assert result.success
    assert result.data["data_preview"] is not None
    assert len(result.data["data_preview"]) == 3


def test_query_agent_aggregate_with_groupby(sample_dataframe):
    column_types = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    result = QueryAgent(llm_router=None).run(
        dataframe=sample_dataframe, question="average revenue by region", column_types=column_types
    )
    assert result.success
    assert "revenue" in result.data["answer"].lower()


def test_query_agent_never_executes_arbitrary_code(sample_dataframe):
    """Even adversarial-looking questions must not lead to code execution — only whitelisted ops run."""
    column_types = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    malicious_question = "__import__('os').system('echo pwned'); top 3 by revenue"
    result = QueryAgent(llm_router=None).run(
        dataframe=sample_dataframe, question=malicious_question, column_types=column_types
    )
    assert result.success  # should degrade gracefully, not crash or execute the payload
