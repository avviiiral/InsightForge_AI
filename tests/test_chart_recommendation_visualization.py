"""Tests for ChartRecommendationAgent and VisualizationAgent."""
from __future__ import annotations

import json

from app.agents.chart_recommendation_agent import ChartRecommendationAgent
from app.agents.schema_detection_agent import SchemaDetectionAgent
from app.agents.visualization_agent import VisualizationAgent


def test_chart_recommendation_produces_valid_specs(sample_dataframe):
    column_types = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    result = ChartRecommendationAgent().run(dataframe=sample_dataframe, column_types=column_types)
    assert result.success
    recs = result.data["recommendations"]
    assert len(recs) > 0
    for spec in recs:
        assert "chart_type" in spec and "title" in spec and "reason" in spec


def test_visualization_agent_renders_all_recommended_charts(sample_dataframe):
    column_types = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    recs = ChartRecommendationAgent().run(dataframe=sample_dataframe, column_types=column_types).data["recommendations"]

    viz_agent = VisualizationAgent()
    for spec in recs:
        result = viz_agent.run(dataframe=sample_dataframe, spec=spec)
        assert result.success, f"Chart '{spec['chart_type']}' failed: {result.error}"
        parsed = json.loads(result.data["figure_json"])
        assert "data" in parsed


def test_visualization_agent_rejects_unknown_chart_type(sample_dataframe):
    result = VisualizationAgent().run(dataframe=sample_dataframe, spec={"chart_type": "not_a_real_chart"})
    assert not result.success
