"""Tests for StatisticsAgent and CorrelationAgent."""
from __future__ import annotations

from app.agents.correlation_agent import CorrelationAgent
from app.agents.statistics_agent import StatisticsAgent


def test_statistics_agent_basic_output(numeric_only_dataframe):
    result = StatisticsAgent().run(dataframe=numeric_only_dataframe)
    assert result.success
    summary = result.data["numeric_summary"]
    assert "x" in summary and "y" in summary and "z" in summary
    for col_stats in summary.values():
        assert "mean" in col_stats and "std" in col_stats and "distribution_shape" in col_stats


def test_correlation_agent_detects_strong_relationship(numeric_only_dataframe):
    result = CorrelationAgent().run(dataframe=numeric_only_dataframe)
    assert result.success
    top_pairs = result.data["top_pairs"]
    xy_pair = next(p for p in top_pairs if {p["column_a"], p["column_b"]} == {"x", "y"})
    assert abs(xy_pair["correlation"]) > 0.8
    assert "strong" in xy_pair["strength"]


def test_correlation_agent_handles_insufficient_columns():
    import pandas as pd
    df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
    result = CorrelationAgent().run(dataframe=df)
    assert result.success
    assert result.data["top_pairs"] == []
