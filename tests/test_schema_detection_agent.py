"""Tests for SchemaDetectionAgent — semantic type inference."""
from __future__ import annotations

from app.agents.schema_detection_agent import SchemaDetectionAgent


def test_detects_expected_semantic_types(sample_dataframe):
    agent = SchemaDetectionAgent()
    result = agent.run(dataframe=sample_dataframe)
    assert result.success

    column_types = result.data["column_types"]
    assert column_types["order_date"]["semantic_type"] == "datetime"
    assert column_types["quantity"]["semantic_type"] == "numeric"
    assert column_types["revenue"]["semantic_type"] in ("numeric", "currency")
    assert column_types["customer_email"]["semantic_type"] == "email"
    assert column_types["is_active"]["semantic_type"] == "boolean"
    assert column_types["category"]["semantic_type"] == "categorical"
    assert column_types["order_id"]["semantic_type"] == "identifier"


def test_summary_groups_columns_by_type(sample_dataframe):
    agent = SchemaDetectionAgent()
    result = agent.run(dataframe=sample_dataframe)
    summary = result.data["summary"]
    assert "datetime" in summary
    assert "order_date" in summary["datetime"]


def test_handles_no_column_name_dependency(sample_dataframe):
    """Renaming columns to meaningless labels must not break detection of core types."""
    renamed = sample_dataframe.rename(columns={"order_date": "col_a", "quantity": "col_b"})
    agent = SchemaDetectionAgent()
    result = agent.run(dataframe=renamed)
    assert result.success
    assert result.data["column_types"]["col_a"]["semantic_type"] == "datetime"
    assert result.data["column_types"]["col_b"]["semantic_type"] == "numeric"
