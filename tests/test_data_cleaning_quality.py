"""Tests for DataCleaningAgent and DataQualityAgent."""
from __future__ import annotations

import numpy as np

from app.agents.data_cleaning_agent import DataCleaningAgent
from app.agents.data_quality_agent import DataQualityAgent
from app.agents.schema_detection_agent import SchemaDetectionAgent


def test_quality_agent_detects_missing_and_duplicates(sample_dataframe):
    agent = DataQualityAgent()
    result = agent.run(dataframe=sample_dataframe)
    assert result.success
    assert result.data["duplicate_rows"] >= 5
    assert result.data["total_missing_cells"] >= 10
    assert 0 <= result.data["health_score"] <= 100


def test_quality_agent_perfect_dataset_scores_high():
    import pandas as pd
    df = pd.DataFrame({"a": range(50), "b": range(50, 100)})
    result = DataQualityAgent().run(dataframe=df)
    assert result.data["health_score"] >= 95


def test_cleaning_agent_removes_duplicates(sample_dataframe):
    schema = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    result = DataCleaningAgent().run(dataframe=sample_dataframe, column_types=schema)
    assert result.success
    cleaned = result.data["cleaned_dataframe"]
    assert cleaned.duplicated().sum() == 0
    assert result.data["rows_after"] <= result.data["rows_before"]


def test_cleaning_agent_imputes_missing_numeric(sample_dataframe):
    schema = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    result = DataCleaningAgent().run(dataframe=sample_dataframe, column_types=schema)
    cleaned = result.data["cleaned_dataframe"]
    assert cleaned["satisfaction_score"].isna().sum() == 0


def test_cleaning_agent_does_not_mutate_original(sample_dataframe):
    original_copy = sample_dataframe.copy(deep=True)
    schema = SchemaDetectionAgent().run(dataframe=sample_dataframe).data["column_types"]
    DataCleaningAgent().run(dataframe=sample_dataframe, column_types=schema)
    pd_equal = sample_dataframe.equals(original_copy)
    assert pd_equal, "DataCleaningAgent must not mutate the input DataFrame in place."
