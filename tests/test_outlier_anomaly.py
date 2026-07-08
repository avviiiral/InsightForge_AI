"""Tests for OutlierDetectionAgent and AnomalyDetectionAgent."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.agents.anomaly_detection_agent import AnomalyDetectionAgent
from app.agents.outlier_detection_agent import OutlierDetectionAgent


def test_outlier_agent_flags_injected_outlier(sample_dataframe):
    result = OutlierDetectionAgent().run(dataframe=sample_dataframe, numeric_columns=["revenue"])
    assert result.success
    assert result.data["columns"]["revenue"]["iqr_outlier_count"] >= 1


def test_outlier_agent_clean_data_has_no_outliers():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"col": rng.normal(0, 1, 500)})
    result = OutlierDetectionAgent().run(dataframe=df)
    assert result.data["columns"]["col"]["iqr_outlier_pct"] < 5


def test_anomaly_agent_requires_minimum_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = AnomalyDetectionAgent().run(dataframe=df)
    assert result.success
    assert result.data["anomaly_count"] == 0
    assert "note" in result.data


def test_anomaly_agent_detects_anomalies_in_larger_dataset():
    rng = np.random.default_rng(0)
    normal = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    anomalies = pd.DataFrame({"a": [50, -50], "b": [50, -50]})
    df = pd.concat([normal, anomalies], ignore_index=True)
    result = AnomalyDetectionAgent().run(dataframe=df, contamination=0.02)
    assert result.success
    assert result.data["anomaly_count"] >= 1
