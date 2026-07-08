"""Tests for KPIAgent and ForecastAgent."""
from __future__ import annotations

from app.agents.forecast_agent import ForecastAgent
from app.agents.kpi_agent import KPIAgent


def test_kpi_agent_generates_kpis(sample_dataframe):
    result = KPIAgent().run(dataframe=sample_dataframe, numeric_columns=["revenue", "quantity"])
    assert result.success
    kpi_names = [k["name"] for k in result.data["kpis"]]
    assert "Total revenue" in kpi_names
    assert "Total Records" in kpi_names


def test_kpi_agent_formats_large_numbers():
    import pandas as pd
    df = pd.DataFrame({"revenue": [2_500_000, 3_000_000]})
    result = KPIAgent().run(dataframe=df, numeric_columns=["revenue"])
    total_kpi = next(k for k in result.data["kpis"] if k["name"] == "Total revenue")
    assert "M" in total_kpi["formatted_value"]


def test_forecast_agent_produces_future_periods(timeseries_dataframe):
    result = ForecastAgent().run(
        dataframe=timeseries_dataframe, date_column="week", value_column="sales", periods_ahead=8
    )
    assert result.success
    assert len(result.data["forecast"]) == 8
    assert all(point["is_forecast"] for point in result.data["forecast"])
    assert all(not point["is_forecast"] for point in result.data["history"])


def test_forecast_agent_detects_seasonality(timeseries_dataframe):
    result = ForecastAgent().run(
        dataframe=timeseries_dataframe, date_column="week", value_column="sales", periods_ahead=4
    )
    assert result.data["seasonal_period"] in (52, None)  # weekly data -> yearly seasonality if enough history


def test_forecast_agent_handles_insufficient_data():
    import pandas as pd
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "value": [1, 2, 3]})
    result = ForecastAgent().run(dataframe=df, date_column="date", value_column="value")
    assert result.success
    assert "error" in result.data
