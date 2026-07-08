"""Forecast tab — lets the user pick a date + value column and view a projection."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.agents.forecast_agent import ForecastAgent
from frontend.components.ui_helpers import render_section_title

forecast_agent = ForecastAgent()


def render_forecast_tab(context: dict[str, Any]) -> None:
    dataframe = context["dataframe"]
    datetime_cols = context.get("datetime_columns", [])
    numeric_cols = context.get("numeric_columns", [])

    render_section_title("📈", "Time-Series Forecast")

    if not datetime_cols:
        st.info("No datetime column was detected in this dataset — forecasting requires one.")
        return
    if not numeric_cols:
        st.info("No numeric column available to forecast.")
        return

    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date column", datetime_cols)
    value_col = c2.selectbox("Value column", numeric_cols)
    periods = c3.slider("Periods to forecast ahead", min_value=1, max_value=24, value=6)

    if st.button("Run Forecast", type="primary"):
        result = forecast_agent.run(
            dataframe=dataframe, date_column=date_col, value_column=value_col, periods_ahead=periods
        )
        if not result.success or result.data.get("error"):
            st.error(result.error or result.data.get("error"))
            return

        data = result.data
        st.session_state["last_forecast"] = data
        history = pd.DataFrame(data["history"])
        forecast = pd.DataFrame(data["forecast"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["period"], y=history["value"], mode="lines+markers",
                                  name="Historical", line=dict(color="#2563eb")))
        fig.add_trace(go.Scatter(x=forecast["period"], y=forecast["value"], mode="lines+markers",
                                  name="Forecast", line=dict(color="#f97316", dash="dash")))
        fig.update_layout(title=f"{value_col} Forecast ({data['method']})", template="plotly_white",
                           xaxis_title="Period", yaxis_title=value_col)
        st.plotly_chart(fig, use_container_width=True, key="forecast_chart")

        st.caption(f"Method: **{data['method']}** · Frequency: **{data['frequency']}** · "
                   f"Seasonal period: **{data.get('seasonal_period') or 'none detected'}**")

        with st.expander("View forecasted values"):
            st.dataframe(forecast, use_container_width=True, hide_index=True)
