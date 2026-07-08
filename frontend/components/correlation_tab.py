"""Correlation & Anomalies tab — correlation heatmap, top pairs, outliers, multivariate anomalies."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.agents.root_cause_agent import RootCauseAgent
from app.agents.visualization_agent import VisualizationAgent
from frontend.components.ui_helpers import render_section_title

visualization_agent = VisualizationAgent()
root_cause_agent = RootCauseAgent()


def render_correlation_tab(context: dict[str, Any]) -> None:
    dataframe = context["dataframe"]
    correlation = context.get("correlation", {})
    outliers = context.get("outliers", {})
    anomalies = context.get("anomalies", {})
    numeric_cols = context.get("numeric_columns", [])

    render_section_title("🔗", "Correlation Heatmap")
    if len(numeric_cols) >= 2:
        fig = visualization_agent.render(dataframe, {"chart_type": "correlation_heatmap", "title": "Correlation Heatmap"})
        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
    else:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")

    if correlation.get("top_pairs"):
        render_section_title("🏆", "Strongest Relationships")
        st.dataframe(pd.DataFrame(correlation["top_pairs"]), use_container_width=True, hide_index=True)

    render_section_title("📉", "Outliers (Single-Column, IQR Method)")
    outlier_cols = outliers.get("columns", {})
    if outlier_cols:
        rows = [
            {"Column": col, "Outliers Found": info["iqr_outlier_count"],
             "% of Data": info["iqr_outlier_pct"], "Valid Range": f"[{info['lower_bound']}, {info['upper_bound']}]"}
            for col, info in outlier_cols.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No significant single-column outliers detected.")

    render_section_title("🕵️", "Multivariate Anomalies (Isolation Forest)")
    if anomalies.get("anomaly_count"):
        st.warning(f"{anomalies['anomaly_count']} anomalous rows detected ({anomalies.get('anomaly_pct')}% of data).")
        top_rows = [{"Row Index": r["row_index"], "Anomaly Score": r["anomaly_score"], **r["values"]}
                    for r in anomalies.get("top_anomalies", [])]
        if top_rows:
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
    else:
        st.info(anomalies.get("note", "No multivariate anomalies detected."))

    if numeric_cols:
        st.divider()
        render_section_title("🎯", "Root Cause Explorer")
        target = st.selectbox("Investigate drivers behind:", numeric_cols)
        if st.button("Analyze Root Cause"):
            result = root_cause_agent.run(
                dataframe=dataframe, target_column=target, correlation=correlation, anomaly_result=anomalies
            )
            if result.success:
                st.info(result.data["narrative"])
                if result.data["candidate_drivers"]:
                    st.dataframe(pd.DataFrame(result.data["candidate_drivers"]), use_container_width=True, hide_index=True)
            else:
                st.error(result.error)
