"""Overview tab — KPIs, dataset snapshot, and the auto-built dashboard."""
from __future__ import annotations

from typing import Any

import streamlit as st

from app.agents.visualization_agent import VisualizationAgent
from frontend.components.ui_helpers import render_kpi_row, render_section_title, plotly_chart

visualization_agent = VisualizationAgent()


def render_overview_tab(context: dict[str, Any]) -> None:
    dataframe = context["dataframe"]
    quality = context.get("quality", {})

    render_section_title("📌", "Key Performance Indicators")
    render_kpi_row(context.get("kpis", []))

    render_section_title("🗂️", "Dataset Snapshot")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(dataframe.head(50), use_container_width=True, height=320)
    with col2:
        st.metric("Rows", f"{quality.get('total_rows', len(dataframe)):,}")
        st.metric("Columns", quality.get("total_columns", dataframe.shape[1]))
        st.metric("Health Score", f"{quality.get('health_score', 'N/A')}/100")
        summary = context.get("schema_summary", {})
        if summary:
            st.caption("Detected column types:")
            for semantic, cols in summary.items():
                st.caption(f"• **{semantic}**: {len(cols)} column(s)")

    render_section_title("📊", "Auto-Generated Dashboard")
    layout = context.get("dashboard_layout", {})
    chart_rows = layout.get("chart_rows", [])
    if not chart_rows:
        st.info("No charts recommended for this dataset yet.")
        return

    for row_idx, row in enumerate(chart_rows):
        cols = st.columns(len(row))
        for col_idx, (col, spec) in enumerate(zip(cols, row)):
            with col:
                try:
                    fig = visualization_agent.render(dataframe, spec)
                    st.plotly_chart(fig, use_container_width=True, key=f"overview_chart_{row_idx}_{col_idx}")
                    st.caption(spec.get("reason", ""))
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Could not render '{spec.get('title')}': {exc}")
