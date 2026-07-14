"""Visualizations tab — the full recommended chart gallery, plus a manual chart builder."""
from __future__ import annotations

from typing import Any

import streamlit as st

from app.agents.visualization_agent import VisualizationAgent
from frontend.components.ui_helpers import render_section_title

visualization_agent = VisualizationAgent()

_CHART_OPTIONS = [
    "bar", "line", "area", "pie", "treemap", "sunburst", "scatter", "bubble",
    "histogram", "violin", "box", "waterfall", "funnel", "radar",
    "parallel_coordinates", "pair_plot", "correlation_heatmap",
]


def render_visualization_tab(context: dict[str, Any]) -> None:
    dataframe = context["dataframe"]
    recommendations = context.get("chart_recommendations", [])

    render_section_title("✨", "AI-Recommended Charts")
    if not recommendations:
        st.info("No chart recommendations available.")
    else:
        for i, spec in enumerate(recommendations):
            
            with st.expander(
                f"{spec['title']}",
                expanded=(i == 0)
            ):

                try:

                    fig = visualization_agent.render(
                        dataframe,
                        spec,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"chart_{i}",
                    )

                    st.caption(
                        f"💡 {spec.get('reason','')}"
                    )

                except Exception as exc:

                    st.warning(str(exc))
                    
    st.divider()
    render_section_title("🛠️", "Build Your Own Chart")
    numeric_cols = context.get("numeric_columns", list(dataframe.select_dtypes("number").columns))
    all_cols = list(dataframe.columns)

    c1, c2, c3 = st.columns(3)
    chart_type = c1.selectbox("Chart type", _CHART_OPTIONS)
    x_col = c2.selectbox("X / category column", [None] + all_cols)
    y_col = c3.selectbox("Y / value column", [None] + numeric_cols)

    if st.button("Generate Chart"):
        spec = {"chart_type": chart_type, "title": f"{chart_type.title()}: {y_col or ''} by {x_col or ''}",
                "x": x_col, "y": y_col}
        try:
            fig = visualization_agent.render(dataframe, spec)
            st.plotly_chart(fig, use_container_width=True, key="custom_chart")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not build chart: {exc}")
