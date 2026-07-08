"""Data Quality tab — health score, missing values, duplicates, cleaning log."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.components.ui_helpers import render_health_badge, render_section_title


def render_quality_tab(context: dict[str, Any]) -> None:
    quality = context.get("quality", {})
    if not quality:
        st.info("Run the analysis to see data quality results.")
        return

    render_section_title("🩺", "Overall Health")
    st.markdown(render_health_badge(quality.get("health_score", 0)), unsafe_allow_html=True)
    st.progress(min(quality.get("health_score", 0) / 100, 1.0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{quality.get('total_rows', 0):,}")
    col2.metric("Duplicate Rows", f"{quality.get('duplicate_rows', 0):,}", f"{quality.get('duplicate_pct', 0)}%")
    col3.metric("Missing Cells", f"{quality.get('total_missing_cells', 0):,}", f"{quality.get('missing_pct', 0)}%")
    col4.metric("Columns w/ Missing", len(quality.get("columns_with_missing", [])))

    if quality.get("warnings"):
        render_section_title("⚠️", "Warnings")
        for w in quality["warnings"]:
            st.warning(w)

    missing_by_col = quality.get("missing_by_column", {})
    if missing_by_col:
        render_section_title("🔍", "Missing Values by Column")
        missing_df = pd.DataFrame(
            [{"Column": k, "Missing Count": v} for k, v in missing_by_col.items()]
        ).sort_values("Missing Count", ascending=False)
        st.bar_chart(missing_df.set_index("Column"))

    if context.get("cleaning_actions"):
        render_section_title("🧹", "Cleaning Actions Applied")
        for action in context["cleaning_actions"]:
            st.write(f"✅ {action}")

    if quality.get("constant_columns"):
        st.info(f"Constant-value columns (little analytical value): {quality['constant_columns']}")
    if quality.get("high_cardinality_columns"):
        st.info(f"Likely unique-identifier columns: {quality['high_cardinality_columns']}")
