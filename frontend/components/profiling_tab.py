"""Profiling tab — detailed per-column statistics with an expander per column."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.components.ui_helpers import render_section_title


def render_profiling_tab(context: dict[str, Any]) -> None:
    profiles = context.get("profiles", {})
    column_types = context.get("column_types", {})
    if not profiles:
        st.info("Run the analysis to see column profiles.")
        return

    render_section_title("🔬", "Column Profiles")
    search = st.text_input("Filter columns", placeholder="Type to filter by column name...")

    for col, profile in profiles.items():
        if search and search.lower() not in col.lower():
            continue
        semantic = column_types.get(col, {}).get("semantic_type", "unknown")
        pandas_dtype = column_types.get(col, {}).get("pandas_dtype", "")
        with st.expander(f"**{col}**  ·  {semantic}  ·  {pandas_dtype}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Count", profile.get("count", 0))
            c2.metric("Missing", f"{profile.get('missing', 0)} ({profile.get('missing_pct', 0)}%)")
            c3.metric("Unique", profile.get("unique", 0))
            c4.metric("Sample", ", ".join(str(v) for v in profile.get("sample_values", [])[:3]))

            stats = profile.get("stats", {})
            if "top_values" in stats:
                st.caption("Top values:")
                st.dataframe(
                    pd.DataFrame(list(stats["top_values"].items()), columns=["Value", "Count"]),
                    use_container_width=True, hide_index=True,
                )
            elif stats:
                stats_display = pd.DataFrame(
                    [{"Metric": k, "Value": "—" if v is None else str(v)} for k, v in stats.items()]
                )
                st.dataframe(stats_display, use_container_width=True, hide_index=True)
