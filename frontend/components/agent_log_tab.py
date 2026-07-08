"""Agent Log tab — transparency into every agent's run, timing, and status."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.components.ui_helpers import render_section_title


def render_agent_log_tab(context: dict[str, Any]) -> None:
    render_section_title("🧠", "Agent Execution Log")
    log = context.get("execution_log", [])
    if not log:
        st.info("Run the analysis to see the agent execution log.")
        return

    total_ms = sum(entry["duration_ms"] for entry in log)
    failed = [entry for entry in log if not entry["success"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Agents Run", len(log))
    c2.metric("Total Time", f"{total_ms:.1f} ms")
    c3.metric("Failures", len(failed))

    st.dataframe(
        pd.DataFrame(log)[["agent_name", "success", "duration_ms", "error"]],
        use_container_width=True, hide_index=True,
    )

    st.caption(f"LLM provider in use: **{context.get('llm_provider', 'deterministic-fallback')}** "
               f"({'enabled' if context.get('llm_enabled') else 'fallback mode'})")
