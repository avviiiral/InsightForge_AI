"""InsightForge-AI — Streamlit dashboard entry point.

Run with:
    streamlit run frontend/streamlit_app.py

This is the primary, fully self-contained user interface. It uses the
shared `app/` core library directly (agents + orchestrator), so it works
even if the FastAPI backend (`backend/main.py`) is not running.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable regardless of the working directory
# Streamlit was launched from.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st  # noqa: E402

from app.agents.orchestrator import Orchestrator  # noqa: E402
from frontend.components.agent_log_tab import render_agent_log_tab  # noqa: E402
from frontend.components.correlation_tab import render_correlation_tab  # noqa: E402
from frontend.components.export_tab import render_export_tab  # noqa: E402
from frontend.components.forecast_tab import render_forecast_tab  # noqa: E402
from frontend.components.insights_tab import render_insights_tab  # noqa: E402
from frontend.components.overview_tab import render_overview_tab  # noqa: E402
from frontend.components.profiling_tab import render_profiling_tab  # noqa: E402
from frontend.components.quality_tab import render_quality_tab  # noqa: E402
from frontend.components.query_tab import render_query_tab  # noqa: E402
from frontend.components.sidebar import render_sidebar  # noqa: E402
from frontend.components.ui_helpers import load_css, render_hero  # noqa: E402
from frontend.components.visualization_tab import render_visualization_tab  # noqa: E402

st.set_page_config(
    page_title="InsightForge-AI | Enterprise Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

if "context" not in st.session_state:
    st.session_state["context"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None
if "pending_df" not in st.session_state:
    st.session_state["pending_df"] = None


def main() -> None:
    with st.sidebar:
        sidebar_state = render_sidebar()

    if sidebar_state["reset_clicked"]:
        for key in ("context", "dataset_name", "pending_df", "chat_history"):
            st.session_state.pop(key, None)
        st.rerun()

    if sidebar_state["loaded_df"] is not None:
        st.session_state["pending_df"] = sidebar_state["loaded_df"]
        st.session_state["dataset_name"] = sidebar_state["dataset_name"]

    render_hero(
        "InsightForge-AI",
        "Upload any dataset — CSV, Excel, JSON, Parquet, SQLite, or a live database — "
        "and get instant, agent-powered analytics, dashboards, forecasts, and reports.",
    )

    if sidebar_state["run_clicked"]:
        if st.session_state["pending_df"] is None:
            st.warning("Please load a dataset from the sidebar first.")
        else:
            with st.spinner("Running the full agent pipeline — this may take a few seconds..."):
                orchestrator = Orchestrator()
                context = orchestrator.run_full_pipeline(
                    st.session_state["pending_df"],
                    dataset_name=st.session_state["dataset_name"] or "dataset",
                    clean_data=sidebar_state["clean_data"],
                )
                st.session_state["context"] = context
            st.toast("Analysis complete!", icon="✅")

    context = st.session_state["context"]
    if context is None:
        st.info("👈 Load a dataset and click **Run Full Analysis** in the sidebar to get started.")
        st.markdown(
            "**Supported sources:** CSV · TSV · Excel · JSON · Parquet · SQLite · PostgreSQL · MySQL\n\n"
            "**What you'll get:** automatic schema & data-type detection, a health score, "
            "correlation & outlier analysis, an auto-built dashboard, forecasts, natural-language "
            "querying, and one-click exports to PDF / Excel / Markdown / PowerPoint."
        )
        return

    tabs = st.tabs([
        "📌 Overview", "🩺 Data Quality", "🔬 Profiling", "📊 Visualizations",
        "🔗 Correlation & Anomalies", "📈 Forecast", "💡 Insights & Recommendations",
        "💬 Ask Your Data", "📤 Export", "🧠 Agent Log",
    ])

    with tabs[0]:
        render_overview_tab(context)
    with tabs[1]:
        render_quality_tab(context)
    with tabs[2]:
        render_profiling_tab(context)
    with tabs[3]:
        render_visualization_tab(context)
    with tabs[4]:
        render_correlation_tab(context)
    with tabs[5]:
        render_forecast_tab(context)
    with tabs[6]:
        render_insights_tab(context)
    with tabs[7]:
        render_query_tab(context)
    with tabs[8]:
        render_export_tab(context, st.session_state["dataset_name"] or "dataset")
    with tabs[9]:
        render_agent_log_tab(context)


if __name__ == "__main__":
    main()
