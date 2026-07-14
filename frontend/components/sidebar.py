"""Sidebar component — data source selection, LLM status, and pipeline controls."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import tempfile
from pathlib import Path

from app.agents.data_loader_agent import DataLoaderAgent
from app.config import SAMPLES_DIR
from app.llm.llm_factory import get_llm_router
from app.utils import db_utils

loader_agent = DataLoaderAgent()
@st.cache_data(show_spinner=False)
def load_dataset(
    file_path: str,
    sheet_name: str | None = None,
    table_name: str | None = None,
):
    result = loader_agent.run(
        file_path=file_path,
        sheet_name=sheet_name,
        table_name=table_name,
    )

    if not result.success:
        raise RuntimeError(result.error)

    return result.data["dataframe"]

def _sample_files() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(SAMPLES_DIR.glob("*.*"))


def render_sidebar() -> dict:
    st.markdown('<div class="if-sidebar-brand">🚀 InsightForge-AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="if-sidebar-tagline">Enterprise AI Data Analytics Agent Platform</div>', unsafe_allow_html=True)

    router = get_llm_router()
    if router.is_enabled():
        st.success(f"LLM Provider: **{router.provider_name}**", icon="🤖")
    else:
        st.info("Running in **deterministic analytics mode** (no LLM configured).", icon="🧮")

    st.divider()
    st.subheader("📂 Data Source")
    source = st.radio(
        "Choose a data source",
        ["Upload File", "Sample Dataset", "Database Connection"],
        label_visibility="collapsed",
    )

    loaded_df: pd.DataFrame | None = None
    dataset_name = None

    if source == "Upload File":
        uploaded = st.file_uploader(
            "Upload CSV, Excel, JSON, Parquet, or SQLite",
            type=["csv", "tsv", "xlsx", "xls", "json", "parquet", "db", "sqlite", "sqlite3"],
        )
        if uploaded is not None:
            temp_dir = Path(tempfile.gettempdir())
            tmp_path = temp_dir / uploaded.name
            tmp_path.write_bytes(uploaded.getbuffer())
            sheet_name = None
            if tmp_path.suffix.lower() in (".xlsx", ".xls"):
                sheets = DataLoaderAgent.list_excel_sheets(str(tmp_path))
                if len(sheets) > 1:
                    sheet_name = st.selectbox("Sheet", sheets)
            table_name = None
            if tmp_path.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
                tables = DataLoaderAgent.list_sqlite_tables(str(tmp_path))
                table_name = st.selectbox("Table", tables)
            try:
                loaded_df = load_dataset(
                    str(tmp_path),
                    sheet_name=sheet_name,
                    table_name=table_name,
                )
                dataset_name = uploaded.name
            except Exception as exc:
                st.error(f"Failed to load file: {exc}")

    elif source == "Sample Dataset":
        samples = _sample_files()
        if not samples:
            st.warning("No sample datasets found in `data/samples/`.")
        else:
            choice = st.selectbox("Choose a sample dataset", [f.name for f in samples])
            chosen_path = next(f for f in samples if f.name == choice)
            try:
                loaded_df = load_dataset(str(chosen_path))
                dataset_name = choice
            except Exception as exc:
                st.error(f"Failed to load sample: {exc}")
    else:  # Database Connection
        db_type = st.selectbox("Database type", ["PostgreSQL", "MySQL"])
        default_uri = "postgresql+psycopg2://user:password@localhost:5432/mydb" if db_type == "PostgreSQL" \
            else "mysql+pymysql://user:password@localhost:3306/mydb"
        conn_uri = st.text_input("Connection URI", value=default_uri)
        mode = st.radio("Read mode", ["Table", "Custom SQL"], horizontal=True)
        if mode == "Table":
            table_name = st.text_input("Table name")
            if st.button("Connect & List Tables"):
                try:
                    st.session_state["_db_tables"] = db_utils.list_tables(conn_uri)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Connection failed: {exc}")
            tables = st.session_state.get("_db_tables", [])
            if tables:
                table_name = st.selectbox("Available tables", tables)
            if table_name and st.button("Load Table"):
                result = loader_agent.run(connection_uri=conn_uri, table_name=table_name)
                if result.success:
                    loaded_df = result.data["dataframe"]
                    dataset_name = table_name
                else:
                    st.error(f"Failed to load table: {result.error}")
        else:
            query = st.text_area("SQL query (SELECT only)", value="SELECT * FROM my_table LIMIT 1000")
            if st.button("Run Query"):
                result = loader_agent.run(connection_uri=conn_uri, sql_query=query)
                if result.success:
                    loaded_df = result.data["dataframe"]
                    dataset_name = "custom_query"
                else:
                    st.error(f"Query failed: {result.error}")

    st.divider()
    clean_data = st.checkbox("Auto-clean data before analysis", value=True)
    run_clicked = st.button("▶ Run Full Analysis", type="primary", use_container_width=True)
    reset_clicked = st.button("↺ Reset Session", use_container_width=True)

    return {
        "loaded_df": loaded_df,
        "dataset_name": dataset_name,
        "clean_data": clean_data,
        "run_clicked": run_clicked,
        "reset_clicked": reset_clicked,
    }
