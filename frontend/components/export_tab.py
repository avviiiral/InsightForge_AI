"""Export tab — download the analysis as PDF, Excel, Markdown, PowerPoint, or CSV."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from app.agents.export_agent import ExportAgent
from app.utils.file_utils import safe_filename
from frontend.components.ui_helpers import render_section_title

export_agent = ExportAgent()

_MIME_TYPES = {
    "pdf": "application/pdf",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "markdown": "text/markdown",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "csv": "text/csv",
}
_EXTENSIONS = {"pdf": "pdf", "excel": "xlsx", "markdown": "md", "pptx": "pptx", "csv": "csv"}


def render_export_tab(context: dict[str, Any], dataset_name: str) -> None:
    render_section_title("📤", "Export Your Analysis")
    st.caption("Generate a shareable report or download the cleaned dataset.")

    stem = safe_filename(Path(dataset_name).stem)
    export_dir = Path(tempfile.gettempdir()) / "insightforge_exports"

    cols = st.columns(5)
    labels = {"pdf": "📄 PDF Report", "excel": "📊 Excel Workbook", "markdown": "📝 Markdown",
              "pptx": "📽️ PowerPoint", "csv": "🧮 Cleaned CSV"}

    for col, (fmt, label) in zip(cols, labels.items()):
        with col:
            if st.button(label, key=f"export_{fmt}", use_container_width=True):
                report = context.get("report")
                result = export_agent.run(
                    export_format=fmt, output_dir=str(export_dir), filename_stem=stem,
                    report=report, dataframe=context.get("dataframe"),
                )
                if result.success:
                    st.session_state[f"export_path_{fmt}"] = result.data["file_path"]
                else:
                    st.error(result.error)

    st.divider()
    for fmt in labels:
        path_key = f"export_path_{fmt}"
        if st.session_state.get(path_key):
            file_path = Path(st.session_state[path_key])
            if file_path.exists():
                st.download_button(
                    f"⬇️ Download {file_path.name}",
                    data=file_path.read_bytes(),
                    file_name=file_path.name,
                    mime=_MIME_TYPES[fmt],
                    key=f"download_{fmt}",
                )
