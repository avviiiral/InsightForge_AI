"""Tests for ExportAgent — every supported export format must produce a real file."""
from __future__ import annotations

from app.agents.export_agent import ExportAgent
from app.agents.orchestrator import Orchestrator


def _build_report(sample_dataframe):
    orchestrator = Orchestrator()
    context = orchestrator.run_full_pipeline(sample_dataframe, dataset_name="test_dataset", clean_data=True)
    return context["report"], context["dataframe"]


def test_export_csv(tmp_path, sample_dataframe):
    result = ExportAgent().run(
        export_format="csv", output_dir=str(tmp_path), filename_stem="out", dataframe=sample_dataframe
    )
    assert result.success
    assert (tmp_path / "out.csv").exists()


def test_export_markdown(tmp_path, sample_dataframe):
    report, _ = _build_report(sample_dataframe)
    result = ExportAgent().run(export_format="markdown", output_dir=str(tmp_path), filename_stem="out", report=report)
    assert result.success
    content = (tmp_path / "out.md").read_text()
    assert "Executive Summary" in content


def test_export_excel(tmp_path, sample_dataframe):
    report, _ = _build_report(sample_dataframe)
    result = ExportAgent().run(export_format="excel", output_dir=str(tmp_path), filename_stem="out", report=report)
    assert result.success
    assert (tmp_path / "out.xlsx").exists()


def test_export_pdf(tmp_path, sample_dataframe):
    report, _ = _build_report(sample_dataframe)
    result = ExportAgent().run(export_format="pdf", output_dir=str(tmp_path), filename_stem="out", report=report)
    assert result.success
    path = tmp_path / "out.pdf"
    assert path.exists()
    assert path.stat().st_size > 0


def test_export_pptx(tmp_path, sample_dataframe):
    report, _ = _build_report(sample_dataframe)
    result = ExportAgent().run(export_format="pptx", output_dir=str(tmp_path), filename_stem="out", report=report)
    assert result.success
    assert (tmp_path / "out.pptx").exists()


def test_export_invalid_format_fails(tmp_path, sample_dataframe):
    result = ExportAgent().run(
        export_format="doc", output_dir=str(tmp_path), filename_stem="out", dataframe=sample_dataframe
    )
    assert not result.success
