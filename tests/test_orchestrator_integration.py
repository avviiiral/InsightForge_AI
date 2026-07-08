"""Integration test — the entire Orchestrator pipeline must run end-to-end without failures."""
from __future__ import annotations

from app.agents.orchestrator import Orchestrator


def test_full_pipeline_runs_without_agent_failures(sample_dataframe):
    orchestrator = Orchestrator()
    context = orchestrator.run_full_pipeline(sample_dataframe, dataset_name="integration_test", clean_data=True)

    failed_steps = [entry for entry in context["execution_log"] if not entry["success"]]
    assert not failed_steps, f"Pipeline steps failed: {failed_steps}"

    # Every major artifact should be present.
    for key in (
        "column_types", "quality", "profiles", "statistics", "correlation", "outliers",
        "anomalies", "chart_recommendations", "kpis", "insights", "recommendations",
        "business_narrative", "executive_summary", "dashboard_layout", "report",
    ):
        assert key in context, f"Missing expected pipeline output: '{key}'"

    assert context["quality"]["health_score"] >= 0
    assert isinstance(context["kpis"], list) and len(context["kpis"]) > 0
    assert isinstance(context["insights"], list)
    assert context["report"]["title"]


def test_pipeline_without_cleaning_still_succeeds(sample_dataframe):
    orchestrator = Orchestrator()
    context = orchestrator.run_full_pipeline(sample_dataframe, dataset_name="no_clean_test", clean_data=False)
    failed_steps = [entry for entry in context["execution_log"] if not entry["success"]]
    assert not failed_steps
    # Without cleaning, duplicate rows injected by the fixture should still be present.
    assert context["dataframe"].duplicated().sum() >= 1


def test_pipeline_is_llm_optional(sample_dataframe):
    orchestrator = Orchestrator()
    context = orchestrator.run_full_pipeline(sample_dataframe, dataset_name="llm_optional_test")
    assert context["llm_provider"] == "deterministic-fallback"
    assert context["llm_enabled"] is False
