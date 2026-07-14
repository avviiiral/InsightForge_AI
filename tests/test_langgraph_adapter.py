"""Tests for app.agents.langgraph_adapter."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd
import pytest

import app.agents.langgraph_adapter as adapter


# ------------------------------------------------------------------
# Fake LangGraph
# ------------------------------------------------------------------

class FakeCompiledGraph:
    def __init__(self, graph):
        self.graph = graph


class FakeStateGraph:
    def __init__(self, state):
        self.state = state
        self.nodes = {}
        self.edges = []
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, start, end):
        self.edges.append((start, end))

    def set_entry_point(self, name):
        self.entry = name

    def compile(self):
        return FakeCompiledGraph(self)


# ------------------------------------------------------------------
# Dummy AgentResult
# ------------------------------------------------------------------

class DummyResult:

    def __init__(self, data):
        self.data = data


# ------------------------------------------------------------------
# Dummy Agents
# ------------------------------------------------------------------

class DummySchemaAgent:

    def run(self, **kwargs):

        return DummyResult(
            {
                "column_types": {
                    "sales": {
                        "semantic_type": "numeric"
                    },
                    "profit": {
                        "semantic_type": "numeric"
                    },
                }
            }
        )


class DummyQualityAgent:

    def run(self, **kwargs):

        return DummyResult(
            {
                "health_score": 100,
            }
        )


class DummyStatisticsAgent:

    def run(self, **kwargs):

        return DummyResult(
            {
                "numeric_summary": {}
            }
        )


class DummyCorrelationAgent:

    def run(self, **kwargs):

        return DummyResult(
            {
                "top_pairs": []
            }
        )


class DummyKPIAgent:

    def run(self, **kwargs):

        return DummyResult(
            {
                "kpis": [
                    {
                        "name": "Total Sales"
                    }
                ]
            }
        )


class DummyInsightAgent:

    def __init__(self, llm_router=None):
        pass

    def run(self, **kwargs):

        return DummyResult(
            {
                "insights": [
                    {
                        "title": "Insight"
                    }
                ]
            }
        )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_df():

    return pd.DataFrame(
        {
            "sales": [10, 20, 30],
            "profit": [1, 2, 3],
        }
    )
# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def install_fake_langgraph(monkeypatch):

    graph_module = ModuleType("langgraph.graph")

    graph_module.END = "__END__"

    graph_module.StateGraph = FakeStateGraph

    langgraph_module = ModuleType("langgraph")

    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module


def install_dummy_agents(monkeypatch):

    monkeypatch.setattr(
        adapter,
        "SchemaDetectionAgent",
        DummySchemaAgent,
    )

    monkeypatch.setattr(
        adapter,
        "DataQualityAgent",
        DummyQualityAgent,
    )

    monkeypatch.setattr(
        adapter,
        "StatisticsAgent",
        DummyStatisticsAgent,
    )

    monkeypatch.setattr(
        adapter,
        "CorrelationAgent",
        DummyCorrelationAgent,
    )

    monkeypatch.setattr(
        adapter,
        "KPIAgent",
        DummyKPIAgent,
    )

    monkeypatch.setattr(
        adapter,
        "InsightAgent",
        DummyInsightAgent,
    )

    monkeypatch.setattr(
        adapter,
        "get_llm_router",
        lambda: MagicMock(),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_build_pipeline_import_error(monkeypatch):

    sys.modules.pop("langgraph", None)
    sys.modules.pop("langgraph.graph", None)

    real_import = __import__

    def fake_import(name, *args, **kwargs):

        if name == "langgraph.graph":
            raise ImportError("missing")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "builtins.__import__",
        fake_import,
    )

    with pytest.raises(ImportError):

        adapter.build_langgraph_pipeline()


def test_pipeline_build(monkeypatch):

    install_fake_langgraph(monkeypatch)

    install_dummy_agents(monkeypatch)

    graph = adapter.build_langgraph_pipeline()

    assert isinstance(
        graph,
        FakeCompiledGraph,
    )

    assert graph.graph.entry == "schema_detection"

    assert "schema_detection" in graph.graph.nodes

    assert "data_quality" in graph.graph.nodes

    assert "statistics" in graph.graph.nodes

    assert "correlation" in graph.graph.nodes

    assert "kpis" in graph.graph.nodes

    assert "insights" in graph.graph.nodes
# ------------------------------------------------------------------
# Execute Every Node
# ------------------------------------------------------------------


def test_execute_all_nodes(
    monkeypatch,
    sample_df,
):

    install_fake_langgraph(monkeypatch)

    install_dummy_agents(monkeypatch)

    compiled = adapter.build_langgraph_pipeline()

    graph = compiled.graph

    state = {
        "dataframe": sample_df,
    }

    # -----------------------------
    # Schema
    # -----------------------------

    schema_state = graph.nodes["schema_detection"](state)

    assert "column_types" in schema_state

    state.update(schema_state)

    # -----------------------------
    # Quality
    # -----------------------------

    quality_state = graph.nodes["data_quality"](state)

    assert quality_state["quality"]["health_score"] == 100

    state.update(quality_state)

    # -----------------------------
    # Statistics
    # -----------------------------

    statistics_state = graph.nodes["statistics"](state)

    assert "statistics" in statistics_state

    state.update(statistics_state)

    # -----------------------------
    # Correlation
    # -----------------------------

    correlation_state = graph.nodes["correlation"](state)

    assert "correlation" in correlation_state

    state.update(correlation_state)

    # -----------------------------
    # KPIs
    # -----------------------------

    kpi_state = graph.nodes["kpis"](state)

    assert len(kpi_state["kpis"]) == 1

    state.update(kpi_state)

    # -----------------------------
    # Insights
    # -----------------------------

    insight_state = graph.nodes["insights"](state)

    assert len(insight_state["insights"]) == 1


# ------------------------------------------------------------------
# Missing DataFrame Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_name",
    [
        "schema_detection",
        "data_quality",
        "statistics",
        "correlation",
        "kpis",
    ],
)
def test_nodes_require_dataframe(
    monkeypatch,
    node_name,
):

    install_fake_langgraph(monkeypatch)

    install_dummy_agents(monkeypatch)

    compiled = adapter.build_langgraph_pipeline()

    graph = compiled.graph

    with pytest.raises(ValueError):

        graph.nodes[node_name]({})