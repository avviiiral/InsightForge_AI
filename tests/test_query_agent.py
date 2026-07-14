from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.agents.query_agent import QueryAgent


@pytest.fixture
def sample_df():

    return pd.DataFrame(
        {
            "Category": [
                "A",
                "A",
                "B",
                "B",
                "C",
            ],
            "Sales": [
                100,
                200,
                300,
                400,
                500,
            ],
            "Profit": [
                10,
                20,
                30,
                40,
                50,
            ],
            "Region": [
                "East",
                "West",
                "East",
                "West",
                "North",
            ],
        }
    )


@pytest.fixture
def column_types():

    return {
        "Category": {
            "semantic_type": "categorical"
        },
        "Sales": {
            "semantic_type": "numeric"
        },
        "Profit": {
            "semantic_type": "numeric"
        },
        "Region": {
            "semantic_type": "categorical"
        },
    }


@pytest.fixture
def agent():

    return QueryAgent()


@pytest.fixture
def llm_agent():

    router = MagicMock()

    router.is_enabled.return_value = True

    router.generate.return_value = """
    {
        "intent":"aggregate",
        "column":"Sales",
        "group_by":null,
        "agg":"sum",
        "n":5
    }
    """

    return QueryAgent(llm_router=router)

# ---------------------------------------------------------
# Intent Parsing
# ---------------------------------------------------------


def test_keyword_aggregate(agent, sample_df, column_types):

    result = agent.run(
        dataframe=sample_df,
        question="What is the total Sales?",
        column_types=column_types,
    )

    assert result.success
    assert result.data["intent"]["intent"] == "aggregate"
    assert result.data["used_llm"] is False


def test_keyword_top_n(agent, sample_df, column_types):

    result = agent.run(
        dataframe=sample_df,
        question="Top 3 Sales",
        column_types=column_types,
    )

    assert result.success
    assert result.data["intent"]["intent"] == "top_n"


def test_keyword_describe(agent, sample_df, column_types):

    result = agent.run(
        dataframe=sample_df,
        question="Describe Sales",
        column_types=column_types,
    )

    assert result.success
    assert result.data["intent"]["intent"] == "describe_column"


def test_keyword_correlation(agent, sample_df, column_types):

    result = agent.run(
        dataframe=sample_df,
        question="Correlation between Sales and Profit",
        column_types=column_types,
    )

    assert result.success
    assert result.data["intent"]["intent"] == "correlation_lookup"


def test_keyword_general_summary(agent, sample_df):

    result = agent.run(
        dataframe=sample_df,
        question="Give me summary",
    )

    assert result.success
    assert result.data["intent"]["intent"] == "general_summary"


# ---------------------------------------------------------
# LLM Parsing
# ---------------------------------------------------------


def test_llm_intent(llm_agent, sample_df):

    result = llm_agent.run(
        dataframe=sample_df,
        question="Total sales",
    )

    assert result.success
    assert result.data["used_llm"] is True
    assert result.data["intent"]["intent"] == "aggregate"
    
# ---------------------------------------------------------
# Intent Execution
# ---------------------------------------------------------


def test_execute_sum(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "aggregate",
            "column": "Sales",
            "group_by": None,
            "agg": "sum",
            "n": 5,
        },
        list(sample_df.columns),
    )

    assert "1500" in answer
    assert preview is None


def test_execute_groupby(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "aggregate",
            "column": "Sales",
            "group_by": "Category",
            "agg": "sum",
            "n": 5,
        },
        list(sample_df.columns),
    )

    assert "grouped by" in answer.lower()
    assert len(preview) == 3


def test_execute_top_rows(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "top_n",
            "column": "Sales",
            "group_by": None,
            "agg": "sum",
            "n": 2,
        },
        list(sample_df.columns),
    )

    assert len(preview) == 2


def test_execute_top_groups(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "top_n",
            "column": "Sales",
            "group_by": "Category",
            "agg": "sum",
            "n": 2,
        },
        list(sample_df.columns),
    )

    assert len(preview) == 2


def test_execute_describe(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "describe_column",
            "column": "Sales",
            "group_by": None,
            "agg": None,
            "n": 5,
        },
        list(sample_df.columns),
    )

    assert preview is not None
    assert len(preview) > 0


def test_execute_correlation(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "correlation_lookup",
            "column": "Sales",
            "group_by": "Profit",
            "agg": None,
            "n": 5,
        },
        list(sample_df.columns),
    )

    assert "correlation" in answer.lower()
    assert preview is None


def test_execute_invalid_correlation(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "correlation_lookup",
            "column": "Sales",
            "group_by": "Category",
            "agg": None,
            "n": 5,
        },
        list(sample_df.columns),
    )

    assert "requires two numeric columns" in answer.lower()


def test_general_summary(agent, sample_df):

    answer, preview = agent._execute_intent(
        sample_df,
        {
            "intent": "general_summary",
            "column": None,
            "group_by": None,
            "agg": None,
            "n": 5,
        },
        list(sample_df.columns),
    )

    assert "rows" in answer.lower()
    assert preview is None


def test_invalid_llm_json(sample_df):

    router = MagicMock()

    router.is_enabled.return_value = True
    router.generate.return_value = "INVALID JSON"

    agent = QueryAgent(llm_router=router)

    result = agent.run(
        dataframe=sample_df,
        question="total sales",
    )

    assert result.success
    assert result.data["used_llm"] is False


def test_validate_intent():

    intent = QueryAgent._validate_intent(
        {
            "intent": "aggregate",
            "column": "Sales",
            "group_by": "Region",
            "agg": "sum",
            "n": 10,
        },
        ["Sales", "Region"],
    )

    assert intent["intent"] == "aggregate"
    assert intent["agg"] == "sum"


def test_validate_bad_column():

    intent = QueryAgent._validate_intent(
        {
            "intent": "aggregate",
            "column": "ABC",
            "group_by": "XYZ",
            "agg": "sum",
            "n": 5,
        },
        ["Sales"],
    )

    assert intent["column"] is None
    assert intent["group_by"] is None