import pandas as pd

from app.agents.feature_engineering_agent import FeatureEngineeringAgent


def test_datetime_suggestion_only():
    df = pd.DataFrame({
        "OrderDate": ["2024-01-01", "2024-02-01"]
    })

    column_types = {
        "OrderDate": {
            "semantic_type": "datetime"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types
    )

    assert result.success is True
    assert result.data["n_suggestions"] == 1
    assert "Extract year" in result.data["suggestions"][0]["suggestion"]


def test_datetime_feature_generation():
    df = pd.DataFrame({
        "OrderDate": ["2024-01-01", "2024-02-10"]
    })

    column_types = {
        "OrderDate": {
            "semantic_type": "datetime"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types,
        apply_transformations=True
    )

    engineered = result.data["engineered_dataframe"]

    assert "OrderDate_year" in engineered.columns
    assert "OrderDate_month" in engineered.columns
    assert "OrderDate_dayofweek" in engineered.columns
    assert "OrderDate_quarter" in engineered.columns

    assert len(result.data["new_columns"]) == 4


def test_numeric_binning_suggestion():
    df = pd.DataFrame({
        "Sales": list(range(20))
    })

    column_types = {
        "Sales": {
            "semantic_type": "numeric"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types
    )

    assert result.success
    assert result.data["n_suggestions"] == 1
    assert "quartile" in result.data["suggestions"][0]["suggestion"]


def test_numeric_bucket_creation():
    df = pd.DataFrame({
        "Sales": list(range(20))
    })

    column_types = {
        "Sales": {
            "semantic_type": "numeric"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types,
        apply_transformations=True
    )

    engineered = result.data["engineered_dataframe"]

    assert "Sales_bucket" in engineered.columns


def test_numeric_no_bucket_when_low_unique():
    df = pd.DataFrame({
        "Sales": [1, 2, 2, 1]
    })

    column_types = {
        "Sales": {
            "semantic_type": "numeric"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types
    )

    assert result.data["n_suggestions"] == 0


def test_currency_column_behaves_like_numeric():
    df = pd.DataFrame({
        "Revenue": list(range(30))
    })

    column_types = {
        "Revenue": {
            "semantic_type": "currency"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types,
        apply_transformations=True
    )

    engineered = result.data["engineered_dataframe"]

    assert "Revenue_bucket" in engineered.columns


def test_categorical_suggestion():
    df = pd.DataFrame({
        "Category": [
            "A", "B", "C",
            "A", "B", "C"
        ]
    })

    column_types = {
        "Category": {
            "semantic_type": "categorical"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types
    )

    assert result.data["n_suggestions"] == 1
    assert "One-hot encode" in result.data["suggestions"][0]["suggestion"]


def test_categorical_high_cardinality_not_suggested():
    df = pd.DataFrame({
        "Category": [f"C{i}" for i in range(20)]
    })

    column_types = {
        "Category": {
            "semantic_type": "categorical"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types
    )

    assert result.data["n_suggestions"] == 0


def test_empty_column_types():
    df = pd.DataFrame({
        "A": [1, 2, 3]
    })

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types={}
    )

    assert result.success
    assert result.data["n_suggestions"] == 0


def test_unknown_semantic_type():
    df = pd.DataFrame({
        "A": [1, 2, 3]
    })

    column_types = {
        "A": {
            "semantic_type": "unknown"
        }
    }

    agent = FeatureEngineeringAgent()

    result = agent.run(
        dataframe=df,
        column_types=column_types
    )

    assert result.success
    assert result.data["n_suggestions"] == 0