import pandas as pd

from app.agents.root_cause_agent import RootCauseAgent


def sample_dataframe():
    return pd.DataFrame(
        {
            "Sales": [100, 120, 140, 160, 180],
            "Profit": [10, 12, 14, 16, 18],
            "Discount": [5, 10, 5, 20, 15],
        }
    )


def test_no_correlation_data():
    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
    )

    assert result.success
    assert result.data["candidate_drivers"] == []
    assert result.data["anomaly_overlap"] is None
    assert "No strongly correlated" in result.data["narrative"]


def test_candidate_drivers_sorted():
    correlation = {
        "top_pairs": [
            {
                "column_a": "Sales",
                "column_b": "Profit",
                "correlation": 0.95,
                "strength": "Very Strong",
            },
            {
                "column_a": "Sales",
                "column_b": "Discount",
                "correlation": 0.65,
                "strength": "Strong",
            },
        ]
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        correlation=correlation,
    )

    drivers = result.data["candidate_drivers"]

    assert len(drivers) == 2
    assert drivers[0]["driver"] == "Profit"
    assert drivers[1]["driver"] == "Discount"


def test_top_n_limit():
    correlation = {
        "top_pairs": [
            {
                "column_a": "Sales",
                "column_b": f"C{i}",
                "correlation": 0.9 - i * 0.01,
                "strength": "Strong",
            }
            for i in range(10)
        ]
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        correlation=correlation,
        top_n=3,
    )

    assert len(result.data["candidate_drivers"]) == 3


def test_anomaly_overlap():
    anomaly = {
        "top_anomalies": [
            {
                "values": {
                    "Sales": 300
                }
            },
            {
                "values": {
                    "Sales": 250
                }
            },
        ]
    }

    correlation = {
        "top_pairs": [
            {
                "column_a": "Sales",
                "column_b": "Profit",
                "correlation": 0.95,
                "strength": "Very Strong",
            }
        ]
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        correlation=correlation,
        anomaly_result=anomaly,
    )

    overlap = result.data["anomaly_overlap"]

    assert overlap["n_anomalous_rows_with_target"] == 2
    assert overlap["avg_target_value_in_anomalies"] == 275.0
    assert overlap["overall_avg_target_value"] == 140.0


def test_anomaly_without_target_column():
    anomaly = {
        "top_anomalies": [
            {
                "values": {
                    "Profit": 99
                }
            }
        ]
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        anomaly_result=anomaly,
    )

    assert result.success
    assert result.data["anomaly_overlap"] is None


def test_narrative_contains_driver():
    correlation = {
        "top_pairs": [
            {
                "column_a": "Sales",
                "column_b": "Profit",
                "correlation": 0.98,
                "strength": "Very Strong",
            }
        ]
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        correlation=correlation,
    )

    narrative = result.data["narrative"]

    assert "Profit" in narrative
    assert "Sales" in narrative
    assert "0.98" in narrative


def test_narrative_with_anomalies():
    correlation = {
        "top_pairs": [
            {
                "column_a": "Sales",
                "column_b": "Profit",
                "correlation": 0.95,
                "strength": "Very Strong",
            }
        ]
    }

    anomaly = {
        "top_anomalies": [
            {
                "values": {
                    "Sales": 500
                }
            }
        ]
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        correlation=correlation,
        anomaly_result=anomaly,
    )

    assert "Anomalous rows average" in result.data["narrative"]


def test_target_not_present_in_pairs():
    correlation = {
        "top_pairs": [
            {
                "column_a": "Profit",
                "column_b": "Discount",
                "correlation": 0.75,
                "strength": "Strong",
            }
        ]
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        correlation=correlation,
    )

    assert result.success
    assert result.data["candidate_drivers"] == []


def test_empty_anomaly_list():
    anomaly = {
        "top_anomalies": []
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        anomaly_result=anomaly,
    )

    assert result.success
    assert result.data["anomaly_overlap"] is None


def test_empty_top_pairs():
    correlation = {
        "top_pairs": []
    }

    agent = RootCauseAgent()

    result = agent.run(
        dataframe=sample_dataframe(),
        target_column="Sales",
        correlation=correlation,
    )

    assert result.success
    assert result.data["candidate_drivers"] == []