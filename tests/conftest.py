"""Shared pytest fixtures for InsightForge-AI's test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """A small, deliberately messy DataFrame covering multiple semantic types."""
    rng = np.random.default_rng(7)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "order_id": [f"ORD-{1000 + i}" for i in range(n)],
        "order_date": dates,
        "region": rng.choice(["North", "South", "East", "West"], size=n),
        "category": rng.choice(["A", "B", "C"], size=n),
        "quantity": rng.integers(1, 50, size=n),
        "revenue": np.round(rng.gamma(2, 50, size=n), 2),
        "customer_email": [f"user{i}@example.com" for i in range(n)],
        "is_active": rng.choice([True, False], size=n),
        "satisfaction_score": np.clip(rng.normal(4, 0.7, size=n), 1, 5).round(2),
    })
    # Inject missing values and an obvious outlier BEFORE duplicating rows, so the
    # duplicated rows remain exact duplicates of their source rows.
    df.loc[rng.choice(n, 10, replace=False), "satisfaction_score"] = np.nan
    df.loc[0, "revenue"] = df["revenue"].max() * 20
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


@pytest.fixture
def numeric_only_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 100
    x = rng.normal(0, 1, size=n)
    y = x * 2 + rng.normal(0, 0.3, size=n)  # strongly correlated with x
    z = rng.normal(10, 5, size=n)  # uncorrelated
    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture
def timeseries_dataframe() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=104, freq="W")
    rng = np.random.default_rng(1)
    trend = np.linspace(100, 300, len(dates))
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
    noise = rng.normal(0, 5, len(dates))
    return pd.DataFrame({"week": dates, "sales": trend + seasonal + noise})
