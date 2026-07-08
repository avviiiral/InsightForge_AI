"""Tests for DataLoaderAgent — multi-format file ingestion."""
from __future__ import annotations

import sqlite3

import pandas as pd

from app.agents.data_loader_agent import DataLoaderAgent


def test_load_csv(tmp_path, sample_dataframe):
    path = tmp_path / "data.csv"
    sample_dataframe.to_csv(path, index=False)
    agent = DataLoaderAgent()
    result = agent.run(file_path=str(path))
    assert result.success
    assert result.data["n_rows"] == len(sample_dataframe)
    assert result.data["n_columns"] == sample_dataframe.shape[1]


def test_load_excel(tmp_path, sample_dataframe):
    path = tmp_path / "data.xlsx"
    sample_dataframe.to_excel(path, index=False)
    agent = DataLoaderAgent()
    result = agent.run(file_path=str(path))
    assert result.success
    assert result.data["n_rows"] == len(sample_dataframe)


def test_load_json(tmp_path, sample_dataframe):
    path = tmp_path / "data.json"
    sample_dataframe.to_json(path, orient="records")
    agent = DataLoaderAgent()
    result = agent.run(file_path=str(path))
    assert result.success
    assert result.data["n_rows"] == len(sample_dataframe)


def test_load_parquet(tmp_path, sample_dataframe):
    path = tmp_path / "data.parquet"
    sample_dataframe.to_parquet(path)
    agent = DataLoaderAgent()
    result = agent.run(file_path=str(path))
    assert result.success
    assert result.data["n_rows"] == len(sample_dataframe)


def test_load_sqlite(tmp_path, sample_dataframe):
    path = tmp_path / "data.db"
    with sqlite3.connect(path) as conn:
        sample_dataframe.to_sql("my_table", conn, index=False)
    agent = DataLoaderAgent()
    result = agent.run(file_path=str(path))
    assert result.success
    assert result.data["n_rows"] == len(sample_dataframe)


def test_load_missing_source_raises():
    agent = DataLoaderAgent()
    result = agent.run()
    assert not result.success
    assert "Provide either" in result.error


def test_load_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    agent = DataLoaderAgent()
    result = agent.run(file_path=str(path))
    assert not result.success
