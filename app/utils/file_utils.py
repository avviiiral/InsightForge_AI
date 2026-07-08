"""File-system helper utilities shared across agents.

Handles safe, dependency-independent reading of the supported flat-file
formats (CSV, Excel, JSON, Parquet, SQLite) into pandas DataFrames.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".db", ".sqlite", ".sqlite3"}


def detect_format(path: Union[str, Path]) -> str:
    """Return a short format tag inferred from a file's extension."""
    ext = Path(path).suffix.lower()
    mapping = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".parquet": "parquet",
        ".db": "sqlite",
        ".sqlite": "sqlite",
        ".sqlite3": "sqlite",
    }
    if ext not in mapping:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return mapping[ext]


def read_any_file(path: Union[str, Path], sheet_name=None, table_name: str | None = None) -> pd.DataFrame:
    """Read any supported file type into a single pandas DataFrame.

    Args:
        path: Path to the source file.
        sheet_name: Optional sheet name/index for Excel files (defaults to the first sheet).
        table_name: Optional table name for SQLite files (defaults to the first table found).

    Returns:
        A pandas DataFrame with the file's contents.
    """
    path = Path(path)
    fmt = detect_format(path)
    logger.info(f"Reading file '{path.name}' as format='{fmt}'")

    if fmt == "csv":
        return pd.read_csv(path, low_memory=False)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if fmt == "excel":
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if fmt == "json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.json_normalize(pd.read_json(path, typ="series"))
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "sqlite":
        return read_sqlite(path, table_name)

    raise ValueError(f"Unhandled format '{fmt}'")  # pragma: no cover


def list_sqlite_tables(path: Union[str, Path]) -> list[str]:
    """Return the list of user tables inside a SQLite database file."""
    with sqlite3.connect(str(path)) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]


def read_sqlite(path: Union[str, Path], table_name: str | None = None) -> pd.DataFrame:
    """Read a table from a SQLite file. Uses the first table if none is given."""
    tables = list_sqlite_tables(path)
    if not tables:
        raise ValueError("No tables found in SQLite database.")
    table = table_name or tables[0]
    if table not in tables:
        raise ValueError(f"Table '{table}' not found. Available: {tables}")
    with sqlite3.connect(str(path)) as conn:
        return pd.read_sql_query(f"SELECT * FROM '{table}'", conn)


def safe_filename(name: str) -> str:
    """Strip characters that are unsafe for filenames while keeping it readable."""
    keep = "-_. "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return cleaned.replace(" ", "_") or "dataset"
