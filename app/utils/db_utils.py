"""Database connectivity helpers (PostgreSQL / MySQL) via SQLAlchemy.

Kept separate from `file_utils.py` because live database connections have
different failure modes (network, auth) than flat-file reads, and because
enterprise deployments often want to swap this module out for a pooled
connection manager.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_engine(connection_uri: str) -> Engine:
    """Create a SQLAlchemy engine for a Postgres or MySQL connection URI."""
    logger.info("Creating SQLAlchemy engine for external database")
    return create_engine(connection_uri, pool_pre_ping=True)


def list_tables(connection_uri: str) -> list[str]:
    """List available tables for a given database connection URI."""
    engine = get_engine(connection_uri)
    inspector = inspect(engine)
    return inspector.get_table_names()


def read_table(connection_uri: str, table_name: str, row_limit: int | None = None) -> pd.DataFrame:
    """Read a table (optionally row-limited) from Postgres/MySQL into a DataFrame."""
    engine = get_engine(connection_uri)
    query = f"SELECT * FROM {table_name}"
    if row_limit:
        query += f" LIMIT {int(row_limit)}"
    logger.info(f"Executing query on external database: {query}")
    return pd.read_sql_query(query, engine)


def read_custom_query(connection_uri: str, sql_query: str) -> pd.DataFrame:
    """Execute an arbitrary read-only SQL query and return the result as a DataFrame."""
    if not sql_query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are permitted for safety.")
    engine = get_engine(connection_uri)
    return pd.read_sql_query(sql_query, engine)
