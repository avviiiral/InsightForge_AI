"""DataLoaderAgent — ingests a dataset from any supported source.

Supports local files (CSV, TSV, Excel, JSON, Parquet, SQLite) as well as
live PostgreSQL/MySQL connections. This is intentionally the *only* agent
that knows about file formats or database drivers — every other agent
downstream just receives a plain pandas DataFrame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.agents.base_agent import BaseAgent
from app.utils import db_utils, file_utils


class DataLoaderAgent(BaseAgent):
    name = "data_loader_agent"
    description = "Loads a dataset from a file path or database connection into a DataFrame."

    def _execute(
        self,
        file_path: Optional[str] = None,
        connection_uri: Optional[str] = None,
        table_name: Optional[str] = None,
        sql_query: Optional[str] = None,
        sheet_name: Any = None,
    ) -> dict[str, Any]:
        if file_path:
            df = file_utils.read_any_file(file_path, sheet_name=sheet_name, table_name=table_name)
            source = Path(file_path).name
        elif connection_uri and sql_query:
            df = db_utils.read_custom_query(connection_uri, sql_query)
            source = "custom_query"
        elif connection_uri and table_name:
            df = db_utils.read_table(connection_uri, table_name)
            source = table_name
        else:
            raise ValueError(
                "Provide either `file_path`, or `connection_uri` with `table_name`/`sql_query`."
            )

        if df.empty:
            raise ValueError("The loaded dataset is empty.")

        # Normalize column names: strip whitespace, keep original casing.
        df.columns = [str(c).strip() for c in df.columns]

        return {
            "dataframe": df,
            "source": source,
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
        }

    @staticmethod
    def list_database_tables(connection_uri: str) -> list[str]:
        return db_utils.list_tables(connection_uri)

    @staticmethod
    def list_excel_sheets(file_path: str) -> list[str]:
        return list(pd.ExcelFile(file_path).sheet_names)

    @staticmethod
    def list_sqlite_tables(file_path: str) -> list[str]:
        return file_utils.list_sqlite_tables(file_path)
