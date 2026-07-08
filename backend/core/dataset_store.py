"""Process-local dataset store for the FastAPI backend.

For a single-process demo/enterprise-skeleton deployment, an in-memory
dict keyed by a UUID is sufficient and avoids introducing Redis/DB
infrastructure just to hold a user's uploaded DataFrame during their
session. Swap this out for a Redis- or database-backed store in a
horizontally-scaled production deployment — every route only depends on
the four methods below.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class DatasetRecord:
    dataset_id: str
    filename: str
    dataframe: pd.DataFrame
    column_types: dict[str, Any] = field(default_factory=dict)
    pipeline_context: dict[str, Any] = field(default_factory=dict)


class DatasetStore:
    def __init__(self):
        self._records: dict[str, DatasetRecord] = {}

    def create(self, filename: str, dataframe: pd.DataFrame) -> DatasetRecord:
        dataset_id = str(uuid.uuid4())
        record = DatasetRecord(dataset_id=dataset_id, filename=filename, dataframe=dataframe)
        self._records[dataset_id] = record
        return record

    def get(self, dataset_id: str) -> Optional[DatasetRecord]:
        return self._records.get(dataset_id)

    def update_context(self, dataset_id: str, context: dict[str, Any]) -> None:
        record = self._records.get(dataset_id)
        if record:
            record.pipeline_context = context
            if "dataframe" in context:
                record.dataframe = context["dataframe"]
            if "column_types" in context:
                record.column_types = context["column_types"]

    def delete(self, dataset_id: str) -> bool:
        return self._records.pop(dataset_id, None) is not None

    def list_ids(self) -> list[str]:
        return list(self._records.keys())


_STORE_SINGLETON: Optional[DatasetStore] = None


def get_dataset_store() -> DatasetStore:
    global _STORE_SINGLETON
    if _STORE_SINGLETON is None:
        _STORE_SINGLETON = DatasetStore()
    return _STORE_SINGLETON
