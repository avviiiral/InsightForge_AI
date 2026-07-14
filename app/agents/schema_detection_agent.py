"""SchemaDetectionAgent — infers data types AND semantic meaning per column.

This is the cornerstone of the "no predefined schema" requirement: nothing
downstream depends on hard-coded column names. Every other agent reads the
`semantic_type` produced here (numeric, categorical, datetime, email,
phone, currency, id, boolean, location, text) to decide how to treat a
column, regardless of what it happens to be called.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^[\+\(]?[0-9][0-9\-\.\s\(\)]{6,17}[0-9]$")
_CURRENCY_RE = re.compile(
    r"^[\$\€\£\¥]\s?-?[\d,]+(\.\d+)?$|^-?[\d,]+(\.\d+)?\s?(USD|EUR|GBP|INR|JPY)$"
)

_ID_NAME_HINTS = (
    "id",
    "_id",
    "uuid",
    "guid",
    "code",
    "key",
)

_LOCATION_NAME_HINTS = (
    "city",
    "state",
    "country",
    "region",
    "address",
    "zip",
    "postal",
    "location",
    "latitude",
    "longitude",
    "lat",
    "lng",
    "province",
    "continent",
)

_CURRENCY_NAME_HINTS = (
    "price",
    "amount",
    "revenue",
    "cost",
    "salary",
    "income",
    "fee",
    "total",
    "budget",
)


class SchemaDetectionAgent(BaseAgent):
    name = "schema_detection_agent"

    description = (
        "Detects dtypes and semantic column types "
        "without relying on fixed column names."
    )

    def _execute(
        self,
        dataframe: pd.DataFrame,
        sample_size: int = 500,
    ) -> dict[str, Any]:

        profiles: dict[str, dict[str, Any]] = {}

        for column in dataframe.columns:
            profiles[column] = self._profile_column(
                dataframe[column],
                column,
                sample_size,
            )

        summary = self._summarize(profiles)

        identifier_analysis = self._detect_identifier_columns(
            dataframe
        )

        return {
            "column_types": profiles,
            "summary": summary,
            "identifier_analysis": identifier_analysis,
        }

    # ---------------------------------------------------------
    # Column Profiling
    # ---------------------------------------------------------

    def _profile_column(
        self,
        series: pd.Series,
        name: str,
        sample_size: int,
    ) -> dict[str, Any]:

        non_null = series.dropna()

        sample = non_null.head(sample_size)

        semantic = self._infer_semantic_type(
            series,
            sample,
            name,
        )

        return {
            "pandas_dtype": str(series.dtype),
            "semantic_type": semantic,
            "is_nullable": bool(series.isna().any()),
        }

    def _infer_semantic_type(
        self,
        series: pd.Series,
        sample: pd.Series,
        name: str,
    ) -> str:

        name_lower = name.lower()

        # ---------------- Boolean ----------------

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        unique_vals = {
            str(v).strip().lower()
            for v in sample.dropna().head(10)
        }

        if (
            unique_vals
            and unique_vals.issubset(
                {
                    "true",
                    "false",
                    "yes",
                    "no",
                    "0",
                    "1",
                    "y",
                    "n",
                }
            )
            and series.nunique(dropna=True) <= 2
        ):
            return "boolean"

        # ---------------- Datetime ----------------

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if series.dtype == object and self._looks_like_date(sample):
            return "datetime"

        # ---------------- Numeric ----------------

        if pd.api.types.is_numeric_dtype(series):

            if any(
                hint in name_lower
                for hint in _CURRENCY_NAME_HINTS
            ):
                return "currency"

            if (
                any(
                    hint in name_lower
                    for hint in _ID_NAME_HINTS
                )
                and self._is_id_like(series)
            ):
                return "identifier"

            return "numeric"

        # ---------------- Object ----------------

        if series.dtype == object:

            str_sample = (
                sample.dropna()
                .astype(str)
                .str.strip()
            )

            if (
                not str_sample.empty
                and str_sample.map(
                    lambda x: bool(_EMAIL_RE.fullmatch(str(x)))
                ).mean()
                > 0.7
            ):
                return "email"

            if any(
                hint in name_lower
                for hint in _LOCATION_NAME_HINTS
            ):
                return "location"

            if (
                not str_sample.empty
                and str_sample.map(
                    lambda x: bool(_CURRENCY_RE.fullmatch(str(x)))
                ).mean()
                > 0.6
            ):
                return "currency"

            if (
                any(
                    hint in name_lower
                    for hint in _ID_NAME_HINTS
                )
                and self._is_id_like(series)
            ):
                return "identifier"

            if (
                not str_sample.empty
                and str_sample.map(
                    lambda x: bool(_PHONE_RE.fullmatch(str(x)))
                ).mean()
                > 0.6
            ):
                return "phone"

            avg_len = (
                str_sample.str.len().mean()
                if not str_sample.empty
                else 0
            )

            n_unique = int(series.nunique(dropna=True))

            n_total = max(
                int(series.dropna().shape[0]),
                1,
            )

            if (
                n_unique / n_total < 0.5
                and n_unique <= max(
                    50,
                    int(n_total * 0.2),
                )
            ):
                return "categorical"

            if avg_len > 30:
                return "text"

            return "categorical"

        return "unknown"
    # ---------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------

    @staticmethod
    def _is_id_like(series: pd.Series) -> bool:
        """
        Determines whether a column behaves like an identifier.
        """

        clean = series.dropna()

        if clean.empty:
            return False

        unique_ratio = clean.nunique(dropna=True) / len(clean)

        return unique_ratio >= 0.90

    @staticmethod
    def _looks_like_date(sample: pd.Series) -> bool:
        """
        Detect datetime-like string columns.
        """

        if sample.empty:
            return False

        str_sample = (
            sample.dropna()
            .astype(str)
            .head(50)
        )

        try:
            parsed = pd.to_datetime(
                str_sample,
                errors="coerce",
                format="mixed",
            )

        except Exception:
            parsed = pd.to_datetime(
                str_sample,
                errors="coerce",
            )

        return parsed.notna().mean() >= 0.70

    @staticmethod
    def _summarize(
        profiles: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:

        summary: dict[str, list[str]] = {}

        for column, info in profiles.items():
            semantic = str(info["semantic_type"])

            summary.setdefault(
                semantic,
                [],
            ).append(column)

        return summary

    # ---------------------------------------------------------
    # Identifier Detection
    # ---------------------------------------------------------

    def _detect_identifier_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Detect:

        • Primary Key candidates
        • Foreign Key candidates
        • Duplicate identifiers
        • Unique columns
        • Sequential IDs
        • UUID columns
        """

        result: dict[str, Any] = {
            "primary_keys": [],
            "foreign_keys": [],
            "duplicate_identifiers": [],
            "unique_columns": [],
        }

        id_keywords = (
            "id",
            "key",
            "uuid",
            "guid",
            "code",
            "number",
            "no",
            "invoice",
            "order",
            "transaction",
            "customer",
            "employee",
            "product",
        )
        
        uuid_regex = re.compile(
            r"^[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}$"
        )

        total_rows = len(dataframe)

        for column in dataframe.columns:

            series = dataframe[column]

            missing = int(series.isna().sum())

            unique = int(series.nunique(dropna=False))

            duplicates = total_rows - unique

            is_unique = unique == total_rows

            lower_name = column.lower()

            looks_like_id = any(
                keyword in lower_name
                for keyword in id_keywords
            )

            # -----------------------------------------
            # UUID Detection
            # -----------------------------------------

            is_uuid = False

            if pd.api.types.is_object_dtype(series):

                values = (
                    series.dropna()
                    .astype(str)
                )

                if not values.empty:
                    is_uuid = bool(
                        values.str.fullmatch(uuid_regex).all()
                    )

            # -----------------------------------------
            # Sequential Integer Detection
            # -----------------------------------------

            sequential = False

            if pd.api.types.is_integer_dtype(series):

                values = (
                    series.dropna()
                    .astype("int64")
                    .sort_values()
                    .to_numpy()
                )

                if values.size > 1:

                    sequential = bool(
                        np.all(np.diff(values) == 1)
                    )

            # -----------------------------------------
            # Confidence Score
            # -----------------------------------------

            confidence = 0

            if is_unique:
                confidence += 40

            if missing == 0:
                confidence += 20

            if looks_like_id:
                confidence += 20

            if sequential:
                confidence += 10

            if is_uuid:
                confidence += 10

            confidence = min(confidence, 100)

            # -----------------------------------------
            # Unique Columns
            # -----------------------------------------

            if is_unique:

                result["unique_columns"].append(
                    {
                        "column": column,
                        "unique_values": unique,
                    }
                )

            # -----------------------------------------
            # Primary Key Candidates
            # -----------------------------------------

            if confidence >= 70:

                result["primary_keys"].append(
                    {
                        "column": column,
                        "confidence": confidence,
                        "missing_values": missing,
                        "duplicates": duplicates,
                        "sequential": sequential,
                        "uuid": is_uuid,
                    }
                )

            # -----------------------------------------
            # Foreign Key Candidates
            # -----------------------------------------

            elif looks_like_id:

                result["foreign_keys"].append(
                    {
                        "column": column,
                        "duplicates": duplicates,
                        "missing_values": missing,
                    }
                )

            # -----------------------------------------
            # Duplicate IDs
            # -----------------------------------------

            if looks_like_id and duplicates > 0:

                result["duplicate_identifiers"].append(
                    {
                        "column": column,
                        "duplicates": duplicates,
                    }
                )

        return result