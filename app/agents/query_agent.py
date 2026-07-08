"""QueryAgent — natural language querying over a DataFrame.

SAFETY NOTE: this agent never `eval()`s or `exec()`s LLM output. Instead,
the LLM (when configured) is asked to translate a question into a small,
strictly-validated JSON "intent" object. A deterministic executor then
maps that intent onto a fixed set of safe pandas operations. If no LLM is
configured, or its output doesn't validate, a keyword-based fallback
parser produces the same kind of intent object. Either way, only a known,
whitelisted set of DataFrame operations ever runs.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import pandas as pd

from app.agents.base_agent import BaseAgent
from app.llm.llm_factory import LLMRouter

_VALID_INTENTS = {"top_n", "aggregate", "describe_column", "filter_count", "correlation_lookup", "general_summary"}
_VALID_AGGS = {"sum", "mean", "median", "min", "max", "count"}


class QueryAgent(BaseAgent):
    name = "query_agent"
    description = "Answers natural-language questions about a dataset using a safe, whitelisted intent executor."

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        super().__init__()
        self.llm_router = llm_router

    def _execute(
        self,
        dataframe: pd.DataFrame,
        question: str,
        column_types: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        columns = list(dataframe.columns)
        column_types = column_types or {}
        intent, used_llm = self._parse_intent(question, columns, column_types)
        answer, preview = self._execute_intent(dataframe, intent, columns)
        return {"answer": answer, "data_preview": preview, "used_llm": used_llm, "intent": intent}

    # -- intent parsing --------------------------------------------------

    def _parse_intent(
        self, question: str, columns: list[str], column_types: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        if self.llm_router and self.llm_router.is_enabled():
            intent = self._llm_intent(question, columns)
            if intent:
                return intent, True
        return self._keyword_intent(question, columns, column_types), False

    def _llm_intent(self, question: str, columns: list[str]) -> Optional[dict[str, Any]]:
        system_prompt = (
            "Translate the user's question about a tabular dataset into a JSON object with this exact "
            'shape: {"intent": one of ["top_n","aggregate","describe_column","filter_count",'
            '"correlation_lookup","general_summary"], "column": <column name or null>, '
            '"group_by": <column name or null>, "agg": one of ["sum","mean","median","min","max","count"] '
            'or null, "n": <integer or null>}. '
            f"Only use column names from this exact list: {columns}. "
            "Respond with ONLY the JSON object, no explanation, no markdown fences."
        )
        raw = self.llm_router.generate(system_prompt, question, max_tokens=200)
        if not raw:
            return None
        try:
            cleaned = raw.strip().strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            intent = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            self.logger.warning("LLM intent output was not valid JSON; falling back to keyword parsing.")
            return None
        return self._validate_intent(intent, columns)

    @staticmethod
    def _validate_intent(intent: dict[str, Any], columns: list[str]) -> Optional[dict[str, Any]]:
        if not isinstance(intent, dict) or intent.get("intent") not in _VALID_INTENTS:
            return None
        for key in ("column", "group_by"):
            if intent.get(key) is not None and intent[key] not in columns:
                intent[key] = None
        if intent.get("agg") not in _VALID_AGGS:
            intent["agg"] = None
        n = intent.get("n")
        intent["n"] = int(n) if isinstance(n, (int, float)) and 0 < n <= 1000 else 5
        return intent

    _AGG_SYNONYMS = {
        "mean": "mean", "average": "mean", "avg": "mean",
        "sum": "sum", "total": "sum",
        "median": "median",
        "min": "min", "minimum": "min", "lowest": "min", "smallest": "min",
        "max": "max", "maximum": "max",
        "count": "count", "number of": "count", "how many": "count",
    }

    @classmethod
    def _keyword_intent(
        cls, question: str, columns: list[str], column_types: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        q = question.lower()
        mentioned = [c for c in columns if c.lower() in q]
        numeric_mentioned = [c for c in mentioned if column_types.get(c, {}).get("semantic_type") in ("numeric", "currency")]
        categorical_mentioned = [c for c in mentioned if c not in numeric_mentioned]

        n_match = re.search(r"top\s+(\d+)", q)
        n = int(n_match.group(1)) if n_match else 5

        agg = next((mapped for phrase, mapped in cls._AGG_SYNONYMS.items() if phrase in q), None)

        # Prefer a numeric column as the metric and a categorical column as the grouping key.
        metric_col = numeric_mentioned[0] if numeric_mentioned else (mentioned[0] if mentioned else None)
        group_col = categorical_mentioned[0] if categorical_mentioned else next(
            (c for c in mentioned if c != metric_col), None
        )

        if "correlat" in q:
            two_numeric = numeric_mentioned[:2] if len(numeric_mentioned) >= 2 else mentioned[:2]
            return {"intent": "correlation_lookup",
                    "column": two_numeric[0] if len(two_numeric) > 0 else None,
                    "group_by": two_numeric[1] if len(two_numeric) > 1 else None, "agg": None, "n": n}
        if "top" in q or "highest" in q or "largest" in q:
            return {"intent": "top_n", "column": metric_col, "group_by": group_col, "agg": agg or "sum", "n": n}
        if agg and metric_col:
            return {"intent": "aggregate", "column": metric_col, "group_by": group_col, "agg": agg, "n": n}
        if mentioned:
            return {"intent": "describe_column", "column": mentioned[0], "group_by": None, "agg": None, "n": n}
        return {"intent": "general_summary", "column": None, "group_by": None, "agg": None, "n": n}

    # -- execution ---------------------------------------------------------

    def _execute_intent(
        self, df: pd.DataFrame, intent: dict[str, Any], columns: list[str]
    ) -> tuple[str, Optional[list[dict[str, Any]]]]:
        kind = intent["intent"]
        column, group_by, agg, n = intent.get("column"), intent.get("group_by"), intent.get("agg"), intent.get("n", 5)

        if kind == "top_n" and column:
            sort_col = column
            if group_by and pd.api.types.is_numeric_dtype(df.get(column, pd.Series(dtype=float))):
                grouped = df.groupby(group_by)[column].agg(agg or "sum").sort_values(ascending=False).head(n)
                preview = grouped.reset_index().to_dict("records")
                return f"Top {n} '{group_by}' groups by {agg or 'sum'} of '{column}'.", preview
            if pd.api.types.is_numeric_dtype(df[sort_col]):
                top = df.nlargest(n, sort_col)
                return f"Top {n} rows ranked by '{sort_col}'.", top.head(n).to_dict("records")
            counts = df[sort_col].value_counts().head(n)
            return f"Top {n} most frequent values in '{sort_col}'.", counts.reset_index().to_dict("records")

        if kind == "aggregate" and column and agg:
            if group_by:
                grouped = df.groupby(group_by)[column].agg(agg).reset_index().sort_values(column, ascending=False)
                return f"{agg.title()} of '{column}' grouped by '{group_by}'.", grouped.head(20).to_dict("records")
            value = getattr(df[column], agg)()
            return f"The {agg} of '{column}' is {round(float(value), 4)}.", None

        if kind == "describe_column" and column:
            desc = df[column].describe(include="all").to_dict()
            preview = [{"statistic": k, "value": str(v)} for k, v in desc.items()]
            return f"Summary statistics for '{column}'.", preview

        if kind == "correlation_lookup" and column and group_by:
            if pd.api.types.is_numeric_dtype(df[column]) and pd.api.types.is_numeric_dtype(df[group_by]):
                corr = df[[column, group_by]].corr().iloc[0, 1]
                return f"The correlation between '{column}' and '{group_by}' is {round(float(corr), 4)}.", None
            return "Correlation lookup requires two numeric columns.", None

        return (
            f"This dataset has {len(df):,} rows and {len(columns)} columns: {', '.join(columns[:10])}"
            + ("..." if len(columns) > 10 else "") + ".",
            None,
        )
