"""ChartRecommendationAgent — decides which chart types best fit the data.

This agent does NOT render anything — it only reasons about column
semantic types and cardinality to produce a ranked list of
`{chart_type, x, y, reason}` specs. `VisualizationAgent` turns those specs
into actual Plotly figures. Separating "what to show" from "how to render
it" keeps both agents small and independently testable.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.agents.base_agent import BaseAgent


class ChartRecommendationAgent(BaseAgent):
    name = "chart_recommendation_agent"
    description = "Recommends the best chart types for a dataset based on column semantics."

    def _execute(
        self, dataframe: pd.DataFrame, column_types: dict[str, dict[str, Any]], max_charts: int = 12
    ) -> dict[str, Any]:
        numeric = [c for c, i in column_types.items() if i["semantic_type"] in ("numeric", "currency")]
        categorical = [c for c, i in column_types.items() if i["semantic_type"] == "categorical"]
        datetime_cols = [c for c, i in column_types.items() if i["semantic_type"] == "datetime"]

        recs: list[dict[str, Any]] = []

        # Time series: datetime + numeric
        for date_col in datetime_cols[:1]:
            for num_col in numeric[:3]:
                recs.append(self._spec("line", f"{num_col} over {date_col}", date_col, num_col,
                                        "Datetime + numeric column pair is ideal for a time-series trend line."))

        # Categorical + numeric aggregation -> bar chart
        for cat_col in categorical[:3]:
            n_unique = dataframe[cat_col].nunique(dropna=True)
            for num_col in numeric[:2]:
                if n_unique <= 12:
                    recs.append(self._spec("bar", f"{num_col} by {cat_col}", cat_col, num_col,
                                            f"'{cat_col}' has {n_unique} categories — a bar chart compares them clearly."))
                else:
                    recs.append(self._spec("treemap", f"{num_col} distribution across {cat_col}", cat_col, num_col,
                                            f"'{cat_col}' has {n_unique} categories — a treemap handles high cardinality better than bars."))

        # Categorical proportion -> pie (only for low cardinality)
        for cat_col in categorical[:2]:
            if dataframe[cat_col].nunique(dropna=True) <= 8:
                recs.append(self._spec("pie", f"Share of {cat_col}", cat_col, None,
                                        f"'{cat_col}' has few categories — a pie chart shows proportional share."))

        # Numeric distribution -> histogram / box
        for num_col in numeric[:4]:
            recs.append(self._spec("histogram", f"Distribution of {num_col}", num_col, None,
                                    "Histograms reveal the shape/spread of a single numeric column."))
        if len(numeric) >= 1 and categorical:
            recs.append(self._spec("box", f"{numeric[0]} spread by {categorical[0]}", categorical[0], numeric[0],
                                    "Box plots compare distributions across categories, including outliers."))

        # Correlation heatmap
        if len(numeric) >= 2:
            recs.append(self._spec("correlation_heatmap", "Correlation heatmap", None, None,
                                    "Two or more numeric columns allow a correlation heatmap to surface relationships."))

        # Scatter / bubble for two-three numeric columns
        if len(numeric) >= 2:
            recs.append(self._spec("scatter", f"{numeric[0]} vs {numeric[1]}", numeric[0], numeric[1],
                                    "Two numeric columns are well suited to a scatter plot for relationship discovery."))
        if len(numeric) >= 3:
            recs.append(self._spec("bubble", f"{numeric[0]} vs {numeric[1]} (sized by {numeric[2]})",
                                    numeric[0], numeric[1],
                                    "A third numeric column can drive bubble size for extra context.",
                                    extra={"size": numeric[2]}))

        # Radar / parallel coordinates for multi-numeric comparison
        if len(numeric) >= 3:
            recs.append(self._spec("parallel_coordinates", "Multi-metric comparison", None, None,
                                    "Three or more numeric columns can be compared simultaneously with parallel coordinates."))

        # Pair plot
        if 2 <= len(numeric) <= 6:
            recs.append(self._spec("pair_plot", "Pairwise numeric relationships", None, None,
                                    "A small set of numeric columns is ideal for a pairwise scatter matrix."))

        return {"recommendations": recs[:max_charts], "numeric_columns": numeric,
                "categorical_columns": categorical, "datetime_columns": datetime_cols}

    @staticmethod
    def _spec(chart_type: str, title: str, x, y, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = {"chart_type": chart_type, "title": title, "x": x, "y": y, "reason": reason}
        if extra:
            spec.update(extra)
        return spec
