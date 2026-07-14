"""VisualizationAgent — renders Plotly figures for a wide catalog of chart types.

Supports: bar, line, area, pie, treemap, sunburst, scatter, heatmap
(correlation), histogram, violin, box, bubble, waterfall, funnel, radar,
parallel coordinates, pair plot, and KPI indicator cards.

Every `build_*` method returns a native `plotly.graph_objects.Figure`.
`VisualizationAgent.render(dataframe, spec)` dispatches by `chart_type` so
callers (Streamlit UI, FastAPI routes, ExportAgent) can stay agnostic of
the individual chart implementations.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.agents.base_agent import BaseAgent

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Bold


class VisualizationAgent(BaseAgent):
    name = "visualization_agent"
    description = "Builds interactive Plotly charts from a chart specification."

    MAX_POINTS = 10000
    MAX_MATRIX_POINTS = 2000

    DISPATCH = {
        "bar": "build_bar",
        "line": "build_line",
        "area": "build_area",
        "pie": "build_pie",
        "treemap": "build_treemap",
        "sunburst": "build_sunburst",
        "scatter": "build_scatter",
        "bubble": "build_bubble",
        "histogram": "build_histogram",
        "violin": "build_violin",
        "box": "build_box",
        "waterfall": "build_waterfall",
        "funnel": "build_funnel",
        "radar": "build_radar",
        "parallel_coordinates": "build_parallel_coordinates",
        "pair_plot": "build_pair_plot",
        "correlation_heatmap": "build_correlation_heatmap",
        "kpi_card": "build_kpi_indicator",
    }
    @classmethod
    def _sample(cls, df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
        if limit is None:
            limit = cls.MAX_POINTS

        if len(df) <= limit:
            return df

        return df.sample(limit, random_state=42)
    def _execute(self, dataframe: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
        fig = self.render(dataframe, spec)
        return {
            "figure_json": fig.to_json(),
            "chart_type": str(spec["chart_type"]),
        }

    def render(self, dataframe: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        chart_type = str(spec["chart_type"])
        method_name = self.DISPATCH.get(chart_type)
        if not method_name:
            raise ValueError(f"Unsupported chart_type '{chart_type}'")
        method = getattr(self, method_name)
        fig = method(dataframe, spec)
        fig.update_layout(template=TEMPLATE, margin=dict(l=30, r=30, t=60, b=30), font=dict(size=13))
        return fig

    # ------------------------------------------------------------------
    # Chart builders
    # ------------------------------------------------------------------

    def build_bar(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec["x"], spec["y"]
        agg = df.groupby(x, dropna=False)[y].sum(numeric_only=True).sort_values(ascending=False).head(20)
        return px.bar(agg, x=agg.index, y=agg.values, title=spec.get("title"),
                      labels={"x": x, "y": y}, color=agg.index, color_discrete_sequence=COLOR_SEQUENCE)

    def build_line(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec["x"], spec["y"]
        data = df[[x, y]].dropna().copy()
        data[x] = pd.to_datetime(data[x], errors="coerce")
        data = data.dropna(subset=[x]).sort_values(x)
        if data.empty:
            data = df[[x, y]].dropna().sort_values(x)
        return px.line(data, x=x, y=y, title=spec.get("title"), markers=True,
                        color_discrete_sequence=COLOR_SEQUENCE)

    def build_area(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec["x"], spec["y"]
        data = df[[x, y]].dropna().sort_values(x)
        return px.area(data, x=x, y=y, title=spec.get("title"), color_discrete_sequence=COLOR_SEQUENCE)

    def build_pie(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        col = spec["x"]
        counts = df[col].value_counts(dropna=True).head(10)
        return px.pie(names=counts.index, values=counts.values, title=spec.get("title"),
                      color_discrete_sequence=COLOR_SEQUENCE, hole=0.35)

    def build_treemap(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec["x"], spec.get("y")
        if y:
            agg = df.groupby(x, dropna=False)[y].sum(numeric_only=True).reset_index()
            return px.treemap(agg, path=[x], values=y, title=spec.get("title"),
                               color_discrete_sequence=COLOR_SEQUENCE)
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, "count"]
        return px.treemap(counts, path=[x], values="count", title=spec.get("title"))

    def build_sunburst(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        path = spec.get("path") or [spec["x"]]
        value_col = spec.get("y")
        if value_col:
            agg = df.groupby(path, dropna=False)[value_col].sum(numeric_only=True).reset_index()
            return px.sunburst(agg, path=path, values=value_col, title=spec.get("title"))
        counts = df.groupby(path, dropna=False).size().reset_index(name="count")
        return px.sunburst(counts, path=path, values="count", title=spec.get("title"))

    def build_scatter(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec["x"], spec["y"]
        data = self._sample(
            df[[x, y]].dropna()
        )
        return px.scatter(data, x=x, y=y, title=spec.get("title"), trendline="ols",
                           color_discrete_sequence=COLOR_SEQUENCE, opacity=0.7)

    def build_bubble(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y, size = spec["x"], spec["y"], spec.get("size")
        cols = [c for c in (x, y, size) if c]
        data = self._sample(
            df[cols].dropna()
        )
        return px.scatter(data, x=x, y=y, size=size if size else None, title=spec.get("title"),
                           color_discrete_sequence=COLOR_SEQUENCE, opacity=0.7, size_max=40)

    def build_histogram(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        col = spec["x"]
        data = self._sample(df[[col]].dropna())

        return px.histogram(
            data,
            x=col,
            title=spec.get("title"),
            nbins=30,
            color_discrete_sequence=COLOR_SEQUENCE,
            marginal="box",
        )

    def build_violin(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec.get("x"), spec["y"]
        data = self._sample(
            df[[c for c in [x, y] if c]].dropna()
        )

        return px.violin(
            data,
            x=x,
            y=y,
            box=True,
            points="outliers",
            title=spec.get("title"),
            color_discrete_sequence=COLOR_SEQUENCE,
        )

    def build_box(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec.get("x"), spec["y"]
        data = self._sample(
        df[[c for c in [x, y] if c]].dropna()
        )

        return px.box(
            data,
            x=x,
            y=y,
            title=spec.get("title"),
            color_discrete_sequence=COLOR_SEQUENCE,
        )

    def build_waterfall(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec["x"], spec["y"]
        agg = df.groupby(x, dropna=False)[y].sum(numeric_only=True).sort_values(ascending=False).head(15)
        fig = go.Figure(go.Waterfall(
            x=agg.index.astype(str), y=agg.values,
            connector={"line": {"color": "#94a3b8"}},
        ))
        fig.update_layout(title=spec.get("title"))
        return fig

    def build_funnel(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        x, y = spec["x"], spec["y"]
        agg = df.groupby(x, dropna=False)[y].sum(numeric_only=True).sort_values(ascending=False).head(10)
        return px.funnel(y=agg.index.astype(str), x=agg.values, title=spec.get("title"))

    def build_radar(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        numeric_cols = spec.get("columns") or list(df.select_dtypes("number").columns)[:6]
        means = df[numeric_cols].mean(numeric_only=True)
        normalized = (means - means.min()) / (means.max() - means.min() + 1e-9)
        fig = go.Figure(go.Scatterpolar(r=normalized.values, theta=normalized.index, fill="toself"))
        fig.update_layout(title=spec.get("title", "Metric Radar"), polar=dict(radialaxis=dict(visible=True)))
        return fig

    def build_parallel_coordinates(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        numeric_cols = spec.get("columns") or list(df.select_dtypes("number").columns)[:6]
        data = self._sample(
            df[numeric_cols].dropna(),
            self.MAX_MATRIX_POINTS,
        )
        return px.parallel_coordinates(data, color=numeric_cols[0] if numeric_cols else None,
                                        title=spec.get("title"))

    def build_pair_plot(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        numeric_cols = spec.get("columns") or list(df.select_dtypes("number").columns)[:5]
        data = self._sample(
            df[numeric_cols].dropna(),
            self.MAX_MATRIX_POINTS,
        )
        fig = px.scatter_matrix(data, dimensions=numeric_cols, title=spec.get("title"),
                                 color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        return fig

    def build_correlation_heatmap(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        numeric_cols = spec.get("columns") or list(df.select_dtypes("number").columns)
        corr = df[numeric_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title=spec.get("title", "Correlation Heatmap"))
        return fig

    def build_kpi_indicator(self, df: pd.DataFrame, spec: dict[str, Any]) -> go.Figure:
        value = spec.get("value", 0)
        delta = spec.get("delta")
        fig = go.Figure(go.Indicator(
            mode="number+delta" if delta is not None else "number",
            value=value,
            delta={"reference": value - (delta or 0)} if delta is not None else None,
            title={"text": spec.get("title", "KPI")},
        ))
        return fig
