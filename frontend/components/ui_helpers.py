"""Reusable Streamlit UI building blocks shared across every dashboard tab."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"

_TREND_ARROWS = {"up": "▲", "down": "▼", "flat": "▶"}
_TREND_CLASS = {"up": "if-kpi-trend-up", "down": "if-kpi-trend-down", "flat": "if-kpi-trend-flat"}


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="if-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_section_title(icon: str, text: str) -> None:
    st.markdown(f'<div class="if-section-title">{icon} {text}</div>', unsafe_allow_html=True)


def render_kpi_row(kpis: list[dict[str, Any]]) -> None:
    if not kpis:
        st.info("No KPIs available yet — run the analysis first.")
        return
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        trend = kpi.get("trend")
        trend_html = ""
        if trend and trend in _TREND_ARROWS:
            trend_html = (
                f'<span class="{_TREND_CLASS[trend]}">{_TREND_ARROWS[trend]} '
                f'{kpi.get("delta", "")}%</span>'
            )
        with col:
            st.markdown(
                f"""<div class="if-card">
                        <div class="if-kpi-label">{kpi['name']}</div>
                        <div class="if-kpi-value">{kpi['formatted_value']}</div>
                        {trend_html}
                    </div>""",
                unsafe_allow_html=True,
            )


def render_insight_card(insight: dict[str, Any]) -> None:
    severity = insight.get("severity", "info")
    badge_class = f"if-badge-{severity}"
    card_class = severity if severity in ("warning", "critical") else ""
    st.markdown(
        f"""<div class="if-insight-card {card_class}">
                <span class="if-badge {badge_class}">{severity}</span>
                <b> {insight.get('title')}</b>
                <div style="margin-top:0.3rem; font-size:0.92rem;">{insight.get('description')}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_health_badge(score: float) -> str:
    if score >= 90:
        return f'<span class="if-health-good">🟢 {score}/100 — Excellent</span>'
    if score >= 70:
        return f'<span class="if-health-warn">🟡 {score}/100 — Fair</span>'
    return f'<span class="if-health-bad">🔴 {score}/100 — Needs Attention</span>'


def plotly_chart(figure_json: str, key: str) -> None:
    import plotly.io as pio

    fig = pio.from_json(figure_json)
    st.plotly_chart(fig, use_container_width=True, key=key)
