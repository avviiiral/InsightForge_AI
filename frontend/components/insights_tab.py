"""Insights tab — insights feed, recommendations, executive summary, business narrative."""
from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.components.ui_helpers import render_insight_card, render_section_title


def render_insights_tab(context: dict[str, Any]) -> None:
    insights = context.get("insights", [])
    recommendations = context.get("recommendations", [])
    executive_summary = context.get("executive_summary", {})
    business_narrative = context.get("business_narrative")

    render_section_title("🧾", "Executive Summary")
    for bullet in executive_summary.get("bullets", []):
        st.markdown(f"- {bullet}")
    if executive_summary.get("ai_bullets"):
        st.markdown("**AI-generated summary:**")
        for bullet in executive_summary["ai_bullets"]:
            if bullet.strip():
                st.markdown(bullet if bullet.strip().startswith("-") else f"- {bullet}")

    if business_narrative:
        render_section_title("💼", "Business Narrative")
        st.write(business_narrative)

    render_section_title("💡", "Insights")
    if not insights:
        st.info("No insights generated yet.")
    severity_filter = st.multiselect("Filter by severity", ["critical", "warning", "info"],
                                       default=["critical", "warning", "info"])
    for insight in insights:
        if insight.get("severity", "info") in severity_filter:
            render_insight_card(insight)

    render_section_title("✅", "Recommendations")
    if not recommendations:
        st.info("No recommendations generated yet.")
    for rec in recommendations:
        st.markdown(f"**{rec.get('title')}** — {rec.get('action')}")
