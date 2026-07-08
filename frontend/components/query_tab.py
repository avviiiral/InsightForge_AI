"""Ask Your Data tab — chat-style natural-language querying over the dataset."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.agents.query_agent import QueryAgent
from app.llm.llm_factory import get_llm_router
from frontend.components.ui_helpers import render_section_title

query_agent = QueryAgent(llm_router=get_llm_router())


def render_query_tab(context: dict[str, Any]) -> None:
    dataframe = context["dataframe"]
    column_types = context.get("column_types", {})

    render_section_title("💬", "Ask Your Data")
    st.caption(
        "Ask things like *'top 5 by revenue'*, *'average price by category'*, "
        "*'correlation between age and income'*, or *'describe salary'*."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("preview") is not None:
                st.dataframe(pd.DataFrame(turn["preview"]), use_container_width=True, hide_index=True)

    question = st.chat_input("Ask a question about your data...")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        result = query_agent.run(dataframe=dataframe, question=question, column_types=column_types)
        with st.chat_message("assistant"):
            if result.success:
                st.write(result.data["answer"])
                preview = result.data.get("data_preview")
                if preview:
                    st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
                if result.data.get("used_llm"):
                    st.caption("🤖 Answered with LLM-assisted intent parsing.")
                else:
                    st.caption("🧮 Answered with deterministic keyword parsing.")
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": result.data["answer"], "preview": preview}
                )
            else:
                st.error(result.error)
                st.session_state["chat_history"].append({"role": "assistant", "content": f"Error: {result.error}"})
