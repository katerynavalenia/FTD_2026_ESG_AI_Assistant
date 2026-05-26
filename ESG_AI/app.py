"""
ESG AI Assistant - Streamlit web UI for CSRD Q&A.
Run with: streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from csrd_chat import (
    AlbertClient,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_TOP_K,
    RAGEngine,
    load_dotenv,
    resolve_api_key,
)

# Index directory can be overridden via environment variable.
# Set INDEX_DIR=data/vector_index_hash to use the offline hash index.
_INDEX_DIR_OVERRIDE = os.getenv("INDEX_DIR", "")
_EFFECTIVE_INDEX_DIR = Path(_INDEX_DIR_OVERRIDE) if _INDEX_DIR_OVERRIDE else DEFAULT_INDEX_DIR

load_dotenv()

st.set_page_config(
    page_title="ESG AI Assistant",
    page_icon="leaf",
    layout="wide",
)

AVAILABLE_MODELS = [
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "openai/gpt-oss-120b",
    "mistralai/Ministral-3-8B-Instruct-2512",
]

EXAMPLE_QUESTIONS = [
    "What are the GHG emissions reduction targets?",
    "How does this company address biodiversity risks?",
    "What social commitments are made for employees?",
    "Describe the governance structure for sustainability.",
    "What progress has been made on the circular economy?",
    "What CSRD-related disclosures are included in the report?",
    "What are the Scope 1, 2, and 3 emission figures?",
    "How does the company manage water consumption?",
]

HOW_TO_USE = """
**How to use this app:**
1. Select a **company** in the sidebar to focus on one report, or keep *All companies* to search across all indexed reports.
2. Choose the **LLM model** and the number of **passages to retrieve** (top-k).
3. Type your CSRD or ESG question in the chat box below, or click one of the example questions.
4. The assistant retrieves the most relevant passages from the indexed sustainability reports and generates a grounded answer with source citations.
5. Expand **Sources** under any answer to see the exact report excerpts used.

> **Note:** answers are strictly grounded in the indexed reports. The assistant will say so if information is not available.
"""


@st.cache_resource(show_spinner="Loading vector index...")
def load_engine():
    """Return (engine, error_message). error_message is empty on success."""
    try:
        api_key = resolve_api_key()
    except RuntimeError as exc:
        return None, str(exc)

    base_url = os.getenv("ALBERT_API_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
    client = AlbertClient(api_key=api_key, base_url=base_url)
    try:
        engine = RAGEngine(
            index_dir=_EFFECTIVE_INDEX_DIR,
            client=client,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
        )
        return engine, ""
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Unexpected error loading index: {exc}"


def render_sources(sources):
    with st.expander(
        f"Sources ({len(sources)} passage{'s' if len(sources) != 1 else ''})",
        expanded=False,
    ):
        for i, chunk in enumerate(sources, 1):
            score = chunk.get("_score", 0.0)
            st.markdown(
                f"**[{i}]** `{chunk['citation']}` &nbsp;*relevance: {score:.3f}*"
            )
            if chunk.get("section_title"):
                st.caption(f"Section: {chunk['section_title']}")
            with st.expander("Show excerpt", expanded=False):
                excerpt = chunk["text"][:800]
                if len(chunk["text"]) > 800:
                    excerpt += "..."
                st.markdown(excerpt)
            st.divider()


def main():
    engine, load_error = load_engine()

    # Sidebar
    with st.sidebar:
        st.title("ESG AI Assistant")
        st.caption("CSRD Q&A over sustainability reports")
        st.divider()

        if engine is not None:
            st.subheader("Filters & Settings")
            company_options = ["All companies"] + engine.companies
            selected = st.selectbox("Company", company_options, index=0)
            company = "" if selected == "All companies" else selected

            chat_model = st.selectbox("LLM model", AVAILABLE_MODELS, index=0)

            top_k = st.slider(
                "Passages to retrieve (top-k)",
                min_value=3,
                max_value=12,
                value=DEFAULT_TOP_K,
                help="More passages = richer context but slower generation.",
            )

            st.divider()
            st.subheader("Indexed companies")
            for c in engine.companies:
                st.markdown(f"- {c}")
            model_label = engine.index_config.get("model", DEFAULT_EMBEDDING_MODEL)
            st.caption(f"{len(engine.metadata):,} chunks - {model_label}")
        else:
            company = ""
            chat_model = AVAILABLE_MODELS[0]
            top_k = DEFAULT_TOP_K

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main area
    st.title("ESG AI Assistant")
    st.markdown(
        "Ask questions about CSRD compliance, GHG emissions, social commitments, "
        "governance, biodiversity targets, and more -- grounded in indexed sustainability reports."
    )

    api_key_present = bool(os.getenv("ALBERT_API_KEY", "").strip())
    if not api_key_present:
        st.warning(
            "**ALBERT_API_KEY is not set.** "
            "Copy `.env.example` to `.env` and add your key, then restart the app.",
            icon="warning",
        )

    if load_error:
        st.error(f"**Could not load the vector index:**\n```\n{load_error}\n```")
        st.info(
            "Build the index first:\n"
            "```bash\npython vector_index.py build --index-dir data/vector_index_api --provider api\n```"
        )
        st.stop()

    with st.expander("How to use this app", expanded=False):
        st.markdown(HOW_TO_USE)

    with st.expander("Example CSRD questions", expanded=False):
        cols = st.columns(2)
        for idx, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[idx % 2].button(q, key=f"ex_{idx}", use_container_width=True):
                st.session_state["prefill"] = q

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    prefill = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask a CSRD or ESG question...") or prefill
    if not question or not question.strip():
        return

    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources = []
        answer = ""
        try:
            # Retrieval is fast and blocking; LLM generation streams token by token.
            with st.spinner("Retrieving relevant passages..."):
                sources, token_stream = engine.stream_answer(
                    question,
                    top_k=top_k,
                    company=company,
                    chat_model=chat_model,
                )
            # st.write_stream renders tokens in real time as they arrive from the API.
            answer = st.write_stream(token_stream)
        except Exception as exc:
            answer = f"An error occurred while generating the answer: {exc}"
            st.error(answer)

        if sources:
            render_sources(sources)
        elif answer and not answer.startswith("An error") and "No relevant passages" not in answer:
            st.info("No relevant passages were found for this question.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
