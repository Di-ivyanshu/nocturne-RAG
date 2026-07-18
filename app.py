"""Streamlit UI — everything from the browser: upload, index, chat, citations, confidence.

Run:  streamlit run app.py
No need for the user to touch the terminal.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src import config, pipeline, vectorstore

st.set_page_config(page_title="Self-Correcting RAG Assistant", page_icon="🧠", layout="wide")


def _save_uploads(files) -> list[Path]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        dest = config.DATA_DIR / f.name
        dest.write_bytes(f.getbuffer())
        saved.append(dest)
    return saved


def _confidence_badge(conf: float, grounded: bool) -> None:
    if grounded and conf >= 0.75:
        st.success(f"✅ High confidence: {conf:.2f}")
    elif conf >= config.CONFIDENCE_THRESHOLD:
        st.warning(f"🟡 Medium confidence: {conf:.2f}")
    else:
        st.error(f"🔴 Low confidence: {conf:.2f} — answer may be incomplete")


# ---- Sidebar: upload + index controls -------------------------------------
with st.sidebar:
    st.header("📂 Documents")

    _has_key = config.GROQ_API_KEY if config.LLM_PROVIDER == "groq" else (
        config.GEMINI_API_KEY and config.GEMINI_API_KEY != "your_key_here"
    )
    if not _has_key:
        st.error(f"No API key for provider '{config.LLM_PROVIDER}'. Add it to a .env file (see .env.example).")
        st.caption("Groq key: https://console.groq.com/keys · Gemini key: https://aistudio.google.com/apikey")

    uploaded = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚙️ Build / Update Index", use_container_width=True, type="primary"):
            if not uploaded:
                st.warning("Upload at least one file first.")
            else:
                paths = _save_uploads(uploaded)
                with st.spinner("Chunking + embedding + indexing..."):
                    stats = pipeline.index_files(paths)
                st.success(
                    f"Indexed {stats['documents']} docs / {stats['pages']} pages → "
                    f"+{stats['chunks_added']} chunks (total {stats['chunks_total']})."
                )
    with col_b:
        if st.button("🗑️ Clear Index", use_container_width=True):
            vectorstore.reset()
            st.info("Index cleared.")

    st.metric("Indexed chunks", vectorstore.count())
    st.caption(
        f"Embeddings: {config.EMBED_MODEL} (local) · "
        f"LLM: {config.active_model()} ({config.LLM_PROVIDER}) · "
        f"Hybrid retrieval: {'on' if config.USE_HYBRID else 'off'}"
    )


# ---- Main: chat -----------------------------------------------------------
st.title("🧠 Self-Correcting RAG Knowledge Assistant")
st.caption(
    "Upload your documents, build the index, then ask questions. Every answer is "
    "self-verified — with citations and a confidence score — and honestly says "
    "\"I don't know\" when the answer isn't in your documents."
)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask something about your documents...")

# replay history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["result"]["answer"])

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if vectorstore.count() == 0:
            st.warning("Please upload documents and build the index first (sidebar).")
        else:
            with st.spinner("Retrieving → answering → self-checking..."):
                result = pipeline.answer(question)

            st.write(result["answer"])
            _confidence_badge(result["confidence"], result["grounded"])

            if result["self_corrected"]:
                st.caption(f"🔁 Self-corrected over {len(result['attempts'])} attempts.")

            if result["citations"]:
                with st.expander("📑 Sources"):
                    for c in result["citations"]:
                        score = f" · score {c['score']:.2f}" if c.get("score") is not None else ""
                        st.markdown(f"- **{c['source']}** — page {c['page']}{score}")

            with st.expander("🔍 Self-correction trace (how the answer was reached)"):
                for a in result["attempts"]:
                    st.markdown(
                        f"**Attempt {a['attempt']}** · confidence {a['confidence']} · "
                        f"grounded={a['grounded']}"
                    )
                    st.caption(f"search query: {a['search_query']}")
                    if a.get("reason"):
                        st.caption(f"critic: {a['reason']}")

            st.session_state.history.append({"question": question, "result": result})
