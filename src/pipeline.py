"""High-level orchestration: build the index and answer questions.

Both the UI and the CLI call these two functions — the rest of the detail lives in src/*.
"""
from __future__ import annotations

from pathlib import Path

from . import config, vectorstore
from .agents import answer_stream, answer_with_self_correction
from .chunker import chunk_documents
from .ingest import Document, load_dir, load_file


def build_index(
    data_dir: Path | str = config.DATA_DIR, *, reset: bool = False
) -> dict:
    """Ingest -> chunk -> embed -> store. Returns a small stats dict."""
    if reset:
        vectorstore.reset()
    docs: list[Document] = load_dir(data_dir)
    chunks = chunk_documents(docs)
    added = vectorstore.add_chunks(chunks)
    return {
        "documents": len({d.metadata.get("source") for d in docs}),
        "pages": len(docs),
        "chunks_added": added,
        "chunks_total": vectorstore.count(),
    }


def index_files(paths: list[Path | str]) -> dict:
    """Index a specific set of files (used by the UI right after upload)."""
    docs: list[Document] = []
    for p in paths:
        docs.extend(load_file(Path(p)))
    chunks = chunk_documents(docs)
    added = vectorstore.add_chunks(chunks)
    return {
        "documents": len({d.metadata.get("source") for d in docs}),
        "pages": len(docs),
        "chunks_added": added,
        "chunks_total": vectorstore.count(),
    }


def _empty_result(message: str) -> dict:
    return {
        "answer": message,
        "confidence": 0.0,
        "grounded": False,
        "citations": [],
        "evidence": [],
        "attempts": [],
        "self_corrected": False,
    }


def _friendly_error(exc: Exception) -> dict:
    msg = str(exc)
    if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
        return _empty_result(
            "⚠️ The LLM free-tier quota/rate limit was hit. Please wait a moment and "
            "try again, or switch to a key with higher limits."
        )
    return _empty_result(f"⚠️ LLM error: {msg}")


def answer(
    question: str, top_k: int = config.TOP_K, history: list[dict] | None = None
) -> dict:
    """Answer a question with the self-correcting RAG loop.

    `history` (list of {question, answer} turns) enables multi-turn follow-ups.
    Gracefully degrades on quota/LLM errors instead of crashing the UI.
    """
    if vectorstore.count() == 0:
        return _empty_result(
            "No documents indexed yet. Upload files and build the index first."
        )
    try:
        return answer_with_self_correction(question, top_k=top_k, history=history)
    except RuntimeError as exc:
        return _friendly_error(exc)


def answer_streaming(
    question: str, top_k: int = config.TOP_K, history: list[dict] | None = None
):
    """Generator yielding ("token", str) then ("meta", dict) for the streaming UI.

    Degrades gracefully: if no docs are indexed or the LLM errors, it yields a
    single ("meta", ...) event carrying a friendly message.
    """
    if vectorstore.count() == 0:
        yield ("meta", _empty_result(
            "No documents indexed yet. Upload files and build the index first."
        ))
        return
    try:
        yield from answer_stream(question, top_k=top_k, history=history)
    except RuntimeError as exc:
        yield ("meta", _friendly_error(exc))


def list_documents() -> list[dict]:
    """Indexed documents with their chunk counts (for the manage-docs panel)."""
    return vectorstore.documents()


def delete_document(source: str) -> dict:
    """Remove one document's chunks from the index and delete its file."""
    removed = vectorstore.delete_by_source(source)
    path = config.DATA_DIR / source
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
    return {"removed": removed, "chunks_total": vectorstore.count()}


def index_stats() -> dict:
    return {"chunks_total": vectorstore.count()}
