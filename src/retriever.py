"""Retrieval: semantic search, optionally fused with BM25 keyword search.

Why hybrid: vector search captures "meaning" but can miss exact keywords / IDs /
numbers (e.g. "section 80C", error codes). BM25 matches keywords. Combining both
improves recall — and makes for a nice benchmark table in the README.
"""
from __future__ import annotations

import re

from . import config, vectorstore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _bm25_scores(query_text: str) -> dict[str, float]:
    """Return {chunk_text: bm25_score} over all stored chunks."""
    from rank_bm25 import BM25Okapi

    corpus = vectorstore.all_documents()
    if not corpus:
        return {}
    tokenized = [_tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query_text))
    return {corpus[i]["text"]: float(scores[i]) for i in range(len(corpus))}


def retrieve(query_text: str, top_k: int = config.TOP_K) -> list[dict]:
    """Return top_k chunks as [{text, metadata, score}], best first.

    Pipeline:  hybrid (semantic + BM25) over a wide candidate pool  ->  optional
    cross-encoder re-rank  ->  top_k. Each stage degrades gracefully if a
    component is unavailable.
    """
    pool = config.RETRIEVE_CANDIDATES if config.USE_RERANK else max(top_k * 2, top_k)
    candidates = _hybrid_candidates(query_text, pool)
    if not candidates:
        return []

    if config.USE_RERANK:
        from . import reranker

        reranked = reranker.rerank(query_text, candidates, top_k)
        return [_clean(c) for c in reranked]

    return [_clean(c) for c in candidates[:top_k]]


def _hybrid_candidates(query_text: str, pool: int) -> list[dict]:
    """Fuse semantic + BM25 scores and return up to `pool` candidates, best first."""
    semantic = vectorstore.query(query_text, top_k=pool)
    if not config.USE_HYBRID or not semantic:
        return semantic

    try:
        bm25_raw = _bm25_scores(query_text)
    except Exception:  # noqa: BLE001 — degrade gracefully to semantic-only
        return semantic

    sem_norm = _min_max_norm({r["text"]: r["score"] for r in semantic})
    bm25_norm = _min_max_norm(bm25_raw)
    meta_by_text = {r["text"]: r["metadata"] for r in semantic}
    alpha = config.HYBRID_ALPHA

    fused: dict[str, float] = {}
    for text in set(sem_norm) | set(bm25_norm):
        fused[text] = alpha * sem_norm.get(text, 0.0) + (1 - alpha) * bm25_norm.get(text, 0.0)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:pool]
    return [
        {"text": t, "metadata": meta_by_text.get(t, {"source": "?", "page": "?"}), "score": round(s, 4)}
        for t, s in ranked
    ]


def _clean(c: dict) -> dict:
    """Normalize a result dict: prefer the rerank score as the headline score."""
    out = {"text": c["text"], "metadata": c["metadata"], "score": c.get("score")}
    if "rerank_score" in c:
        out["rerank_score"] = round(c["rerank_score"], 4)
    return out


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citation-friendly context block."""
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        src = ch["metadata"].get("source", "?")
        page = ch["metadata"].get("page", "?")
        lines.append(f"[{i}] (source: {src}, page: {page})\n{ch['text']}")
    return "\n\n".join(lines)
