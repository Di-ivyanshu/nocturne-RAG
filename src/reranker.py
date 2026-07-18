"""Cross-encoder re-ranking — a second, more precise pass over candidates.

Bi-encoder retrieval (the embeddings search) is fast but approximate: it scores
the query and each chunk *independently*. A cross-encoder instead reads the query
and a chunk *together*, so it judges relevance far more accurately — but it's too
slow to run over the whole corpus. The standard pattern is therefore:

    retrieve a wide candidate pool (cheap bi-encoder)  ->  re-rank it (cross-encoder)

The model is lazy-loaded and cached so startup stays fast.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-score (query, chunk) pairs with a cross-encoder and return the best top_k.

    Each chunk dict gets a ``rerank_score`` added. Falls back to the input order
    (trimmed to top_k) if the model can't be loaded.
    """
    if not chunks:
        return []
    try:
        model = _get_model()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)
    except Exception:  # noqa: BLE001 — degrade gracefully to the original ranking
        return chunks[:top_k]

    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
