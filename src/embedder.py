"""Local embeddings via sentence-transformers (free, offline, no rate limits).

An embedding turns text into numbers (a vector) so "meaning" can be compared.
Texts with similar meaning get vectors that sit close together. This is the heart of RAG.

The model is lazy-loaded (downloaded/cached on first use, then reused) to keep the app fast.
"""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _get_model():
    # imported here so importing this module stays cheap until embeddings needed
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts -> list of float vectors."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-friendly
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single query string.

    For instruction-tuned models (BGE/E5) a short prefix is prepended to the
    query only — this measurably improves retrieval. Passages stay un-prefixed.
    """
    return embed_texts([config.EMBED_QUERY_PREFIX + text])[0]


def embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()
