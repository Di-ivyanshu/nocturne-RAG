"""ChromaDB persistent vector store wrapper.

A vector DB stores embeddings + their text + metadata, and provides "nearest
neighbour" search (the chunks closest to the query vector). Persistent so the
index doesn't have to be rebuilt every time.
"""
from __future__ import annotations

from . import config
from .chunker import Chunk
from .embedder import embed_query, embed_texts


def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _collection():
    # cosine space matches our normalized embeddings
    return _client().get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def reset() -> None:
    """Delete the whole collection (used by 'Clear index')."""
    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # noqa: BLE001 — already absent is fine
        pass


def add_chunks(chunks: list[Chunk]) -> int:
    """Embed and store chunks. Returns number added."""
    if not chunks:
        return 0
    col = _collection()
    existing = col.count()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"chunk-{existing + i}" for i in range(len(chunks))]
    col.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def count() -> int:
    """How many chunks are indexed right now."""
    try:
        return _collection().count()
    except Exception:  # noqa: BLE001
        return 0


def query(text: str, top_k: int = config.TOP_K) -> list[dict]:
    """Semantic search. Returns list of {text, metadata, score} (score 1=best)."""
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embed_query(text)],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out: list[dict] = []
    for text_i, meta_i, dist_i in zip(docs, metas, dists):
        # cosine distance -> similarity score in [0,1]
        out.append({"text": text_i, "metadata": meta_i, "score": 1.0 - float(dist_i)})
    return out


def all_documents() -> list[dict]:
    """Return every stored chunk (used to build the BM25 keyword index)."""
    col = _collection()
    if col.count() == 0:
        return []
    res = col.get(include=["documents", "metadatas"])
    return [
        {"text": t, "metadata": m}
        for t, m in zip(res["documents"], res["metadatas"])
    ]


def documents() -> list[dict]:
    """Group indexed chunks by source file: [{source, chunks}], most chunks first."""
    col = _collection()
    if col.count() == 0:
        return []
    res = col.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for m in res["metadatas"]:
        src = m.get("source", "?")
        counts[src] = counts.get(src, 0) + 1
    return [
        {"source": s, "chunks": n}
        for s, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]


def delete_by_source(source: str) -> int:
    """Delete all chunks belonging to one source file. Returns how many were removed."""
    col = _collection()
    before = col.count()
    col.delete(where={"source": source})
    return before - col.count()
