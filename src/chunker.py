"""Chunking: split documents into overlapping chunks, keeping metadata.

Why we chunk: LLMs and embeddings have limited context, and retrieval is only
precise when text is in small, focused pieces. Overlap ensures a sentence/idea
isn't split across two chunks and lose its meaning.

Strategy: respect paragraph boundaries (\\n\\n) and greedily fill up to ~CHUNK_SIZE
chars; if a single paragraph is larger, hard-split it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .ingest import Document


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Hard character-window split for text bigger than `size`."""
    pieces: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        pieces.append(text[start : start + size])
        start += step
    return pieces


def chunk_document(
    doc: Document,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split one Document into overlapping chunks."""
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""

    def flush(buf: str) -> None:
        buf = buf.strip()
        if buf:
            chunks.append(Chunk(text=buf, metadata=dict(doc.metadata)))

    for para in paragraphs:
        if len(para) > chunk_size:
            flush(buffer)
            buffer = ""
            for piece in _split_long(para, chunk_size, overlap):
                flush(piece)
            continue

        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            flush(buffer)
            # start new buffer with a tail of the old one for overlap continuity
            tail = buffer[-overlap:] if overlap and buffer else ""
            buffer = f"{tail}\n\n{para}".strip() if tail else para

    flush(buffer)

    # stamp a stable per-chunk index into metadata
    for i, ch in enumerate(chunks):
        ch.metadata = {**ch.metadata, "chunk": i}
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Chunk a list of documents into a flat list of chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
