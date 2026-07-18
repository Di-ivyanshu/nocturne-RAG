"""Chunking: sizing, overlap, and metadata preservation."""
from src.chunker import chunk_document, chunk_documents
from src.ingest import Document


def test_small_doc_is_single_chunk():
    doc = Document(text="A short note.", metadata={"source": "a.txt", "page": 1})
    chunks = chunk_document(doc, chunk_size=800, overlap=120)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "a.txt"
    assert chunks[0].metadata["chunk"] == 0


def test_long_paragraph_is_split():
    doc = Document(text="word " * 400, metadata={"source": "b.txt", "page": 1})
    chunks = chunk_document(doc, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    # no chunk wildly exceeds the target size
    assert all(len(c.text) <= 520 for c in chunks)


def test_metadata_preserved_and_chunk_indexed():
    doc = Document(text="A\n\nB\n\nC", metadata={"source": "c.md", "page": 2})
    chunks = chunk_documents([doc])
    assert all(c.metadata["source"] == "c.md" for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))
