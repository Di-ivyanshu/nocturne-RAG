"""Ingestion: loaders, metadata, and unsupported-file handling."""
from src.ingest import load_dir, load_file


def test_load_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("Hello world", encoding="utf-8")
    docs = load_file(p)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "note.txt"
    assert "Hello" in docs[0].text


def test_unsupported_extension_returns_empty(tmp_path):
    p = tmp_path / "archive.zip"
    p.write_text("not a document", encoding="utf-8")
    assert load_file(p) == []


def test_load_dir_picks_up_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nBody", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "c.csv").write_text("ignored", encoding="utf-8")
    docs = load_dir(tmp_path)
    assert {d.metadata["source"] for d in docs} == {"a.md", "b.txt"}
