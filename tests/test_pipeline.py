"""Pipeline: empty-index handling, error mapping, and delegation."""
from src import pipeline


def test_answer_without_documents(monkeypatch):
    monkeypatch.setattr(pipeline.vectorstore, "count", lambda: 0)
    r = pipeline.answer("anything")
    assert r["confidence"] == 0.0
    assert "No documents" in r["answer"]


def test_friendly_error_maps_quota(monkeypatch):
    r = pipeline._friendly_error(RuntimeError("429 quota exceeded for free tier"))
    assert "quota" in r["answer"].lower() or "limit" in r["answer"].lower()


def test_friendly_error_generic():
    r = pipeline._friendly_error(RuntimeError("boom"))
    assert "boom" in r["answer"]


def test_answer_delegates_to_self_correction(monkeypatch):
    monkeypatch.setattr(pipeline.vectorstore, "count", lambda: 5)
    monkeypatch.setattr(
        pipeline, "answer_with_self_correction",
        lambda question, top_k, history: {"answer": "A", "history_len": len(history or [])},
    )
    r = pipeline.answer("q", history=[{"question": "x", "answer": "y"}])
    assert r["answer"] == "A" and r["history_len"] == 1
