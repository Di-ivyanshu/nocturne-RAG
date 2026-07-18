"""Retriever helpers: tokenizing, score normalization, result shaping, context."""
from src.retriever import _clean, _min_max_norm, _tokenize, format_context


def test_tokenize_lowercases_and_keeps_alnum():
    assert _tokenize("Hello, World! 80C") == ["hello", "world", "80c"]


def test_min_max_norm_scales_to_unit_range():
    out = _min_max_norm({"a": 1.0, "b": 3.0})
    assert out["a"] == 0.0 and out["b"] == 1.0


def test_min_max_norm_handles_equal_scores():
    assert _min_max_norm({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}


def test_clean_surfaces_rerank_score():
    c = {"text": "t", "metadata": {"source": "s", "page": 1}, "score": 0.5, "rerank_score": 2.3456}
    out = _clean(c)
    assert out["rerank_score"] == 2.3456
    assert out["text"] == "t" and out["metadata"]["source"] == "s"


def test_format_context_is_numbered_and_cited():
    ctx = format_context([{"text": "hello", "metadata": {"source": "a.md", "page": 3}}])
    assert "[1]" in ctx and "a.md" in ctx and "page: 3" in ctx
