"""Re-ranker: graceful fallback when the cross-encoder is unavailable."""
from src import reranker


def test_rerank_empty_input():
    assert reranker.rerank("q", [], top_k=5) == []


def test_rerank_falls_back_to_input_order(monkeypatch):
    def boom():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(reranker, "_get_model", boom)
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert reranker.rerank("q", chunks, top_k=2) == chunks[:2]


def test_rerank_orders_by_model_score(monkeypatch):
    class FakeModel:
        def predict(self, pairs):
            # score by length of the chunk text — longest is most "relevant"
            return [len(t) for _, t in pairs]

    monkeypatch.setattr(reranker, "_get_model", lambda: FakeModel())
    chunks = [{"text": "short"}, {"text": "the longest passage here"}, {"text": "mid one"}]
    out = reranker.rerank("q", chunks, top_k=2)
    assert out[0]["text"] == "the longest passage here"
    assert "rerank_score" in out[0]
