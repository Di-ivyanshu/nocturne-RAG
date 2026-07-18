"""Agents: critic normalization, answer pass-through, and follow-up condensing.

The LLM layer is monkeypatched so these run offline with no API key.
"""
from src import agents


def test_condense_returns_question_when_no_history():
    assert agents.condense_question("What is X?", []) == "What is X?"


def test_condense_uses_llm_with_history(monkeypatch):
    monkeypatch.setattr(agents.llm, "generate", lambda *a, **k: "What is the sick leave limit?")
    out = agents.condense_question("and sick leave?", [{"question": "leave policy?", "answer": "20 days"}])
    assert out == "What is the sick leave limit?"


def test_critic_normalizes_messy_output(monkeypatch):
    monkeypatch.setattr(
        agents.llm, "generate_json",
        lambda *a, **k: {"grounded": 1, "confidence": "0.8", "reason": "ok", "missing": ""},
    )
    v = agents.critic_agent("q", "ctx", "ans")
    assert v["grounded"] is True
    assert abs(v["confidence"] - 0.8) < 1e-9


def test_critic_defaults_on_empty(monkeypatch):
    monkeypatch.setattr(agents.llm, "generate_json", lambda *a, **k: {})
    v = agents.critic_agent("q", "ctx", "ans")
    assert v["grounded"] is False and v["confidence"] == 0.0


def test_answer_agent_passes_through(monkeypatch):
    monkeypatch.setattr(agents.llm, "generate", lambda *a, **k: "  the answer  ")
    assert agents.answer_agent("q", "ctx") == "  the answer  "
