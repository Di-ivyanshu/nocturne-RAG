"""Multi-agent self-correction — the differentiator of this project.

Teen role:
  1. answer_agent : produces a cited answer from the context; says "I don't know"
                    when the context doesn't contain the answer.
  2. critic_agent : verifies the answer against the context -> grounded? confidence?
  3. rewrite_query: if confidence is low, rewrites the query and retrieves again.

Loop: answer -> critique -> (if weak) rewrite + re-retrieve -> answer ... (up to max tries).
This reduces hallucination and attaches an honest confidence to every answer.
"""
from __future__ import annotations

from . import config, llm
from .retriever import format_context, retrieve

# ---- Prompts ---------------------------------------------------------------

_ANSWER_PROMPT = """You are a precise knowledge assistant. Answer the user's QUESTION
using ONLY the CONTEXT below. Rules:
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Do NOT use outside knowledge or guess.
- Cite the sources you used with bracket numbers like [1], [2] that match the context blocks.
- Be concise and direct.

Formatting (use clean Markdown):
- Open with a one-line summary sentence when helpful.
- When listing multiple items, put EACH item on its OWN line as a "- " bullet or a
  "1." numbered item. Never cram a list into a single running sentence.
- Bold the key fact in each point with **double asterisks**.
- Keep paragraphs short; separate them with a blank line.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

_CRITIC_PROMPT = """You are a strict fact-checker. Decide whether the ANSWER is fully
supported by the CONTEXT (no hallucination, no outside facts).

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
{answer}

Respond ONLY with JSON of this exact shape:
{{"grounded": true/false, "confidence": 0.0-1.0, "reason": "<short>", "missing": "<what info is missing, or empty>"}}
- grounded=true only if every claim in the answer is backed by the context.
- An honest "I don't know" answer is grounded=true with high confidence.
- confidence reflects how well the context supports the answer."""

_REWRITE_PROMPT = """The previous search did not retrieve enough information to answer the
question. Rewrite the QUESTION into a single improved search query that may match the
documents better (add synonyms / key terms, remove fluff). Return ONLY the query text.

QUESTION: {question}
WHAT WAS MISSING: {missing}

IMPROVED QUERY:"""

_CONDENSE_PROMPT = """Given the CONVERSATION so far and a FOLLOW-UP question, rewrite the
follow-up into a standalone question that makes sense on its own (resolve pronouns like
"it"/"that" and implied context). If the follow-up is already standalone, return it
unchanged. Return ONLY the rewritten question.

CONVERSATION:
{history}

FOLLOW-UP: {question}

STANDALONE QUESTION:"""


# ---- Agents ----------------------------------------------------------------

def answer_agent(question: str, context: str) -> str:
    return llm.generate(_ANSWER_PROMPT.format(context=context, question=question))


def critic_agent(question: str, context: str, answer: str) -> dict:
    result = llm.generate_json(
        _CRITIC_PROMPT.format(context=context, question=question, answer=answer)
    )
    # defensive defaults if the model returned something odd
    return {
        "grounded": bool(result.get("grounded", False)),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "reason": str(result.get("reason", "")),
        "missing": str(result.get("missing", "")),
    }


def rewrite_query(question: str, missing: str) -> str:
    new_q = llm.generate(
        _REWRITE_PROMPT.format(question=question, missing=missing), temperature=0.3
    )
    return new_q.strip() or question


def condense_question(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a standalone question using recent chat history.

    `history` is a list of {"question", "answer"} turns. Returns the original
    question unchanged when there's no history.
    """
    if not history:
        return question
    recent = history[-config.HISTORY_TURNS:]
    convo = "\n".join(f"User: {t['question']}\nAssistant: {t['answer']}" for t in recent)
    rewritten = llm.generate(
        _CONDENSE_PROMPT.format(history=convo, question=question), temperature=0.0
    )
    return rewritten.strip() or question


# ---- Orchestration ---------------------------------------------------------

def answer_with_self_correction(
    question: str, top_k: int = config.TOP_K, history: list[dict] | None = None
) -> dict:
    """Run the full retrieve -> answer -> critique -> refine loop.

    With `history`, a follow-up question is first condensed into a standalone one
    so retrieval works on the resolved intent. Returns a dict with: answer,
    confidence, grounded, citations, evidence, attempts (trace), self_corrected,
    and the standalone question actually used.
    """
    standalone = condense_question(question, history or [])
    search_query = standalone
    attempts: list[dict] = []
    best: dict | None = None

    for attempt_no in range(1, config.MAX_REFINE_TRIES + 2):  # 1 initial + N refines
        chunks = retrieve(search_query, top_k=top_k)
        context = format_context(chunks) if chunks else "(no documents retrieved)"
        answer = answer_agent(standalone, context)
        verdict = critic_agent(standalone, context, answer)

        attempt = {
            "attempt": attempt_no,
            "search_query": search_query,
            "answer": answer,
            "confidence": round(verdict["confidence"], 2),
            "grounded": verdict["grounded"],
            "reason": verdict["reason"],
            "chunks": chunks,
        }
        attempts.append(attempt)

        if best is None or verdict["confidence"] > best["confidence"]:
            best = attempt

        # good enough -> stop early
        if verdict["grounded"] and verdict["confidence"] >= config.CONFIDENCE_THRESHOLD:
            break

        # otherwise try to refine the search (unless we're out of tries)
        if attempt_no <= config.MAX_REFINE_TRIES:
            search_query = rewrite_query(standalone, verdict["missing"])

    assert best is not None
    return {
        "answer": best["answer"],
        "confidence": best["confidence"],
        "grounded": best["grounded"],
        "citations": _collect_citations(best["chunks"]),
        "evidence": _collect_evidence(best["chunks"]),
        "attempts": attempts,
        "self_corrected": len(attempts) > 1,
        "standalone_question": standalone,
    }


def answer_stream(
    question: str, top_k: int = config.TOP_K, history: list[dict] | None = None
):
    """Stream the answer token-by-token, then yield final verification metadata.

    Yields ("token", text) events as the answer is generated, then one
    ("meta", result_dict) event after the critic has scored it. This path is a
    single pass (no refine loop) so the streamed text stays coherent.
    """
    standalone = condense_question(question, history or [])
    chunks = retrieve(standalone, top_k=top_k)
    context = format_context(chunks) if chunks else "(no documents retrieved)"

    parts: list[str] = []
    for tok in llm.generate_stream(_ANSWER_PROMPT.format(context=context, question=standalone)):
        parts.append(tok)
        yield ("token", tok)

    answer = "".join(parts).strip()
    verdict = critic_agent(standalone, context, answer)
    yield ("meta", {
        "answer": answer,
        "confidence": round(verdict["confidence"], 2),
        "grounded": verdict["grounded"],
        "citations": _collect_citations(chunks),
        "evidence": _collect_evidence(chunks),
        "attempts": [{
            "attempt": 1,
            "search_query": standalone,
            "answer": answer,
            "confidence": round(verdict["confidence"], 2),
            "grounded": verdict["grounded"],
            "reason": verdict["reason"],
            "chunks": chunks,
        }],
        "self_corrected": False,
        "standalone_question": standalone,
    })


def _collect_citations(chunks: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    citations: list[dict] = []
    for ch in chunks:
        src = ch["metadata"].get("source", "?")
        page = ch["metadata"].get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            citations.append({"source": src, "page": page, "score": ch.get("score")})
    return citations


def _collect_evidence(chunks: list[dict]) -> list[dict]:
    """The exact passages used to ground the answer — powers click-to-inspect."""
    evidence: list[dict] = []
    for i, ch in enumerate(chunks, start=1):
        evidence.append(
            {
                "n": i,
                "source": ch["metadata"].get("source", "?"),
                "page": ch["metadata"].get("page", "?"),
                "score": ch.get("score"),
                "rerank_score": ch.get("rerank_score"),
                "text": ch["text"],
            }
        )
    return evidence
