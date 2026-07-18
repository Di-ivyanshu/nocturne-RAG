"""Evaluation harness — proves RAG quality in numbers (the core of "recognition").

What it measures:
  - Retrieval hit-rate : did the correct source document show up in the top results?
  - Answer correctness  : LLM-as-judge comparison (expected vs actual) -> correct%
  - "I don't know" check : for questions not answerable from the docs, did the model stay honest?
  - Avg confidence + self-correction rate

Run:  python eval/evaluate.py
(It first indexes sample_docs/ into a fresh evaluation collection.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows console (cp1252) chokes on emoji — force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 — older interpreters / redirected streams
    pass

# allow "from src import ..." when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm, pipeline, vectorstore  # noqa: E402

GOLDEN = Path(__file__).parent / "golden_dataset.json"
SAMPLE_DOCS = config.ROOT_DIR / "sample_docs"

_JUDGE_PROMPT = """You are grading a RAG system. Compare the PREDICTED answer to the
EXPECTED answer for the QUESTION. They match if the predicted answer conveys the same
key facts as expected (wording may differ).

QUESTION: {question}
EXPECTED: {expected}
PREDICTED: {predicted}

Respond ONLY with JSON: {{"correct": true/false, "reason": "<short>"}}"""


def judge_llm(question: str, expected: str, predicted: str) -> bool:
    res = llm.generate_json(
        _JUDGE_PROMPT.format(question=question, expected=expected, predicted=predicted)
    )
    return bool(res.get("correct", False))


def judge_keyword(_q: str, expected: str, predicted: str) -> bool:
    """No-LLM judge: do the key tokens of the expected answer appear in prediction?

    Saves one Gemini call per question (helpful on tight free-tier quotas).
    Less precise than the LLM judge but good enough for a quick signal.
    """
    import re

    stop = {"the", "a", "an", "of", "to", "is", "are", "per", "and", "or", "i",
            "based", "on", "provided", "documents", "for", "in", "at", "up", "days"}
    exp_tokens = {t for t in re.findall(r"\w+", expected.lower()) if t not in stop and len(t) > 1}
    pred_low = predicted.lower()
    if not exp_tokens:
        return True
    hits = sum(1 for t in exp_tokens if t in pred_low)
    return hits / len(exp_tokens) >= 0.6


def main() -> None:
    config.require_api_key()  # fail early with a clear message

    fast = "--fast" in sys.argv
    judge = judge_keyword if fast else judge_llm
    if fast:
        print("(--fast mode: keyword judge, no extra LLM calls)\n")

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    print("Re-indexing sample_docs/ for a clean evaluation...")
    vectorstore.reset()
    stats = pipeline.build_index(SAMPLE_DOCS, reset=True)
    print(f"Indexed {stats['chunks_total']} chunks.\n")

    n = len(golden)
    correct = 0
    retrieval_hits = 0
    retrieval_total = 0
    idk_correct = 0
    idk_total = 0
    confidences: list[float] = []
    self_corrected = 0

    print(f"Running {n} evaluation questions...\n")
    evaluated = 0
    for i, item in enumerate(golden, start=1):
        q = item["question"]
        expected = item["expected_answer"]
        src = item.get("source")

        result = pipeline.answer(q)
        predicted = result["answer"]
        # quota exhausted mid-run -> stop cleanly and report what we have
        if "quota is exhausted" in predicted or predicted.startswith("⚠️"):
            print(f"\n  ! Stopped at Q{i}: {predicted}")
            print("  (Free-tier quota hit — partial results below. Retry later.)\n")
            break

        evaluated += 1
        confidences.append(result["confidence"])
        if result["self_corrected"]:
            self_corrected += 1

        try:
            is_correct = judge(q, expected, predicted)
        except RuntimeError:
            print(f"\n  ! Judge quota hit at Q{i} — partial results below.\n")
            evaluated -= 1
            confidences.pop()
            break
        correct += int(is_correct)

        # retrieval hit-rate (only for answerable questions with a known source)
        if src:
            retrieval_total += 1
            cited = {c["source"] for c in result["citations"]}
            if src in cited:
                retrieval_hits += 1
        else:
            # "should say I don't know" cases
            idk_total += 1
            if "don't know" in predicted.lower() or "do not know" in predicted.lower():
                idk_correct += 1

        mark = "OK " if is_correct else "XX "
        print(f"  [{mark}] Q{i}: {q[:60]}")

    denom = evaluated or 1
    print("\n" + "=" * 52)
    print("              EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Questions evaluated    : {evaluated}/{n}")
    print(f"  Answer correctness     : {correct}/{evaluated}  ({correct / denom:.0%})")
    if retrieval_total:
        print(
            f"  Retrieval hit-rate     : {retrieval_hits}/{retrieval_total}  "
            f"({retrieval_hits / retrieval_total:.0%})"
        )
    if idk_total:
        print(
            f"  'I don't know' honesty : {idk_correct}/{idk_total}  "
            f"({idk_correct / idk_total:.0%})"
        )
    if confidences:
        print(f"  Avg confidence         : {sum(confidences) / len(confidences):.2f}")
    print(f"  Self-correction rate   : {self_corrected}/{denom}  ({self_corrected / denom:.0%})")
    print("=" * 52)

    # persist for the in-app eval dashboard (/api/eval)
    metrics = {
        "available": True,
        "total": n,
        "evaluated": evaluated,
        "answer_correctness": {"correct": correct, "of": evaluated, "pct": round(correct / denom * 100)},
        "retrieval_hit_rate": {"hits": retrieval_hits, "of": retrieval_total,
                               "pct": round(retrieval_hits / retrieval_total * 100) if retrieval_total else None},
        "idk_honesty": {"correct": idk_correct, "of": idk_total,
                        "pct": round(idk_correct / idk_total * 100) if idk_total else None},
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "self_correction_rate": {"count": self_corrected, "of": evaluated,
                                 "pct": round(self_corrected / denom * 100)},
    }
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics -> {out}")


if __name__ == "__main__":
    main()
