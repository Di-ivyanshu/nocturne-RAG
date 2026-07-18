"""Command-line interface (optional, dev/testing ke liye).

Usage:
    python cli.py index [--reset]      # index the data/ folder
    python cli.py ask "your question"  # ask a question
    python cli.py stats                # how many chunks are indexed
"""
from __future__ import annotations

import argparse
import sys

try:  # Windows console (cp1252) chokes on non-ASCII; force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from src import pipeline


def _cmd_index(args: argparse.Namespace) -> None:
    print("Building index from data/ ...")
    stats = pipeline.build_index(reset=args.reset)
    print(
        f"Done. {stats['documents']} documents, {stats['pages']} pages, "
        f"+{stats['chunks_added']} chunks (total {stats['chunks_total']})."
    )


def _cmd_ask(args: argparse.Namespace) -> None:
    result = pipeline.answer(args.question)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']:.2f}  |  Grounded: {result['grounded']}", end="")
    if result["self_corrected"]:
        print(f"  |  self-corrected over {len(result['attempts'])} attempts", end="")
    print()
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  - {c['source']} (page {c['page']})")


def _cmd_stats(_: argparse.Namespace) -> None:
    print(f"Indexed chunks: {pipeline.index_stats()['chunks_total']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Correcting RAG Assistant — CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index documents in data/")
    p_index.add_argument("--reset", action="store_true", help="clear existing index first")
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question", help="the question to ask")
    p_ask.set_defaults(func=_cmd_ask)

    p_stats = sub.add_parser("stats", help="show index stats")
    p_stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
