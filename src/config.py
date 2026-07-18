"""Central configuration. Reads .env and exposes tunable settings.

All knobs in one place — chunk size, model names, retrieval top_k, self-correction
threshold. Tweak here; no need to touch the rest of the code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"          # uploaded / source documents
CHROMA_DIR = ROOT_DIR / "chroma_db"   # persistent vector store

# Load .env from project root
load_dotenv(ROOT_DIR / ".env")

# --- LLM provider selection ---
# "groq" (generous free tier) or "gemini". If unset, auto-pick based on which
# API key is present (Groq preferred when both exist).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _auto_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"groq", "gemini"}:
        return explicit
    if GROQ_API_KEY:
        return "groq"
    return "gemini"


LLM_PROVIDER = _auto_provider()

# --- Embeddings (local, free) ---
# BGE small: stronger retrieval quality than MiniLM, still 384-dim and fast.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# BGE recommends prepending this instruction to the QUERY (not passages) for retrieval.
EMBED_QUERY_PREFIX = os.getenv(
    "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)

# --- Vector store ---
COLLECTION_NAME = "knowledge_base"

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      # chars per chunk (approx)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # chars of overlap

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))
USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() == "true"
# weight for semantic (vector) score in hybrid merge; (1-w) for BM25
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))
# candidate pool fetched before re-ranking trims it down to TOP_K
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))

# --- Re-ranker (cross-encoder, local) ---
USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Conversation memory ---
# how many previous (question, answer) turns to use when condensing a follow-up
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))

# --- Self-correction loop ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MAX_REFINE_TRIES = int(os.getenv("MAX_REFINE_TRIES", "2"))

# Supported document extensions
SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}


def require_api_key() -> str:
    """Return the active provider's API key or raise a clear, actionable error."""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY missing. Get a FREE key at https://console.groq.com/keys "
                "and put it in your .env file (copy .env.example -> .env)."
            )
        return GROQ_API_KEY
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY missing. Get a FREE key at "
            "https://aistudio.google.com/apikey and put it in a .env file "
            "(copy .env.example -> .env)."
        )
    return GEMINI_API_KEY


def active_model() -> str:
    return GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
