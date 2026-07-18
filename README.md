# 🧠 Self-Correcting RAG Knowledge Assistant

![CI](https://github.com/Di-ivyanshu/nocturne-RAG/actions/workflows/ci.yml/badge.svg)

A production-minded **Retrieval-Augmented Generation (RAG)** assistant that answers
questions over your own documents — and, unlike most "chat with PDF" demos, it
**verifies its own answers**, reports a **confidence score with source citations**,
and honestly says **"I don't know"** when the answer isn't in your documents.

Built to be genuinely useful *and* to demonstrate the engineering rigor that
distinguishes a real AI engineer from a tutorial follower.

> **100% free stack** — local embeddings (`sentence-transformers`) + a free LLM
> (**Groq** by default, **Gemini** also supported). No paid APIs, no Docker required.

---

## ✨ What makes this different

Most RAG projects stop at "retrieve → answer". This one adds the things that
95% of beginner projects skip — and that interviewers actually ask about:

| Feature | Why it matters |
|---|---|
| 🔁 **Self-correction loop** (multi-agent) | An *answer agent* drafts the reply, a *critic agent* fact-checks it against the retrieved context, and if confidence is low the query is rewritten and retrieval is retried. Cuts hallucination. |
| 📑 **Structured ingestion** | PyMuPDF extracts text *and* detects tables, converting them to Markdown so rows/columns survive chunking — plus accurate page numbers for citations. |
| 🎯 **Hybrid retrieval + cross-encoder re-ranker** | BGE embeddings + BM25 keyword fetch a wide candidate pool; a cross-encoder then re-reads each (query, passage) pair to rank precisely. |
| 📊 **Confidence + citations + "I don't know"** | Every answer ships with a confidence score, the exact source file + page, and refuses to guess when the docs don't contain the answer. |
| 🔍 **Evidence inspection** | Click any citation to reveal the exact passage that grounds the claim — full traceability. |
| 💬 **Conversation memory** | Follow-up questions are condensed into standalone queries using chat history, so "and what about sick leave?" just works. |
| ⚡ **Token streaming** | Answers stream token-by-token (SSE), then settle into the verified result with its confidence gauge. |
| 🗂️ **Document management** | See every indexed file with its chunk count; remove one without rebuilding the whole index. |
| 🧪 **Evaluation harness + in-app dashboard** | A golden Q&A set measures answer correctness, retrieval hit-rate, and "I don't know" honesty — surfaced live in the UI's **Benchmark** panel. |
| ✅ **Tested + CI** | A `pytest` suite (mocked LLM, runs offline) gates every push via GitHub Actions. |

---

## 🏗️ Architecture

```
        Documents (PDF / DOCX / TXT / MD)
                     │
                     ▼
        Ingest (PyMuPDF: text + tables→Markdown, page metadata)
                     │
                     ▼
            Chunking  ──►  Embeddings (BGE, local)
                                          │
                                          ▼
                                 Vector DB (ChromaDB)
                                          │
   (follow-up) ─► condense with history   │
        │                                  ▼
        └──► Retrieval (semantic + BM25 hybrid) ─► Cross-encoder re-rank ─► top-k ◄─┐
                                  │                                                  │
                                  ▼                                                  │
   Answer Agent (streams) ──► Critic Agent (grounded? confidence?)                   │
        ▲                                  │                                         │
        │                  low confidence  │ → rewrite query ───────────────────────┘
        └──────────────────────────────────┘
                     │
                     ▼
        Final Answer + Confidence + Citations + Evidence
```

---

## 🚀 Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Add a free LLM key
```bash
cp .env.example .env      # Windows: copy .env.example .env
```
Then paste **one** key into `.env`:
- **Groq** (recommended, ~14k req/day) — get it at <https://console.groq.com/keys> → `GROQ_API_KEY=`
- **Gemini** — get it at <https://aistudio.google.com/apikey> → `GEMINI_API_KEY=`

The provider is auto-selected (Groq if its key is present).

### 3. Run the web app

**Option A — Nocturne Archive UI (recommended, custom animated frontend):**
```bash
python -m uvicorn server:app --port 8000
```
Open <http://localhost:8000> → drop documents → **Build index** → ask. Each answer
renders with a live verification pipeline, an animated confidence gauge, and source
cards. Powered by a small FastAPI backend (`server.py`) over the same `src/` pipeline.

**Option B — Streamlit UI (simple, zero frontend code):**
```bash
streamlit run app.py     # if 'streamlit' isn't found: python -m streamlit run app.py
```
Upload documents in the sidebar → click **Build / Update Index** → ask questions.

### 4. Or use the CLI
```bash
python cli.py index                       # index everything in data/
python cli.py ask "What is the leave policy?"
python cli.py stats
```

---

## 🧪 Evaluation

```bash
python eval/evaluate.py
```
This indexes the bundled `sample_docs/`, runs a golden Q&A set, and prints:

```
====================================================
              EVALUATION RESULTS
====================================================
  Questions evaluated    : 16/16
  Answer correctness     : 16/16  (100%)
  Retrieval hit-rate     : 15/15  (100%)
  'I don't know' honesty : 1/1    (100%)
  Avg confidence         : 1.00
  Self-correction rate   : 0/16   (0%)
====================================================
```
*Measured on the bundled `sample_docs/` with the Groq provider. 15 answerable
questions + 1 out-of-scope question (capital of France) — the assistant correctly
answered all 15 with the right source and correctly said "I don't know" for the
out-of-scope one. (Numbers will vary with your own documents.)*

**Tip:** `python eval/evaluate.py --fast` uses a keyword-based judge instead of an
LLM judge — one fewer API call per question, handy on tight free-tier quotas.

The metrics are also written to `eval/results.json` and shown live in the UI's
**Benchmark** panel.

---

## ✅ Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
The suite mocks the LLM and embedding layers, so it runs **offline in well under a
second** — no API key, network, or model downloads. It covers chunking, ingestion,
retrieval scoring, the re-ranker fallback, critic normalization, and pipeline error
handling. Every push runs it via GitHub Actions (`.github/workflows/ci.yml`).

---

## 🗂️ Project structure

```
rag1/
├── server.py               # FastAPI backend for the Nocturne Archive UI
├── web/                    # custom animated frontend (HTML / CSS / JS)
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── app.py                  # alternative Streamlit UI
├── cli.py                  # command-line interface
├── src/
│   ├── config.py           # all tunable settings
│   ├── ingest.py           # load PDF/DOCX/TXT/MD + metadata
│   ├── chunker.py          # overlapping chunking
│   ├── embedder.py         # local sentence-transformers embeddings
│   ├── vectorstore.py      # ChromaDB store + per-document management
│   ├── retriever.py        # hybrid retrieval (semantic + BM25)
│   ├── reranker.py         # cross-encoder re-ranking
│   ├── llm.py              # pluggable Groq/Gemini wrapper (+ streaming)
│   ├── agents.py           # answer + critic + self-correction + memory
│   └── pipeline.py         # build_index() / answer() orchestration
├── eval/
│   ├── golden_dataset.json # Q&A ground truth
│   ├── evaluate.py         # metrics harness
│   └── results.json        # latest metrics (powers the in-app dashboard)
├── tests/                  # pytest suite (mocked, offline)
├── .github/workflows/ci.yml
└── sample_docs/            # demo document
```

---

## 🛠️ Tech stack

Python · **PyMuPDF** (text + table extraction) · `sentence-transformers` with
**BGE** embeddings + cross-encoder re-ranker · ChromaDB (vector DB) · Groq / Google
Gemini (LLM, pluggable, streaming) · `rank-bm25` (hybrid search) · FastAPI + vanilla
JS UI · `pytest` + GitHub Actions.

---

## 📝 Résumé bullet

> Built a self-correcting RAG knowledge assistant in Python (sentence-transformers
> embeddings, ChromaDB vector store, hybrid semantic+BM25 retrieval, pluggable
> Groq/Gemini LLM) with a multi-agent answer→critique→refine loop that grounds answers
> in source citations, reports confidence, and is measured by an automated evaluation
> harness (100% answer correctness on a 16-question benchmark).

---

## 🔮 Possible extensions
- Streaming responses · conversation memory · reranker model
- Docker + CI/CD · auth & multi-user · cloud deploy (Streamlit Community Cloud is free)
