# Nocturne Archive

Retrieval-augmented generation over your own documents.

Point it at a folder of PDFs, Word files or notes, ask a question, and get an answer grounded in those documents with the source file and page attached. Before an answer is shown it is checked against the passages it was built from, and when the documents do not contain the answer the system says so instead of guessing.

Embeddings run locally and the LLM runs on a free tier, so there are no paid APIs and no Docker requirement.

[![CI](https://github.com/Di-ivyanshu/nocturne-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Di-ivyanshu/nocturne-RAG/actions/workflows/ci.yml)

## How it works

```
data/  (PDF, DOCX, TXT, MD)
  |
  ingest      PyMuPDF text + table extraction, tables to Markdown, page metadata
  chunker     overlapping chunks
  embedder    sentence-transformers BGE, computed locally
  vectorstore ChromaDB
  |
  retriever   dense similarity + BM25, wide candidate pool
  reranker    cross-encoder re-scores every (query, passage) pair
  |
  agents      answer drafted from retrieved context, then checked against it
              low confidence: query rewritten, retrieval retried
  |
  answer + confidence + citations (file, page)
```

## Ingestion and indexing

PDF, DOCX, TXT and MD are parsed with PyMuPDF. Tables are detected and converted to Markdown so rows and columns survive chunking instead of collapsing into a run-on line. Page numbers travel with each chunk as metadata, which is what makes file-and-page citations possible later.

Text is split into overlapping chunks and embedded locally with a sentence-transformers BGE model, so no embedding API is called and document text is not sent anywhere during indexing. Vectors are persisted in ChromaDB, and individual documents can be removed without rebuilding the whole index.

## Retrieval

Retrieval is hybrid. Dense vector similarity and BM25 lexical search each contribute to a wide candidate pool, which recovers exact-term matches that pure semantic search tends to miss. A cross-encoder re-ranker then re-scores every (query, passage) pair and reorders the pool before the context is assembled, so recall comes from the bi-encoder and precision comes from the cross-encoder.

## Answering and verification

The LLM layer is pluggable. Groq or Google Gemini is selected automatically from whichever key is present, and generation is constrained to the retrieved context with inline source references.

A second pass evaluates the drafted answer against the retrieved passages and produces a confidence signal. When it falls below the threshold the query is reformulated and retrieval runs again before anything is finalised. If the retrieved context does not support an answer at all, the system returns that rather than filling the gap.

## Interface

Follow-up questions are condensed into standalone queries using the conversation history, so a question like "and what about sick leave?" retrieves correctly on its own. Answers stream token by token over Server-Sent Events from a FastAPI backend and then settle into the verified result with its confidence and sources. Clicking a citation reveals the passage it came from.

There are three ways in: the custom frontend in `web/` served by `server.py`, a Streamlit app in `app.py`, and a CLI.

## Quickstart

**1. Install**

```
pip install -r requirements.txt
```

**2. Add a key**

```
cp .env.example .env      # Windows: copy .env.example .env
```

Put one of these in `.env`. Groq is used by default when its key is present.

- Groq: <https://console.groq.com/keys> as `GROQ_API_KEY`
- Gemini: <https://aistudio.google.com/apikey> as `GEMINI_API_KEY`

**3. Run**

```
python -m uvicorn server:app --port 8000
```

Open <http://localhost:8000>, drop in documents, build the index, and ask.

Streamlit alternative:

```
streamlit run app.py
```

Or from the command line:

```
python cli.py index                        # index everything in data/
python cli.py ask "What is the leave policy?"
python cli.py stats
```

## Evaluation

```
python eval/evaluate.py
```

This indexes `sample_docs/`, runs the question set in `eval/golden_dataset.json`, and reports answer correctness, retrieval hit rate, honesty on out-of-scope questions, average confidence, and how often the self-correction loop fired. Results are written to `eval/results.json` and surfaced in the Benchmark panel in the UI.

The bundled set is deliberately small and hand-written: 15 answerable questions plus one out-of-scope question, measured against `sample_docs/` with Groq. It exists as a regression check for this repo, not as a benchmark claim, and numbers will differ on your own documents.

Separating retrieval hit rate from answer correctness is the point of the harness. When an answer is wrong, the two metrics together show whether the retriever failed to surface the right passage or the model failed to use it.

`--fast` swaps the LLM judge for a keyword judge, which saves one API call per question on tight free-tier quotas.

## Tests

```
pip install -r requirements-dev.txt
pytest -q
```

The suite mocks the LLM and embedding layers, so it runs offline in well under a second with no API key, network access or model downloads. It covers chunking, ingestion, retrieval scoring, the re-ranker fallback path, critic normalisation and pipeline error handling. Every push runs it through GitHub Actions (`.github/workflows/ci.yml`).

## Project layout

```
nocturne-RAG/
├── server.py               FastAPI backend for the web UI
├── web/                    frontend (HTML / CSS / JS)
├── app.py                  Streamlit UI
├── cli.py                  command-line interface
├── src/
│   ├── config.py           tunable settings
│   ├── ingest.py           load PDF/DOCX/TXT/MD with metadata
│   ├── chunker.py          overlapping chunking
│   ├── embedder.py         local sentence-transformers embeddings
│   ├── vectorstore.py      ChromaDB store, per-document management
│   ├── retriever.py        hybrid retrieval (dense + BM25)
│   ├── reranker.py         cross-encoder re-ranking
│   ├── llm.py              pluggable Groq/Gemini wrapper with streaming
│   ├── agents.py           answer, verification, retry loop, memory
│   └── pipeline.py         build_index() and answer() orchestration
├── eval/
│   ├── golden_dataset.json question set with ground truth
│   ├── evaluate.py         metrics harness
│   └── results.json        latest metrics, powers the in-app panel
├── tests/                  pytest suite, mocked and offline
├── data/                   documents to index
├── sample_docs/            demo document used by the evaluation
└── .github/workflows/ci.yml
```

## Stack

Python, PyMuPDF, sentence-transformers with BGE embeddings and a cross-encoder re-ranker, ChromaDB, `rank-bm25`, Groq or Google Gemini, FastAPI with a vanilla JS frontend, Streamlit, pytest, GitHub Actions.

## Not done yet

- No authentication or multi-user separation. The index is shared.
- No Docker image or deployment configuration.
- The evaluation set is small and hand-written, which makes it useful for catching regressions but not for comparing against other systems.
