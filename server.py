"""FastAPI backend for the Nocturne Archive UI.

Serves the custom animated frontend (web/) and exposes a small JSON API that
reuses the existing RAG pipeline in src/. Run:

    python -m uvicorn server:app --reload --port 8000

then open http://localhost:8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config, pipeline, vectorstore

app = FastAPI(title="Self-Correcting RAG Assistant")

WEB_DIR = Path(__file__).parent / "web"
EVAL_RESULTS = Path(__file__).parent / "eval" / "results.json"


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    history: list[Turn] = []


def _provider_info() -> dict:
    return {
        "provider": config.LLM_PROVIDER,
        "model": config.active_model(),
        "embed_model": config.EMBED_MODEL,
        "hybrid": config.USE_HYBRID,
    }


@app.get("/api/stats")
def stats() -> dict:
    return {"chunks": vectorstore.count(), **_provider_info()}


@app.post("/api/upload")
async def upload(files: list[UploadFile]) -> dict:
    """Save uploaded files to the data dir and index them."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    paths: list[Path] = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in config.SUPPORTED_EXTS:
            continue
        dest = config.DATA_DIR / f.filename
        dest.write_bytes(await f.read())
        saved.append(f.filename)
        paths.append(dest)

    if not paths:
        return JSONResponse(
            status_code=400,
            content={"error": "No supported files (PDF, DOCX, TXT, MD)."},
        )

    t0 = time.perf_counter()
    result = pipeline.index_files(paths)
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    return {"saved": saved, **result, "chunks": vectorstore.count()}


@app.post("/api/clear")
def clear() -> dict:
    vectorstore.reset()
    return {"chunks": vectorstore.count()}


def _history(req: AskRequest) -> list[dict]:
    return [{"question": t.question, "answer": t.answer} for t in req.history]


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    question = (req.question or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "Empty question."})
    chunks_searched = vectorstore.count()
    t0 = time.perf_counter()
    result = pipeline.answer(question, history=_history(req))
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    result["chunks_searched"] = chunks_searched
    return result


@app.post("/api/ask_stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events: stream answer tokens, then a final 'meta' event."""
    question = (req.question or "").strip()
    history = _history(req)
    chunks_searched = vectorstore.count()

    def events():
        if not question:
            yield _sse({"type": "meta", "answer": "Empty question.", "confidence": 0,
                        "grounded": False, "citations": [], "evidence": [], "attempts": []})
            return
        t0 = time.perf_counter()
        for kind, payload in pipeline.answer_streaming(question, history=history):
            if kind == "token":
                yield _sse({"type": "token", "text": payload})
            else:  # meta
                payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
                payload["chunks_searched"] = chunks_searched
                yield _sse({"type": "meta", **payload})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": pipeline.list_documents()}


@app.delete("/api/documents/{source}")
def delete_document(source: str) -> dict:
    return pipeline.delete_document(source)


@app.get("/api/eval")
def eval_results() -> dict:
    """Latest benchmark results written by eval/evaluate.py (if present)."""
    if EVAL_RESULTS.exists():
        return json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))
    return {"available": False}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# Static frontend (declared last so /api routes win).
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
