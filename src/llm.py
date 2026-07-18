"""LLM wrapper — single place that talks to the model.

Supports two free providers, chosen by config.LLM_PROVIDER:
  - "groq"   : generous free tier (~14k req/day), uses llama-3.x
  - "gemini" : Google free tier (small daily cap on some keys)

A JSON helper lives here too, since the critic/judge need structured output.
Clients (provider SDKs) configure lazily so importing this module is cheap.
"""
from __future__ import annotations

import json
import os
import re
import time

from . import config

_gemini_ready = False
_groq_client = None

# Throttle between calls (seconds). Gemini free tier is ~10/min, so default 7s.
# Groq is far more generous, so we drop the gap to keep things fast.
_DEFAULT_INTERVAL = "0.5" if config.LLM_PROVIDER == "groq" else "7"
MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", _DEFAULT_INTERVAL))
_last_call_ts = 0.0


def _throttle() -> None:
    """Space out calls to respect provider rate limits."""
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_call_ts = time.monotonic()


def _retry_delay(err: Exception, default: float) -> float:
    """Pull the server-suggested wait (seconds) out of a 429 error message."""
    m = re.search(r"retry in (\d+(?:\.\d+)?)", str(err))
    if m:
        return float(m.group(1)) + 1
    m = re.search(r"seconds:\s*(\d+)", str(err))
    if m:
        return float(m.group(1)) + 1
    return default


# --- Provider backends ------------------------------------------------------

def _gemini_generate(prompt: str, temperature: float) -> str:
    global _gemini_ready
    import google.generativeai as genai

    if not _gemini_ready:
        genai.configure(api_key=config.require_api_key())
        _gemini_ready = True
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    resp = model.generate_content(prompt, generation_config={"temperature": temperature})
    return (resp.text or "").strip()


def _groq_generate(prompt: str, temperature: float) -> str:
    global _groq_client
    from groq import Groq

    if _groq_client is None:
        _groq_client = Groq(api_key=config.require_api_key())
    resp = _groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _backend():
    return _groq_generate if config.LLM_PROVIDER == "groq" else _gemini_generate


def _groq_stream(prompt: str, temperature: float):
    global _groq_client
    from groq import Groq

    if _groq_client is None:
        _groq_client = Groq(api_key=config.require_api_key())
    stream = _groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _gemini_stream(prompt: str, temperature: float):
    global _gemini_ready
    import google.generativeai as genai

    if not _gemini_ready:
        genai.configure(api_key=config.require_api_key())
        _gemini_ready = True
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    for chunk in model.generate_content(
        prompt, generation_config={"temperature": temperature}, stream=True
    ):
        if getattr(chunk, "text", ""):
            yield chunk.text


def generate_stream(prompt: str, *, temperature: float = 0.2):
    """Yield answer text token-by-token from the active LLM.

    Throttled once up front; on failure it yields nothing and the caller falls
    back to the non-streaming path.
    """
    _throttle()
    backend = _groq_stream if config.LLM_PROVIDER == "groq" else _gemini_stream
    yield from backend(prompt, temperature)


def generate(prompt: str, *, temperature: float = 0.2, retries: int = 3) -> str:
    """Send a prompt to the active LLM and return plain text.

    Throttles to respect rate limits; on a 429 waits the server-suggested delay
    before retrying.
    """
    backend = _backend()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            _throttle()
            return backend(prompt, temperature)
        except Exception as exc:  # noqa: BLE001 — surface after retries
            last_err = exc
            if attempt < retries:
                is_429 = "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower()
                wait = _retry_delay(exc, default=2.0 * (attempt + 1)) if is_429 else 1.5 * (attempt + 1)
                time.sleep(min(wait, 65))  # cap so we never hang too long
    raise RuntimeError(
        f"{config.LLM_PROVIDER} call failed after {retries + 1} tries: {last_err}"
    )


def generate_json(prompt: str, *, temperature: float = 0.0) -> dict:
    """Ask the LLM for JSON and parse it robustly (handles ```json fences)."""
    raw = generate(prompt, temperature=temperature)
    text = raw.strip()
    if text.startswith("```"):
        # strip a ```json ... ``` fence
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...}
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}
