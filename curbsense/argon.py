"""Thin wrapper over the Gemini Interactions API for every Argon call in the pipeline.

- ARGON_MODEL env picks the model (default gemini-4-argon; set gemini-3.8-flash for the A/B).
- Every call is appended to runs/argon_log.jsonl so the demo UI can show what Argon saw and said.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = os.environ.get("ARGON_MODEL", "gemini-4-argon")
LOG = Path(__file__).resolve().parents[1] / "runs" / "argon_log.jsonl"
INLINE_LIMIT = 15 * 1024 * 1024  # docs: inline requests must stay under 20 MB total

T = TypeVar("T", bound=BaseModel)
_client = None
_lock = threading.Lock()


def client():
    global _client
    with _lock:  # one shared client; parallel threads must not each create (and close) their own
        if _client is None:
            from google import genai
            _client = genai.Client()  # reads GEMINI_API_KEY
    return _client


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def video(path: str | Path, fps: float | None = None) -> dict:
    """Video part: inline if small, else uploaded through the Files API."""
    path = Path(path)
    if path.stat().st_size < INLINE_LIMIT:
        part = {"type": "video", "data": base64.b64encode(path.read_bytes()).decode(), "mime_type": "video/mp4"}
    else:
        f = client().files.upload(file=str(path))
        while not f.state or f.state.name != "ACTIVE":
            time.sleep(3)
            f = client().files.get(name=f.name)
        part = {"type": "video", "uri": f.uri, "mime_type": f.mime_type}
    if fps:
        part["processing"] = {"type": "static", "fps": fps}
    return part


def image(path: str | Path) -> dict:
    path = Path(path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image", "data": base64.b64encode(path.read_bytes()).decode(), "mime_type": mime}


def ask(parts: list[dict] | str, schema: type[T] | None = None, *, system: str | None = None,
        thinking: str = "medium", model: str | None = None, background: bool = False,
        max_output_tokens: int | None = None, tag: str = "") -> T | str:
    """One Argon call. With `schema`, returns a validated pydantic object (structured JSON output)."""
    model = model or MODEL
    kwargs: dict[str, Any] = {"model": model, "input": parts,
                              "generation_config": {"thinking_level": thinking}}
    if max_output_tokens:
        kwargs["generation_config"]["max_output_tokens"] = max_output_tokens
    if system:
        kwargs["system_instruction"] = system
    if schema is not None:
        kwargs["response_format"] = {"type": "text", "mime_type": "application/json",
                                     "schema": schema.model_json_schema()}
    if background:
        kwargs["background"] = True
    t0 = time.time()
    it = client().interactions.create(**kwargs)
    while getattr(it, "status", "completed") in ("in_progress", "queued"):
        time.sleep(3)
        it = client().interactions.get(id=it.id)
    out = it.output_text or ""
    if not out:  # surface why (failed / incomplete / cut off) instead of failing later on empty text
        out_status = getattr(it, "status", None)
        err = getattr(it, "error", None)
        print(f"[argon] empty output for {tag}: status={out_status} error={err}", flush=True)
    _log(tag, model, thinking, parts, out, time.time() - t0, getattr(it, "id", None))
    if schema is None:
        return out
    return schema.model_validate_json(_strip_fences(out))


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    return s


def _log(tag, model, thinking, parts, out, secs, iid):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    seen = parts if isinstance(parts, str) else [
        p["text"][:2000] if p["type"] == "text" else f"<{p['type']}>" for p in parts]
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "tag": tag, "model": model, "thinking": thinking,
                            "secs": round(secs, 1), "interaction_id": iid, "input": seen, "output": out}) + "\n")


if __name__ == "__main__":  # smoke test: uv run python -m curbsense.argon
    print(MODEL, "->", ask("In one sentence: why do delivery robots tip over at curbs?", thinking="low", tag="smoke"))
