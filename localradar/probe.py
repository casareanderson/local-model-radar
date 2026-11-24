"""Ollama client + hardware fit + the tool-call probe.

Three jobs:
  1. Talk to Ollama: list tags, show model metadata (size, context), pull, and
     run a chat completion with tools.
  2. Classify whether a model FITS THE HARDWARE: fully in VRAM (fast), partly
     offloaded (slower), or CPU-only (rejected).
  3. The tool-call probe: does the model actually make the get_weather call?
     A model that answers in prose cannot drive an agent.

The probe returns (ok, kind, detail, tok_s, ctx) where kind is 'ok', 'hard' or
'transient' -- the same three-way classification model-radar uses.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

TOOL = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

CITY = "Reykjavik"
PROMPT = "What is the weather in %s? Use the tool." % CITY

HARD_MARKERS = (
    "does not exist", "not found", "no such model", "manifest", "unknown model",
)


def _post_json(url: str, payload: dict, timeout: int = 120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def ollama_tags(base_url: str, tags_path: str) -> list[dict]:
    d = json.load(urllib.request.urlopen(base_url + tags_path, timeout=15))
    return d.get("models", [])


def ollama_show(base_url: str, show_path: str, model: str) -> dict:
    d = json.load(_post_json(base_url + show_path, {"name": model}, timeout=30))
    return d


def ollama_pull(base_url: str, pull_path: str, model: str, timeout: int = 1800):
    """Pull a model. Returns True on success. Pull can be slow; allow it."""
    d = json.load(_post_json(base_url + pull_path, {"name": model}, timeout=timeout))
    return d.get("status") == "success"


def ollama_ps(base_url: str, ps_path: str) -> list[dict]:
    d = json.load(urllib.request.urlopen(base_url + ps_path, timeout=15))
    return d.get("models", [])


def classify(status: int, body: str) -> str:
    if status in (408, 409, 429) or status >= 500:
        return "transient"
    low = body.lower()
    if any(m in low for m in HARD_MARKERS):
        return "hard"
    return "hard" if status in (400, 401, 403, 404) else "transient"


def size_mib(size_bytes: int | None) -> int:
    return int((size_bytes or 0) / (1024 * 1024))


def fits(size_bytes: int | None, vram_budget_mib: int) -> str:
    """'vram' (fully in GPU), 'partial' (offloads), or 'cpu' (too big)."""
    mib = size_mib(size_bytes)
    if mib == 0:
        return "unknown"
    if mib <= vram_budget_mib:
        return "vram"
    # Partial: it will offload to the 31GB of RAM, but is slower. Still usable.
    return "partial"


def probe(base_url: str, chat_path: str, model: str, timeout: int = 300):
    """(ok, kind, detail, tok_s, ctx). ok None = transient."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOL, "tool_choice": "auto", "max_tokens": 256,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base_url + chat_path, data=body, headers={
        "Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raw = e.read()[:200].decode(errors="replace")
        kind = classify(e.code, raw)
        return (None if kind == "transient" else False), kind, \
            "HTTP %s: %s" % (e.code, raw), 0.0, 0
    except Exception as e:                                    # noqa: BLE001
        return None, "transient", "%s: %s" % (type(e).__name__, e), 0.0, 0

    dt = max(time.time() - t0, 0.01)
    usage = d.get("usage") or {}
    ctx = int((d.get("prompt_tokens") or 0) and usage.get("prompt_tokens", 0)
              or 0)  # placeholder; real ctx comes from /api/show
    out_tok = usage.get("completion_tokens") or 0
    tok_s = round(out_tok / dt, 2) if out_tok else 0.0
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, "hard", "no tool_call", tok_s, ctx
    fn = calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:                                         # noqa: BLE001
        return False, "hard", "unparseable tool arguments", tok_s, ctx
    if fn.get("name") != "get_weather" or \
            CITY.lower() not in str(args.get("city", "")).lower():
        return False, "hard", "wrong tool call", tok_s, ctx
    return True, "ok", "ok", tok_s, ctx


def model_context(model_show: dict) -> int:
    """Advertised context length from /api/show, or 0 if unknown."""
    info = model_show.get("model_info") or {}
    for k, v in info.items():
        if "context" in k.lower():
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0


def model_size(model_show: dict, tags: dict | None = None) -> int:
    """Total on-disk size in bytes. /api/show has no size field on newer
    Ollama, so prefer a name->size map built from /api/tags; fall back to any
    size-ish field on the show dict."""
    if tags:
        for name, size in tags.items():
            if name == model_show.get("model") or name == model_show.get("name"):
                return int(size)
    for key in ("size", "model_size"):
        if key in model_show:
            try:
                return int(model_show[key])
            except (TypeError, ValueError):
                continue
    return 0


def tags_size_map(tags_list: list[dict]) -> dict[str, int]:
    """name -> size bytes, from the /api/tags models list."""
    return {m.get("name"): m.get("size", 0) for m in tags_list}
