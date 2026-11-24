"""Candidate discovery for local models.

Unlike model-radar (which reads a provider's /models with prices), the Ollama
library has no single machine-readable catalogue endpoint. So candidates come
from three sources, merged and deduped:

  1. the library's model list (scraped from ollama.com/library -- names only),
  2. config `radar.candidates` (explicit tags the operator wants watched),
  3. the `--candidates` / `--include` CLI args.

A candidate is just a name + optional hint. Size/context are read from
/api/show AFTER a pull, so filtering by hardware happens post-show in radar.py.
"""
from __future__ import annotations

import re
import urllib.request


def library_names(timeout: int = 30) -> list[str]:
    """Scrape ollama.com/library for model names. Best-effort, not a verdict."""
    try:
        req = urllib.request.Request(
            "https://ollama.com/library",
            headers={"User-Agent": "curl/8.5.0"})
        html = urllib.request.urlopen(req, timeout=timeout).read().decode(
            errors="replace")
    except Exception:                                        # noqa: BLE001
        return []
    names = re.findall(r"/library/([a-z0-9][a-z0-9.-]+)", html)
    return sorted(set(names))


def merge(blocklist: list[str], configured: list[str], extra: list[str]) -> list[str]:
    out: dict[str, str] = {}
    for n in configured:
        out[n] = n
    for n in extra:
        out[n] = n
    blocked = set(blocklist)
    return [n for n in out if n not in blocked]
