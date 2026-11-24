"""Configuration for local-model-radar.

Loads defaults, then a TOML file, then secrets from the environment. No secrets
in the TOML. Mirrors model-radar's config module.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "state_dir": "~/.local/state/local-model-radar",
    "ollama": {
        # Local Ollama endpoint (OpenAI-compatible /v1 + native /api).
        "base_url": "http://192.168.8.152:11434",
        "tags": "/api/tags",
        "show": "/api/show",
        "pull": "/api/pull",
        "generate": "/api/generate",
        "chat": "/v1/chat/completions",
        "ps": "/api/ps",
    },
    "hardware": {
        # GPU VRAM budget in MiB that a model must fit under to be "in VRAM".
        # The RTX 2060 has 6144 MiB; leave headroom for KV cache + ComfyUI.
        "vram_budget_mib": 5500,
        "min_free_disk_mib": 4096,
    },
    "radar": {
        "min_ctx": 65536,
        "max_probes_per_run": 3,
        "adopt": False,
        "adopt_targets": [],
        "adopt_confirms": 3,
        "adopt_confirm_gap_s": 8,
        "adopt_tokens_per_sec_floor": 8.0,
        "adopt_cooldown_days": 7,
        # Tokens-per-second incumbent factor for the latency gate, same idea as
        # model-radar's adopt_latency_factor but on tok/s (higher is better).
        "adopt_speed_factor": 0.7,
        # Models to skip even if they appear in the library.
        "blocklist": [],
        # Extra candidates to always consider, keyed by model tag.
        "candidates": [],
    },
    "canary": {
        "targets": [],
        "fails_before_failover": 2,
        "ladder": [],
        "local": {"model": ""},
    },
    "adapter": {"name": "hermes", "home": "~/.hermes"},
    "notify": {"discord_webhook_env": "LOCALRADAR_DISCORD_WEBHOOK"},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load(path: str | None = None) -> dict:
    path = path or os.environ.get("LOCALRADAR_CONFIG") or "config.toml"
    p = Path(path).expanduser()
    cfg = _merge(DEFAULTS, tomllib.loads(p.read_text()) if p.exists() else {})
    cfg["_path"] = str(p) if p.exists() else "(defaults only)"
    cfg["state_dir"] = str(Path(cfg["state_dir"]).expanduser())
    Path(cfg["state_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
