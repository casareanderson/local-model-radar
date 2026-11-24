"""The adapter boundary -- repoint Hermes profiles at a local model.

A target is a Hermes profile ('global' or a name under profiles/<name>/). The
local-model-radar only ever writes the PRIMARY to a LOCAL (custom/Ollama) model;
it never touches cloud fallbacks. Same traps as model-radar: duplicate-key
refusal and the one-slot providers.custom rule.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Primary:
    provider: str
    model: str
    context_length: int = 0
    fallbacks: list = field(default_factory=list)


class _NoDupLoader(yaml.SafeLoader):
    pass


def _no_dup(loader, node, deep=False):
    keys = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in keys:
            raise ValueError("duplicate key in YAML: %r" % (key,))
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_NoDupLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup)


def _load(text: str):
    return _NoDupLoader(text).get_single_data()


class HermesAdapter:
    def __init__(self, options: dict):
        self.home = Path(options.get("home", "~/.hermes")).expanduser()
        self.base_url = options.get("base_url", "http://192.168.8.152:11434")

    def path(self, target: str) -> Path:
        return (self.home / "config.yaml" if target == "global"
                else self.home / "profiles" / target / "config.yaml")

    def targets(self) -> list[str]:
        out = ["global"] if (self.home / "config.yaml").exists() else []
        pdir = self.home / "profiles"
        if pdir.is_dir():
            out += sorted(p.name for p in pdir.iterdir()
                          if (p / "config.yaml").exists())
        return out

    def read(self, target: str) -> Primary | None:
        p = self.path(target)
        if not p.exists():
            return None
        cfg = _load(p.read_text()) or {}
        m = cfg.get("model") or {}
        if not m.get("default"):
            return None
        return Primary(provider=m.get("provider", ""), model=m["default"],
                       context_length=int(m.get("context_length") or 0),
                       fallbacks=cfg.get("fallback_providers") or [])

    def write(self, target: str, primary: Primary, tag: str) -> None:
        p = self.path(target)
        cfg = _load(p.read_text()) or {}
        shutil.copy2(p, "%s.bak-%s-%s" % (p, tag, time.strftime("%Y%m%d%H%M")))

        # Local primary: provider custom -> Ollama. Keep the providers.custom
        # base_url and the existing cloud fallback chain intact.
        cfg["model"] = {"provider": primary.provider, "default": primary.model,
                        "context_length": primary.context_length}
        prov = cfg.setdefault("providers", {})
        prov.setdefault("custom", {})["base_url"] = self.base_url
        if primary.fallbacks:
            cfg["fallback_providers"] = primary.fallbacks
        p.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        _load(p.read_text())   # re-read or die


def get(name: str, options: dict):
    if name == "hermes":
        return HermesAdapter(options)
    raise SystemExit("unknown adapter %r (have: hermes)" % name)
