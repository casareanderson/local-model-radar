"""tidy -- report (and optionally remove) local models that are orphaned or stale.

The garbage-collector counterpart of the radar. Discovery/probe/adopt ask "is
there something better?"; tidy asks "what is on disk that nobody uses any more?"

Classification is deterministic, read from facts only:

  * in-use   -- referenced by a Hermes profile's primary or fallback chain, or
                by the local-model-radar canary ladder/local fallback.
  * orphan   -- on disk, not referenced anywhere above.
  * dup      -- same digest as another tag (a near-duplicate alias sharing one
                blob). The tag is removable; the blob is kept while one name
                still references it.

tidy never deletes without --rm. With --rm it removes ONLY orphan/dup tags,
prints each name it deletes, and skips anything in-use. It does not judge
quality -- that is the radar's job. It only answers "is this still referenced?"
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import config as cfgmod
from . import probe as probemod
from .adapters import hermes as hermes_adapt
from .notify import log, notify


def referenced_models(adapter, cfg) -> set[str]:
    """Every model name any Hermes profile (or this tool's config) points at."""
    names: set[str] = set()
    for t in adapter.targets():
        p = adapter.read(t)
        if not p:
            continue
        names.add(p.model)
        for fb in p.fallbacks or []:
            if fb.get("model"):
                names.add(fb["model"])
    for rung in cfg["canary"].get("ladder", []):
        if rung.get("model"):
            names.add(rung["model"])
    if cfg["canary"].get("local", {}).get("model"):
        names.add(cfg["canary"]["local"]["model"])
    return names


def classify(tags: list[dict], referenced: set[str]) -> dict:
    """name -> 'in-use' | 'dup' | 'orphan'."""
    by_digest: dict[str, list[str]] = {}
    sizes = {}
    for m in tags:
        name = m.get("name", "")
        sizes[name] = m.get("size", 0)
        by_digest.setdefault(m.get("digest", ""), []).append(name)

    out = {}
    for m in tags:
        name = m.get("name", "")
        if name in referenced:
            out[name] = "in-use"
            continue
        # A duplicate alias: more than one tag shares this digest, and at least
        # one of those tags is referenced -- this one is a removable alias.
        peers = by_digest.get(m.get("digest", ""), [])
        if len(peers) > 1 and any(p in referenced for p in peers):
            out[name] = "dup"
        else:
            out[name] = "orphan"
    return out, sizes


def delete_model(base_url: str, model: str) -> bool:
    try:
        probemod._post_json(base_url + "/api/delete", {"name": model}, timeout=120)
        return True
    except Exception as e:                                    # noqa: BLE001
        log("    delete %s FAILED: %s" % (model, type(e).__name__))
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="local-model-tidy",
        description="report (and with --rm, remove) orphaned/stale local models")
    ap.add_argument("--config")
    ap.add_argument("--rm", action="store_true",
                    help="actually delete orphan/dup tags (never in-use ones)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, delete nothing (default behaviour)")
    args = ap.parse_args(argv)

    cfg = cfgmod.load(args.config)
    adapter = hermes_adapt.get(cfg["adapter"]["name"], cfg["adapter"])
    base = cfg["ollama"]["base_url"]

    try:
        tags = probemod.ollama_tags(base, cfg["ollama"]["tags"])
    except Exception as e:                                    # noqa: BLE001
        log("Ollama unreachable: %s" % type(e).__name__)
        return 1

    referenced = referenced_models(adapter, cfg)
    kinds, sizes = classify(tags, referenced)

    total = 0
    orphans, dups = [], []
    for m in tags:
        name = m.get("name", "")
        gb = sizes.get(name, 0) / 1e9
        kind = kinds.get(name, "?")
        log("  %-28s %-7s %6.1f GB" % (name, kind, gb))
        if kind == "orphan":
            orphans.append(name)
            total += sizes.get(name, 0)
        elif kind == "dup":
            dups.append(name)

    log("%d model(s), %d orphan, %d duplicate alias" % (len(tags), len(orphans), len(dups)))
    reclaimable = total / 1e9
    log("reclaimable if removed: %.1f GB" % reclaimable)

    if not (args.rm or args.dry_run):
        log("nothing deleted (pass --rm to delete orphan/dup tags)")
        return 0

    if args.rm:
        for name in orphans + dups:
            log("  deleting %s ..." % name)
            delete_model(base, name)
        notify(cfg, "**Local model tidy** removed %d orphan + %d duplicate tag(s) "
                    "from `%s` (~%.1f GB)." % (len(orphans), len(dups), base, reclaimable))
    else:
        notify(cfg, "**Local model tidy (dry-run)** %d orphan + %d duplicate "
                    "tag(s) on `%s` (~%.1f GB reclaimable). No deletion."
                    % (len(orphans), len(dups), base, reclaimable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
