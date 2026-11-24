"""canary -- is the local model still resident and serving?

The local counterpart of model-canary. Failure modes differ from a hosted API:

  * the GPU can die (NVML driver mismatch -- the exact 2026-07-17 outage),
  * the resident model can be evicted by another consumer (Honcho/ComfyUI),
  * the model can silently stop tool-calling after an Ollama upgrade.

So it checks: is Ollama up? Is the target still pointing at a local model? Does
that model serve a tool call? After N consecutive hard failures it walks a local
ladder (smaller models first), then drops to the configured local fallback.

Unreachable is not dead: transients never count toward failover.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config as cfgmod
from . import probe as probemod
from .adapters import hermes as hermes_adapt
from .notify import log, notify


def state_file(cfg) -> Path:
    return Path(cfg["state_dir"]) / "canary.json"


def get_state(cfg) -> dict:
    try:
        return json.loads(state_file(cfg).read_text())
    except Exception:                                         # noqa: BLE001
        return {}


def put_state(cfg, st) -> None:
    state_file(cfg).write_text(json.dumps(st, indent=1, sort_keys=True))


def ollama_up(cfg) -> tuple[bool, str]:
    try:
        probemod.ollama_tags(cfg["ollama"]["base_url"], cfg["ollama"]["tags"])
        return True, "ok"
    except Exception as e:                                    # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, e)


def switch(cfg, adapter, targets, model, ctx, tag, dry):
    changed = []
    for t in targets:
        cur = adapter.read(t)
        if cur is None:
            continue
        if cur.provider != "custom":
            log("  %s: on provider %r -- left alone" % (t, cur.provider))
            continue
        if cur.model == model:
            continue
        if dry:
            log("  DRY RUN would repoint %s -> %s" % (t, model))
            changed.append(t)
            continue
        adapter.write(t, hermes_adapt.Primary(
            provider="custom", model=model, context_length=ctx,
            fallbacks=cur.fallbacks), tag)
        changed.append(t)
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="local-model-canary",
                                 description="is the local model still serving?")
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = cfgmod.load(args.config)
    c = cfg["canary"]
    base = cfg["ollama"]["base_url"]
    paths = cfg["ollama"]
    adapter = hermes_adapt.get(cfg["adapter"]["name"], cfg["adapter"])
    targets = c["targets"] or adapter.targets()
    st = get_state(cfg)

    up, up_detail = ollama_up(cfg)
    if not up:
        log("Ollama unreachable (%s) -- treated as transient, no failover" % up_detail)
        return 0
    log("Ollama up")

    inc = adapter.read(targets[0]) if targets else None
    if not inc:
        log("no readable primary -- nothing to probe")
        return 0
    if inc.provider != "custom":
        log("primary is on %r, not local -- canary only speaks for the local model"
            % inc.provider)
        return 0

    ok, kind, detail, tok_s, _ctx = probemod.probe(base, paths["chat"], inc.model)
    if ok:
        if st.pop("hard_fails", None):
            put_state(cfg, st)
            log("primary %s recovered" % inc.model)
        log("primary %s ok (%.1f tok/s)" % (inc.model, tok_s))
        return 0
    if kind == "transient":
        log("primary %s transient (%s) -- not counted" % (inc.model, detail[:120]))
        return 0

    n = int(st.get("hard_fails", 0)) + 1
    st["hard_fails"] = n
    st["last_hard"] = detail[:300]
    if not args.dry_run:
        put_state(cfg, st)
    log("hard failure #%d for %s -- %s" % (n, inc.model, detail[:160]))
    if n < c["fails_before_failover"]:
        return 0

    for rung in c["ladder"]:
        model, ctx = rung["model"], int(rung.get("context_length") or 0)
        if model == inc.model:
            continue
        rok, _k, rdetail, rtok, _c = probemod.probe(base, paths["chat"], model)
        log("  ladder %-30s %s %s" % (model, "PASS" if rok else "FAIL", rdetail[:50]))
        if not rok:
            continue
        moved = switch(cfg, adapter, targets, model, ctx, "localcanary-failover",
                       args.dry_run)
        st["hard_fails"] = 0
        if not args.dry_run:
            put_state(cfg, st)
        notify(cfg, "**Local primary failed over** `%s` -> `%s` (%.1f tok/s).%s"
                    % (inc.model, model, rtok, " (DRY RUN)" if args.dry_run else ""))
        return 0

    local = c["local"]
    if local.get("model"):
        moved = switch(cfg, adapter, targets, local["model"], 0,
                       "localcanary-fallback", args.dry_run)
        st["hard_fails"] = 0
        if not args.dry_run:
            put_state(cfg, st)
        notify(cfg, "**No ladder model served** - dropped to `%s`.%s"
                    % (local["model"], " (DRY RUN)" if args.dry_run else ""))
        return 0

    notify(cfg, "**Local primary is down and NOTHING served** - `%s`: %s"
                % (inc.model, detail[:250]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
