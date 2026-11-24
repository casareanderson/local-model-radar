"""radar -- find local models that fit the GPU and really tool-call.

The local version of model-radar's radar. The discovery source is the Ollama
library, but the gate is hardware fit, not price:

  * a candidate must fit under the VRAM budget to be considered (CPU-only is
    never adopted -- it is slower than whatever is already resident),
  * THREE consecutive clean tool-call probes, spaced,
  * tok/s above the floor AND within adopt_speed_factor of the incumbent
    measured in the same run,
  * context at least the incumbent's,
  * a cooldown.

Deterministic: it gathers facts (sizes, context, tok/s, tool calls) and acts on
them. It never asks a model its opinion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config as cfgmod
from . import discover
from . import probe as probemod
from .adapters import hermes as hermes_adapt
from .notify import log, notify


def state_file(cfg) -> Path:
    return Path(cfg["state_dir"]) / "radar.json"


def get_state(cfg) -> dict:
    try:
        return json.loads(state_file(cfg).read_text())
    except Exception:                                         # noqa: BLE001
        return {"seen": {}}


def put_state(cfg, st) -> None:
    state_file(cfg).write_text(json.dumps(st, indent=1, sort_keys=True))


def show_and_probe(base, paths, model):
    """Pull if needed, then return (show_dict, probe_result)."""
    show = {}
    try:
        show = probemod.ollama_show(base, paths["show"], model)
    except Exception:                                         # noqa: BLE001
        # Not pulled yet -- pull it, then re-show.
        try:
            probemod.ollama_pull(base, paths["pull"], model)
            show = probemod.ollama_show(base, paths["show"], model)
        except Exception as e:                                # noqa: BLE001
            return None, (None, "transient", "pull/show failed: %s" % type(e).__name__, 0.0, 0)
    ok, kind, note, tok_s, _ = probemod.probe(base, paths["chat"], model)
    return show, (ok, kind, note, tok_s, probemod.model_context(show))


def confirm(cfg, base, paths, model, n):
    worst = 0.0
    for i in range(n):
        if i:
            time.sleep(cfg["radar"]["adopt_confirm_gap_s"])
        ok, _kind, note, tok_s, _ctx = probemod.probe(base, paths["chat"], model)
        log("    confirm %d/%d %-30s %s %stok/s %s"
            % (i + 1, n, model, "BUSY" if ok is None else ("PASS" if ok else "FAIL"),
               tok_s, note[:50]))
        if not ok:
            return False, worst, ("busy/rate-limited" if ok is None else note[:80])
        worst = max(worst, tok_s)
    return True, worst, "ok"


def try_adopt(cfg, adapter, hits, st, dry):
    r = cfg["radar"]
    targets = r["adopt_targets"]
    if not targets:
        return None, [], "no adopt_targets configured"

    last = st.get("last_adopt_ts") or 0
    if last and (time.time() - last) / 86400.0 < r["adopt_cooldown_days"]:
        return None, [], "cooldown"

    inc = adapter.read(targets[0])
    if not inc or inc.provider != "custom":
        return None, [], ("%s is not on a local/custom primary -- not adopting "
                          "over a cloud model" % targets[0])

    inc_tok_s = 0.0
    iok, _k, _n, inc_tok_s, _c = probemod.probe(
        cfg["ollama"]["base_url"], cfg["ollama"]["chat"], inc.model)
    if not iok:
        return None, [], ("incumbent %s would not probe cleanly -- no honest "
                          "baseline" % inc.model)
    log("  incumbent %s baseline %.1f tok/s" % (inc.model, inc_tok_s))
    floor = max(r["adopt_tokens_per_sec_floor"], inc_tok_s * r["adopt_speed_factor"])

    for h in sorted(hits, key=lambda x: -(x["tok_s"] or 0)):
        if (h["ctx"] or 0) < inc.context_length:
            continue
        if h["tok_s"] < floor:
            continue
        ok, worst, note = confirm(cfg, cfg["ollama"]["base_url"],
                                  cfg["ollama"], h["model"], r["adopt_confirms"])
        if not ok:
            continue
        if worst < floor:
            continue

        done = []
        for t in targets:
            cur = adapter.read(t)
            if cur is None:
                continue
            fbs = [e for e in (cur.fallbacks or [])
                   if e.get("model") != h["model"]]
            new = hermes_adapt.Primary(
                provider="custom", model=h["model"],
                context_length=h["ctx"] or 0, fallbacks=fbs)
            if dry:
                log("  DRY RUN would repoint %s -> %s" % (t, h["model"]))
                done.append(t)
                continue
            try:
                adapter.write(t, new, "localradar-adopt")
                done.append(t)
            except Exception as e:                            # noqa: BLE001
                log("  %s: WRITE FAILED (%s) -- stopping" % (t, e))
                break
        if not done:
            return None, [], "no target repointed"
        if not dry:
            st["last_adopt_ts"] = time.time()
            st.setdefault("adopted", []).append(
                {"model": h["model"], "ts": time.time(), "targets": done,
                 "displaced": inc.model, "tok_s": h["tok_s"], "ctx": h["ctx"]})
        return h, done, None
    return None, [], "no candidate cleared the gate"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="local-model-radar",
                                 description="find local models that fit the GPU and tool-call")
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-adopt", action="store_true")
    ap.add_argument("--include", nargs="*", default=[],
                    help="extra model tags to consider")
    ap.add_argument("--library", action="store_true",
                    help="also scrape ollama.com/library for candidates")
    args = ap.parse_args(argv)

    cfg = cfgmod.load(args.config)
    base = cfg["ollama"]["base_url"]
    paths = cfg["ollama"]
    adapter = hermes_adapt.get(cfg["adapter"]["name"], cfg["adapter"])
    st = get_state(cfg)
    seen = st.setdefault("seen", {})

    names = discover.merge(cfg["radar"]["blocklist"],
                           cfg["radar"]["candidates"], args.include)
    if args.library:
        names += discover.library_names()
    names = sorted(set(names))
    log("%d candidate(s) to consider" % len(names))

    fresh = [n for n in names if n not in seen]
    log("%d new candidate(s) since last run" % len(fresh))

    try:
        tags = probemod.tags_size_map(
            probemod.ollama_tags(base, paths["tags"]))
    except Exception:                                         # noqa: BLE001
        tags = {}

    hits = []
    for n in fresh[:cfg["radar"]["max_probes_per_run"]]:
        show, pr = show_and_probe(base, paths, n)
        ok, kind, note, tok_s, ctx = pr
        fit = probemod.fits(probemod.model_size(show, tags),
                            cfg["hardware"]["vram_budget_mib"]) if show else "unknown"
        if ok is None:
            log("  probe %-30s BUSY fit=%-8s %s (left unseen)" % (n, fit, note[:50]))
            continue
        seen[n] = {"ts": time.time(), "ok": ok, "note": note[:120],
                   "tok_s": tok_s, "ctx": ctx, "fit": fit}
        log("  probe %-30s %s fit=%-8s %stok/s ctx=%d %s"
            % (n, "PASS" if ok else "FAIL", fit, tok_s, ctx, note[:50]))
        if ok and fit == "vram":
            hits.append({"model": n, "tok_s": tok_s, "ctx": ctx, "fit": fit})

    if not args.dry_run:
        put_state(cfg, st)

    if not hits:
        log("nothing fits the GPU and tool-calls")
        return 0

    if args.no_adopt or not cfg["radar"]["adopt"]:
        for h in hits:
            notify(cfg, "**Local model fits + tool-calls**: `%s` (%.1f tok/s, "
                        "%dk ctx). Not adopted (adoption disabled)."
                        % (h["model"], h["tok_s"], (h["ctx"] or 0) // 1000))
        return 0

    won, tgts, why = try_adopt(cfg, adapter, hits, st, args.dry_run)
    if not args.dry_run:
        put_state(cfg, st)
    if won:
        notify(cfg, "**Local model ADOPTED** `%s` -> `%s` on %s (%.1f tok/s, "
                    "%dk ctx).%s"
                    % (won["model"], won["model"], ", ".join(tgts), won["tok_s"],
                       (won["ctx"] or 0) // 1000,
                       " (DRY RUN)" if args.dry_run else ""))
    else:
        notify(cfg, "**Local model fits but not adopted**: %s" % why)
    return 0


if __name__ == "__main__":
    sys.exit(main())
