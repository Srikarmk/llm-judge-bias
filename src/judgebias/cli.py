"""judgebias command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

from . import config as C


def _client(args):
    from .llm import Cache, Client
    os.makedirs(os.path.dirname(C.CACHE_DB), exist_ok=True)
    return Client(Cache(C.CACHE_DB), cache_only=getattr(args, "cache_only", False))


def cmd_fetch(args):
    from .data import fetch
    for name, meta in fetch().items():
        print(f"{name}: {meta['repo']}@{meta['revision'][:10]} {meta['path']} {meta['bytes']} bytes")


def cmd_items(args):
    from collections import Counter

    from .data import write_items
    items = write_items()
    print(f"wrote {len(items)} items to data/items.jsonl:",
          dict(Counter(f"{i['task']}/{i['split']}" for i in items)))


def _record_model(model: str):
    from .llm import model_info, think_flag
    path = os.path.join(C.RESULTS, "models.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta[model] = {**model_info(model), "think": think_flag(model),
                   "options": {k: C.options(k, model) for k in C.NUM_PREDICT}}
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)


def _model_stages(args, model: str) -> list[tuple[str, int]]:
    """(stage, first N items per task) for one model under --grid, or --stage with --per-task."""
    from .runner import STAGE_ORDER
    if args.grid:
        grid = C.THINK_GRID if model in C.THINKING_ONLY else C.GRID
        return [(st, grid[st]) for st in STAGE_ORDER if st in grid]
    stages = STAGE_ORDER if args.stage == "all" else args.stage.split(",")
    for st in stages:
        if st not in STAGE_ORDER:
            raise SystemExit(f"unknown stage {st}")
    out = [(st, args.per_task) for st in STAGE_ORDER if st in stages]
    if model not in C.MAIN_MODELS:
        out = [(st, n) for st, n in out if st not in ("gen", "self")]
    return out


def cmd_run(args):
    """Run stages for every model. Each model runs its own stages in order, in its own thread, so a fast
    model does not wait for a slow one; the self-preference stage waits until every main model in the run
    has finished generating. Models in --late start once every model in --late-after has finished."""
    from .runner import run_stage
    client = _client(args)
    models = args.models.split(",")
    late = [m for m in args.late_models.split(",") if m]
    after = [m for m in args.late_after.split(",") if m] or models
    if not args.cache_only:
        for m in models + late:
            _record_model(m)
    plans = {m: _model_stages(args, m) for m in models + late}
    gen_done = {m: threading.Event() for m in models + late}
    finished = {m: threading.Event() for m in models + late}
    errors = []

    def go(m):
        try:
            for st, n in plans[m]:
                if st == "self":
                    for o in models + late:
                        if any(s == "gen" for s, _ in plans[o]):
                            gen_done[o].wait()
                rows = run_stage(st, client, m, workers=args.workers, split=args.split, per_task=n,
                                 skip_missing=args.skip_missing)
                size = f"first {n} per set" if n else "all items"
                print(f"{m} {st} ({size}): {len(rows)} rows", flush=True)
                if st == "gen":
                    gen_done[m].set()
        except Exception as e:      # report; the other models carry on
            errors.append((m, e))
            print(f"{m}: FAILED {e!r}", flush=True)
        finally:
            gen_done[m].set()
            finished[m].set()

    def go_late(m):
        for a in after:
            if a in finished:
                finished[a].wait()
        go(m)

    threads = [threading.Thread(target=go, args=(m,)) for m in models]
    threads += [threading.Thread(target=go_late, args=(m,)) for m in late]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("cache", client.stats, flush=True)
    if errors:
        raise SystemExit(f"failed: {[m for m, _ in errors]}")


def cmd_plan(args):
    """Calls per stage and model for the planned grid, and hours at given seconds per call. Before the
    generations exist, self-preference is counted as 3 pairs per problem in both orders (an upper bound:
    pairs where both solutions are wrong are dropped)."""
    from .data import load_items
    from .runner import jobs_for
    rates = json.loads(args.rates) if args.rates else {}
    models = args.models.split(",")
    for m in models:
        grid = C.THINK_GRID if m in C.THINKING_ONLY else C.GRID
        total_calls, secs = 0, 0.0
        parts = []
        for st, n in grid.items():
            k = len(jobs_for(st, args.split, n))
            if st == "self" and k == 0:
                items = load_items(args.split, ["gen"])
                k = 6 * (min(n, len(items)) if n else len(items))
            r = rates.get(m, {})
            secs += k * r.get(st, r.get("default", 0))
            total_calls += k
            parts.append(f"{st}={k}")
        print(f"{m}: {total_calls} calls ({', '.join(parts)})" + (f", {secs / 3600:.1f} h" if rates else ""))


def cmd_show(args):
    """Print the rendered prompts for one item, and the stored outputs if there are any."""
    from .data import load_items
    from .prompts import ORDERS, PAIR_CONDS, POINT_CONDS, render
    from .runner import read_rows
    it = next(i for i in load_items(None) if i["id"] == args.id)
    rows = {(r["mode"], r["cond"], r["order"]): r for r in read_rows(args.model, it["split"]) if r["id"] == it["id"]}
    calls = [("pair", c, o) for c in PAIR_CONDS for o in ORDERS] + [("point", c, "") for c in POINT_CONDS]
    if it["task"] == "gen":
        calls = [("gen", "gen", "")]
    if args.cond:
        calls = [c for c in calls if c[1] in args.cond.split(",")]
    for mode, cond, order in calls:
        rd = render(it, mode, cond, order)
        print(f"===== {mode} {cond} {order} slots={rd.slots}")
        for m in rd.messages:
            print(f"[{m['role']}]\n{m['content']}")
        r = rows.get((mode, cond, order))
        if r:
            keys = ("verdict", "pick", "correct", "score", "answer", "method")
            print(f"--- {args.model}: " + " ".join(f"{k}={r[k]}" for k in keys if k in r) + f"\n{r['output']}")
        print()


def cmd_report(args):
    from .report import write_report
    write_report(split=args.split)


def cmd_handcheck(args):
    from . import handcheck
    if args.action == "sample":
        handcheck.sample(args.n, split=args.split)
    else:
        handcheck.score()


def main(argv=None):
    p = argparse.ArgumentParser(prog="judgebias")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the pinned data files")
    sub.add_parser("items", help="build data/items.jsonl from the downloaded files")
    r = sub.add_parser("run", help="run judging stages")
    r.add_argument("--grid", action="store_true", help="the planned grid in config.GRID / config.THINK_GRID")
    r.add_argument("--stage", default="all", help="comma-separated stages, or all (ignored with --grid)")
    r.add_argument("--models", default=",".join(C.MAIN_MODELS))
    r.add_argument("--split", default="main", choices=["main", "pilot"])
    r.add_argument("--per-task", type=int, default=0, help="only the first N items of each task")
    r.add_argument("--late-models", default="", help="models to start once --late-after models finish")
    r.add_argument("--late-after", default="", help="default: all of --models")
    r.add_argument("--workers", type=int, default=1, help="concurrent calls per model (the server runs them "
                                                          "one at a time per model, so 1 is enough)")
    r.add_argument("--cache-only", action="store_true")
    r.add_argument("--skip-missing", action="store_true", help="with --cache-only, skip calls not in the cache")
    pl = sub.add_parser("plan", help="count the calls of a grid")
    pl.add_argument("--split", default="main", choices=["main", "pilot"])
    pl.add_argument("--models", default=",".join(C.MODELS))
    pl.add_argument("--rates", default="", help='JSON {model: {stage or "default": seconds per call}}')
    s = sub.add_parser("show", help="print an item's rendered prompts and stored outputs")
    s.add_argument("id")
    s.add_argument("--model", default=C.MODELS[0])
    s.add_argument("--cond", default="")
    rp = sub.add_parser("report", help="write reports/results.md and reports/figures/")
    rp.add_argument("--split", default="main", choices=["main", "pilot"])
    hc = sub.add_parser("handcheck", help="sample judge outputs for labelling, or score the labels")
    hc.add_argument("action", choices=["sample", "score"])
    hc.add_argument("--n", type=int, default=60)
    hc.add_argument("--split", default="main", choices=["main", "pilot"])
    args = p.parse_args(argv)
    fn = {"fetch": cmd_fetch, "items": cmd_items, "run": cmd_run, "plan": cmd_plan, "show": cmd_show,
          "report": cmd_report, "handcheck": cmd_handcheck}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
