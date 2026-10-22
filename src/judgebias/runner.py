"""Send every judging call of a stage to a model, parse it, and append one row per call.

The run is split into stages. Each model runs them in this order, so that a cut leaves the most useful
part done:

  gen      each main model solves the `gen` problems              (answers for self-preference)
  core     pairwise, base and A/A repeat, both orders            (accuracy, position bias, noise floor)
  style_w  pairwise, each manipulation applied to the wrong answer, both orders   (accuracy cost)
  self     pairwise, pairs of solutions by two different main models, both orders
  style_r  pairwise, each manipulation applied to the right answer, both orders   (control)
  mtb      pairwise, MT-bench pairs with a human majority, base, both orders
  point    single-answer scoring: right, wrong, A/A repeats, the three manipulated wrong answers

Rows are appended as they finish, so an interrupted stage keeps what it did; every call is also in the
cache, so re-running a stage only sends the calls that are missing.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from . import config as C
from .data import load_items
from .llm import CacheMiss, Client
from .parse import canon_number, parse_gsm8k, parse_score, parse_verdict
from .prompts import ORDERS, PAIR_CONDS, POINT_CONDS, SELF_ORDERS, render

STAGES = {
    "gen": {"mode": "gen", "tasks": ["gen"], "conds": ["gen"]},
    "core": {"mode": "pair", "tasks": C.KNOWN, "conds": ["base", "aa"]},
    "style_w": {"mode": "pair", "tasks": C.KNOWN, "conds": ["verbose_w", "confident_w", "markdown_w"]},
    "self": {"mode": "self", "tasks": ["gen"], "conds": ["self"]},
    "style_r": {"mode": "pair", "tasks": C.KNOWN, "conds": ["verbose_r", "confident_r", "markdown_r"]},
    "mtb": {"mode": "pair", "tasks": ["mtb"], "conds": ["base"]},
    "point": {"mode": "point", "tasks": C.KNOWN, "conds": POINT_CONDS},
}
STAGE_ORDER = list(STAGES)
assert STAGE_ORDER == list(C.GRID)
assert set(PAIR_CONDS) == set(STAGES["core"]["conds"] + STAGES["style_w"]["conds"] + STAGES["style_r"]["conds"])


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def rows_path(model: str, split: str = "main") -> str:
    name = slug(model) + ("" if split == "main" else f"_{split}")
    return os.path.join(C.RESULTS, "rows", f"{name}.jsonl")


def read_rows(model: str, split: str = "main") -> list[dict]:
    """Rows of one model, the last row kept for each call key."""
    path = rows_path(model, split)
    if not os.path.exists(path):
        return []
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["id"], r["mode"], r["cond"], r["order"])] = r
    return list(out.values())


class Progress:
    def __init__(self, total: int, label: str, every: int = 50):
        self.total, self.label, self.every = total, label, every
        self.done, self.t0, self.lock = 0, time.time(), threading.Lock()

    def tick(self, client: Client):
        with self.lock:
            self.done += 1
            if self.done % self.every == 0 or self.done == self.total:
                el = time.time() - self.t0
                print(f"[{self.label}] {self.done}/{self.total} new_calls={client.stats['misses']} "
                      f"hits={client.stats['hits']} elapsed={el:.0f}s", flush=True)


def first_n(items: list[dict], per_task: int) -> list[dict]:
    if not per_task:
        return items
    out = []
    for t in C.TASKS:
        out += [it for it in items if it["task"] == t][:per_task]
    return out


def self_pairs(split: str = "main", per_task: int = 0) -> list[dict]:
    """Pairs of solutions to the same `gen` problem by two different main models.

    Kept when both outputs ended normally (not cut by the length cap) and at least one is right:
    `both_right` pairs reach the same correct number, so the judge has no correctness signal and any
    preference is about the writing; `one_right` pairs have a known better answer."""
    gens = {m: {r["id"]: r for r in read_rows(m, split) if r["mode"] == "gen"} for m in C.MAIN_MODELS}
    pairs = []
    for it in first_n(load_items(split, ["gen"]), per_task):
        outs = {m: gens[m].get(it["id"]) for m in C.MAIN_MODELS}
        if any(o is None or o["truncated"] for o in outs.values()):
            continue
        for x, y in combinations(C.MAIN_MODELS, 2):
            cx, cy = outs[x]["correct"], outs[y]["correct"]
            if not (cx or cy):
                continue
            kind = "both_right" if cx and cy else "one_right"
            pairs.append({"id": f"{it['id']}|{x}|{y}", "task": "self", "split": split, "item": it["id"],
                          "question": it["question"], "authors": [x, y],
                          "texts": {x: outs[x]["output"], y: outs[y]["output"]},
                          "right_author": None if kind == "both_right" else (x if cx else y),
                          "pair_type": kind})
    return pairs


def jobs_for(stage: str, split: str = "main", per_task: int = 0) -> list[tuple]:
    """(item, mode, cond, order) for one stage, item by item; the calls of an item are in a seeded
    shuffled order, so a base prompt and its A/A repeat are not sent back to back."""
    st = STAGES[stage]
    rng = random.Random(f"{C.SEED}-{stage}")
    if st["mode"] == "self":
        units = self_pairs(split, per_task)
    else:
        units = first_n(load_items(split, st["tasks"]), per_task)
    jobs = []
    for it in units:
        if st["mode"] == "gen":
            calls = [("gen", "gen", "")]
        elif st["mode"] == "point":
            calls = [("point", c, "") for c in st["conds"]]
        elif st["mode"] == "self":
            calls = [("self", "self", o) for o in SELF_ORDERS]
        else:
            calls = [("pair", c, o) for c in st["conds"] for o in ORDERS]
        rng.shuffle(calls)
        jobs += [(it, *c) for c in calls]
    return jobs


def run_one(client: Client, model: str, it: dict, mode: str, cond: str, order: str) -> dict:
    rd = render(it, mode, cond, order)
    out = client.chat(model, rd.messages, C.options(rd.kind, model), tag=rd.tag)
    row = {"model": model, "id": it["id"], "task": it["task"], "split": it["split"], "mode": mode, "cond": cond,
           "order": order, "output": out["text"], "truncated": out.get("done_reason") == "length",
           "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"],
           "seconds": out["seconds"], "wall_seconds": out.get("wall_seconds"), "cached": out["cached"],
           "thinking_chars": len(out.get("thinking") or "")}
    row.update(score_row(it, mode, cond, order, out["text"]))
    return row


def score_row(it: dict, mode: str, cond: str, order: str, text: str) -> dict:
    """The parsed fields of one output; also used to re-parse stored rows with the current parser."""
    if mode == "gen":
        a, how = parse_gsm8k(text)
        return {"answer": a, "correct": a is not None and a == canon_number(it["gold"]), "method": how}
    rd = render(it, mode, cond, order)
    if mode == "point":
        s, how = parse_score(text)
        return {"score": s, "target": rd.slots["S"], "method": how}
    v, how = parse_verdict(text)
    pick = rd.slots.get(v) if v else None
    out = {"verdict": v, "pick": pick, "method": how, "slot_a": rd.slots["A"]}
    if mode == "self":
        out["pair_type"] = it["pair_type"]
        out["correct"] = None if it["right_author"] is None else pick == it["right_author"]
    else:
        out["correct"] = pick == "right"
    return out


def run_stage(stage: str, client: Client, model: str, workers: int = 4, split: str = "main", per_task: int = 0,
              skip_missing: bool = False) -> list[dict]:
    jobs = jobs_for(stage, split, per_task)
    if not jobs:
        print(f"[{model} {stage}] nothing to run", flush=True)
        return []
    prog = Progress(len(jobs), f"{model} {stage}")
    path = rows_path(model, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = threading.Lock()

    def work(job):
        try:
            row = run_one(client, model, *job)
        except CacheMiss:
            if not skip_missing:
                raise
            row = None
        if row is not None:
            with lock, open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
        prog.tick(client)
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(work, jobs) if r is not None]
    return rows
