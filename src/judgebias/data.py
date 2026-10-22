"""Download the pinned files and build the item set deterministically.

Four item sets, each drawn with a fixed seed, pilot items first:

- gsm8k: a GSM8K test problem, its reference solution (right) and the same solution with one planted
  arithmetic error carried through to a wrong final answer (wrong). See plant.py.
- tqa:   a TruthfulQA question, its best answer (right) and one of its listed incorrect answers (wrong).
- gen:   GSM8K test problems (disjoint from gsm8k) that each main model solves; the solutions are paired
         by author for the self-preference test.
- mtb:   MT-bench first-turn answer pairs where most human judges preferred one answer (right = the one
         humans preferred).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config as C
from .plant import PlantError, plant_any

MTB_CATEGORY = {81: "writing", 91: "roleplay", 101: "reasoning", 111: "math", 121: "coding",
                131: "extraction", 141: "stem", 151: "humanities"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_path(name: str) -> str:
    return os.path.join(C.RAW, f"{name}.parquet")


def fetch() -> dict:
    os.makedirs(C.RAW, exist_ok=True)
    out = {}
    for name, (repo, rev, path, pinned) in C.SOURCES.items():
        dest = raw_path(name)
        if not os.path.exists(dest):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
        got = sha256(dest)
        if pinned and got != pinned:
            raise RuntimeError(f"{dest}: sha256 {got} does not match pinned {pinned}")
        out[name] = {"repo": repo, "revision": rev, "path": path, "bytes": os.path.getsize(dest), "sha256": got}
    with open(os.path.join(C.DATA, "sources.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def sentence(s: str) -> str:
    s = " ".join(str(s).split())
    return s if s.endswith((".", "!", "?")) else s + "."


def mtb_category(qid: int) -> str:
    return MTB_CATEGORY[max(k for k in MTB_CATEGORY if k <= qid)]


def _perm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def _split(rank: int, task: str) -> str:
    return "pilot" if rank < C.N_PILOT[task] else "main"


def build_gsm8k(g: pd.DataFrame) -> tuple[list[dict], set[int]]:
    items, used = [], set()
    need = C.N_PILOT["gsm8k"] + C.N_MAIN["gsm8k"]
    for i in _perm(len(g), C.SEED):
        if len(items) == need:
            break
        r = g.iloc[i]
        used.add(int(i))
        try:
            p = plant_any(r.answer, r.question, C.SEED * 100_000 + int(i))
        except PlantError:
            continue
        items.append({"id": f"gsm8k-{i:04d}", "task": "gsm8k", "split": _split(len(items), "gsm8k"),
                      "question": r.question.strip(), "right": p["right"], "wrong": p["wrong"],
                      "gold": gsm8k_gold(r.answer), "wrong_final": str(p["wrong_final"]),
                      "plant": {k: p[k] for k in ("step", "n_steps", "old", "new", "changed")},
                      "source_index": int(i)})
    return items, used


def build_gen(g: pd.DataFrame, used: set[int]) -> list[dict]:
    items = []
    need = C.N_PILOT["gen"] + C.N_MAIN["gen"]
    for i in _perm(len(g), C.SEED + 1):
        if len(items) == need:
            break
        if int(i) in used:
            continue
        r = g.iloc[i]
        items.append({"id": f"gen-{i:04d}", "task": "gen", "split": _split(len(items), "gen"),
                      "question": r.question.strip(), "gold": gsm8k_gold(r.answer), "source_index": int(i)})
    return items


def tqa_usable(r) -> bool:
    best = str(r.best_answer).lower()
    if "no comment" in best or not len(r.incorrect_answers):
        return False
    return any("no comment" not in str(a).lower() and sentence(a) != sentence(r.best_answer)
               for a in r.incorrect_answers)


def build_tqa(t: pd.DataFrame) -> list[dict]:
    items = []
    need = C.N_PILOT["tqa"] + C.N_MAIN["tqa"]
    for i in _perm(len(t), C.SEED + 2):
        if len(items) == need:
            break
        r = t.iloc[i]
        if not tqa_usable(r):
            continue
        wrongs = [a for a in r.incorrect_answers if "no comment" not in str(a).lower()
                  and sentence(a) != sentence(r.best_answer)]
        w = wrongs[int(np.random.default_rng(C.SEED * 1000 + int(i)).integers(len(wrongs)))]
        items.append({"id": f"tqa-{i:03d}", "task": "tqa", "split": _split(len(items), "tqa"),
                      "question": r.question.strip(), "right": sentence(r.best_answer), "wrong": sentence(w),
                      "category": r.category, "source_index": int(i)})
    return items


def mtb_pairs(m: pd.DataFrame) -> list[dict]:
    """First-turn pairs, votes pooled over human judges; kept when one answer has more than half of all
    votes (ties count against both) and both answers are at most MTB_MAX_CHARS long."""
    votes = defaultdict(lambda: defaultdict(int))
    texts = {}
    for r in m[m.turn == 1].itertuples():
        key = (int(r.question_id), *sorted([r.model_a, r.model_b]))
        winner = {"model_a": r.model_a, "model_b": r.model_b}.get(r.winner, "tie")
        votes[key][winner] += 1
        texts[(key[0], r.model_a)] = r.conversation_a[1]["content"]
        texts[(key[0], r.model_b)] = r.conversation_b[1]["content"]
        texts[(key[0], "question")] = r.conversation_a[0]["content"]
    out = []
    for key, v in sorted(votes.items()):
        qid, x, y = key
        n = sum(v.values())
        best = max((x, y), key=lambda k: v[k])
        if v[best] * 2 <= n:
            continue
        a, b = texts[(qid, x)].strip(), texts[(qid, y)].strip()
        if max(len(a), len(b)) > C.MTB_MAX_CHARS or a == b:
            continue
        worse = y if best == x else x
        out.append({"qid": qid, "better": best, "worse": worse, "votes": dict(v), "n_votes": n,
                    "question": texts[(qid, "question")].strip(),
                    "right": texts[(qid, best)].strip(), "wrong": texts[(qid, worse)].strip()})
    return out


def build_mtb(m: pd.DataFrame) -> list[dict]:
    """One pair per question (seeded), then questions in seeded order."""
    by_q = defaultdict(list)
    for p in mtb_pairs(m):
        by_q[p["qid"]].append(p)
    qids = sorted(by_q)
    chosen = []
    for qid in qids:
        ps = by_q[qid]
        chosen.append(ps[int(np.random.default_rng(C.SEED * 10 + qid).integers(len(ps)))])
    items = []
    need = C.N_PILOT["mtb"] + C.N_MAIN["mtb"]
    for j in _perm(len(chosen), C.SEED + 3)[:need]:
        p = chosen[j]
        items.append({"id": f"mtb-{p['qid']}-{p['better']}-{p['worse']}", "task": "mtb",
                      "split": _split(len(items), "mtb"), "question": p["question"], "right": p["right"],
                      "wrong": p["wrong"], "better_model": p["better"], "worse_model": p["worse"],
                      "votes": p["votes"], "n_votes": p["n_votes"], "category": mtb_category(p["qid"]),
                      "source_index": p["qid"]})
    return items


def build_items() -> list[dict]:
    g = pd.read_parquet(raw_path("gsm8k_test")).reset_index(drop=True)
    gsm, used = build_gsm8k(g)
    gen = build_gen(g, used)
    tqa = build_tqa(pd.read_parquet(raw_path("truthfulqa_gen")).reset_index(drop=True))
    mtb = build_mtb(pd.read_parquet(raw_path("mtbench_human")).reset_index(drop=True))
    return gsm + tqa + gen + mtb


def write_items() -> list[dict]:
    items = build_items()
    with open(C.ITEMS, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items


def load_items(split: str | None = "main", tasks: list[str] | None = None) -> list[dict]:
    with open(C.ITEMS) as f:
        items = [json.loads(line) for line in f]
    return [it for it in items if (split is None or it["split"] == split) and (tasks is None or it["task"] in tasks)]
