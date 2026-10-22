"""Check of the verdict and score parsers against a hand-labelled sample.

`sample` writes handcheck/sample.csv: a seeded random sample of main-run judge outputs, stratified by
mode (pairwise, self-preference pairs, single-answer scores), plus a sample of outputs where the parser
needed a rule other than the requested `Verdict:` or `Score:` line, where mistakes are likelier. It holds
the raw output but not the parser's reading. The `label` column is left empty for a human labeller
(Srikar) to fill: for a pairwise output the slot the judge commits to (`A` or `B`), for a scored output
the score (1 to 10), or `none` if the output commits to nothing. Saved as handcheck/labels.csv, `score`
compares the labels with the parser and writes handcheck/agreement.json.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import config as C
from .analysis import load_rows

COLS = ["id", "subset", "model", "task", "mode", "cond", "order", "item", "output"]


def sample(n: int = 60, n_fallback: int = 20, seed: int = 11, split: str = "main") -> None:
    os.makedirs(C.HANDCHECK, exist_ok=True)
    df = load_rows(split)
    pre = "" if split == "main" else f"{split}_"
    df = df[df["mode"].isin(["pair", "self", "point"])].reset_index(drop=True)
    modes = ["pair", "self", "point"]
    share = {"pair": 0.5, "self": 0.25, "point": 0.25}
    parts = []
    for m in modes:
        d = df[df["mode"] == m]
        k = min(len(d), int(round(n * share[m])))
        if k:
            parts.append(d.sample(n=k, random_state=seed))
    rand = pd.concat(parts).sample(frac=1, random_state=seed).assign(subset="random")
    rest = df.drop(rand.index)
    fb = rest[~rest.method.isin(["cue"])]
    fb = fb.sample(n=min(n_fallback, len(fb)), random_state=seed).assign(subset="fallback rule")
    s = pd.concat([rand, fb]).reset_index(drop=True)
    s = s.drop(columns=[c for c in ("item",) if c in s]).rename(columns={"id": "item"})
    s["id"] = range(1, len(s) + 1)
    s[COLS].assign(label="").to_csv(os.path.join(C.HANDCHECK, f"{pre}sample.csv"), index=False)
    # the parser's reading, kept apart so the labeller does not see it
    s[["id", "verdict", "score"]].to_csv(os.path.join(C.HANDCHECK, f"{pre}parser_key.csv"), index=False)
    print(f"wrote {len(s)} rows to handcheck/{pre}sample.csv ({len(rand)} random, {len(fb)} fallback); labels empty")


def _norm(mode: str, x) -> str:
    x = "" if pd.isna(x) else str(x).strip()
    if x.lower() in ("", "none", "nan"):
        return "none"
    if mode == "point":
        try:
            return str(int(float(x)))
        except ValueError:
            return x
    return x.upper()


def score() -> dict:
    path = os.path.join(C.HANDCHECK, "labels.csv")
    if not os.path.exists(path):
        raise SystemExit("handcheck/labels.csv not found: copy sample.csv to labels.csv and fill the label column")
    lab = pd.read_csv(path, keep_default_na=False)
    if (lab.label.astype(str).str.strip() == "").all():
        raise SystemExit("no labels filled in yet")
    key = pd.read_csv(os.path.join(C.HANDCHECK, "parser_key.csv"), keep_default_na=False)
    lab = lab.merge(key, on="id")
    lab["parser"] = [_norm(r.mode, r.score if r.mode == "point" else r.verdict) for r in lab.itertuples()]
    lab["label_n"] = [_norm(r.mode, r.label) for r in lab.itertuples()]
    lab = lab[lab.label.astype(str).str.strip() != ""]
    lab["agree"] = lab.parser == lab.label_n
    out = {"n": int(len(lab)), "agree": int(lab.agree.sum()), "by_subset": {}}
    for sub, g in lab.groupby("subset", sort=False):
        out["by_subset"][sub] = {"n": int(len(g)), "agree": int(g.agree.sum())}
    out["disagreements"] = [{"id": int(r.id), "subset": r.subset, "mode": r.mode, "label": r.label,
                             "parser": r.parser, "output": r.output[-300:]} for r in lab[~lab.agree].itertuples()]
    parts = [f"{sub}: {v['agree']}/{v['n']}" for sub, v in out["by_subset"].items()]
    out["summary"] = ("A labelling pass by Srikar read each sampled output without seeing the parser's reading and "
                      "wrote down the verdict or score it commits to. Agreement with the parser: " + "; ".join(parts)
                      + ".")
    with open(os.path.join(C.HANDCHECK, "agreement.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out["summary"])
    return out
