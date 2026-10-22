"""Tables computed from results/rows/: accuracy, position bias, flips under each manipulation, pointwise
scores, self-preference and agreement with human judges."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import load_items
from .prompts import COND_STYLE
from .runner import read_rows, score_row, self_pairs
from .stats import item_ci, mean_ci, paired_ci

NONE = "<none>"
STYLE_CONDS = ["verbose_w", "confident_w", "markdown_w", "verbose_r", "confident_r", "markdown_r"]
STYLES = ["verbose", "confident", "markdown"]
PT_STYLE = {"verbose": "pt_verbose_w", "confident": "pt_confident_w", "markdown": "pt_markdown_w"}


def load_rows(split: str = "main") -> pd.DataFrame:
    rows = []
    for m in C.MODELS:
        rows += read_rows(m, split)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # re-parse every stored output with the current parser, so all rows are scored by the same rules
    units = {it["id"]: it for it in load_items(split)}
    units.update({p["id"]: p for p in self_pairs(split)})
    parsed = [score_row(units[r["id"]], r["mode"], r["cond"], r["order"], r["output"]) if r["id"] in units else None
              for r in df.to_dict("records")]
    df = df[[p is not None for p in parsed]].reset_index(drop=True)
    parsed = [p for p in parsed if p is not None]
    for k in sorted({k for p in parsed for k in p}):
        df[k] = [p.get(k) for p in parsed]
    if "self" in set(df["mode"]):
        meta = {p["id"]: p for p in self_pairs(split)}
        s = df["mode"] == "self"
        df.loc[s, "item"] = df.loc[s, "id"].map(lambda i: meta[i]["item"])
    return df


def models_in(df: pd.DataFrame, mode: str | None = None) -> list[str]:
    d = df if mode is None else df[df["mode"] == mode]
    return [m for m in C.MODELS if m in set(d.model)]


def mlabel(m: str) -> str:
    return f"{m} (thinking)" if m in C.THINKING_ONLY else m


# ---------------------------------------------------------------- pairwise, known better answer

def pair_rows(df: pd.DataFrame, tasks=None) -> pd.DataFrame:
    d = df[(df["mode"] == "pair") & df.task.isin(tasks or C.KNOWN + ["mtb"])].copy()
    d["pick_s"] = d["pick"].fillna(NONE)
    d["correct_f"] = d["correct"].fillna(False).astype(float)
    d["is_a"] = (d["verdict"] == "A").astype(float)
    d["parsed"] = d["verdict"].notna().astype(float)
    return d


def per_item(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, task, id, cond) with both orders present."""
    g = d.groupby(["model", "task", "id", "cond"])
    out = g.agg(acc=("correct_f", "mean"), p_a=("is_a", "mean"), n_orders=("order", "nunique"),
                parsed=("parsed", "min")).reset_index()
    picks = d.pivot_table(index=["model", "task", "id", "cond"], columns="order", values="pick_s", aggfunc="first")
    verd = d.assign(v=d.verdict.fillna(NONE)).pivot_table(index=["model", "task", "id", "cond"], columns="order",
                                                         values="v", aggfunc="first")
    picks = picks.reindex(columns=["RW", "WR"])
    verd = verd.reindex(columns=["RW", "WR"])
    out = out.set_index(["model", "task", "id", "cond"])
    out["consistent"] = ((picks["RW"] == picks["WR"]) & (picks["RW"] != NONE)).astype(float)
    out["always_a"] = ((verd["RW"] == "A") & (verd["WR"] == "A")).astype(float)
    out["always_b"] = ((verd["RW"] == "B") & (verd["WR"] == "B")).astype(float)
    out["acc_rw"] = d[d.order == "RW"].set_index(["model", "task", "id", "cond"]).correct_f
    out["acc_wr"] = d[d.order == "WR"].set_index(["model", "task", "id", "cond"]).correct_f
    out = out.reset_index()
    return out[out.n_orders == 2].copy()


def flip_rows(d: pd.DataFrame, cond: str) -> pd.DataFrame:
    """Per (model, task, id): share of the two orders in which the pick under `cond` differs from base."""
    key = ["model", "task", "id", "order"]
    a = d[d.cond == cond].set_index(key).pick_s
    b = d[d.cond == "base"].set_index(key).pick_s
    m = pd.concat([a.rename("x"), b.rename("b")], axis=1, join="inner").reset_index()
    m["flip"] = (m.x != m.b).astype(float)
    full = m.groupby(["model", "task", "id"]).order.transform("nunique") == 2
    return m[full].groupby(["model", "task", "id"]).flip.mean().reset_index()


def sel(d: pd.DataFrame, model: str, task: str | None) -> pd.DataFrame:
    x = d[d.model == model]
    return x if task is None else x[x.task == task]


def known_tables(df: pd.DataFrame, model: str, task: str | None) -> dict:
    """All pairwise numbers for one judge on gsm8k, tqa, or both pooled (task None)."""
    d = sel(pair_rows(df, C.KNOWN), model, task)
    if d.empty:
        return {}
    pi = per_item(d)
    base = pi[pi.cond == "base"]
    out = {"n_items": int(base.id.nunique()), "acc": {}, "cost": {}, "flip": {}, "excess": {}, "pull": {}}
    out["position"] = {
        "consistent": item_ci(base, "consistent"), "always_a": item_ci(base, "always_a"),
        "always_b": item_ci(base, "always_b"), "p_a": item_ci(base, "p_a"),
        "acc_rw": item_ci(base, "acc_rw"), "acc_wr": item_ci(base, "acc_wr"),
        "rw_minus_wr": paired_ci(base, "acc_rw", base, "acc_wr"),
    }
    out["parsed"] = float(d[d.cond == "base"].parsed.mean())
    aa_flip = flip_rows(d, "aa")
    out["floor"] = item_ci(aa_flip, "flip")
    out["floor_n_flips"] = int(round((aa_flip.flip * 2).sum()))
    for c in ["base", "aa"] + STYLE_CONDS:
        pc = pi[pi.cond == c]
        if pc.empty:
            continue
        out["acc"][c] = item_ci(pc, "acc")
        if c != "base":
            out["cost"][c] = paired_ci(pc, "acc", base[base.id.isin(set(pc.id))])
            fl = flip_rows(d, c)
            out["flip"][c] = item_ci(fl, "flip")
            if c != "aa":
                out["excess"][c] = paired_ci(fl, "flip", aa_flip[aa_flip.id.isin(set(fl.id))])
    # pull of a style: how much more often the styled answer is picked than the same answer plain.
    # Applied to the wrong answer: base accuracy minus accuracy; applied to the right one: the reverse.
    for s in STYLES:
        w, r = pi[pi.cond == f"{s}_w"], pi[pi.cond == f"{s}_r"]
        if not w.empty:
            out["pull"][f"{s}_w"] = paired_ci(base[base.id.isin(set(w.id))], "acc", w)
        if not r.empty:
            out["pull"][f"{s}_r"] = paired_ci(r, "acc", base[base.id.isin(set(r.id))])
        if not w.empty and not r.empty:
            both = r.merge(w, on=["model", "task", "id"], suffixes=("_r", "_w"))
            both["pull"] = (both.acc_r - both.acc_w) / 2
            out["pull"][s] = mean_ci(both.pull.values, both.task.values)
    return out


def consistency_by_cond(df: pd.DataFrame, model: str, task: str | None) -> dict:
    d = sel(pair_rows(df, C.KNOWN), model, task)
    pi = per_item(d)
    return {c: item_ci(pi[pi.cond == c], "consistent") for c in ["base", "aa"] + STYLE_CONDS
            if not pi[pi.cond == c].empty}


def gsm8k_by_plant_position(df: pd.DataFrame, model: str, split: str = "main") -> dict:
    """Base accuracy on GSM8K when the planted error is in the last calculation vs an earlier one."""
    items = {it["id"]: it for it in load_items(split, ["gsm8k"])}
    d = sel(pair_rows(df, ["gsm8k"]), model, "gsm8k")
    pi = per_item(d)
    base = pi[pi.cond == "base"].copy()
    if base.empty:
        return {}
    base["last"] = base.id.map(lambda i: items[i]["plant"]["step"] == items[i]["plant"]["n_steps"] - 1
                               if i in items else np.nan)
    return {"last": item_ci(base[base["last"] == True], "acc"),  # noqa: E712
            "earlier": item_ci(base[base["last"] == False], "acc")}  # noqa: E712


# ---------------------------------------------------------------- pointwise

def point_items(df: pd.DataFrame, model: str, task: str | None) -> pd.DataFrame:
    d = df[(df["mode"] == "point") & (df.model == model) & df.task.isin(C.KNOWN)]
    if task:
        d = d[d.task == task]
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index=["task", "id"], columns="cond", values="score", aggfunc="first").reset_index()


def _cmp(a, b):
    """1 if a > b, 0.5 if equal, 0 if less, nan if either is missing."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return 1.0 if a > b else 0.5 if a == b else 0.0


def point_tables(df: pd.DataFrame, model: str, task: str | None) -> dict:
    p = point_items(df, model, task)
    if p.empty or "pt_right" not in p or "pt_wrong" not in p:
        return {}
    out = {"n_items": int(len(p)), "parsed": float(p[["pt_right", "pt_wrong"]].notna().values.mean())}
    p["acc"] = [_cmp(a, b) for a, b in zip(p.pt_right, p.pt_wrong)]
    out["acc"] = mean_ci(p.acc.values, p.task.values)
    out["mean_right"] = mean_ci(p.pt_right.values, p.task.values)
    out["mean_wrong"] = mean_ci(p.pt_wrong.values, p.task.values)
    for src, aa in (("pt_right", "pt_right_aa"), ("pt_wrong", "pt_wrong_aa")):
        if aa in p:
            ch = (p[src] != p[aa]).astype(float).where(p[src].notna() & p[aa].notna())
            out[f"aa_{src}"] = mean_ci(ch.values, p.task.values)
    out["style"] = {}
    for s, c in PT_STYLE.items():
        if c not in p:
            continue
        shift = (p[c] - p.pt_wrong).values
        acc = np.array([_cmp(a, b) for a, b in zip(p.pt_right, p[c])])
        out["style"][s] = {"shift": mean_ci(shift, p.task.values), "acc": mean_ci(acc, p.task.values),
                           "cost": mean_ci(acc - p.acc.values, p.task.values)}
    return out


def pair_vs_point(df: pd.DataFrame, model: str, task: str | None) -> dict:
    """Pairwise base accuracy minus pointwise accuracy (right scored above wrong), paired by item."""
    d = sel(pair_rows(df, C.KNOWN), model, task)
    pi = per_item(d)
    base = pi[pi.cond == "base"][["task", "id", "acc"]]
    p = point_items(df, model, task)
    if base.empty or p.empty or "pt_wrong" not in p:
        return {}
    p["pt_acc"] = [_cmp(a, b) for a, b in zip(p.pt_right, p.pt_wrong)]
    m = base.merge(p[["task", "id", "pt_acc"]], on=["task", "id"])
    return mean_ci((m.acc - m.pt_acc).values, m.task.values)


# ---------------------------------------------------------------- self-preference

def self_rows(df: pd.DataFrame, split: str = "main", meta: dict | None = None) -> pd.DataFrame:
    """One row per (judge, pair) where the judge wrote one of the two answers: how often the judge picks its
    own answer, how often the third (neutral) model picks that same answer, over both orders."""
    d = df[df["mode"] == "self"].copy()
    if d.empty:
        return pd.DataFrame()
    meta = meta or {p["id"]: p for p in self_pairs(split)}
    d["pick_s"] = d["pick"].fillna(NONE)
    out = []
    for pid, g in d.groupby("id"):
        p = meta[pid]
        x, y = p["authors"]
        z = next(m for m in C.MAIN_MODELS if m not in (x, y))
        by = {m: gg for m, gg in g.groupby("model")}
        if z not in by or by[z].order.nunique() < 2:
            continue
        for j in (x, y):
            if j not in by or by[j].order.nunique() < 2:
                continue
            own = float((by[j].pick_s == j).mean())
            neutral = float((by[z].pick_s == j).mean())
            other_author = y if j == x else x
            row = {"task": "self", "id": pid, "item": p["item"], "judge": j, "other": other_author, "neutral": z,
                   "pair_type": p["pair_type"], "own": own, "neutral_pick": neutral, "diff": own - neutral,
                   "own_right": None if p["right_author"] is None else p["right_author"] == j,
                   "consistent": float(by[j].pick_s.nunique() == 1 and by[j].pick_s.iloc[0] != NONE)}
            out.append(row)
    return pd.DataFrame(out)


def self_tables(sr: pd.DataFrame) -> dict:
    if sr.empty:
        return {}
    out = {}
    for j in C.MAIN_MODELS:
        s = sr[sr.judge == j]
        if s.empty:
            continue
        r = {}
        b = s[s.pair_type == "both_right"]
        r["both_right"] = {"n_pairs": int(len(b)), "n_items": int(b.item.nunique()),
                           "own": item_ci(b, "own", unit="item"), "neutral": item_ci(b, "neutral_pick", unit="item"),
                           "diff": item_ci(b, "diff", unit="item")}
        for flag, name in ((True, "own_right"), (False, "own_wrong")):
            o = s[(s.pair_type == "one_right") & (s.own_right == flag)]
            r[name] = {"n_pairs": int(len(o)), "own": item_ci(o, "own", unit="item"),
                       "neutral": item_ci(o, "neutral_pick", unit="item"), "diff": item_ci(o, "diff", unit="item")}
        out[j] = r
    return out


def gen_accuracy(df: pd.DataFrame) -> dict:
    g = df[df["mode"] == "gen"].copy()
    out = {}
    for m in C.MAIN_MODELS:
        x = g[g.model == m]
        if not x.empty:
            x = x.assign(c=x.correct.astype(float))
            out[m] = {**item_ci(x, "c"), "truncated": float(x.truncated.mean()),
                      "chars": float(x.output.str.len().median())}
    return out


# ---------------------------------------------------------------- MT-bench human agreement

def mtb_tables(df: pd.DataFrame, model: str, split: str = "main") -> dict:
    d = sel(pair_rows(df, ["mtb"]), model, "mtb")
    if d.empty:
        return {}
    items = {it["id"]: it for it in load_items(split, ["mtb"])}
    pi = per_item(d)
    base = pi[pi.cond == "base"].copy()
    base["multi"] = base.id.map(lambda i: items[i]["n_votes"] >= 2)
    base["right_longer"] = base.id.map(lambda i: len(items[i]["right"]) > len(items[i]["wrong"]))
    # how often the judge picks the longer answer, averaged over orders
    base["pick_longer"] = np.where(base.right_longer, base.acc, 1 - base.acc)
    return {"n_items": int(len(base)), "agree": item_ci(base, "acc"),
            "agree_multi": item_ci(base[base.multi], "acc"), "n_multi": int(base.multi.sum()),
            "consistent": item_ci(base, "consistent"), "p_a": item_ci(base, "p_a"),
            "pick_longer": item_ci(base, "pick_longer"),
            "human_longer": float(base.right_longer.mean())}


# ---------------------------------------------------------------- run facts

def run_facts(df: pd.DataFrame) -> list[dict]:
    out = []
    for m in models_in(df):
        d = df[df.model == m]
        for mode in ["pair", "self", "point", "gen"]:
            x = d[d["mode"] == mode]
            if x.empty:
                continue
            parsed = (x.verdict.notna() if mode in ("pair", "self") else x.score.notna() if mode == "point"
                      else x.answer.notna())
            out.append({"model": m, "mode": mode, "calls": int(len(x)), "seconds": float(x.seconds.median()),
                        "output_tokens": float(x.output_tokens.mean()), "truncated": float(x.truncated.mean()),
                        "parsed": float(parsed.mean()),
                        "methods": x.method.value_counts(normalize=True).to_dict()})
    return out


def style_of(cond: str) -> tuple[str, str]:
    return COND_STYLE.get(cond, ("", ""))
