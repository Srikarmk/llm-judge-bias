"""reports/results.md and reports/figures/, generated from results/ and the item file only."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis as A  # noqa: E402
from . import config as C  # noqa: E402
from .data import load_items  # noqa: E402
from .hypotheses import hypothesis_lines  # noqa: E402
from .prompts import render  # noqa: E402

MODEL_COLOR = {"qwen3:8b": "#2a78d6", "llama3.1:8b": "#eb6834", "gemma3:12b": "#1baf7a", "qwen3:30b-a3b": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
TASK_LABEL = {"gsm8k": "GSM8K planted error", "tqa": "TruthfulQA", None: "both tasks", "mtb": "MT-bench (human)"}
COND_LABEL = {"base": "base", "aa": "A/A repeat", "verbose_w": "verbose wrong answer",
              "confident_w": "confident wrong answer", "markdown_w": "markdown wrong answer",
              "verbose_r": "verbose right answer", "confident_r": "confident right answer",
              "markdown_r": "markdown right answer"}
mlabel = A.mlabel
FIGDIR = {"path": C.FIGURES, "link": "figures"}


def pct(x, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def ci(r: dict, d: int = 1) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{pct(r['mean'], d)} [{100 * r['lo']:.{d}f}, {100 * r['hi']:.{d}f}]"


def pp(r: dict, d: int = 1) -> str:
    if not r or not r.get("n"):
        return "n/a"
    star = " *" if r["lo"] > 0 or r["hi"] < 0 else ""
    return f"{100 * r['mean']:+.{d}f} pp [{100 * r['lo']:+.{d}f}, {100 * r['hi']:+.{d}f}]{star}"


def num(r: dict, d: int = 2) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{r['mean']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}]"


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]


def style_axes(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(FIGDIR["path"], exist_ok=True)
    fig.savefig(os.path.join(FIGDIR["path"], name), dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figures

def fig_accuracy(kt: dict, models: list[str]):
    conds = ["base", "aa", "verbose_w", "confident_w", "markdown_w", "verbose_r", "confident_r", "markdown_r"]
    tasks = ["gsm8k", "tqa"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    n = len(models)
    for ax, t in zip(axes, tasks):
        style_axes(ax)
        for j, m in enumerate(models):
            r = kt.get((m, t), {}).get("acc", {})
            ys, xs, lo, hi = [], [], [], []
            for i, c in enumerate(conds):
                if c in r and r[c]["n"]:
                    ys.append(i + (j - (n - 1) / 2) * 0.17)
                    xs.append(100 * r[c]["mean"])
                    lo.append(100 * (r[c]["mean"] - r[c]["lo"]))
                    hi.append(100 * (r[c]["hi"] - r[c]["mean"]))
            ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", color=MODEL_COLOR[m], markersize=5, elinewidth=1.3,
                        capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.0, label=mlabel(m))
        ax.axvline(50, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_yticks(range(len(conds)), [COND_LABEL[c] for c in conds])
        ax.set_xlim(0, 100)
        ax.set_title(TASK_LABEL[t], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("judge picks the right answer (% of items, both orders, 95% CI)", fontsize=8, color=INK2)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, frameon=False, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Pairwise judge accuracy under each manipulation (dashed: chance)", fontsize=10, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "accuracy_by_condition.png")


def fig_position(kt: dict, models: list[str]):
    cells = [(m, t) for m in models for t in ["gsm8k", "tqa"] if kt.get((m, t))]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(cells) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for i, (m, t) in enumerate(cells):
        p = kt[(m, t)]["position"]
        c, a, b = 100 * p["consistent"]["mean"], 100 * p["always_a"]["mean"], 100 * p["always_b"]["mean"]
        rest = 100 - c - a - b
        ax.barh(i, c, color=MODEL_COLOR[m], height=0.62)
        ax.barh(i, a, left=c, color="#8d8a83", height=0.62)
        ax.barh(i, b, left=c + a, color="#c9c6be", height=0.62)
        ax.barh(i, rest, left=c + a + b, color=GRID, height=0.62)
    ax.set_yticks(range(len(cells)), [f"{mlabel(m)}, {TASK_LABEL[t]}" for m, t in cells])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of items (base prompt, both orders)", fontsize=8, color=INK2)
    ax.set_title("Same answer picked in both orders (colour), slot A both times (dark grey), slot B both times "
                 "(light grey)", fontsize=8.5, color=INK, loc="left")
    save(fig, "position.png")


def fig_pull(kt: dict, models: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    n = len(models)
    for ax, t in zip(axes, ["gsm8k", "tqa"]):
        style_axes(ax)
        for j, m in enumerate(models):
            pl = kt.get((m, t), {}).get("pull", {})
            for i, s in enumerate(A.STYLES):
                for key, mk in ((f"{s}_w", "o"), (f"{s}_r", "s")):
                    r = pl.get(key)
                    if not r or not r["n"]:
                        continue
                    y = i + (j - (n - 1) / 2) * 0.2 + (0.05 if mk == "s" else -0.05)
                    ax.errorbar(100 * r["mean"], y, xerr=[[100 * (r["mean"] - r["lo"])], [100 * (r["hi"] - r["mean"])]],
                                fmt=mk, color=MODEL_COLOR[m], markersize=5, elinewidth=1.2, capsize=0,
                                markerfacecolor=MODEL_COLOR[m] if mk == "o" else SURFACE, markeredgewidth=1.2,
                                label=mlabel(m) if (i == 0 and mk == "o") else None)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(A.STYLES)), A.STYLES)
        ax.set_title(TASK_LABEL[t], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("change in how often the styled answer is picked (pp)", fontsize=8, color=INK2)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, frameon=False, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Pull of each style: filled = applied to the wrong answer, open = applied to the right answer",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "style_pull.png")


def fig_self(st: dict):
    js = [j for j in C.MAIN_MODELS if j in st]
    if not js:
        return
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for i, j in enumerate(js):
        r = st[j]["both_right"]
        for key, mk, off in (("own", "o", -0.12), ("neutral", "s", 0.12)):
            x = r[key]
            if not x["n"]:
                continue
            ax.errorbar(100 * x["mean"], i + off, xerr=[[100 * (x["mean"] - x["lo"])], [100 * (x["hi"] - x["mean"])]],
                        fmt=mk, color=MODEL_COLOR[j], markersize=6, elinewidth=1.2, capsize=0,
                        markerfacecolor=MODEL_COLOR[j] if mk == "o" else SURFACE, markeredgewidth=1.4)
    ax.axvline(50, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_yticks(range(len(js)), [f"{j}'s solutions" for j in js])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of pairs in which that solution is picked (both solutions correct)", fontsize=8, color=INK2)
    ax.set_title("Self-preference: filled = picked by its own author, open = picked by the neutral third model",
                 fontsize=9, color=INK, loc="left")
    save(fig, "self_preference.png")


# ---------------------------------------------------------------- report

def write_report(split: str = "main") -> None:
    df = A.load_rows(split)
    if df.empty:
        raise SystemExit("no results yet")
    models = A.models_in(df, "pair")
    main = [m for m in models if m not in C.THINKING_ONLY]
    kt = {(m, t): A.known_tables(df, m, t) for m in models for t in ["gsm8k", "tqa", None]}
    pt = {(m, t): A.point_tables(df, m, t) for m in models for t in ["gsm8k", "tqa", None]}
    pvp = {(m, t): A.pair_vs_point(df, m, t) for m in models for t in ["gsm8k", "tqa", None]}
    sr = A.self_rows(df, split)
    st = A.self_tables(sr)
    mt = {m: A.mtb_tables(df, m, split) for m in models}
    facts = A.run_facts(df)
    gacc = A.gen_accuracy(df)
    meta_path = os.path.join(C.RESULTS, "models.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if split == "pilot":
        FIGDIR.update(path=os.path.join(C.REPORTS, "pilot_figures"), link="pilot_figures")
    else:
        FIGDIR.update(path=C.FIGURES, link="figures")
    fl = FIGDIR["link"]

    fig_accuracy(kt, models)
    fig_position(kt, models)
    fig_pull(kt, models)
    fig_self(st)

    L = ["# Results", "",
         f"Generated by `uv run judgebias report{' --split pilot' if split == 'pilot' else ''}` from "
         "`results/rows/*.jsonl` and `data/items.jsonl`. Every number below is computed by code from cached "
         "model outputs, re-parsed with the current parser. Intervals are 95% percentile bootstraps "
         f"({C.BOOT:,} resamples) over items (both orders of an item, and all its conditions, move together), "
         "stratified by task when tasks are pooled. `pp` = percentage points; `*` marks a difference whose "
         "interval excludes zero.", ""]
    if split == "pilot":
        L += ["**Pilot split.** These are the pilot items, used only to check templates and parsers. They are "
              "excluded from every reported result.", ""]

    # headline
    L += ["## Headline", ""]
    for m in models:
        k = kt.get((m, None))
        if not k:
            continue
        p = k["position"]
        L.append(f"- **{mlabel(m)}** ({k['n_items']} items, both tasks): picks the right answer in "
                 f"{ci(k['acc']['base'])} of item-orders; same pick in both orders on {ci(p['consistent'])} of "
                 f"items, slot A both times on {ci(p['always_a'])}, slot B both times on {ci(p['always_b'])}. "
                 f"A/A repeat changed {ci(k['floor'])} of verdicts.")
    L.append("")
    for m in models:
        k = kt.get((m, None))
        if not k or not k["cost"]:
            continue
        parts = [f"{COND_LABEL[c]} {pp(k['cost'][c])}" for c in ["verbose_w", "confident_w", "markdown_w"]
                 if c in k["cost"]]
        if parts:
            L.append(f"- **{mlabel(m)}**, accuracy change when the wrong answer is restyled: " + "; ".join(parts) + ".")
    L.append("")

    # 1 setup
    L += ["## 1. Setup", ""]
    rows = []
    for f in facts:
        info = meta.get(f["model"], {})
        rows.append([mlabel(f["model"]), info.get("digest", "")[:12], f["mode"], f["calls"], f"{f['seconds']:.1f}",
                     f"{f['output_tokens']:.0f}", pct(f["truncated"]), pct(f["parsed"]),
                     ", ".join(f"{k} {100 * v:.0f}%" for k, v in f["methods"].items())])
    L += md_table(["model", "digest", "mode", "calls", "median server s/call", "mean output tokens",
                   "hit length cap", "parsed", "parse rule"], rows)
    items = load_items(split)
    from collections import Counter
    cnt = Counter(it["task"] for it in items)
    L += [f"Items in this split: {cnt.get('gsm8k', 0)} GSM8K planted-error pairs, {cnt.get('tqa', 0)} TruthfulQA "
          f"pairs, {cnt.get('gen', 0)} GSM8K problems for self-preference, {cnt.get('mtb', 0)} MT-bench pairs. "
          "Each table uses the items on which every condition it compares was answered in both orders.", ""]

    # 2 accuracy and position
    L += ["## 2. Accuracy and position bias (base prompt)", "",
          "Accuracy = share of (item, order) calls in which the judge picks the right answer; an unparsed verdict "
          "counts as wrong. Consistent = the same answer picked in both orders. `RW` = right answer in slot A, "
          "`WR` = right answer in slot B.", "", f"![position]({fl}/position.png)", ""]
    rows = []
    for m in models:
        for t in ["gsm8k", "tqa", None]:
            k = kt.get((m, t))
            if not k:
                continue
            p = k["position"]
            rows.append([mlabel(m), TASK_LABEL[t], k["n_items"], ci(k["acc"]["base"]), ci(p["consistent"]),
                         ci(p["always_a"]), ci(p["always_b"]), ci(p["p_a"]), ci(p["acc_rw"]), ci(p["acc_wr"]),
                         pp(p["rw_minus_wr"])])
    L += md_table(["judge", "task", "items", "accuracy", "consistent", "slot A both", "slot B both", "P(verdict A)",
                   "accuracy RW", "accuracy WR", "RW minus WR"], rows)
    rows = []
    for m in models:
        g = A.gsm8k_by_plant_position(df, m, split)
        if g:
            rows.append([mlabel(m), ci(g["last"]), g["last"]["n"], ci(g["earlier"]), g["earlier"]["n"]])
    if rows:
        L += ["GSM8K base accuracy by where the planted error is (exploratory):", ""]
        L += md_table(["judge", "error in the last calculation", "items", "error earlier", "items"], rows)

    # 3 floor
    L += ["## 3. Noise floor: the identical prompt sent twice", "",
          "Each base prompt is sent a second time as a separate call with its own cache entry, in both orders, "
          "spread through the run. A manipulation only counts as moving verdicts when its flip rate minus this "
          "repeat's has an interval above zero (paired by item).", ""]
    rows = []
    for m in models:
        for t in ["gsm8k", "tqa", None]:
            k = kt.get((m, t))
            if k:
                rows.append([mlabel(m), TASK_LABEL[t], k["n_items"], k["floor_n_flips"], ci(k["floor"]),
                             pp(k["cost"].get("aa", {}))])
    L += md_table(["judge", "task", "items", "verdicts changed (of 2 per item)", "flip rate", "accuracy change"], rows)

    # 4 manipulations
    L += ["## 4. Manipulations", "",
          "One answer is restyled at a time, in both orders. Flip rate = share of item-orders whose pick differs "
          "from the base prompt's. Excess = flip rate minus the A/A repeat's on the same items. Accuracy change "
          "is paired by item against the base prompt.", "", f"![accuracy]({fl}/accuracy_by_condition.png)", ""]
    for m in models:
        for t in ["gsm8k", "tqa", None]:
            k = kt.get((m, t))
            if not k:
                continue
            cons = A.consistency_by_cond(df, m, t)
            rows = []
            for c in ["aa"] + A.STYLE_CONDS:
                if c not in k["acc"]:
                    continue
                rows.append([COND_LABEL[c], k["acc"][c]["n"], ci(k["acc"][c]), pp(k["cost"].get(c, {})),
                             ci(k["flip"].get(c, {})), pp(k["excess"].get(c, {})) if c != "aa" else "",
                             ci(cons.get(c, {}))])
            L += [f"**{mlabel(m)}, {TASK_LABEL[t]}** (base accuracy {ci(k['acc']['base'])})", ""]
            L += md_table(["condition", "items", "accuracy", "change vs base", "flip rate", "excess over A/A",
                           "consistent"], rows)

    # 5 pull
    L += ["## 5. Pull of each style", "",
          "Pull = how much more often an answer is picked when it carries the style than when it is plain. "
          "On the wrong answer it equals base accuracy minus accuracy; on the right answer, accuracy minus base "
          "accuracy. `both` is the mean of the two, paired by item: a preference for the style itself, as "
          "distinct from a style that only confuses the judge (which would lower accuracy whichever answer "
          "carries it, giving a positive pull on the wrong answer and a negative one on the right answer).", "",
          f"![pull]({fl}/style_pull.png)", ""]
    rows = []
    for m in models:
        for t in ["gsm8k", "tqa", None]:
            k = kt.get((m, t))
            if not k:
                continue
            for s in A.STYLES:
                pl = k["pull"]
                if f"{s}_w" in pl or f"{s}_r" in pl:
                    rows.append([mlabel(m), TASK_LABEL[t], s, pp(pl.get(f"{s}_w", {})), pp(pl.get(f"{s}_r", {})),
                                 pp(pl.get(s, {}))])
    L += md_table(["judge", "task", "style", "on the wrong answer", "on the right answer", "both"], rows)

    # 6 pointwise
    L += ["## 6. Single-answer scoring", "",
          "Each answer is scored alone from 1 to 10. Pointwise accuracy = the right answer scored above the wrong "
          "one (a tie counts half). Shift = score of the restyled wrong answer minus the plain wrong answer.", ""]
    rows = []
    for m in models:
        for t in ["gsm8k", "tqa", None]:
            p = pt.get((m, t))
            k = kt.get((m, t))
            if not p:
                continue
            rows.append([mlabel(m), TASK_LABEL[t], p["n_items"], f"{p['mean_right']['mean']:.2f}",
                         f"{p['mean_wrong']['mean']:.2f}", ci(p["acc"]), ci(k["acc"]["base"]) if k else "n/a",
                         pp(pvp.get((m, t), {})), ci(p.get("aa_pt_right", {})), ci(p.get("aa_pt_wrong", {}))])
    L += md_table(["judge", "task", "items", "mean score right", "mean score wrong", "pointwise accuracy",
                   "pairwise accuracy", "pairwise minus pointwise", "A/A score changed (right)",
                   "A/A score changed (wrong)"], rows)
    rows = []
    for m in models:
        for t in ["gsm8k", "tqa", None]:
            p = pt.get((m, t))
            if not p:
                continue
            for s, r in p["style"].items():
                rows.append([mlabel(m), TASK_LABEL[t], s, num(r["shift"]), ci(r["acc"]), pp(r["cost"])])
    if rows:
        L += md_table(["judge", "task", "style on the wrong answer", "score shift", "pointwise accuracy",
                       "change vs plain"], rows)

    # 7 self-preference
    L += ["## 7. Self-preference", "",
          "Each main model solved the same GSM8K problems. Pairs of solutions by two different models are judged "
          "by all three main models in both orders. In a pair by models X and Y, the third model Z is neutral. "
          "Self-preference of X = how often X picks its own solution minus how often Z picks X's solution, on "
          "the same pairs, so the quality of X's writing as seen by an outside judge is held fixed. On "
          "`both right` pairs both solutions reach the gold number, so correctness gives no reason to prefer "
          "either.", ""]
    if gacc:
        L += md_table(["author", "problems", "solved", "hit length cap", "median characters"],
                      [[m, r["n"], ci(r), pct(r["truncated"]), f"{r['chars']:.0f}"] for m, r in gacc.items()])
    if st:
        rows = []
        for j, r in st.items():
            b = r["both_right"]
            rows.append([j, "both right", b["n_pairs"], ci(b["own"]), ci(b["neutral"]), pp(b["diff"])])
            for name, lab in (("own_right", "one right: own is right"), ("own_wrong", "one right: own is wrong")):
                x = r[name]
                if x["n_pairs"]:
                    rows.append([j, lab, x["n_pairs"], ci(x["own"]), ci(x["neutral"]), pp(x["diff"])])
        L += [f"![self]({fl}/self_preference.png)", ""]
        L += md_table(["judge (author)", "pairs", "n", "judge picks own", "neutral picks it", "self-preference"], rows)
        L += ["`one right` rows mix authorship with knowledge: a judge that got a problem wrong itself may also "
              "fail to see the error when judging, so `own is wrong` is read as exploratory.", ""]
    else:
        L += ["Not run yet.", ""]

    # 8 MT-bench
    L += ["## 8. Agreement with human judges (MT-bench, first turn)", "",
          "Pairs from `lmsys/mt_bench_human_judgments` where more than half of the human votes went to one answer "
          "(ties count against both). Agreement = the judge picks the human-preferred answer. Zheng et al. "
          "report about 66% agreement between two humans on non-tie MT-bench votes.", ""]
    rows = [[mlabel(m), r["n_items"], ci(r["agree"]), f"{ci(r['agree_multi'])} (n={r['n_multi']})",
             ci(r["consistent"]), ci(r["p_a"]), ci(r["pick_longer"]), pct(r["human_longer"])]
            for m, r in mt.items() if r]
    L += md_table(["judge", "pairs", "agrees with humans", "pairs with 2+ votes", "consistent", "P(verdict A)",
                   "picks the longer answer", "humans picked the longer"], rows) if rows else ["Not run yet.", ""]

    # 9 predictions
    if split == "main":
        L += ["## 9. Pre-registered predictions", "", "Scored against `HYPOTHESES.md` (written before the full run).", ""]
        L += hypothesis_lines(df, kt, pt, pvp, st, mt)

    # 10 sample prompts
    L += ["## 10. Sample rendered prompts", ""]
    sample = {t: next((it for it in items if it["task"] == t), None) for t in C.KNOWN}
    for t, it in sample.items():
        if not it:
            continue
        for mode, cond, order in (("pair", "base", "RW"), ("pair", "verbose_w", "WR"), ("pair", "confident_w", "RW"),
                                  ("pair", "markdown_w", "RW"), ("point", "pt_wrong", "")):
            rd = render(it, mode, cond, order)
            L += [f"**{it['id']}, {mode} {cond} {order}**", "", "```", rd.messages[-1]["content"], "```", ""]

    # 11 hand-check
    L += ["## 11. Hand-check of the verdict parser", ""]
    ag = os.path.join(C.HANDCHECK, "agreement.json")
    if os.path.exists(ag):
        L += [json.load(open(ag))["summary"], ""]
    else:
        L += ["`handcheck/sample.csv` holds a seeded sample of judge outputs with an empty `label` column. It has "
              "not been labelled yet; once Srikar labels it, `uv run judgebias handcheck score` writes the "
              "agreement here.", ""]

    os.makedirs(C.REPORTS, exist_ok=True)
    name = "results.md" if split == "main" else "pilot.md"
    with open(os.path.join(C.REPORTS, name), "w") as f:
        f.write("\n".join(L).rstrip() + "\n")
    print(f"wrote reports/{name} and figures")
