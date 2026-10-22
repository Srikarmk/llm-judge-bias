"""Scoring of the pre-registered predictions in HYPOTHESES.md against the computed tables."""

from __future__ import annotations

import numpy as np

from . import config as C
from .analysis import STYLE_CONDS, STYLES, pair_rows, per_item
from .stats import mean_ci


def _pct(r: dict) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{100 * r['mean']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"


def _pp(r: dict) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{100 * r['mean']:+.1f} pp [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"


def _v(ok: bool | None) -> str:
    return "not testable" if ok is None else ("held" if ok else "did not hold")


def hypothesis_lines(df, kt, pt, pvp, st, mt) -> list[str]:
    main = [m for m in C.MAIN_MODELS if kt.get((m, None))]
    out = []
    if not main:
        return ["No main-judge results yet.", ""]

    # 1 position bias common, slot A favoured
    parts, n_same, n_a = [], 0, 0
    for m in main:
        d = per_item(pair_rows(df[df.model == m], C.KNOWN))
        b = d[d.cond == "base"]
        same = kt[(m, None)]["position"]["always_a"]["mean"] + kt[(m, None)]["position"]["always_b"]["mean"]
        diff = mean_ci((b.always_a - b.always_b).values, b.task.values)
        n_same += same >= 0.20
        n_a += diff["lo"] > 0
        parts.append(f"{m}: same slot {100 * same:.0f}%, A-both minus B-both {_pp(diff)}")
    out.append(f"1. Same slot in both orders on at least 20% of items for at least two main judges, and slot A "
               f"favoured (interval above zero) for at least two: {_v(n_same >= 2 and n_a >= 2)} "
               f"({'; '.join(parts)}).")

    # 2 position bias larger on GSM8K
    parts, n = [], 0
    for m in main:
        g, t = kt.get((m, "gsm8k")), kt.get((m, "tqa"))
        if g and t:
            cg, ct = g["position"]["consistent"]["mean"], t["position"]["consistent"]["mean"]
            n += cg < ct
            parts.append(f"{m}: consistent {100 * cg:.0f}% GSM8K vs {100 * ct:.0f}% TruthfulQA")
    out.append(f"2. Lower consistency on GSM8K than TruthfulQA for at least two main judges: {_v(n >= 2)} "
               f"({'; '.join(parts)}).")

    # 3 A/A floor under 5%
    js = [m for m in C.MODELS if kt.get((m, None))]
    ok = all(kt[(m, None)]["floor"]["mean"] < 0.05 for m in js)
    out.append(f"3. A/A flip rate under 5% for every judge: {_v(ok)} ("
               + "; ".join(f"{m} {_pct(kt[(m, None)]['floor'])}" for m in js) + ").")

    # 4 verbose wrong lowers accuracy
    cells = {m: kt[(m, None)]["cost"].get("verbose_w") for m in main}
    n = sum(bool(c and c["hi"] < 0) for c in cells.values())
    out.append(f"4. Verbose wrong answer lowers accuracy (interval below zero) for at least two main judges: "
               f"{_v(n >= 2)} (" + "; ".join(f"{m} {_pp(c)}" for m, c in cells.items()) + ").")

    # 5 confidence backfires
    cost = {m: kt[(m, None)]["cost"].get("confident_w") for m in main}
    pull = {m: kt[(m, None)]["pull"].get("confident") for m in main}
    ok1 = all(c and c["mean"] >= 0 for c in cost.values())
    n = sum(bool(p and p["hi"] < 0) for p in pull.values())
    out.append(f"5. Confident wrong answer does not lower accuracy for any main judge, and mean pull of "
               f"confidence below zero for at least two: {_v(ok1 and n >= 2)} (accuracy change: "
               + "; ".join(f"{m} {_pp(c)}" for m, c in cost.items()) + "; mean pull: "
               + "; ".join(f"{m} {_pp(p)}" for m, p in pull.items()) + ").")

    # 6 markdown smallest absolute pull
    n, parts, testable = 0, [], True
    for m in main:
        pl = kt[(m, None)]["pull"]
        if not all(s in pl for s in STYLES):
            testable = False
            continue
        a = {s: abs(pl[s]["mean"]) for s in STYLES}
        n += a["markdown"] <= min(a.values())
        parts.append(f"{m}: " + ", ".join(f"{s} {100 * pl[s]['mean']:+.1f}" for s in STYLES))
    out.append(f"6. Markdown has the smallest absolute mean pull for at least two main judges: "
               f"{_v(n >= 2 if testable else None)} ({'; '.join(parts)} pp).")

    # 7 excess above floor in more than half of 18 cells
    cells = [(m, c, kt[(m, None)]["excess"].get(c)) for m in main for c in STYLE_CONDS]
    have = [x for x in cells if x[2] and x[2]["n"]]
    n = sum(x[2]["lo"] > 0 for x in have)
    out.append(f"7. Excess over A/A above zero in more than half of the main judge x style cells: "
               f"{_v(n > len(cells) / 2 if len(have) == len(cells) else None)} "
               f"({n} of the {len(have)} cells run have an excess interval above zero; {len(cells)} planned).")

    # 8 self-preference
    if st:
        d = {j: r["both_right"]["diff"] for j, r in st.items() if r["both_right"]["diff"]["n"]}
        mean = np.mean([x["mean"] for x in d.values()]) if d else np.nan
        ok = bool(d) and mean > 0 and any(x["lo"] > 0 for x in d.values())
        out.append(f"8. Mean self-preference above zero and at least one judge's interval above zero (both-right "
                   f"pairs): {_v(ok if d else None)} (mean {100 * mean:+.1f} pp; "
                   + "; ".join(f"{j} {_pp(x)}" for j, x in d.items()) + ").")
    else:
        out.append("8. Self-preference: not testable (not run).")

    # 9 pointwise misses planted errors
    sc = {m: pt.get((m, "gsm8k")) for m in main}
    if all(sc.values()):
        ok1 = all(s["mean_wrong"]["mean"] >= 6 for s in sc.values())
        diffs = {m: pvp.get((m, "gsm8k")) for m in main}
        n = sum(bool(x and x.get("n") and x["lo"] > 0) for x in diffs.values())
        out.append(f"9. GSM8K planted-error solution scored at least 6/10 on average by every main judge, and "
                   f"pairwise above pointwise accuracy for at least two: {_v(ok1 and n >= 2)} (mean score of the "
                   "wrong solution: " + "; ".join(f"{m} {s['mean_wrong']['mean']:.1f}" for m, s in sc.items())
                   + "; pairwise minus pointwise: " + "; ".join(f"{m} {_pp(x)}" for m, x in diffs.items()) + ").")
    else:
        out.append("9. Single-answer scoring: not testable (not run for every main judge).")

    # 10 MT-bench agreement
    ag = {m: mt.get(m) for m in main if mt.get(m)}
    if len(ag) == len(main):
        ok = all(0.55 <= r["agree"]["mean"] <= 0.80 for r in ag.values())
        out.append(f"10. Agreement with the MT-bench human majority between 55% and 80% for every main judge: "
                   f"{_v(ok)} (" + "; ".join(f"{m} {_pct(r['agree'])}" for m, r in ag.items()) + ").")
    else:
        out.append("10. MT-bench agreement: not testable (not run for every main judge).")

    # 11 gemma most accurate
    acc = {m: kt[(m, None)]["acc"]["base"]["mean"] for m in main}
    out.append(f"11. gemma3:12b has the highest pooled base accuracy: "
               f"{_v(max(acc, key=acc.get) == 'gemma3:12b' if 'gemma3:12b' in acc and len(acc) == 3 else None)} ("
               + "; ".join(f"{m} {100 * a:.1f}%" for m, a in acc.items()) + ").")

    # 12 thinking model vs qwen3:8b on the same items
    big = "qwen3:30b-a3b"
    if kt.get((big, None)) and kt.get(("qwen3:8b", None)):
        d = per_item(pair_rows(df[df.model.isin([big, "qwen3:8b"])], C.KNOWN))
        b = d[d.cond == "base"].pivot_table(index=["task", "id"], columns="model", values=["acc", "consistent"])
        b = b.dropna()
        da = mean_ci((b["acc"][big] - b["acc"]["qwen3:8b"]).values, b.index.get_level_values(0))
        dc = mean_ci((b["consistent"][big] - b["consistent"]["qwen3:8b"]).values, b.index.get_level_values(0))
        ok = da["lo"] > 0 and dc["lo"] > 0
        out.append(f"12. qwen3:30b-a3b (thinking) more accurate and more consistent than qwen3:8b on the same "
                   f"{len(b)} items: {_v(ok)} (accuracy {_pp(da)}, consistency {_pp(dc)}; confounded with "
                   "thinking).")
    else:
        out.append("12. Thinking model: not testable (not run).")
    return [x + "\n" for x in out]

