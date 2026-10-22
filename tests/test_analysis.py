import pandas as pd

from judgebias.analysis import flip_rows, pair_rows, per_item, self_rows


def row(id_, cond, order, verdict, task="tqa", model="m"):
    slots = {"RW": {"A": "right", "B": "wrong"}, "WR": {"A": "wrong", "B": "right"}}[order]
    pick = slots.get(verdict)
    return {"model": model, "task": task, "id": id_, "mode": "pair", "cond": cond, "order": order,
            "verdict": verdict, "pick": pick, "correct": pick == "right"}


def test_per_item_position_measures():
    df = pd.DataFrame([
        row("i1", "base", "RW", "A"), row("i1", "base", "WR", "B"),    # consistent and right
        row("i2", "base", "RW", "A"), row("i2", "base", "WR", "A"),    # slot A both times
        row("i3", "base", "RW", "B"), row("i3", "base", "WR", None),   # wrong, then unparsed
        row("i4", "base", "RW", "A"),                                   # one order only: dropped
    ])
    pi = per_item(pair_rows(df)).set_index("id")
    assert list(pi.index) == ["i1", "i2", "i3"]
    assert pi.loc["i1", "consistent"] == 1 and pi.loc["i1", "acc"] == 1
    assert pi.loc["i2", "always_a"] == 1 and pi.loc["i2", "consistent"] == 0 and pi.loc["i2", "acc"] == 0.5
    assert pi.loc["i3", "consistent"] == 0 and pi.loc["i3", "acc"] == 0
    assert pi.loc["i2", "acc_rw"] == 1 and pi.loc["i2", "acc_wr"] == 0


def test_flip_rows_pair_by_order():
    df = pd.DataFrame([
        row("i1", "base", "RW", "A"), row("i1", "base", "WR", "B"),
        row("i1", "verbose_w", "RW", "B"), row("i1", "verbose_w", "WR", "B"),   # flips in RW only
        row("i2", "base", "RW", "A"), row("i2", "base", "WR", "A"),
        row("i2", "verbose_w", "RW", "A"), row("i2", "verbose_w", "WR", None),  # unparsed counts as a change
    ])
    f = flip_rows(pair_rows(df), "verbose_w").set_index("id").flip
    assert f["i1"] == 0.5 and f["i2"] == 0.5


def test_self_preference_against_the_neutral_judge():
    pid = "gen-1|qwen3:8b|llama3.1:8b"
    meta = {pid: {"id": pid, "item": "gen-1", "authors": ["qwen3:8b", "llama3.1:8b"], "pair_type": "both_right",
                  "right_author": None}}
    rows = []
    picks = {"qwen3:8b": ["qwen3:8b", "qwen3:8b"], "llama3.1:8b": ["llama3.1:8b", "qwen3:8b"],
             "gemma3:12b": ["qwen3:8b", "llama3.1:8b"]}
    for judge, (p1, p2) in picks.items():
        for order, p in (("XY", p1), ("YX", p2)):
            rows.append({"model": judge, "task": "self", "id": pid, "mode": "self", "cond": "self", "order": order,
                         "pick": p})
    sr = self_rows(pd.DataFrame(rows), meta=meta).set_index("judge")
    # qwen picks itself 2/2, the neutral gemma picks qwen 1/2
    assert sr.loc["qwen3:8b", "own"] == 1 and sr.loc["qwen3:8b", "neutral_pick"] == 0.5
    assert sr.loc["qwen3:8b", "diff"] == 0.5
    # llama picks itself 1/2, gemma picks llama 1/2
    assert sr.loc["llama3.1:8b", "diff"] == 0
    assert "gemma3:12b" not in sr.index
