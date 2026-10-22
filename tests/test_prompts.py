import re

import pytest

from judgebias.prompts import (AA, AA_SOURCE, COND_STYLE, PAIR_CONDS, POINT_CONDS, POINT_TARGET, confident,
                               markdown, render, verbose)

GSM = {"id": "gsm8k-0001", "task": "gsm8k", "split": "main", "question": "Tom has 3 bags of 4 apples. How many?",
       "right": "Tom has 3 * 4 = 12 apples.\nThe answer is 12.",
       "wrong": "Tom has 3 * 4 = 14 apples.\nThe answer is 14."}
TQA = {"id": "tqa-001", "task": "tqa", "split": "main", "question": "What happens if you crack your knuckles a lot?",
       "right": "Nothing in particular happens if you crack your knuckles a lot.",
       "wrong": "If you crack your knuckles a lot, you will develop arthritis."}
SELF = {"id": "gen-0001|qwen3:8b|llama3.1:8b", "task": "self", "split": "main", "question": "Q?",
        "authors": ["qwen3:8b", "llama3.1:8b"], "texts": {"qwen3:8b": "X says 5", "llama3.1:8b": "Y says 5"},
        "right_author": None, "pair_type": "both_right"}


def nums(s):
    return sorted(re.findall(r"\d+", s))


@pytest.mark.parametrize("item", [GSM, TQA])
@pytest.mark.parametrize("style", [verbose, confident, markdown])
def test_styles_keep_the_answer_and_add_no_numbers(item, style):
    for which in ("right", "wrong"):
        out = style(item[which], item)
        assert out != item[which]
        # every number of the answer survives, and the only new numbers come from restating the question
        allowed = set(nums(item[which])) | set(nums(item["question"])) | {str(i) for i in range(1, 10)}
        assert set(nums(item[which])) <= set(nums(out))
        assert set(nums(out)) <= allowed
        if item["task"] == "gsm8k":
            final = item[which].split("\n")[-1]
            assert final in out
        else:
            assert item[which] in out


def test_markdown_list_keeps_every_line():
    out = markdown(GSM["wrong"], GSM)
    assert "1. Tom has 3 * 4 = 14 apples." in out and "**The answer is 14.**" in out


@pytest.mark.parametrize("item", [GSM, TQA])
def test_pair_slots_and_orders(item):
    a = render(item, "pair", "base", "RW")
    b = render(item, "pair", "base", "WR")
    assert a.slots == {"A": "right", "B": "wrong"} and b.slots == {"A": "wrong", "B": "right"}
    ta, tb = a.messages[0]["content"], b.messages[0]["content"]
    assert ta.index(item["right"]) < ta.index(item["wrong"])
    assert tb.index(item["wrong"]) < tb.index(item["right"])
    assert len(a.messages) == 1 and a.messages[0]["role"] == "user"
    assert 'exactly "Verdict: A" or "Verdict: B"' in ta


@pytest.mark.parametrize("cond", [c for c in PAIR_CONDS if c in COND_STYLE])
def test_each_manipulation_changes_exactly_one_answer(cond):
    style, target = COND_STYLE[cond]
    other = "wrong" if target == "right" else "right"
    for order in ("RW", "WR"):
        r = render(GSM, "pair", cond, order)
        text = r.messages[0]["content"]
        assert GSM[other] in text
        assert text != render(GSM, "pair", "base", order).messages[0]["content"]


def test_aa_is_the_identical_prompt_with_its_own_tag():
    for c in AA:
        src = AA_SOURCE[c]
        mode = "point" if c.startswith("pt_") else "pair"
        order = "" if mode == "point" else "WR"
        a, b = render(TQA, mode, c, order), render(TQA, mode, src, order)
        assert a.messages == b.messages and a.tag == "aa" and b.tag == ""


def test_point_targets():
    for c in POINT_CONDS:
        r = render(GSM, "point", c)
        assert r.slots == {"S": POINT_TARGET[c]} and "[Response]" in r.messages[0]["content"]
        text = r.messages[0]["content"]
        plain_target = GSM[POINT_TARGET[c]].split("\n")[-1]
        assert plain_target in text
    assert "14" not in render(GSM, "point", "pt_right").messages[0]["content"]


def test_self_pair_orders():
    a, b = render(SELF, "self", "self", "XY"), render(SELF, "self", "self", "YX")
    assert a.slots == {"A": "qwen3:8b", "B": "llama3.1:8b"} and b.slots == {"A": "llama3.1:8b", "B": "qwen3:8b"}
    assert a.messages[0]["content"].index("X says") < a.messages[0]["content"].index("Y says")


def test_gen_prompt():
    r = render({"question": "What is 2+2?"}, "gen", "gen")
    assert r.kind == "gen" and "Answer: <number>" in r.messages[0]["content"]
