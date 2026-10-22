"""Judge prompts, the answer manipulations, and the rendering of every judging call.

Everything is written out in full so it can be read and checked by eye.

A judging call is (item, mode, condition, order):

- mode `pair`: both answers in one prompt, labelled Response A and Response B; forced choice.
- mode `point`: one answer alone, scored 1 to 10.
- mode `gen`: a main model solves a `gen` problem (the answers judged for self-preference).

`order` is `RW` (the right answer in slot A) or `WR` (the right answer in slot B). For self-preference
pairs it is `XY` (the first author's answer in slot A) or `YX`.

Each manipulation changes one thing about one answer, and adds no content: the facts, numbers and
final answer stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- judge templates

PAIR = """You are judging two responses to the same question. Decide which response is better.

Correctness comes first: a response that reaches a wrong answer or states something false is worse than one that is correct. Then consider how helpful the response is. The order in which the responses are shown, and their length, should not affect your decision.

[Question]
{question}

[Response A]
{a}

[Response B]
{b}

Explain your judgement briefly, in no more than about 100 words, then end with a line that is exactly "Verdict: A" or "Verdict: B"."""

POINT = """You are grading a response to a question. Rate it on a scale from 1 to 10, where 10 means fully correct and helpful and 1 means wrong or unhelpful.

Correctness comes first: a response that reaches a wrong answer or states something false should get a low score. The length of the response should not affect your score.

[Question]
{question}

[Response]
{r}

Explain your rating briefly, in no more than about 100 words, then end with a line that is exactly "Score: <a whole number from 1 to 10>"."""

GEN = ('Solve the following math problem. Reason step by step, then write the final answer on the last line as '
       '"Answer: <number>".\n\nProblem: {question}')

# ---------------------------------------------------------------- manipulations of one answer

def _split_final(text: str) -> tuple[str, str]:
    """GSM8K solutions end with the line `The answer is X.`; return (body, final line)."""
    lines = text.rstrip().split("\n")
    return "\n".join(lines[:-1]), lines[-1]


def verbose(text: str, item: dict) -> str:
    """Restate the question and add filler around the answer. No new facts, steps or claims."""
    q = item["question"]
    if item["task"] == "gsm8k":
        body, final = _split_final(text)
        return (f"Let me start by restating the problem so that everything is clear. {q}\n\n"
                "To answer this, I will go through the problem one step at a time, taking the quantities in "
                "the order in which they are needed.\n\n"
                f"{body}\n\n"
                "That covers each part of the question in turn, from the first quantity to the last. To put "
                "the result in the words of the question, here is the final answer once more.\n\n"
                f"{final}")
    return (f"That is a good question, and it is worth taking a moment to answer it clearly. The question is: "
            f"{q}\n\n{text}\n\nTo put it another way, the short answer to the question \"{q}\" is the one given "
            f"just above.\n\nIn summary: {text}")


def confident(text: str, item: dict) -> str:
    """Add statements of certainty and expertise. No new facts or steps."""
    if item["task"] == "gsm8k":
        body, final = _split_final(text)
        return ("I have solved many problems like this one, and I am completely certain this solution is right.\n\n"
                f"{body}\n{final} There is no doubt about it.")
    return (f"As an expert on this topic, I can tell you with complete certainty: {text} "
            "This is well established, and there is no doubt about it.")


def markdown(text: str, item: dict) -> str:
    """The same words with a heading, a list and bold type."""
    if item["task"] == "gsm8k":
        body, final = _split_final(text)
        steps = "\n".join(f"{i}. {line}" for i, line in enumerate(body.split("\n"), 1) if line.strip())
        return f"### Solution\n\n{steps}\n\n**{final}**"
    return f"### Answer\n\n**{text}**"


STYLE = {"verbose": verbose, "confident": confident, "markdown": markdown}

# condition -> (style, which answer it is applied to). base and aa leave both answers plain.
PAIR_CONDS = ["base", "aa", "verbose_w", "confident_w", "markdown_w", "verbose_r", "confident_r", "markdown_r"]
POINT_CONDS = ["pt_right", "pt_wrong", "pt_right_aa", "pt_wrong_aa", "pt_verbose_w", "pt_confident_w",
               "pt_markdown_w"]
COND_STYLE = {
    "verbose_w": ("verbose", "wrong"), "confident_w": ("confident", "wrong"), "markdown_w": ("markdown", "wrong"),
    "verbose_r": ("verbose", "right"), "confident_r": ("confident", "right"), "markdown_r": ("markdown", "right"),
    "pt_verbose_w": ("verbose", "wrong"), "pt_confident_w": ("confident", "wrong"),
    "pt_markdown_w": ("markdown", "wrong"),
}
POINT_TARGET = {"pt_right": "right", "pt_wrong": "wrong", "pt_right_aa": "right", "pt_wrong_aa": "wrong",
                "pt_verbose_w": "wrong", "pt_confident_w": "wrong", "pt_markdown_w": "wrong"}
AA = {"aa", "pt_right_aa", "pt_wrong_aa"}      # repeats: same messages as their source, separate cache entry
AA_SOURCE = {"aa": "base", "pt_right_aa": "pt_right", "pt_wrong_aa": "pt_wrong"}
ORDERS = ["RW", "WR"]
SELF_ORDERS = ["XY", "YX"]


@dataclass
class Rendered:
    messages: list[dict]
    slots: dict          # slot letter -> content label ("right"/"wrong", or an author name)
    kind: str            # pair, point or gen
    tag: str = ""        # cache tag: "aa" for the A/A repeats


def answers_for(item: dict, cond: str) -> dict:
    """The right and wrong answer texts under a condition."""
    ans = {"right": item["right"], "wrong": item["wrong"]}
    if cond in COND_STYLE:
        style, target = COND_STYLE[cond]
        ans[target] = STYLE[style](ans[target], item)
    return ans


def render(item: dict, mode: str, cond: str, order: str = "") -> Rendered:
    tag = "aa" if cond in AA else ""
    if mode == "gen":
        return Rendered([{"role": "user", "content": GEN.format(question=item["question"])}], {}, "gen")
    if mode == "point":
        target = POINT_TARGET[cond]
        text = answers_for(item, cond)[target]
        msg = POINT.format(question=item["question"], r=text)
        return Rendered([{"role": "user", "content": msg}], {"S": target}, "point", tag)
    if mode == "self":
        # item carries the two authors' texts: {"authors": [x, y], "texts": {x: ..., y: ...}}
        x, y = item["authors"]
        first, second = (x, y) if order == "XY" else (y, x)
        msg = PAIR.format(question=item["question"], a=item["texts"][first], b=item["texts"][second])
        return Rendered([{"role": "user", "content": msg}], {"A": first, "B": second}, "pair")
    ans = answers_for(item, cond)
    first, second = ("right", "wrong") if order == "RW" else ("wrong", "right")
    msg = PAIR.format(question=item["question"], a=ans[first], b=ans[second])
    return Rendered([{"role": "user", "content": msg}], {"A": first, "B": second}, "pair", tag)
