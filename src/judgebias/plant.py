"""Plant one arithmetic error in a GSM8K reference solution and carry it forward.

GSM8K reference solutions mark every calculation as `<<expr=result>>` followed by the visible result.
One calculation (chosen with a seeded generator) gets a wrong result: a single digit of the result is
moved by 1 or 2, the kind of slip a person makes when multiplying or adding by hand. Every later
calculation that uses that value is recomputed with the wrong value, so the rest of the solution is
internally consistent and the final answer is wrong. The written expression of the planted step is left
as it is, so the error is visible to anyone who redoes that one calculation.

An item is used only if the rewrite is unambiguous: every annotated calculation evaluates to its stated
integer result, the final answer equals the last calculation, the changed values do not occur in the
problem text and are not results of other steps, and the final answer changes. Anything else is skipped
(the caller tries another step, then another item), so no solution is edited by guesswork.
"""

from __future__ import annotations

import ast
import operator
import random
import re
from fractions import Fraction

ANN = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
NUMTOK = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\d.,])\d+(?:\.\d+)?(?![\d])")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


class PlantError(ValueError):
    pass


def evaluate(expr: str) -> Fraction:
    """Exact value of a GSM8K calculator expression (+ - * / and brackets on decimal numbers)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Fraction(str(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div) and b == 0:
                raise PlantError("division by zero")
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise PlantError(f"unsupported expression {expr!r}")
    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except SyntaxError as e:
        raise PlantError(str(e)) from e


def to_int(x: Fraction) -> int | None:
    return int(x) if x.denominator == 1 else None


def numbers_in(text: str) -> set[int]:
    out = set()
    for m in NUMTOK.finditer(text):
        s = m.group(0).replace(",", "")
        if "." not in s:
            out.add(int(s))
    return out


def fmt_like(new: int, old_visible: str) -> str:
    """Write `new` the way the old visible number was written (thousands separators or not)."""
    return f"{new:,}" if "," in old_visible else str(new)


def slip(value: int, rng: random.Random) -> int:
    """Move one digit of `value` by 1 or 2 (never the leading digit to 0), so the length is unchanged."""
    s = str(value)
    for _ in range(50):
        pos = rng.randrange(len(s))
        d = int(s[pos]) + rng.choice([-2, -1, 1, 2])
        if not 0 <= d <= 9 or (pos == 0 and d == 0 and len(s) > 1) or (len(s) == 1 and d == 0):
            continue
        new = int(s[:pos] + str(d) + s[pos + 1:])
        if new != value:
            return new
    raise PlantError("no slip found")


def strip(solution: str) -> str:
    """Remove the calculator annotations and turn `#### X` into a closing sentence."""
    lines = []
    for line in solution.strip().split("\n"):
        if line.startswith("####"):
            lines.append(f"The answer is {line[4:].strip()}.")
        else:
            lines.append(ANN.sub("", line).rstrip())
    return "\n".join(lines)


def parse_steps(solution: str, question: str) -> tuple[list[str], list[dict], int]:
    lines = solution.strip().split("\n")
    if not lines[-1].startswith("####"):
        raise PlantError("no final line")
    final = int(lines[-1][4:].strip().replace(",", ""))
    steps = []
    for li, line in enumerate(lines[:-1]):
        for m in ANN.finditer(line):
            res = Fraction(m.group(2).strip().replace(",", "")) if re.fullmatch(r"-?[\d,]+(?:\.\d+)?",
                                                                                 m.group(2).strip()) else None
            if res is None or to_int(res) is None:
                raise PlantError("non-integer or odd result")
            if evaluate(m.group(1)) != res:
                raise PlantError("annotation does not evaluate to its result")
            steps.append({"line": li, "start": m.start(), "end": m.end(), "expr": m.group(1).strip(),
                          "result": to_int(res)})
    if len(steps) < 2:
        raise PlantError("fewer than two calculations")
    if steps[-1]["result"] != final:
        raise PlantError("final answer is not the last calculation")
    return lines, steps, final


def plant(solution: str, question: str, k: int, rng: random.Random) -> dict:
    """Plant the error in calculation `k`. Returns the wrong solution (annotations stripped) and a record."""
    lines, steps, final = parse_steps(solution, question)
    results = [s["result"] for s in steps]
    if len(set(results)) != len(results):
        raise PlantError("repeated intermediate results")
    qnums = numbers_in(question)
    old = steps[k]["result"]
    new = slip(old, rng)
    mapping = {old: new}
    new_results = {k: new}
    for j in range(k + 1, len(steps)):
        expr = steps[j]["expr"]
        toks = [int(t) for t in re.findall(r"(?<![\d.])\d+(?![\d.])", expr)]
        if not any(t in mapping for t in toks):
            continue
        e2 = re.sub(r"(?<![\d.])\d+(?![\d.])", lambda m: str(mapping.get(int(m.group(0)), m.group(0))), expr)
        r2 = to_int(evaluate(e2))
        if r2 is None or r2 <= 0:
            raise PlantError("carried error gives a non-integer or non-positive value")
        steps[j]["expr2"] = e2
        if r2 != steps[j]["result"]:
            mapping[steps[j]["result"]] = r2
            new_results[j] = r2
    # every value that changes must be traceable: not in the question, not another step's result, and the
    # new values must not collide with numbers already in the text
    all_text_nums = numbers_in(question) | numbers_in(strip(solution))
    kinv = {s["result"]: i for i, s in enumerate(steps)}
    for o, n in mapping.items():
        if o in qnums or results.count(o) != 1:
            raise PlantError("changed value is ambiguous")
        st = steps[kinv[o]]
        before = "\n".join(lines[:st["line"]] + [lines[st["line"]][:st["start"]]])
        if o in numbers_in(ANN.sub("", before)):
            raise PlantError("changed value is also a constant used earlier")
        if n in all_text_nums:
            raise PlantError("new value collides with a number in the text")
    if len(set(mapping.values())) != len(mapping):
        raise PlantError("two changed values collide")
    new_final = mapping.get(final, final)
    if new_final == final:
        raise PlantError("the error does not reach the final answer")

    # rewrite the visible text line by line: on the planted line only the visible result (and later mentions
    # of it) change; on later lines every mention of a changed value changes
    out = []
    first_line = steps[k]["line"]
    for li, line in enumerate(lines[:-1]):
        if li < first_line:
            out.append(ANN.sub("", line).rstrip())
            continue
        cut = steps[k]["end"] if li == first_line else 0
        head, tail = line[:cut], line[cut:]
        head = ANN.sub("", head)
        tail = ANN.sub("", tail)

        def sub(m):
            v = m.group(0).replace(",", "")
            if "." in v:
                return m.group(0)
            return fmt_like(mapping[int(v)], m.group(0)) if int(v) in mapping else m.group(0)
        if li == first_line:
            # only the old result and values derived from it may change after the planted calculation
            tail = NUMTOK.sub(sub, tail)
            out.append((head + tail).rstrip())
        else:
            out.append(NUMTOK.sub(sub, ANN.sub("", line)).rstrip())
    out.append(f"The answer is {new_final}.")
    wrong = "\n".join(out)
    right = strip(solution)
    if wrong == right or right.count("\n") != wrong.count("\n"):
        raise PlantError("rewrite failed")
    return {"wrong": wrong, "right": right, "step": k, "n_steps": len(steps), "old": old, "new": new,
            "final": final, "wrong_final": new_final, "changed": {str(a): b for a, b in mapping.items()}}


def plant_any(solution: str, question: str, seed: int) -> dict:
    """Try the calculations in a seeded order and return the first clean plant."""
    rng = random.Random(seed)
    _, steps, _ = parse_steps(solution, question)
    order = list(range(len(steps)))
    rng.shuffle(order)
    last = None
    for k in order:
        try:
            return plant(solution, question, k, rng)
        except PlantError as e:
            last = e
    raise PlantError(f"no clean plant: {last}")
