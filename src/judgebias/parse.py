"""Deterministic parsing of judge outputs and of generated solutions.

Each parser returns (value, method). `value` is None when nothing usable was found; `method` records
which rule fired, so the report can show how often the fallbacks were needed.
"""

from __future__ import annotations

import re

NUM = r"-?\d[\d,]*(?:\.\d+)?"
SLOT = r"(?:response|assistant|answer|option)?\s*\(?\[?\[?(?-i:([AB]))\b"


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("**", "").replace("__", "").replace("`", "")


def parse_verdict(text: str) -> tuple[str | None, str]:
    t = clean(text)
    # 1. the requested "Verdict: A", on its own line or at the end of a sentence (also "Final verdict:
    # Response B", "# Verdict: [[A]]")
    cues = re.findall(rf"(?i)\bverdict\s*[:：]\s*{SLOT}", t)
    if cues:
        return cues[-1].upper(), "cue"
    # 2. the MT-bench style token [[A]]
    br = re.findall(r"\[\[([AB])\]\]", t)
    if br:
        return br[-1], "brackets"
    # 3. "verdict" inside a sentence: "my verdict is Response B"
    m = re.findall(rf"(?i)\bverdict\b[^.\n]{{0,20}}?\bis\s*{SLOT}", t)
    if m:
        return m[-1].upper(), "inline verdict"
    # 4. "Response A is better" (the last such statement, and only if it does not name both)
    m = re.findall(r"(?i)\bresponse\s+([AB])\s+is\s+(?:the\s+)?(?:better|correct|stronger|preferred)", t)
    if m and len(set(m)) == 1:
        return m[-1].upper(), "is better"
    # 5. a last line that is only the letter
    last = t.strip().split("\n")[-1].strip().strip(".:*()[] ")
    if last in ("A", "B"):
        return last, "bare letter"
    return None, "none"


def parse_score(text: str) -> tuple[int | None, str]:
    t = clean(text)
    # the requested "Score: N", on its own line or at the end of a sentence
    cues = re.findall(r"(?i)\b(?:score|rating)\s*[:：]\s*\[?\[?(\d{1,2}(?:\.\d+)?)", t)
    if cues:
        v = _score(cues[-1])
        if v is not None:
            return v, "cue"
    m = re.findall(r"(?i)\b(?:score|rating)\b[^.\n]{0,20}?\b(?:is|of)\s*(\d{1,2}(?:\.\d+)?)", t)
    if m and _score(m[-1]) is not None:
        return _score(m[-1]), "inline score"
    m = re.findall(r"\b(\d{1,2}(?:\.\d+)?)\s*/\s*10\b", t)
    if m and _score(m[-1]) is not None:
        return _score(m[-1]), "out of 10"
    return None, "none"


def _score(s: str) -> int | None:
    x = float(s)
    if not 1 <= x <= 10:
        return None
    return int(round(x))


def canon_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if x == int(x):
        return str(int(x))
    return f"{x:.6g}"


def parse_gsm8k(text: str) -> tuple[str | None, str]:
    """Final number of a worked solution (from prompt-brittleness)."""
    t = clean(text).replace("\\(", "").replace("\\)", "")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", t)
    cues = list(re.finditer(r"(?m)^\s*[#>*\- ]*(?i:final answer|answer)\s*[:：]\s*(.*)$", t))
    if cues:
        m = re.search(NUM, cues[-1].group(1).replace("$", ""))
        if m:
            return canon_number(m.group(0)), "cue"
    if boxed:
        m = re.search(NUM, boxed[-1])
        if m:
            return canon_number(m.group(0)), "boxed"
    nums = re.findall(NUM, t.replace("$", ""))
    if nums:
        return canon_number(nums[-1]), "last number"
    return None, "none"
