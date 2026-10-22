import pytest

from judgebias.parse import parse_gsm8k, parse_score, parse_verdict


@pytest.mark.parametrize("text,want,how", [
    ("Response A is right.\n\nVerdict: A", "A", "cue"),
    ("B has the error. Verdict: B", "B", "cue"),
    ("**Verdict: B**", "B", "cue"),
    ("Verdict: Response A", "A", "cue"),
    ("Final verdict: [[B]]", "B", "cue"),
    ("Verdict:A", "A", "cue"),
    ("Verdict: A\n\nOn reflection, Verdict: B", "B", "cue"),
    ("I prefer the second. [[B]]", "B", "brackets"),
    ("My verdict is Response B.", "B", "inline verdict"),
    ("Response A is better because it is correct.", "A", "is better"),
    ("Analysis...\nB", "B", "bare letter"),
    ("Both are fine.", None, "none"),
    # an article "a" is not a verdict, and "Both" does not read as B
    ("The verdict is a tie.", None, "none"),
    ("Verdict: Both", None, "none"),
    # mixed "is better" statements are not read
    ("Response A is better in tone, but Response B is correct.", None, "none"),
])
def test_verdict(text, want, how):
    assert parse_verdict(text) == (want, how)


@pytest.mark.parametrize("text,want,how", [
    ("Good answer.\n\nScore: 8", 8, "cue"),
    ("**Score: 10**", 10, "cue"),
    ("Score: 3/10", 3, "cue"),
    ("Rating: 7", 7, "cue"),
    ("I would give a score of 6.", 6, "inline score"),
    ("Overall 4/10.", 4, "out of 10"),
    ("Score: 0", None, "none"),
    ("No score.", None, "none"),
    ("Score: 2\nScore: 9", 9, "cue"),
    ("The logic is sound. Score: 9", 9, "cue"),
])
def test_score(text, want, how):
    assert parse_score(text) == (want, how)


@pytest.mark.parametrize("text,want,how", [
    ("15 * 1/5 = 3\nAnswer: 18", "18", "cue"),
    ("**Answer: $1,250**", "1250", "cue"),
    ("Answer: 18.00", "18", "cue"),
    ("The answer is \\boxed{12}.", "12", "boxed"),
    ("So 7 in total.", "7", "last number"),
    ("No idea.", None, "none"),
])
def test_gsm8k(text, want, how):
    assert parse_gsm8k(text) == (want, how)
