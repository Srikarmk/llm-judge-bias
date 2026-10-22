import random
from fractions import Fraction

import pytest

from judgebias.plant import PlantError, evaluate, numbers_in, plant, plant_any, slip, strip

JANET_Q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for "
           "her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh "
           "duck egg. How much in dollars does she make every day at the farmers' market?")
JANET_A = ("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
           "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18")
JILL_Q = ("Jill gets paid $20 per hour to teach and $30 to be a cheerleading coach. If she works 50 weeks a year, "
          "35 hours a week as a teacher and 15 hours a week as a coach, what's her annual salary?")
JILL_A = ("Teaching: $20/hour * 35 hours/week = $<<20*35=700>>700/week\n"
          "Coaching: $30/hour * 15 hours/week = $<<30*15=450>>450/week\n"
          "Per week: $700/week + $450/week = $<<700+450=1150>>1150/week\n"
          "Per year: $1150/week * 50 weeks/year = $<<1150*50=57500>>57,500\n#### 57500")


def test_evaluate_is_exact():
    assert evaluate("60/100*5") == 3
    assert evaluate("1/3*3") == 1
    assert evaluate("(2+3)*4") == 20
    assert evaluate("2.5*2") == Fraction(5)
    with pytest.raises(PlantError):
        evaluate("2**10")


def test_slip_keeps_length_and_changes_value():
    rng = random.Random(0)
    for v in [3, 9, 18, 450, 1150, 57500]:
        for _ in range(20):
            w = slip(v, rng)
            assert w != v and len(str(w)) == len(str(v)) and w > 0


def test_strip_removes_annotations():
    s = strip(JANET_A)
    assert "<<" not in s and s.endswith("The answer is 18.")
    assert "16 - 3 - 4 = 9 duck eggs" in s


def test_plant_first_step_carries_forward():
    p = plant(JANET_A, JANET_Q, 0, random.Random(1))
    new = p["new"]
    assert p["right"] == strip(JANET_A)
    assert f"16 - 3 - 4 = {new} duck eggs" in p["wrong"]
    assert f"She makes {new} * 2 = ${new * 2} every day" in p["wrong"]
    assert p["wrong"].endswith(f"The answer is {new * 2}.") and p["wrong_final"] == new * 2
    # the planted expression is still written correctly, only its result is wrong
    assert p["wrong"].split("\n")[0].startswith("Janet sells 16 - 3 - 4 = ")


def test_plant_middle_step_keeps_earlier_lines_and_thousands_format():
    p = plant(JILL_A, JILL_Q, 1, random.Random(3))
    r, w = p["right"].split("\n"), p["wrong"].split("\n")
    assert r[0] == w[0]                       # before the planted step: unchanged
    assert r[1] != w[1] and "$30/hour * 15 hours/week = $" in w[1]
    total = 700 + p["new"]
    assert f"$700/week + ${p['new']}/week = ${total}/week" in w[2]
    assert f"{total * 50:,}" in w[3]          # written with a thousands separator, as in the original
    assert p["wrong_final"] == total * 50
    assert len(r) == len(w)


def test_plant_last_step_changes_only_the_end():
    p = plant(JILL_A, JILL_Q, 3, random.Random(5))
    r, w = p["right"].split("\n"), p["wrong"].split("\n")
    assert r[:3] == w[:3] and r[3] != w[3]
    assert p["wrong_final"] == p["new"] != 57500


def test_ambiguous_values_are_refused():
    # 9 appears in the question as a given number, so changing the result 9 cannot be traced safely
    q = "A box has 9 apples and 16 pears; 3 and 4 pears are eaten."
    with pytest.raises(PlantError):
        plant(JANET_A, q, 0, random.Random(1))
    with pytest.raises(PlantError):
        plant_any("It is 2+2=<<2+2=4>>4\n#### 4", "What is 2+2?", 1)   # one calculation only


def test_numbers_in_reads_separators():
    assert numbers_in("costs $1,350 and 6 more, 2.5 kg") == {1350, 6}


def test_plant_any_is_seeded():
    a = plant_any(JILL_A, JILL_Q, 42)
    b = plant_any(JILL_A, JILL_Q, 42)
    assert a == b and a["wrong"] != a["right"]
