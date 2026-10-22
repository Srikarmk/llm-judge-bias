# Hypotheses and analysis plan

Written 2026-10-22 at 17:47 EDT, before the full run. Nothing below is edited after the full run
starts; corrections go in a dated addendum at the end. The commit that adds this file precedes every
full-run call, so the git history shows the order.

What had been seen when this was written: a pilot on the pilot items only (8 GSM8K planted-error pairs,
8 TruthfulQA pairs, 6 self-preference problems, 4 MT-bench pairs), excluded from every reported number.
`qwen3:8b` and `llama3.1:8b` ran every stage; `gemma3:12b` ran generation, base, A/A and
self-preference on all pilot items and the other stages on 4 items per set; `qwen3:30b-a3b` ran base and
A/A on 3 items per set and the other stages on 4. The pilot was used to check templates, parsers, timing
and the grid size. On far too few items to estimate anything, it showed:

- every pairwise verdict of the three main judges parsed, all by the requested `Verdict:` rule; one
  single-answer output of 112 for `llama3.1:8b` gave no score; `qwen3:30b-a3b` used up its 4,096-token
  budget thinking about one MT-bench pair (both orders) and gave no verdict (it does not judge MT-bench in
  the full grid); no other output hit a length cap;
- a strong pull toward slot A: the same slot in both orders on 44% (`qwen3:8b`), 69% (`llama3.1:8b`) and
  19% (`gemma3:12b`) of 16 items, almost always slot A; `llama3.1:8b` picked slot A in both orders on
  all 8 GSM8K pairs;
- base accuracy (both sets, both orders) of 72%, 66% and 84% for the three main judges, and 6 of 6 for
  the thinking model;
- the A/A repeat changed 1 verdict in 96 for the three main judges, but the judge's text differed in 21
  of 96 repeats: temperature 0 with a fixed seed is not bit-for-bit repeatable on this server;
- the confident framing was penalised rather than rewarded: with it on the wrong answer, accuracy rose
  for both 8B judges, and with it on the right answer it fell;
- verbose padding of the wrong answer cost `llama3.1:8b` 22 pp; markdown changed little;
- on GSM8K, the planted-error solution scored alone got a mean of 8.1 of 10 from both 8B judges;
- self-preference on 18 both-right pairs was mixed in sign.

Predictions informed by the pilot are marked (P).

## Primary question

When an LLM judges which of two answers is better and the better one is known, how often does the
verdict follow something that should not matter (the answer's position, length, confident tone,
markdown formatting, or the judge's own authorship) rather than correctness, and how much accuracy does
each cost, measured against the noise of sending the identical prompt twice?

## Definitions

- *Accuracy*: share of (item, order) calls whose verdict picks the right answer; an unparsed verdict
  counts as wrong.
- *Consistent*: same answer picked in both orders. *Slot A both* / *slot B both*: the verdict follows the
  slot rather than the content.
- *Flip rate* of a condition: share of (item, order) calls whose pick differs from the base call in the
  same order (an unparsed verdict is its own value). *Floor*: the A/A repeat's flip rate.
- *Excess*: flip rate minus the floor on the same items, paired by item.
- *Pull* of a style: how much more often an answer is picked when it carries the style than when plain.
  On the wrong answer: base accuracy minus accuracy; on the right answer: accuracy minus base accuracy;
  *mean pull*: the average of the two, paired by item.
- *Self-preference* of judge X: on pairs of solutions by X and Y, how often X picks its own solution
  minus how often the third main model Z (neutral) picks X's solution, both orders.
- *Pointwise accuracy*: share of items where the right answer, scored alone from 1 to 10, scores above
  the wrong one (a tie counts half).
- "Main judges" = `qwen3:8b`, `llama3.1:8b`, `gemma3:12b`. "Pooled" = GSM8K and TruthfulQA together,
  resampled within set.

## Predictions

1. (P) Position bias is common and favours the first slot: on the base prompt, pooled, at least two of
   the three main judges give the same slot in both orders on at least 20% of items, and for at least
   two of them `slot A both` exceeds `slot B both` with the interval of the paired difference above
   zero.
2. (P) Position bias is larger on GSM8K than on TruthfulQA (lower consistency) for at least two of the
   three main judges.
3. The A/A flip rate is under 5% for every judge, pooled.
4. (P) Verbose padding of the wrong answer lowers accuracy, pooled, with the interval below zero, for at
   least two of the three main judges.
5. (P) Confident framing backfires: applied to the wrong answer it does not lower accuracy for any main
   judge (point estimate at or above zero, pooled), and its mean pull is negative, with the interval
   below zero, for at least two of the three main judges.
6. Markdown formatting has the smallest (or tied smallest) absolute mean pull of the three styles for at
   least two of the three main judges, pooled.
7. Restyling moves verdicts beyond the noise floor: of the 18 cells (3 main judges x 6 style
   conditions, pooled), the excess over the A/A repeat has an interval above zero in more than half.
8. Self-preference: on both-right pairs, the mean of the three main judges' self-preference is above
   zero, and at least one judge's interval is above zero.
9. Single-answer scoring misses planted errors: on GSM8K, the planted-error solution scored alone gets a
   mean of at least 6 of 10 from every main judge, and pairwise accuracy exceeds pointwise accuracy
   (interval of the paired difference above zero) for at least two of the three main judges. No
   direction is predicted for TruthfulQA.
10. Agreement with the MT-bench human majority is between 55% and 80% for every main judge.
11. (P) `gemma3:12b` has the highest pooled base accuracy of the three main judges.
12. (P) `qwen3:30b-a3b` (thinking on) has higher base accuracy and higher consistency than `qwen3:8b` on
    the same items. This is confounded with size and thinking.

What would contradict "LLM judges follow things that should not matter": consistency near 100%, flip
rates within the A/A floor for every style, pulls near zero, and self-preference near zero.

## Analysis plan (fixed in advance)

- Unit of resampling: item (problem, for self-preference). Both orders and every condition of an item
  move together. 95% percentile bootstrap, 2,000 draws; pooled over sets, items are resampled within
  each set.
- Every comparison is paired by item and uses the items on which both conditions were answered in both
  orders (relevant if the run is cut).
- Primary metrics: accuracy, consistency, slot A both / slot B both, flip rate and excess over A/A,
  accuracy change, pull, self-preference on both-right pairs, pointwise accuracy and mean scores,
  agreement with humans. Secondary: accuracy with the right answer in slot A vs slot B, accuracy by
  position of the planted error, single-answer score shifts under each style, self-preference on pairs
  with one right answer (exploratory: it mixes authorship with the judge's own knowledge), MT-bench
  agreement on pairs with two or more votes and the share of picks of the longer answer.
- Floor rule: a condition is said to move verdicts only if its excess over the A/A repeat has an
  interval above zero.
- Self-preference pairs: both solutions must end normally (not at the length cap); pairs where both are
  wrong are dropped. `both right` = both reach the gold number.
- MT-bench: first turn; votes pooled over human judges; kept when one answer has more than half of all
  votes (ties count against both) and both answers are at most 2,500 characters; one pair per question.
- Parsing: verdicts and scores are parsed by code; rule shares are reported. A seeded sample of judge
  outputs goes to `handcheck/` for a labelling pass by Srikar after the run; until it is labelled, the
  parser check is reported as not done.
- Multiple comparisons: no correction. Predictions 1 to 10 are read as confirmatory; 11 and 12 and
  everything else as exploratory.
- Exclusions: pilot items. A failed call is retried five times; if it still fails, that model's run
  stops and the others carry on.

## Run settings

- Ollama 0.40.6 on an Apple M5 Pro (48 GB). Models: `qwen3:8b` (digest `500a1f067a9f`, think off),
  `llama3.1:8b` (`46e0c10c039e`), `gemma3:12b` (`b92b218582c5`), `qwen3:30b-a3b` (`ad815644918f`,
  think on; the installed tag is the thinking-only release).
- Temperature 0, seed 7, `num_ctx` 4096, `num_predict` 320 for judging and 512 for the self-preference
  solutions; `num_ctx` 8192 and `num_predict` 4096 for the thinking model. Only the final answer (after
  thinking) is parsed.
- One call at a time per model; the three main judges run at the same time, each through its own
  stages; `qwen3:30b-a3b` starts after all three finish (in the pilot, running it beside `gemma3:12b`
  slowed both by more than it saved).

## Exact grid

Items are the first N of each set in the seeded order of `data/items.jsonl`, so each subset is a random
draw. Stages run in this order for each model, so a cut leaves the most important part done:

| stage | what | items | calls per main judge |
|---|---|---|---|
| gen | each main model solves GSM8K problems (worked solution) | 60 problems | 60 |
| core | pairwise base and A/A repeat, both orders | 80 GSM8K + 80 TruthfulQA | 640 |
| style_w | verbose, confident, markdown on the wrong answer, both orders | same 160 | 960 |
| self | pairs of solutions by two different main models, both orders | pairs from the 60 problems | up to 360 |
| style_r | verbose, confident, markdown on the right answer, both orders | first 40 + 40 | 480 |
| mtb | MT-bench pairs, base, both orders | 74 | 148 |
| point | right, wrong, A/A of each, three restyled wrong answers, scored alone | first 40 + 40 | 560 |

`qwen3:30b-a3b`: core and style_w on the first 25 + 25 items (500 calls).

In all, about 10,100 calls. Expected duration from the pilot's seconds per call (about 1.7 s for a
pairwise call by `qwen3:8b` alone and 2.8 to 3.2 s for each 8B judge with two running, about 4 s for
`gemma3:12b` alone, about 6 to 7 s for the thinking model alone; three models at once are assumed to
run at about 40% of their solo speed each): the two 8B judges finish in about 5 hours,
`gemma3:12b` in about 7.5 to 8, and the thinking model about 1 hour after that, so about 8.5 hours in
all (7 to 10 hours given the spread of the pilot timings). If the run is cut, whatever finished is
reported and the analysed grid is stated in an addendum.

## Addenda
