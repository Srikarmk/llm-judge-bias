# llm-judge-bias

<a href="https://www.srikarmk.dev/series/measuring-llm-systems"><img alt="Series" src="https://img.shields.io/badge/series-Measuring%20LLM%20Systems-5b4fc9" /></a> <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue" /></a> <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue" /></a>

**Question.** An LLM is asked to judge which of two answers is better, and the better answer is known.
Does the verdict change when something that should not matter changes: which answer is shown first, how
long it is, how confident it sounds, whether it is formatted in markdown, or whether the judge wrote it
itself? How much accuracy does each of these cost, measured against the noise of asking the identical
question twice?

This repository is a measurement study with local open-weight models, the next in the series after
prompt-brittleness, memory-rot and rag-poisoning. Every number in [reports/results.md](reports/results.md)
is produced by the code here from cached model calls, so it regenerates exactly. Pre-registered
predictions, with dated addenda: [HYPOTHESES.md](HYPOTHESES.md).

## Findings

Numbers are from [reports/results.md](reports/results.md): 95% bootstrap intervals over items, `pp` =
percentage points. The planned grid ran in full: `qwen3:8b`, `llama3.1:8b` and `gemma3:12b` judged 80
GSM8K planted-error pairs and 80 TruthfulQA pairs in both orders under every condition, plus the
right-answer controls, single-answer scoring (40 + 40), self-preference (60 problems) and 74 MT-bench
pairs; `qwen3:30b-a3b` (thinking) judged 25 + 25 pairs in the base and wrong-answer conditions. 10,052
model calls in all.

- **Order decides many verdicts.** Shown the same two answers in both orders, `llama3.1:8b` picked
  whichever came first on 96% [91, 100] of GSM8K pairs, so its accuracy there was 50.6%, a coin flip.
  `qwen3:8b` picked slot A both times on 42.5% of GSM8K pairs; `gemma3:12b` had no lean to either slot
  (+0.6 pp [-5.6, +6.9]). With the right answer in slot A the 8B judges were right 90 to 93% of the time;
  in slot B, 32 to 63%.
- **Thinking fixed it, on few items.** `qwen3:30b-a3b` with reasoning on was right on 94% of 50 pairs and
  consistent across orders on 96%, against 70% and 60% for `qwen3:8b` on the same items (confounded
  with thinking and size).
- **Noise floor is low.** The identical prompt sent twice changed 0.6% [0.0, 1.6] of verdicts for each
  main judge, though the judge's wording changed more often. Every style effect below clears it: 18 of
  18 judge-by-style cells have a flip rate above the repeat's.
- **Padding a wrong answer helps it.** Restating the wrong answer at length, with no new content, cut
  accuracy by 18.4 pp [13.8, 23.4] for `llama3.1:8b` and 8.1 pp [3.4, 12.8] for `gemma3:12b`
  (`qwen3:8b` -3.8 pp, interval touching zero).
- **Confidence backfires.** An openly confident wrong answer ("I am certain ...") was picked less, not
  more: accuracy rose 8 to 13 pp for every main judge. The same framing on the right answer cost 4 to
  33 pp. Read across both, the judges count stated certainty against an answer (mean pull -7.5 to
  -22.2 pp).
- **Markdown pulls a little.** Formatting an answer as markdown made it 4 to 8 pp more likely to be
  picked, averaged over putting it on the wrong answer and on the right one.
- **Two of three judges prefer their own work.** On pairs where both solutions are right, `qwen3:8b` and
  `llama3.1:8b` picked their own about 12 pp more often than a neutral judge picked the same solution;
  `gemma3:12b` picked its own 11 pp less often.
- **Scored alone, wrong maths looks fine.** A solution with a planted arithmetic error still averaged
  7.9 to 8.2 out of 10, against 8.8 to 9.3 for the correct one. Single-answer scoring ranked the right
  answer higher on 62 to 71% of items; pairwise comparison did better for `gemma3:12b` (+10 pp
  [+3.8, +16.9]), and about the same for `qwen3:8b` and `llama3.1:8b` (for `llama3.1:8b`, worse on GSM8K,
  where its pairwise verdicts were mostly slot A).
- **Agreement with humans.** On MT-bench pairs with a human majority, the judges agreed with it 68 to 80%
  of the time, around the 66% two humans agree with each other in Zheng et al.; humans and judges both
  preferred the longer answer about 70% of the time.

How the 12 pre-registered predictions fared (details in results.md): 10 held (position bias, lower
consistency on GSM8K, the A/A floor, the verbosity cost, confidence not helping, styles beating the
floor, self-preference, human agreement in range, `gemma3:12b` most accurate, the thinking model best).
Two did not: markdown was not the weakest style (for two judges verbosity pulled less), and on GSM8K
pairwise judging beat single-answer scoring clearly for only one judge, where the prediction needed two.

The verdict parser read 100% of pairwise verdicts by the requested `Verdict:` line for two judges and
99% for the third. A hand-labelled check of 80 sampled outputs is drawn in `handcheck/` and pending;
see Hand-check.

## Design

**Pairs with a known better answer.** Bias is easiest to see when the right verdict is known, so most
pairs are built so that one answer is right by construction:

| set | source | items sampled (in the grid; pilot) | right answer | wrong answer |
|---|---|---|---|---|
| GSM8K planted error | GSM8K test | 100 (80, 40 for the controls and scoring; 8) | the reference worked solution | the same solution with one planted arithmetic error, carried forward to a wrong final answer |
| TruthfulQA | TruthfulQA generation, validation | 100 (80, 40; 8) | the dataset's best answer | one of the dataset's listed incorrect answers (seeded choice) |
| self-preference | GSM8K test, disjoint problems | 100 (60; 6) | solutions written by the three main models, paired by author | |
| MT-bench | `lmsys/mt_bench_human_judgments`, first turn | 74 (74; 4) | the answer most human judges preferred | the other answer |

*Planted errors* ([src/judgebias/plant.py](src/judgebias/plant.py)). GSM8K reference solutions mark every
calculation as `<<48/2=24>>24`. One calculation, chosen with a seeded generator, gets a wrong result: one
digit moved by 1 or 2 (a hand-arithmetic slip, same number of digits). Every later calculation that uses
the value is recomputed, so the rest of the solution is internally consistent and the final answer is
wrong; the written expression of the planted step stays correct, so the error is visible to anyone who
redoes that one calculation. Both answers then lose the annotations and end with `The answer is X.`, so
the pair differs only in the planted result and what follows from it. A problem is used only when the
rewrite is unambiguous (every annotation evaluates exactly to its integer result, the changed values do
not occur in the problem text or as constants, the final answer changes); 881 of 1,319 test problems
qualify, and items are drawn from them in seeded order. 50 of the 108 drawn errors are in the last
calculation; the report splits accuracy by this.

*TruthfulQA* pairs are short factual answers where the wrong one is a common misconception written by the
dataset's authors ("you will develop arthritis"). Items whose best answer is "I have no comment" are
skipped. Short answers make each style manipulation a large part of the text.

**Judge prompts** ([src/judgebias/prompts.py](src/judgebias/prompts.py)). One pairwise template for every
set: the question, `[Response A]`, `[Response B]`, the instruction that correctness comes first, then
helpfulness, and that order and length should not affect the decision, about 100 words of reasoning, and
a last line `Verdict: A` or `Verdict: B`. Forced choice: every pair has a better answer. The instruction
to ignore order and length is in the prompt on purpose, as in common judge prompts (MT-bench's among
them), so every effect below is measured against an explicit instruction not to have it. The
single-answer template scores one answer from 1 to 10 with the same correctness-first wording and ends
with `Score: N`.

**Conditions, one change at a time.** Every pairwise call is made in both orders (`RW`: right answer in
slot A; `WR`: right answer in slot B), so position never confounds a manipulation.

| condition | what changes |
|---|---|
| base | both answers plain |
| A/A repeat | nothing: the base prompt sent again as a separate call with its own cache entry (the noise floor) |
| verbose | the answer is padded: the question restated, two sentences of process filler, the final answer restated; no new step, fact or claim |
| confident | a sentence of certainty and expertise before the answer and a "no doubt about it" after it; no new step or fact |
| markdown | the same words under a heading, as a numbered list (GSM8K), with the final line in bold |
| self-preference | the two answers are solutions written by two different main models |

Each of verbose, confident and markdown is applied to the wrong answer (the accuracy cost) and,
separately, to the right answer (the control). A real preference for the style raises how often the
styled answer is picked in both cases; a style that only confuses the judge lowers accuracy in both.
Tests check that a manipulation keeps every number and the final answer of the text it styles, adds no
number that is not in the question, and leaves the other answer untouched.

**Self-preference.** Each main model solves the same 60 GSM8K problems (worked solution, final line
`Answer: N`). For every problem, the three pairs of solutions by two different authors are judged by all
three main models in both orders. In a pair by X and Y the third model Z is neutral. X's self-preference
is how often X picks its own solution minus how often Z picks X's solution on the same pairs, so the
quality of X's writing as an outside judge sees it is held fixed. The main test uses pairs where both
solutions reach the gold number, so correctness gives no reason to prefer either. Pairs where exactly one
is right are reported separately and read as exploratory, because a judge that could not solve a problem
may also miss the error when judging it, which mixes authorship with knowledge.

**Single-answer scoring.** On the first 40 items of each known-answer set: the right answer alone, the
wrong answer alone, an A/A repeat of each, and each of the three restyled wrong answers alone. Pointwise
accuracy is the share of items where the right answer scores above the wrong one (a tie counts half);
pointwise scoring cannot have a position bias, and it is compared with pairwise accuracy on the same
items.

**Agreement with humans.** MT-bench first-turn answer pairs, votes pooled over the human judges in
`lmsys/mt_bench_human_judgments`, kept when one answer has more than half of all votes (ties count against
both) and both answers are at most 2,500 characters; one pair per question, seeded. Agreement = the judge
picks the human-preferred answer, in both orders. 45 of the 78 qualifying pairs (74 main, 4 pilot) rest on a single human vote, so the
report also gives agreement on the pairs with two or more votes, and how often judge and humans pick the
longer answer (humans did on 73% of pairs).

**Measures.** Per judge, set and condition, with every comparison paired by item:

- *accuracy*: share of (item, order) calls that pick the right answer; an unparsed verdict counts as
  wrong;
- *position*: share of items with the same pick in both orders (consistent), the same slot both times
  (`A` both or `B` both), P(verdict A), and accuracy with the right answer in slot A vs slot B;
- *flip rate*: share of (item, order) calls whose pick differs from the base call in the same order;
  the A/A repeat's flip rate is the noise floor, and a manipulation counts as moving verdicts only when
  its flip rate minus the A/A rate on the same items has an interval above zero;
- *accuracy cost*: accuracy under a manipulation minus base accuracy;
- *pull*: how much more often an answer is picked when it carries a style than when plain, on the wrong
  answer, on the right answer, and their mean;
- *self-preference*: own-pick rate minus the neutral judge's rate, as above;
- *pointwise*: mean scores, pointwise accuracy, the score shift of each restyled wrong answer, and
  pairwise minus pointwise accuracy.

**Statistics.** 95% percentile bootstraps over items (2,000 draws), stratified by set when sets are
pooled; both orders and all conditions of an item move together. Self-preference resamples problems.
The bootstrap and A/A rule follow prompt-brittleness, memory-rot and agent-interface-lab in the same
series.

**Parsing.** Verdicts are read by code ([src/judgebias/parse.py](src/judgebias/parse.py)): the last
`Verdict: A|B` (also `Response B`, `[[A]]`), then fallbacks (`[[A]]` alone, "my verdict is B", "Response A
is better" when only one response is so named, a last line that is only the letter); scores likewise from
`Score: N`, then "a score of N", then "N/10". The report lists how often each rule fired. A seeded sample
of judge outputs, with the parser's reading held apart, is written to `handcheck/` for a labelling pass;
see Hand-check below.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) with the four models pulled.

```sh
uv sync
uv run judgebias fetch          # pinned Hugging Face files, sha256-checked (1.4 MB)
uv run judgebias items          # data/items.jsonl: 374 main items and 26 pilot items, seeded
uv run judgebias show gsm8k-0259 --cond base,verbose_w   # rendered prompts (and outputs, once run)
# pilot (excluded from results): every stage on the pilot items
uv run judgebias run --stage gen --models qwen3:8b,llama3.1:8b,gemma3:12b --split pilot
uv run judgebias run --stage all --models qwen3:8b,llama3.1:8b --split pilot
uv run judgebias report --split pilot        # reports/pilot.md
uv run judgebias plan                        # calls per model for the planned grid
# the full grid (needs mains power; about 8.5 hours, see RUN.md)
pmset -g batt | grep -q "AC Power" && caffeinate -dims \
  uv run judgebias run --grid --models qwen3:8b,llama3.1:8b,gemma3:12b --late-models qwen3:30b-a3b \
  2>&1 | tee -a logs/full.log
uv run judgebias handcheck sample   # handcheck/sample.csv, labels left empty
uv run judgebias report             # reports/results.md and reports/figures/
uv run pytest
```

Every model call is cached in `cache/llm.sqlite`, keyed by model, messages, options, think flag and a
repeat tag (each A/A repeat has its own entry, so it is a real second call with the identical request).
With the cache present, `judgebias run --grid ... --cache-only` rebuilds every row in `results/rows/`
without a server, and `report` re-parses every stored output, so changing a parser rule updates every
number. An interrupted run resumes from the cache.

## Data and models

| Item | Source | Version | Size | Licence |
|---|---|---|---|---|
| GSM8K test | Hugging Face `openai/gsm8k`, `main/test-00000-of-00001.parquet` | commit `740312add88f781978c0658806c59bc2815b9866` | 419,088 bytes | MIT |
| TruthfulQA generation | Hugging Face `truthfulqa/truthful_qa`, `generation/validation-00000-of-00001.parquet` | commit `741b8276f2d1982aa3d5b832d3ee81ed3b896490` | 222,649 bytes | Apache 2.0 |
| MT-bench human judgments | Hugging Face `lmsys/mt_bench_human_judgments`, `data/human-00000-of-00001-25f4910818759289.parquet` | commit `f7d2896d2cc5d80f8b55c2bbc722613555233c25` | 738,612 bytes | CC BY 4.0 |
| Models | Ollama 0.40.6 library tags `qwen3:8b`, `llama3.1:8b`, `gemma3:12b`, `qwen3:30b-a3b` (Q4_K_M) | digests in `results/models.json` | 4.9 to 18.6 GB | Apache 2.0 (Qwen3), Llama 3.1 Community Licence, Gemma Terms of Use |

sha256 values are pinned in `src/judgebias/config.py` and recorded in `data/sources.json`. Settings:
temperature 0, seed 7, `num_ctx` 4096, at most 320 new tokens per judgement and 512 per self-preference
solution; `think: false` for `qwen3:8b` and `think: true` with `num_ctx` 8192 and a 4,096-token cap for
`qwen3:30b-a3b`, the thinking-only release, which keeps reasoning in its visible answer when thinking is
turned off (found in memory-rot); only its final answer is parsed. One call at a time per model.

## Layout

```
src/judgebias/  config (settings, grid), data (fetch, item sets), plant (planted GSM8K errors), prompts
                (judge templates, manipulations, rendering), parse (verdicts, scores, numbers), llm (Ollama
                + disk cache), runner (stages), stats (item bootstrap), analysis, hypotheses (scoring of
                the predictions), report, handcheck, cli
tests/          planted errors, templates and manipulations, parsers, cache, bootstrap, analysis
data/           items.jsonl (the item sets) and sources.json (files, revisions, sha256)
results/        rows/<model>.jsonl (one row per call) and models.json (digests, settings)
reports/        results.md and figures/ (after the full run); pilot.md and pilot_figures/
handcheck/      pilot_sample.csv now; sample.csv after the full run (label column empty), parser_key*.csv
cache/          llm.sqlite: every model call
```

## Hand-check

`handcheck/pilot_sample.csv` holds every pilot judge output that needed a fallback parsing rule and a
seeded random sample of the rest, with the raw output and an empty `label` column; the parser's reading
is kept apart in `handcheck/pilot_parser_key.csv`. After the full run, `judgebias handcheck sample`
writes the same for the main run (`handcheck/sample.csv`, 60 random outputs stratified by mode plus up to
20 that needed a fallback rule). None of these has been labelled: the labelling pass is Srikar's, and no
agreement figure is reported until it is done.

## Limits

- **A small grid.** Sized from the pilot to about 8.5 hours of local compute (it ran in full): 80 items per known-answer
  set for the main conditions, 40 for the right-answer controls and single-answer scoring, 60 problems
  for self-preference, and 25 per set for the thinking model. Intervals on the smaller cells are wide.
- Small local models, 4-bit quantised, one judge prompt, temperature 0 and one seed. A hosted judge, a
  different prompt or sampling could behave differently.
- The judge prompt tells the judge that order and length should not matter. Effects are measured against
  that instruction; without it they could be larger.
- Each style is one fixed, hand-written template, not a sample of all verbose, confident or markdown
  answers. "Confident" means these two sentences, which state certainty openly; subtler confidence
  (assertive wording with no hedges) is not tested.
- Planted errors are arithmetic slips only (one digit of one result), not reasoning errors, and some are
  easy to spot. The right and wrong solutions are otherwise identical, which isolates the error but is
  not how two independent solutions differ.
- TruthfulQA's best answers are often hedged ("Nothing in particular happens") and its incorrect answers
  assertive, so tone differs between right and wrong answers even in the base condition; the dataset is
  public and may be in the models' training data.
- MT-bench pairs come from 2023 models; 45 of 78 rest on one human vote, and humans preferred the longer
  answer in 73% of them, so agreement with humans is partly agreement about length.
- Self-preference uses three authors on one task. The neutral judge differs from pair to pair, and a
  judge's preference for its own style cannot be told apart from recognising itself.
- Temperature 0 with a fixed seed is not bit-for-bit repeatable here: in the pilot the judge's text
  changed in about one repeat in five, though the verdict rarely did. The A/A repeat measures this
  floor under the same conditions as every manipulation.
- The templates and the parser were written by the same person who will label the hand-check sample.

## Prior work

- Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 Datasets and
  Benchmarks ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)). Names position, verbosity and
  self-enhancement bias in LLM judges, swaps positions to test consistency, and releases the MT-bench
  human judgments used here.
- Wang et al. *Large Language Models are not Fair Evaluators.* ACL 2024
  ([arXiv:2305.17926](https://arxiv.org/abs/2305.17926)). Swapping the order of two answers changes
  GPT-4 and ChatGPT verdicts; proposes balanced position calibration.
- Shi et al. *Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative
  Assessments by LLMs* ([arXiv:2406.07791](https://arxiv.org/abs/2406.07791), 2024). Position
  consistency across many judges and tasks; bias is strongest when the two answers are close in quality.
- Saito et al. *Verbosity Bias in Preference Labeling by Large Language Models*
  ([arXiv:2310.10076](https://arxiv.org/abs/2310.10076), 2023). LLM judges prefer longer answers more
  than humans do.
- Panickssery, Bowman and Feng. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024
  ([arXiv:2404.13076](https://arxiv.org/abs/2404.13076)). Self-preference, linked to self-recognition.
- Chen et al. *Humans or LLMs as the Judge? A Study on Judgement Biases.* EMNLP 2024
  ([arXiv:2402.10669](https://arxiv.org/abs/2402.10669)). Authority and formatting ("beauty") biases,
  tested by perturbing answers, with humans and LLM judges.
- Ye et al. *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge*
  ([arXiv:2410.02736](https://arxiv.org/abs/2410.02736), 2024). A catalogue of judge biases with
  automated perturbations.
- Zeng et al. *Evaluating Large Language Models at Evaluating Instruction Following* (LLMBar). ICLR 2024
  ([arXiv:2310.07641](https://arxiv.org/abs/2310.07641)). Pairs with an objectively better answer, built
  so that surface quality misleads.
- Hosking, Blunsom and Bartolo. *Human Feedback is not Gold Standard.* ICLR 2024
  ([arXiv:2309.16349](https://arxiv.org/abs/2309.16349)). Assertive answers are rated as more correct by
  human annotators.
- Datasets: Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K,
  [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)); Lin, Hilton and Evans, *TruthfulQA: Measuring How
  Models Mimic Human Falsehoods* (ACL 2022, [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)).

What this study adds, if the run goes as planned: every bias measured on the same judges and items with
ground truth, against an A/A floor, with each style applied to the right answer as well as the wrong one
(so a preference can be told apart from confusion), and self-preference measured against a neutral judge
on pairs where both answers are right.

## License

MIT, Kameswara Sai Srikar Manda, 2026. GSM8K is MIT-licensed, TruthfulQA Apache 2.0, and the MT-bench
human judgments CC BY 4.0. `data/items.jsonl` holds the derived items (including GSM8K solutions with a
planted error, TruthfulQA answers and MT-bench answers) under their original licences.
