# Running the full grid

The full run has not been started. It needs mains power: on battery the machine may slow down or sleep
(prompt-brittleness lost five hours that way), so the start command below checks for AC power and keeps
the machine awake with `caffeinate`.

## Before starting

```sh
pmset -g batt                     # must say "Now drawing from 'AC Power'"
curl -s localhost:11434/api/tags  # Ollama is up, with qwen3:8b, llama3.1:8b, gemma3:12b, qwen3:30b-a3b
git log --oneline -- HYPOTHESES.md   # the pre-registration is committed before any full-run call
uv sync && uv run pytest -q
```

Nothing else should be using the GPU during the run.

## Start

From the repository root:

```sh
mkdir -p logs
pmset -g batt | grep -q "AC Power" && caffeinate -dims \
  uv run judgebias run --grid --models qwen3:8b,llama3.1:8b,gemma3:12b --late-models qwen3:30b-a3b \
  2>&1 | tee -a logs/full.log
```

`--grid` runs the planned grid in `src/judgebias/config.py` (`GRID` for the three main judges,
`THINK_GRID` for `qwen3:30b-a3b`). Each main judge runs its own stages in order (gen, core, style_w,
self, style_r, mtb, point) in its own thread; the self-preference stage waits until all three have
finished generating. `qwen3:30b-a3b` starts once all three main judges are done (running it beside
`gemma3:12b` in the pilot slowed both).

## Expected duration

About 10,100 calls. From the pilot's seconds per call:

| model | calls | alone | with the other main judges running |
|---|---|---|---|
| qwen3:8b | 3,208 | about 1.9 h | finishes after about 5 h |
| llama3.1:8b | 3,208 | about 1.9 h | finishes after about 5 h |
| gemma3:12b | 3,208 | about 4.6 h | finishes after about 7.5 to 8 h (alone for the last 2.5 to 3) |
| qwen3:30b-a3b | 500 | about 1 h | starts when gemma3:12b finishes |

Total: about 8.5 hours, plausibly 7 to 10. The pilot ran partly on battery; mains power may be a little
faster. Progress lines (`[model stage] done/total ... elapsed=`) are printed every 50 calls.

## If it stops

Every call is cached in `cache/llm.sqlite` and every row is appended to `results/rows/<model>.jsonl` as
it finishes. Re-running the same command resumes: cached calls return at once and only missing calls go
to the server. To stop early, interrupt it; whatever finished is analysed (each comparison uses the items
answered under both of its conditions in both orders), and the cut is recorded as a dated addendum in
`HYPOTHESES.md`.

## After the run

```sh
uv run judgebias handcheck sample   # handcheck/sample.csv, label column left empty for Srikar
uv run judgebias report             # reports/results.md and reports/figures/
```

Then fill in the Findings section of the README from `reports/results.md`. Once Srikar has labelled the
sample (copy `handcheck/sample.csv` to `handcheck/labels.csv` and fill `label`), `uv run judgebias
handcheck score` writes `handcheck/agreement.json` and the next `report` includes it.
