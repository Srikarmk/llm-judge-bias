"""Fixed settings for every run. Changing any of these changes the cache keys or the item sample."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
ITEMS = os.path.join(DATA, "items.jsonl")
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")
HANDCHECK = os.path.join(ROOT, "handcheck")
CACHE_DB = os.path.join(ROOT, "cache", "llm.sqlite")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELS = ["qwen3:8b", "llama3.1:8b", "gemma3:12b", "qwen3:30b-a3b"]
MAIN_MODELS = MODELS[:3]          # judges on the full grid, and the authors for self-preference

SEED = 7
BASE_OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 4096}
NUM_PREDICT = {"pair": 320, "point": 320, "gen": 512}

# The installed qwen3:30b-a3b is the thinking-only release; think: false does not stop it reasoning
# (found in memory-rot). It runs with thinking on and a larger cap; only the final answer is parsed.
THINKING_ONLY = {"qwen3:30b-a3b"}
THINKING_NUM_PREDICT = 4096
THINKING_NUM_CTX = 8192

# Pinned Hugging Face sources: (repo, revision, file path inside the repo, sha256 of the file).
SOURCES = {
    "gsm8k_test": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
                   "main/test-00000-of-00001.parquet",
                   "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "truthfulqa_gen": ("truthfulqa/truthful_qa", "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
                       "generation/validation-00000-of-00001.parquet",
                       "dfb1004b8ab83b22e8e476c76d5ac6074ff35c43946a724e810de3d83c3e21a5"),
    "mtbench_human": ("lmsys/mt_bench_human_judgments", "f7d2896d2cc5d80f8b55c2bbc722613555233c25",
                      "data/human-00000-of-00001-25f4910818759289.parquet",
                      "4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769"),
}

# Item counts per task (main items; the pilot items are drawn first and are disjoint).
# gsm8k: reference solution vs the same solution with one planted arithmetic error
# tqa:   TruthfulQA best answer vs one of its listed incorrect answers
# gen:   GSM8K problems solved by each main model, for self-preference
# mtb:   MT-bench first-turn pairs with a clear human majority
N_MAIN = {"gsm8k": 100, "tqa": 100, "gen": 100, "mtb": 74}
N_PILOT = {"gsm8k": 8, "tqa": 8, "gen": 6, "mtb": 4}
TASKS = ["gsm8k", "tqa", "gen", "mtb"]
KNOWN = ["gsm8k", "tqa"]          # tasks with a constructed right and wrong answer
MTB_MAX_CHARS = 2500              # skip MT-bench answers longer than this, to keep prompts short

# The planned grid: stage -> first N items per task (0 = all), sized from the pilot's seconds per call to
# about 8 to 9 hours. Items are taken in their seeded order, so each subset is a random draw. Stages run
# in this order for each model.
GRID = {"gen": 60, "core": 80, "style_w": 80, "self": 60, "style_r": 40, "mtb": 0, "point": 40}
# qwen3:30b-a3b (thinking) runs a reduced grid: base, A/A and the wrong-answer manipulations.
THINK_GRID = {"core": 25, "style_w": 25}

BOOT = 2000


def options(kind: str, model: str = "") -> dict:
    """`kind` is pair, point (judging) or gen (answer generation for self-preference)."""
    if model in THINKING_ONLY:
        return {**BASE_OPTIONS, "num_ctx": THINKING_NUM_CTX, "num_predict": THINKING_NUM_PREDICT}
    return {**BASE_OPTIONS, "num_predict": NUM_PREDICT[kind]}
