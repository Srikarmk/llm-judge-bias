import pytest

from judgebias import config as C
from judgebias.llm import Cache, CacheMiss, Client, cache_key, request_body
from judgebias.prompts import render
from judgebias.runner import run_one

ITEM = {"id": "tqa-001", "task": "tqa", "split": "main", "question": "Q?", "right": "Yes.", "wrong": "No."}
RESP = {"text": "Verdict: A", "raw": "", "prompt_tokens": 3, "output_tokens": 3, "seconds": 0.1, "done_reason": "stop"}


def test_key_depends_on_everything():
    m = [{"role": "user", "content": "hi"}]
    o = {"temperature": 0, "seed": 7}
    k = cache_key("qwen3:8b", m, o)
    assert k == cache_key("qwen3:8b", m, dict(o))
    assert k != cache_key("llama3.1:8b", m, o)
    assert k != cache_key("qwen3:8b", m, {**o, "seed": 8})
    assert k != cache_key("qwen3:8b", [{"role": "user", "content": "hi!"}], o)
    assert k != cache_key("qwen3:8b", m, o, tag="aa")


def test_think_flag_and_options():
    assert request_body("qwen3:8b", [], {})["think"] is False
    assert request_body("qwen3:30b-a3b", [], {})["think"] is True
    assert "think" not in request_body("gemma3:12b", [], {})
    assert C.options("pair", "qwen3:8b")["temperature"] == 0
    assert C.options("pair", "qwen3:30b-a3b")["num_predict"] == C.THINKING_NUM_PREDICT


def test_cache_roundtrip_and_no_network_on_hit(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    m, o = [{"role": "user", "content": "x"}], {"temperature": 0}
    cache.put(cache_key("qwen3:8b", m, o), "qwen3:8b", RESP)
    client = Client(cache, url="http://127.0.0.1:9", cache_only=True)
    r = client.chat("qwen3:8b", m, o)
    assert r["text"] == "Verdict: A" and r["cached"] and client.stats["hits"] == 1
    with pytest.raises(CacheMiss):
        client.chat("qwen3:8b", m, o, tag="aa")
    assert len(Cache(str(tmp_path / "c.sqlite"))) == 1


def test_aa_repeat_gets_its_own_cache_entry_and_rows_map_slots(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    msgs = render(ITEM, "pair", "base", "WR").messages
    cache.put(cache_key("qwen3:8b", msgs, C.options("pair", "qwen3:8b")), "qwen3:8b", RESP)
    client = Client(cache, url="http://127.0.0.1:9", cache_only=True)
    row = run_one(client, "qwen3:8b", ITEM, "pair", "base", "WR")
    # right answer was in slot B, so verdict A picks the wrong answer
    assert row["verdict"] == "A" and row["pick"] == "wrong" and row["correct"] is False
    with pytest.raises(CacheMiss):
        run_one(client, "qwen3:8b", ITEM, "pair", "aa", "WR")
    cache.put(cache_key("qwen3:8b", msgs, C.options("pair", "qwen3:8b"), tag="aa"), "qwen3:8b",
              {**RESP, "text": "Verdict: B"})
    row = run_one(client, "qwen3:8b", ITEM, "pair", "aa", "WR")
    assert row["pick"] == "right" and row["correct"] is True
