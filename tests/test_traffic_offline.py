"""Offline batches from a production trace (gitm.traffic.offline).

The loop measured every candidate on 256 copies of a five-token prompt asking
for 512 tokens: pure decode, nothing shared. On that, stock vLLM sat at its
ceiling, and Kimi K2.6 on H200 rejected all 17 candidates. A trace's real
lengths and prefix sharing put prefill, prefix caching and KV pressure back in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gitm.traffic.offline import batch_from_trace, load_batch, parse_spec
from gitm.traffic.schema import CanonicalRequest, Trace, TraceMeta

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "traffic_replay" / "fixtures"
MOONCAKE = FIXTURES / "mooncake_slice.jsonl"
BURSTGPT = FIXTURES / "burstgpt_slice.csv"


def _trace(reqs, block=4):
    meta = TraceMeta(source="t", path="t", sha256="0", rows_read=len(reqs),
                     rows_emitted=len(reqs), prefix_block_tokens=block,
                     has_prefix_identity=True)
    return Trace(meta=meta, requests=list(reqs))


def test_requests_that_shared_a_prefix_share_exactly_those_tokens():
    a = CanonicalRequest(arrival_s=0, input_tokens=10, output_tokens=5, prefix_blocks=(7, 8, 9))
    b = CanonicalRequest(arrival_s=1, input_tokens=10, output_tokens=6, prefix_blocks=(7, 3, 4))
    batch = batch_from_trace(_trace([a, b]), n=10, vocab_size=50_000)
    pa, pb = batch.prompt_token_ids
    assert len(pa) == len(pb) == 10
    assert pa[:4] == pb[:4]            # block 7, shared
    assert pa[4:8] != pb[4:8]          # blocks 8 and 3, not
    assert batch.output_tokens == [5, 6]
    assert all(1000 <= t < 50_000 for t in pa + pb)


def test_the_same_trace_gives_the_same_batch_every_time():
    """Every engine in an A/B has to be handed identical work."""
    r = [CanonicalRequest(arrival_s=0, input_tokens=9, output_tokens=2, prefix_blocks=(1, 2, 3))]
    one = batch_from_trace(_trace(r), n=1, vocab_size=50_000)
    two = batch_from_trace(_trace(r), n=1, vocab_size=50_000)
    assert one.prompt_token_ids == two.prompt_token_ids


def test_long_requests_are_skipped_and_counted_never_trimmed():
    reqs = [CanonicalRequest(arrival_s=0, input_tokens=100, output_tokens=10, prefix_blocks=()),
            CanonicalRequest(arrival_s=1, input_tokens=5000, output_tokens=10, prefix_blocks=()),
            CanonicalRequest(arrival_s=2, input_tokens=50, output_tokens=None, prefix_blocks=())]
    batch = batch_from_trace(_trace(reqs), n=10, vocab_size=50_000, max_model_len=1000)
    assert [len(p) for p in batch.prompt_token_ids] == [100]
    assert batch.skipped == {"over_max_model_len": 1, "no_output_length": 1}


def test_requests_without_prefix_identity_share_nothing():
    reqs = [CanonicalRequest(arrival_s=i, input_tokens=8, output_tokens=1) for i in range(2)]
    batch = batch_from_trace(_trace(reqs, block=None), n=2, vocab_size=50_000)
    assert batch.prompt_token_ids[0] != batch.prompt_token_ids[1]


def test_the_mooncake_slice_keeps_its_shape_and_its_shared_system_prompt():
    batch = load_batch(f"mooncake:{MOONCAKE}", n=64, vocab_size=151_936, max_model_len=32_768)
    s = batch.summary()
    assert s["requests"] == 64 and s["source"] == "mooncake"
    assert s["input_tokens_mean"] > 1000              # prefill-heavy, unlike 5-token toys
    # Every request in the slice starts with block 0: one shared 512-token prefix.
    first = {tuple(p[:512]) for p in batch.prompt_token_ids}
    assert len(first) == 1


def test_burstgpt_loads_without_prefix_sharing():
    batch = load_batch(f"burstgpt:{BURSTGPT}", n=20, vocab_size=151_936)
    assert batch.summary()["requests"] == 20


def test_a_bad_spec_names_what_is_accepted():
    with pytest.raises(ValueError, match="mooncake"):
        parse_spec("sharegpt:/x")
    with pytest.raises(ValueError):
        parse_spec("mooncake")


def test_the_decode_workload_sends_the_traces_requests(tmp_path, monkeypatch):
    """With GITM_VLLM_TRAFFIC set, the engine gets token-id prompts with each
    request's own output length, generated to exactly that length."""
    import sys
    import types

    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    sent: list = []

    class _Tok:
        vocab_size = 151_936

    class _LLM:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs
            self.llm_engine = types.SimpleNamespace(
                model_config=types.SimpleNamespace(max_model_len=32_768))

        def get_tokenizer(self):
            return _Tok()

        def reset_prefix_cache(self):
            sent.append("reset")

        def generate(self, prompts, params):
            sent.append((prompts, params))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
                token_ids=list(range(p.max_tokens)))]) for p in params]

    class _SamplingParams:
        def __init__(self, max_tokens=0, temperature=0.0, ignore_eos=False):
            self.max_tokens, self.ignore_eos = max_tokens, ignore_eos

    fake = types.ModuleType("vllm")
    fake.LLM, fake.SamplingParams = _LLM, _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.delenv("GITM_VLLM_SYNTHETIC", raising=False)
    monkeypatch.setenv("GITM_VLLM_TRAFFIC", f"mooncake:{MOONCAKE}")
    monkeypatch.setenv("GITM_VLLM_PROMPTS", "12")

    runner = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode"))
    out = runner()
    # The cache is emptied before the measured batch, so no measurement starts
    # warm from the one before it (review finding).
    assert sent[-2] == "reset"
    prompts, params = sent[-1]
    assert len(prompts) == len(params) == 12
    assert all(isinstance(p, dict) and len(p["prompt_token_ids"]) > 100 for p in prompts)
    assert all(p.ignore_eos for p in params)
    assert out["traffic"]["source"] == "mooncake" and out["traffic"]["requests"] == 12
    assert out["generated_tokens"] == sum(p.max_tokens for p in params)


def test_a_private_tail_never_collides_with_a_trace_block():
    """Review finding: tails used negative ids, and a trace may carry negative
    block ids too, so two unrelated requests could share tokens."""
    shared_with_nobody = CanonicalRequest(arrival_s=0, input_tokens=8, output_tokens=1)
    negative_block = CanonicalRequest(arrival_s=1, input_tokens=8, output_tokens=1,
                                      prefix_blocks=(-1, -2))
    batch = batch_from_trace(_trace([shared_with_nobody, negative_block]), n=2,
                             vocab_size=50_000)
    a, b = batch.prompt_token_ids
    assert a[:4] != b[:4]
