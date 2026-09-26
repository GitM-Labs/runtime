"""Registered hypotheses: derivation, applicability gating, and the proposal path.

The derivation tests check the arithmetic from first principles (bytes per cache
entry, the efficiency band) rather than the planner's own printout. The proposal
tests run the same ``autoresearch_v0`` path every other proposer uses, with a
``DictApplicator``: they show a candidate is generated and gated, not that it
wins. A performance verdict needs a measured run on hardware.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gitm.agents.autoresearch import autoresearch_v0
from gitm.agents.hypotheses import (
    H001,
    H002,
    HypothesisProposer,
    Workload,
    mxfp4_variant,
)
from gitm.agents.policy import Policy
from gitm.optimizer.apply import DictApplicator, apply_intervention
from gitm.optimizer.preconditions import GateContext
from gitm.optimizer.replay import predict_delta
from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.glm_graph import model_weight_bytes
from gitm.planner.model_catalogue import load_spec
from gitm.planner.roofline import BatchConfig, ShardingConfig
from gitm.planner.roofline import BatchConfig as _BC  # noqa: F401

from .conftest import make_kernel, make_trace


def _hw(sku: str):
    return hardware_spec_for(peak_for_sku(sku))


def _k25(sku: str = "MI355X", *, batch: int = 64, kv_len: int = 4352, serving=None) -> Workload:
    """The team's loop headline: rag 4096/512 at c=64, mid-generation cache,
    served as deploy/k8s/mi355x-kimi-loop.yaml does (AITER on)."""
    return Workload(
        "kimi-k2.5", load_spec("kimi-k2.5"), _hw(sku),
        BatchConfig(batch=batch, kv_cache_len=kv_len), ShardingConfig(tp=8),
        {"VLLM_ROCM_USE_AITER": "1"} if serving is None else serving,
    )


def _pass(spec):
    """A correctness gate that passes: the accuracy sentinel is a hardware run."""
    return None


def _prop(w: Workload, **kw) -> HypothesisProposer:
    return HypothesisProposer(w, correctness_gate=_pass, **kw)


def _ctx(**kw) -> GateContext:
    base = dict(workload="vllm-decode", dtype="bf16", hardware="AMD Instinct MI355X",
                kv_cache_len=262144, num_gpus=8, has_collective=True,
                has_interconnect=True)
    return GateContext(**{**base, **kw})


def _attention_heavy_trace():
    """A synthetic decode window where the MLA kernel is 30% of GPU time.

    The name must classify to ``attn_score_value`` or coverage is zero and the
    gate ranks the candidate at nothing. ``flash_mla`` does; whether AITER's MLA
    decode kernel does is a pre-run check in the H-002 spec."""
    return make_trace(events=[
        make_kernel("flash_mla_fwd_kernel", start_ns=0, end_ns=300),
        make_kernel("fused_moe_kernel", start_ns=300, end_ns=1000),
    ])


# ── H-002 derivation: the bf16 MLA cache ─────────────────────────────────────


def test_h002_cache_entry_halves():
    """576 elements per entry: bf16 is 1,152 B, generic fp8 is 576 B."""
    p = H002.predict(_k25())
    assert "1152 B stored vs 576 B as generic fp8" in p.arithmetic[0]


def test_h002_same_kernel_band_halves_the_core_at_every_end():
    """On MI355X (and Blackwell) the backend does not change with the cache
    dtype, so one efficiency prices both sides and the kernel time halves
    whatever that efficiency is. Only the step share moves across the band."""
    p = H002.predict(_k25())
    assert p.op_delta == pytest.approx((0.5, 0.5, 0.5))
    lo, mean, hi = p.step_delta
    assert lo < mean < hi
    assert mean == pytest.approx(0.111, abs=0.001)


def test_h002_switch_band_on_hopper():
    """On H200 fp8 moves the core from FA-MLA to FlashMLA, so each end of the
    band is independent: lo keeps (1/0.95 - 0.5/0.55) of the old kernel time."""
    lo, mean, hi = H002.predict(_k25("H200", batch=32, kv_len=8192)).op_delta
    assert lo == pytest.approx(1 - (0.5 / 0.55) / (1 / 0.95))
    assert mean == pytest.approx(0.5)
    assert hi == pytest.approx(1 - (0.5 / 0.95) / (1 / 0.55))


def test_h002_step_band_uses_one_basis_per_end():
    """The baseline at each end is the unchanged rest at mid-band plus the core
    at that end's efficiency, so the saving and its denominator agree."""
    w = _k25("H200", batch=32, kv_len=8192)
    p = H002.predict(w)
    rest = (p.baseline_step_s - 3.838e-3) / 0.75
    assert p.step_delta[2] == pytest.approx(
        p.recoverable_s[2] / (rest + 3.838e-3 / 0.55), rel=1e-3)


def test_h002_saving_scales_with_cache_read_and_is_flat_in_tp():
    """The replicated latent: tokens read per step scale the saving, TP does not."""
    base = H002.predict(_k25())
    double = H002.predict(_k25(batch=128))
    assert double.total_overhead_s == pytest.approx(2 * base.total_overhead_s, rel=1e-6)
    tp4 = H002.predict(replace(_k25(), sharding=ShardingConfig(tp=4)))
    assert tp4.total_overhead_s == pytest.approx(base.total_overhead_s)


@pytest.mark.parametrize(
    ("workload", "reason"),
    [
        (lambda: _k25(serving={"kv_cache_dtype": "fp8", "VLLM_ROCM_USE_AITER": "1"}),
         "already storing an fp8"),
        # The prediction is for ROCM_AITER_MLA; without AITER, ROCm selects
        # TRITON_MLA (platforms/rocm.py:324-326), a kernel this does not price.
        (lambda: _k25(serving={}), "VLLM_ROCM_USE_AITER"),
        # K2.6-NVFP4 declares a static fp8 kv_cache_scheme, which vLLM resolves
        # 'auto' to (utils/torch_utils.py:262-342), so its default is fp8.
        (lambda: replace(_k25(), spec=load_spec("kimi-k2.6")), "already storing an fp8"),
        (lambda: replace(_k25(), hw=_hw("A100")), "no fp8 MLA decode backend"),
        (lambda: replace(_k25(), batch=BatchConfig(batch=0, prefill_tokens=4096)),
         "decode-only"),
    ],
)
def test_h002_applicability_says_why_not(workload, reason):
    assert reason in H002.applies(workload())


# ── H-001 derivation: the 4-bit scale stream ─────────────────────────────────


def test_h001_variant_matches_the_amd_checkpoint_size():
    """amd/Kimi-K2.6-MXFP4 is 558,995,180,568 B against NVFP4's 595,148,192,736.
    Routed experts at -1/32 B/weight plus shared expert and dense layer 0 at
    bf16 -> MXFP4 predict the 36.15 GB gap to within 0.1%."""
    k26 = load_spec("kimi-k2.6")
    gap = model_weight_bytes(k26) - model_weight_bytes(mxfp4_variant(k26))
    assert gap == pytest.approx(595_148_192_736 - 558_995_180_568, rel=1e-3)


def test_h001_recovers_half_the_scale_stream_not_all_of_it():
    p = H001.predict(_k25(batch=128, kv_len=1024))
    line = next(a for a in p.arithmetic if a.startswith("routed scale stream"))
    stream = float(line.split("stream ")[1].split()[0])
    removed = float(line.split("removes ")[1].split()[0])
    assert removed == pytest.approx(stream / 2, rel=0.01)


def test_h001_latency_effect_is_inside_its_own_band():
    """A few percent of step against a two-kernel efficiency band of tens of
    percent: the spec therefore verdicts on MoE bytes, not on ITL alone."""
    lo, mean, hi = H001.predict(_k25(batch=128, kv_len=1024)).step_delta
    assert 0.02 < mean < 0.05
    assert lo < 0 < mean < hi


def test_h001_is_excluded_where_mxfp4_emulates():
    """Quark OCP-MX MoE emulates wherever supports_mx() is False, which is every
    CUDA part: a full bf16 dequant of every local expert per forward."""
    for sku in ("H200", "B200"):
        why = H001.applies(_k25(sku))
        assert why is not None and "emulates" in why
    assert H001.applies(_k25()) is None


# ── the proposal path ────────────────────────────────────────────────────────


def test_proposer_emits_h002_and_skips_h001_with_reasons():
    prop = _prop(_k25())
    specs = prop.propose("memory_bound")
    assert [s.name for s in specs] == ["hypothesis:H-002:kv_cache_dtype=fp8"]
    s = specs[0]
    assert (s.knob, s.value) == ("kv_cache_dtype", "fp8")
    assert s.applies_to_kernels == list(H002.kernel_scope)
    assert s.expected_delta_mean == pytest.approx(0.5)
    assert s.source == H002.doc and s.safety.tier == "moderate"
    assert dict(prop.skipped)["H-001"].startswith("not proposable")
    assert set(prop.predictions) == {"H-002"}


@pytest.mark.parametrize(
    ("cls", "target", "noise", "reason"),
    [
        ("compute_bound", None, 0.0, "targets memory_bound"),
        ("memory_bound", "moe_routed", 0.0, "largest residual is moe_routed"),
        ("memory_bound", None, 0.20, "under the 20.0% noise floor"),
    ],
)
def test_proposer_gates_before_emitting(cls, target, noise, reason):
    prop = _prop(_k25(), noise_floor=noise)
    assert prop.propose(cls, target_op=target) == []
    assert reason in dict(prop.skipped)["H-002"]


def test_proposer_state_describes_only_the_latest_call():
    prop = _prop(_k25())
    prop.propose("memory_bound")
    prop.propose("compute_bound")
    assert prop.predictions == {}


def test_proposer_emits_nothing_for_a_checkpoint_already_on_fp8():
    prop = _prop(replace(_k25("H200"), spec=load_spec("kimi-k2.6")))
    assert prop.propose("memory_bound") == []
    assert "already storing an fp8" in dict(prop.skipped)["H-002"]


def test_candidate_passes_the_existing_gate_and_is_measured():
    config = {"kv_cache_dtype": "auto"}
    results = autoresearch_v0(
        _attention_heavy_trace(), "memory_bound",
        applicator=DictApplicator(config, measure_fn=lambda spec: 0.12),
        policy=Policy(), proposer=_prop(_k25()), ctx=_ctx(),
    )
    assert len(results) == 1 and results[0].applicable
    assert config["kv_cache_dtype"] == "fp8"
    # predict_delta = coverage (0.3) x the covered-op mean (0.5).
    assert results[0].predicted_delta == pytest.approx(0.3 * 0.5, abs=1e-3)


@pytest.mark.parametrize(
    ("ctx", "why"),
    [
        (_ctx(hardware="NVIDIA A100-SXM4-80GB"), "hardware"),
        (_ctx(dtype="fp16"), "dtype"),
        (_ctx(workload="hft-lob"), "workload"),
    ],
)
def test_candidate_is_rejected_by_the_existing_gate_off_its_scope(ctx, why):
    config = {"kv_cache_dtype": "auto"}
    results = autoresearch_v0(
        _attention_heavy_trace(), "memory_bound",
        applicator=DictApplicator(config, measure_fn=lambda spec: 0.12),
        policy=Policy(), proposer=_prop(_k25()), ctx=ctx,
    )
    assert len(results) == 1 and not results[0].applicable
    assert results[0].rejected_reason.startswith("not_applicable")
    assert why in results[0].rejected_reason.lower()
    assert config == {"kv_cache_dtype": "auto"}, "a gated candidate never touches config"


def test_a_measured_loss_rolls_back():
    config = {"kv_cache_dtype": "auto"}
    results = autoresearch_v0(
        _attention_heavy_trace(), "memory_bound",
        applicator=DictApplicator(config, measure_fn=lambda spec: -0.02),
        policy=Policy(), proposer=_prop(_k25()), ctx=_ctx(),
    )
    assert results[0].rolled_back and config == {"kv_cache_dtype": "auto"}


# ── correctness gate and replay scope ────────────────────────────────────────


def test_h002_is_not_emitted_without_a_correctness_gate():
    """The live keep gate measures throughput only (optimizer/apply.py), so a
    lever that can change model output needs its own gate wired first."""
    prop = HypothesisProposer(_k25())
    assert prop.propose("memory_bound") == []
    assert "requires a correctness gate" in dict(prop.skipped)["H-002"]


def test_a_faster_but_wrong_candidate_is_rolled_back():
    """Throughput says +12%; the accuracy sentinel fails. The gate rides on the
    emitted spec, so a caller passing a plain applicator cannot bypass it:
    apply_intervention restores and the reason lands in the result."""
    prop = HypothesisProposer(_k25(), correctness_gate=lambda spec: "gsm8k -3.1 pts")
    config = {"kv_cache_dtype": "auto"}
    results = autoresearch_v0(
        _attention_heavy_trace(), "memory_bound",
        applicator=DictApplicator(config, measure_fn=lambda spec: 0.12),
        policy=Policy(), proposer=prop, ctx=_ctx(),
    )
    r = results[0]
    assert r.rolled_back and r.measured_delta == pytest.approx(0.12)
    assert "correctness gate failed: gsm8k -3.1 pts" in (r.apply_error or "")
    assert config == {"kv_cache_dtype": "auto"}


def test_the_gate_is_on_the_spec_not_the_proposer():
    """apply_intervention alone, no autoresearch, no wrapper: the spec's gate
    still runs before keep, and a passing gate keeps the measured win."""
    calls: list[str] = []

    def gate(spec):
        calls.append(spec.knob)
        return "needle recall 71%"

    spec = _prop(_k25()).propose("memory_bound")[0]
    assert spec.correctness_gate is not None
    assert "correctness_gate" not in spec.model_dump()
    failing = spec.model_copy(update={"correctness_gate": gate})
    config = {"kv_cache_dtype": "auto"}
    res = apply_intervention(failing, DictApplicator(config, measure_fn=lambda s: 0.3))
    assert calls == ["kv_cache_dtype"] and res.rolled_back
    assert config == {"kv_cache_dtype": "auto"}
    res = apply_intervention(spec, DictApplicator(config, measure_fn=lambda s: 0.3))
    assert not res.rolled_back and config["kv_cache_dtype"] == "fp8"


def test_h002_replay_credits_only_mla_decode_kernels():
    """classify_op files reshape_and_cache and slot_mapping under attn_score_value
    as well; the fp8 cache does not halve those, so the spec's kernel scope must
    leave them out of the replay's coverage."""
    spec = _prop(_k25()).propose("memory_bound")[0]
    insert_only = make_trace(events=[
        make_kernel("reshape_and_cache_kernel", start_ns=0, end_ns=500),
        make_kernel("slot_mapping_kernel", start_ns=500, end_ns=1000),
    ])
    assert predict_delta(insert_only, spec) == 0.0
    assert predict_delta(_attention_heavy_trace(), spec) == pytest.approx(0.15, abs=1e-3)


# ── Jalon's review: regression tests ─────────────────────────────────────────


@pytest.mark.parametrize("exc", [TimeoutError("lm-eval hung"), ConnectionError("server gone"),
                                 RuntimeError("harness crashed")])
def test_a_crashing_gate_rolls_back_instead_of_leaving_the_change_applied(exc):
    """Finding 1. The gate is a live benchmark and can raise. Before the fix the
    exception escaped apply_intervention after `apply` had already run, so the
    fp8 cache stayed on with nobody having judged it."""
    def gate(spec):
        raise exc

    spec = _prop(_k25()).propose("memory_bound")[0].model_copy(update={"correctness_gate": gate})
    config = {"kv_cache_dtype": "auto"}
    res = apply_intervention(spec, DictApplicator(config, measure_fn=lambda s: 0.3))
    assert res.rolled_back and res.applied
    assert res.measured_delta == pytest.approx(0.3)
    assert "correctness gate crashed" in (res.error or "") and str(exc) in (res.error or "")
    assert config == {"kv_cache_dtype": "auto"}


def test_a_crashing_gate_is_reported_through_autoresearch():
    prop = HypothesisProposer(_k25(), correctness_gate=lambda s: (_ for _ in ()).throw(TimeoutError("x")))
    config = {"kv_cache_dtype": "auto"}
    results = autoresearch_v0(
        _attention_heavy_trace(), "memory_bound",
        applicator=DictApplicator(config, measure_fn=lambda s: 0.12),
        policy=Policy(), proposer=prop, ctx=_ctx(),
    )
    assert results[0].rolled_back and "crashed" in (results[0].apply_error or "")
    assert config == {"kv_cache_dtype": "auto"}


def test_h002_refuses_sparse_mla():
    """Finding 4. GLM-5.2 has kv_lora_rank > 0 too, but its DSA indexer puts an
    fp8 cache on the fp8_ds_mla layout (656 B, not 576) and a sparse backend."""
    glm = replace(_k25("H200", batch=32, kv_len=8192), spec=load_spec("glm-5.2"))
    why = H002.applies(glm)
    assert why is not None and "sparse MLA" in why


@pytest.mark.parametrize(
    ("serving", "fragment"),
    [
        ({"VLLM_ROCM_USE_AITER": "1", "attention_backend": "TRITON_MLA"}, "not the pair"),
        ({"VLLM_ROCM_USE_AITER": "1", "kv_cache_dtype": "fp8_e5m2"}, "already storing an fp8"),
        ({"VLLM_ROCM_USE_AITER": "1", "kv_cache_dtype": "fp32"}, "neither bf16/fp16 nor fp8"),
    ],
)
def test_h002_refuses_configs_the_derivation_does_not_price(serving, fragment):
    assert fragment in H002.applies(_k25(serving=serving))


def test_h002_honours_an_explicit_backend_inside_the_priced_pair():
    """On H200 an explicit FLASHMLA with a bf16 cache is arm B of the spec: the
    fp8 switch then keeps the kernel, so the band is the same-kernel one."""
    forced = _k25("H200", batch=32, kv_len=8192, serving={"attention_backend": "FLASHMLA"})
    assert H002.applies(forced) is None
    p = H002.predict(forced)
    assert p.op_delta == pytest.approx((0.5, 0.5, 0.5))
    default = H002.predict(_k25("H200", batch=32, kv_len=8192))
    assert default.op_delta[0] < 0.5 < default.op_delta[2]


@pytest.mark.parametrize("case", ["sparse_mla", "unsupported_backend", "unsupported_cache"])
def test_h002_unsupported_configs_never_reach_the_proposal_path(case):
    workload = _k25()
    if case == "sparse_mla":
        workload = replace(workload, spec=load_spec("glm-5.2"))
    elif case == "unsupported_backend":
        workload = _k25(serving={"VLLM_ROCM_USE_AITER": "1", "attention_backend": "TRITON_MLA"})
    else:
        workload = _k25(serving={"VLLM_ROCM_USE_AITER": "1", "kv_cache_dtype": "fp32"})
    proposer = _prop(workload, hypotheses=(H002,))

    assert proposer.propose("memory_bound") == []
    assert proposer.predictions == {}
    assert proposer.skipped == [("H-002", H002.applies(workload))]
