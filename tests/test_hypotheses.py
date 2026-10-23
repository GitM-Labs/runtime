"""Registered hypotheses: derivation, applicability gating, and the proposal path.

The derivation tests check the arithmetic from first principles (bytes per cache
entry, the efficiency band) rather than the planner's own printout. The proposal
tests run the same ``autoresearch_v0`` path every other proposer uses, with a
``DictApplicator``: they show a candidate is generated and gated, not that it
wins. A performance verdict needs a measured run on hardware.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

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


#: The loop's rag and chat request lengths (prompt, output), keyed by their
#: mid-generation cached tokens, prompt + output // 2 (predict_sweep.py).
_LOOP_LENGTHS = {4352: (4096, 512), 1152: (1024, 256)}


def _k25(sku: str = "MI355X", *, batch: int = 64, kv_len: int = 4352, serving=None,
         lengths="loop") -> Workload:
    """The team's loop headline: rag 4096/512 at c=64, mid-generation cache,
    served as deploy/k8s/mi355x-kimi-loop.yaml does (AITER on), loaded by
    vllm bench serve (every request the same lengths)."""
    return Workload(
        "kimi-k2.5", load_spec("kimi-k2.5"), _hw(sku),
        BatchConfig(batch=batch, kv_cache_len=kv_len), ShardingConfig(tp=8),
        {"VLLM_ROCM_USE_AITER": "1"} if serving is None else serving,
        _LOOP_LENGTHS.get(kv_len) if lengths == "loop" else lengths,
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


#: A real MoE kernel to fill the rest of a window: the INT4 routed experts on
#: the MI355X loop (vLLM v0.29.0 model_executor/layers/fused_moe/fused_moe.py:65).
_MOE = "fused_moe_kernel_gptq_awq"


def _attention_heavy_trace():
    """A decode window where the MLA decode kernel is 30% of GPU time.

    ``_mla_gluon`` is what the bf16 baseline runs on the MI355X loop at 8 heads
    per rank (AITER v0.1.19 aiter/ops/triton/gluon/mla_gluon.py:57, selected by
    vLLM v0.29.0 rocm_aiter_mla.py:1334-1353). Replay runs on the baseline
    trace, so this is the name the scope has to match."""
    return make_trace(events=[
        make_kernel("_mla_gluon", start_ns=0, end_ns=300),
        make_kernel(_MOE, start_ns=300, end_ns=1000),
    ])


# ── H-002 derivation: the bf16 MLA cache ─────────────────────────────────────

#: The attention-core floor the planner prices: sequences x cached entries x
#: 1,152 B x 61 layers over HBM bandwidth (gitm/planner/context.py: MI355X
#: 8.0 TB/s, H200 4.8 TB/s).
_ATTN_RAG64 = 64 * 4352 * 1152 * 61 / 8.0e12
_ATTN_H200 = 32 * 8192 * 1152 * 61 / 4.8e12
#: The kernels the fp8 arm runs and the bf16 arm does not, at the planner's 2 us
#: launch floor, put on the measured basis (/ 0.75) like the rest of the step.
#: MI355X at 8 heads per rank: query concat+quant, 8->16 head pad and
#: mla_reduce_v1 per layer, plus get_mla_metadata_v1 once per step. H200's
#: default FA-MLA arm: the concat+quant alone, because FA-MLA unpacks the bf16
#: query tuple (vLLM v0.27.1 flashattn_mla.py:328-329).
#: Review round 3: the fp8 arm's asm stage 1 also writes split-KV partials,
#: each tile 16 heads x (512 fp32 output + 1 fp32 LSE) (AITER v0.1.19
#: v1_2_device.cuh:121-208; aiter/mla.py:607-616). Review round 4: the 64
#: sequences decode in lockstep through 4,097 to 4,607 cached tokens, and AITER's
#: schedule makes 256 to 317 tiles across those 511 steps, 153,472 in all (the
#: reviewer's literal transcription of phase 1 agrees at every length). The
#: reduce reads them back plus its bf16 output, 1.36 us at the mean, under the
#: 2 us launch it is already charged.
_TILES_RAG64 = 153472 / 511
_PARTIALS_RAG64 = _TILES_RAG64 * 16 * (512 * 4 + 4)
#: Review round 4: the launches sit on the stream and are charged at mid
#: efficiency at every end; the writes are stage 1's own traffic, so they are
#: priced with the fp8 kernel, at its efficiency at each end.
_LAUNCHES_MI355X = (3 * 61 + 1) * 2e-6
_WRITES_RAG64 = 61 * _PARTIALS_RAG64 / 8.0e12
_EXTRA_MI355X = (_LAUNCHES_MI355X + _WRITES_RAG64) / 0.75  # at the mean
_EXTRA_H200 = 1 * 61 * 2e-6 / 0.75
#: The covered-op mean the emitted spec carries at rag c=64: half the core,
#: less the extra kernels as a share of the core (both measured at 0.75).
_OP_MEAN_RAG64 = 0.5 - _EXTRA_MI355X * 0.75 / _ATTN_RAG64


def _line(p, prefix: str) -> str:
    return next(a for a in p.arithmetic if a.startswith(prefix))


def test_h002_cache_entry_halves():
    """576 elements per entry: bf16 is 1,152 B, generic fp8 is 576 B."""
    p = H002.predict(_k25())
    assert "1152 B stored vs 576 B as generic fp8" in _line(p, "cache entry")


def test_h002_cdna4_is_a_kernel_switch_not_a_same_kernel_case():
    """The ROCM_AITER_MLA backend serves both dtypes, but at 8 heads per rank
    the bf16 cache runs AITER's Gluon kernel and fp8 runs an asm kernel (vLLM
    v0.29.0 rocm_aiter_mla.py:1334-1353). So each end of the band is
    independent, as on Hopper. The cache bytes still halve at the mean; the
    extra kernels come off that, scaled to the core's time at each end. Stage
    1's partial writes ride with the fp8 kernel at its own efficiency."""
    p = H002.predict(_k25())

    def op(e_old, e_new):
        new = (_ATTN_RAG64 / 2 + _WRITES_RAG64) / e_new + _LAUNCHES_MI355X / 0.75
        return 1 - new / (_ATTN_RAG64 / e_old)

    assert p.op_delta == pytest.approx((op(0.95, 0.55), op(0.75, 0.75), op(0.55, 0.95)),
                                       rel=1e-6)
    assert "different kernel" in _line(p, "backend")


@pytest.mark.parametrize(
    ("batch", "kv_len", "registered"),
    [
        (64, 4352, (-0.0197, 0.0708, 0.1632)),  # rag c=64, the headline: lo regresses
        (64, 1152, (-0.0432, -0.0123, 0.0217)),  # chat c=64: the mean regresses
        (128, 4352, (0.0035, 0.1301, 0.2515)),   # rag c=128
    ],
)
def test_h002_charges_the_fp8_only_kernels_at_all_three_ends(batch, kv_len, registered):
    """E66 and review round 2. At 8 heads per rank the fp8 arm quantises and
    pads the query and reduces its split-KV partials every layer, and builds
    AITER's schedule once per step; the bf16 Gluon arm does none of it. That is
    (3 x 61 + 1) x 2 us = 0.368 ms of floor, plus stage 1's partial writes
    (review round 3; 0.075 ms at the headline). The launches sit on the
    stream with nothing to hide them, 0.491 ms at mid efficiency at every end;
    the writes are stage 1's traffic and move with the fp8 kernel's efficiency
    (review round 4). The tile count is each row's lockstep-window mean. The
    registered MI355X numbers."""
    p = H002.predict(_k25(batch=batch, kv_len=kv_len))
    assert p.step_delta == pytest.approx(registered, abs=5e-5)
    assert "= 0.368 ms floor" in _line(p, "fp8-only kernels: 3 per layer")
    assert "kernels 0.368 ms / 0.75 = 0.491 ms at lo, mean and hi" in _line(p, "fp8-only total")


def test_h002_predicted_step_is_the_fp8_floor_plus_its_extra_kernels():
    """Prediction.predicted_step_s is a floor like baseline_step_s: the fp8
    cache halves the core's floor and the extra kernels add their launch
    floor, 184 x 2 us, plus stage 1's partial writes."""
    p = H002.predict(_k25())
    assert p.predicted_step_s == pytest.approx(
        p.baseline_step_s - _ATTN_RAG64 / 2 + _EXTRA_MI355X * 0.75, rel=1e-9)


# B200's bandwidth is read from the planner, not written here. This line used
# to say 8.0e12: #186 (ENG-59) wrote it while the planner priced B200 at
# GB200's 8.0 TB/s, and #184 (ENG-57) moved the planner to HGX B200's 7.7 TB/s
# the same day. Each passed CI on its own; merged together, the prediction
# priced attention at 7.7 TB/s and this expectation at 8.0, and the B200 case
# below failed on main by 0.06 ms (3.9%, 8.0 / 7.7). Asking the planner keeps
# the two from disagreeing again when a figure changes.
_ATTN_32X8192_B200 = 32 * 8192 * 1152 * 61 / peak_for_sku("B200").peak_bw_bytes_s


@pytest.mark.parametrize(
    ("workload", "attn", "floor_s"),
    [
        # 64 heads / TP8 = 8 per rank: bf16 runs Gluon with one KV split and no
        # reduce (min_kv_seq_len = 1, vLLM v0.29.0 rocm_aiter_mla.py:1090, :1798;
        # AITER mla_gluon.py:943-945, :1080-1082) and no schedule. fp8 adds
        # concat+quant, pad and mla_reduce_v1 per layer (aiter/mla.py:684-694)
        # and get_mla_metadata_v1 per step (rocm_aiter_mla.py:1053-1077).
        # Stage 1 also writes split-KV partial tiles, 300.34 per layer averaged
        # over the rag window (256 at the aligned 4,352 tokens alone).
        (lambda: _k25(), _ATTN_RAG64, _EXTRA_MI355X * 0.75),
        # 16 per rank: asm on both sides. bf16 concatenates the query tuple
        # (rocm_aiter_mla.py:1892-1893), builds the same schedule and reduces
        # too, so nothing is extra.
        # Partials: at 64 sequences AITER moves bf16 off the persistent kernel
        # (aiter/mla.py:222-267, bf16 only) and splits each sequence 4 ways for
        # occupancy (:150-185), 256 tiles of the same 32,832 B, so the arms'
        # partials match at the aligned length. Over the rag window fp8 averages
        # 300.3 tiles, about 0.011 ms more floor per step, not priced (the doc).
        (lambda: replace(_k25(), sharding=ShardingConfig(tp=4)), _ATTN_RAG64, 0),
        # 12 per rank: 16 is not a multiple, so bf16 takes the padded asm path
        # too (rocm_aiter_mla.py:1353): same concat, pad, schedule and reduce.
        (lambda: replace(_k25(), spec=replace(load_spec("kimi-k2.5"), n_heads=48),
                         sharding=ShardingConfig(tp=4)), _ATTN_RAG64, 0),
        # H200 default: FA-MLA unpacks the tuple (v0.27.1 flashattn_mla.py:
        # 328-329), so the fp8 arm's concat+quant is extra.
        (lambda: _k25("H200", batch=32, kv_len=8192), _ATTN_H200, 61 * 2e-6),
        # H200 pinned FLASHMLA and B200 FLASHINFER_MLA concatenate the bf16
        # tuple themselves (flashmla.py:277-278, flashinfer_mla.py:273-275).
        (lambda: _k25("H200", batch=32, kv_len=8192,
                      serving={"attention_backend": "FLASHMLA"}), _ATTN_H200, 0),
        (lambda: _k25("B200", batch=32, kv_len=8192), _ATTN_32X8192_B200, 0),
    ],
)
def test_h002_extra_kernel_count_follows_the_bf16_arms_kernel(workload, attn, floor_s):
    """The mean saving is the halved core on the measured basis (attn / 2 / 0.75)
    minus the fp8-only floor (launches x 2 us, plus partial writes where the
    arms' schedules differ), also on the measured basis (/ 0.75)."""
    p = H002.predict(workload())
    charged = attn / 2 / 0.75 - p.recoverable_s[1]
    assert charged == pytest.approx(floor_s / 0.75, rel=1e-3, abs=1e-12)


def test_h002_switch_band_on_hopper():
    """On H200 fp8 moves the core from FA-MLA to FlashMLA, so each end of the
    band is independent: lo keeps (1/0.95 - 0.5/0.55) of the old kernel time,
    less the query concat+quant at that end's scale."""
    lo, mean, hi = H002.predict(_k25("H200", batch=32, kv_len=8192)).op_delta
    x = _EXTRA_H200 / _ATTN_H200
    assert lo == pytest.approx(1 - (0.5 / 0.55) / (1 / 0.95) - x * 0.95, rel=1e-4)
    assert mean == pytest.approx(0.5 - x * 0.75, rel=1e-4)
    assert hi == pytest.approx(1 - (0.5 / 0.95) / (1 / 0.55) - x * 0.55, rel=1e-4)


@pytest.mark.parametrize("sku", ["MI355X", "H200", "B200"])
def test_every_prediction_names_the_gpu_it_was_priced_on(sku):
    w = _k25(sku, batch=32, kv_len=8192)
    p = H002.predict(w)
    assert p.hardware == sku and p.arithmetic[0].startswith(f"{sku}, TP8:")
    assert H001.predict(_k25(batch=128, kv_len=1024)).hardware == "MI355X"
    spec = _prop(_k25()).propose("memory_bound")[0]
    assert spec.applicability.other.startswith("kimi-k2.5 on MI355X;")


def test_h002_step_band_uses_one_basis_per_end():
    """The baseline at each end is the unchanged rest at mid-band plus the core
    at that end's efficiency, so the saving and its denominator agree."""
    w = _k25("H200", batch=32, kv_len=8192)
    p = H002.predict(w)
    rest = (p.baseline_step_s - 3.838e-3) / 0.75
    assert p.step_delta[2] == pytest.approx(
        p.recoverable_s[2] / (rest + 3.838e-3 / 0.55), rel=1e-3)


def test_h002_cache_read_scales_with_tokens_and_is_flat_in_tp():
    """The replicated latent: tokens read per step scale the cache-read
    overhead, TP does not. The extra kernels do follow TP: at TP4 (16 heads per
    rank) the bf16 arm already runs the padded asm path, so TP8's 184 extra
    launches and its partial writes are its whole difference from TP4 at the
    mean."""
    base = H002.predict(_k25())
    double = H002.predict(_k25(batch=128))
    assert double.total_overhead_s == pytest.approx(2 * base.total_overhead_s, rel=1e-6)
    tp4 = H002.predict(replace(_k25(), sharding=ShardingConfig(tp=4)))
    assert tp4.total_overhead_s == pytest.approx(base.total_overhead_s)
    assert tp4.recoverable_s[1] - base.recoverable_s[1] == pytest.approx(
        _EXTRA_MI355X, rel=1e-6)


@pytest.mark.parametrize(
    ("workload", "reason"),
    [
        (lambda: _k25(serving={"kv_cache_dtype": "fp8", "VLLM_ROCM_USE_AITER": "1"}),
         "already storing an fp8"),
        # The prediction is for ROCM_AITER_MLA; without AITER, ROCm selects
        # TRITON_MLA (vLLM v0.29.0 platforms/rocm.py:469-479), a kernel this does
        # not price.
        (lambda: _k25(serving={}), "VLLM_ROCM_USE_AITER"),
        # K2.6-NVFP4 declares a static fp8 kv_cache_scheme, which vLLM resolves
        # 'auto' to (vLLM v0.29.0 utils/torch_utils.py:452-467), so its default
        # is fp8.
        (lambda: replace(_k25(), spec=load_spec("kimi-k2.6")), "already storing an fp8"),
        (lambda: replace(_k25(), hw=_hw("A100")), "no fp8 MLA decode backend"),
        (lambda: replace(_k25(lengths=None), batch=BatchConfig(batch=0, prefill_tokens=4096)),
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
    assert s.expected_delta_mean == pytest.approx(_OP_MEAN_RAG64, abs=1e-4)
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
    # predict_delta = coverage (0.3) x the covered-op mean.
    assert results[0].predicted_delta == pytest.approx(0.3 * _OP_MEAN_RAG64, abs=1e-3)


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
    """classify_op files cache inserts under attn_score_value as well; the fp8
    cache does not halve those, so the spec's kernel scope must leave them out
    of the replay's coverage. Names from vLLM v0.29.0
    csrc/libtorch_stable/cache_kernels.cu:261 and :409."""
    spec = _prop(_k25()).propose("memory_bound")[0]
    insert_only = make_trace(events=[
        make_kernel("reshape_and_cache_kernel", start_ns=0, end_ns=500),
        make_kernel("concat_and_cache_mla_kernel", start_ns=500, end_ns=1000),
    ])
    assert predict_delta(insert_only, spec) == 0.0
    assert predict_delta(_attention_heavy_trace(), spec) == pytest.approx(
        0.3 * _OP_MEAN_RAG64, abs=1e-3)


@pytest.mark.parametrize(
    ("kernel", "credited"),
    [
        # MI355X bf16, 8 heads per rank: the loop's baseline. AITER v0.1.19
        # aiter/ops/triton/gluon/mla_gluon.py:57, reducer at :723.
        ("_mla_gluon", True),
        ("_mla_softmax_reducev_kernel", False),
        # MI355X bf16, 16 heads per rank: non-persistent asm at batch >= 32
        # (aiter/mla.py:245-267; csrc/py_itfs_cu/asm_mla.cu:875-878 ->
        # hsa/gfx950/mla/mla_asm.csv:5), reduced by aiter/mla.py:21; persistent
        # below 32 (asm_mla.cu:974-976 -> csv:43).
        ("_ZN5aiter39mla_dec_stage1_bf16_a16w16_subQ16_mqa16E", True),
        ("_fwd_kernel_stage2_asm", False),
        ("_ZN5aiter41mla_a16w16_qh64_qseqlen1_gqaratio64_v3_psE", True),
        # MI355X fp8: csv:9, then mla_reduce_v1 (csrc/kernels/mla/reduce.cu:687).
        ("_ZN5aiter36mla_a8w8_qh16_qseqlen1_gqaratio16_psE", True),
        ("kn_mla_reduce_v1_ps", False),
        # H200 FlashMLA (a8f794d): bf16 csrc/sm90/decode/dense/splitkv_mla.cuh:12,
        # :961; fp8 csrc/extension/sm90/dense_fp8/flash_fwd_mla_kernel.h:162,
        # :547; combines at flash_fwd_mla_kernel.h:593 and
        # csrc/smxx/decode/combine/combine.cu:16, :20.
        ("sm90::flash_fwd_splitkv_mla_kernel", True),
        ("flash::flash_fwd_splitkv_mla_kernel", True),
        ("flash::flash_fwd_splitkv_mla_combine_kernel", False),
        ("smxx::decode::flash_fwd_mla_combine_kernel", False),
        # B200 CUTLASS_MLA (vLLM v0.27.1 csrc/libtorch_stable/attention/mla/
        # cutlass_sm100_mla/kernel/): sm100_fmha_mla_tma_warpspecialized.hpp:53,
        # :70; sm100_fmha_mla_reduction.hpp:44, :55.
        ("cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized", True),
        ("cutlass::fmha::kernel::Sm100FmhaMlaReductionKernel", False),
        # H200's default bf16 arm, FA3 (vllm-flash-attn 28e862d
        # hopper/flash_fwd_kernel_sm90.h:28). Prefill runs it too, so a name
        # match would credit prefill: a known gap that needs op identity.
        ("flash::FlashAttnFwdSm90", False),
        # AITER's Triton split-KV reducer (aiter/ops/triton/_triton_kernels/
        # attention/mla.py:663). Its trace name is this base plus constexpr
        # suffixes (:650-660; utils/_triton/kernel_repr.py:28-56), so a
        # "mla_decode" needle would credit it.
        ("_mla_decode_fwd_reduce_kernel", False),
    ],
)
def test_h002_replay_scope_matches_real_baseline_decode_kernels(kernel, credited):
    """E65. Each name is a real kernel from source, 10% of a window whose rest
    is the MoE kernel. Credited names get 10% of the covered-op mean."""
    spec = _prop(_k25()).propose("memory_bound")[0]
    trace = make_trace(events=[
        make_kernel(kernel, start_ns=0, end_ns=100),
        make_kernel(_MOE, start_ns=100, end_ns=1000),
    ])
    expected = 0.1 * spec.expected_delta_mean if credited else 0.0
    assert predict_delta(trace, spec) == pytest.approx(expected, abs=1e-6)


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
        # Verify steps carry 1 + k query tokens per sequence, which turns
        # Gluon decode off (vLLM v0.29.0 rocm_aiter_mla.py:1339-1340). EngineArgs
        # names (engine/arg_utils.py:692-695) and gitm's knob name.
        ({"VLLM_ROCM_USE_AITER": "1",
          "speculative_config": {"method": "ngram", "num_speculative_tokens": 5}},
         "speculative decoding"),
        ({"VLLM_ROCM_USE_AITER": "1", "spec_tokens": 3}, "speculative decoding"),
        ({"VLLM_ROCM_USE_AITER": "1", "num_speculative_tokens": 5}, "speculative decoding"),
        # DCP multiplies the decode heads per rank (rocm_aiter_mla.py:1426-1428).
        ({"VLLM_ROCM_USE_AITER": "1", "decode_context_parallel_size": 2},
         "decode_context_parallel_size=2"),
        # DP groups step together on a combined token count (forward_context.py:80-99).
        ({"VLLM_ROCM_USE_AITER": "1", "data_parallel_size": 2}, "data_parallel_size=2"),
    ],
)
def test_h002_refuses_configs_the_derivation_does_not_price(serving, fragment):
    assert fragment in H002.applies(_k25(serving=serving))


def test_h002_refuses_a_speculative_batch_and_accepts_the_off_values():
    spec_batch = replace(_k25(), batch=BatchConfig(batch=64, kv_cache_len=4352,
                                                   speculative_tokens=3))
    assert "speculative decoding" in H002.applies(spec_batch)
    dp = replace(_k25(), sharding=ShardingConfig(tp=4, dp=2))
    assert "data_parallel_size=2" in H002.applies(dp)
    off = {"VLLM_ROCM_USE_AITER": "1", "speculative_config": None,
           "num_speculative_tokens": 0, "decode_context_parallel_size": 1,
           "data_parallel_size": 1}
    assert H002.applies(_k25(serving=off)) is None


def test_h002_honours_an_explicit_backend_inside_the_priced_pair():
    """E44. On H200 an explicit FLASHMLA with a bf16 cache is the spec's
    pinned-backend arm. It is accepted, but it is not a same-kernel case:
    FlashMLA runs sm90::flash_fwd_splitkv_mla_kernel (_flashmla_C) for bf16 and
    flash::flash_fwd_splitkv_mla_kernel (_flashmla_extension_C) for fp8, so the
    band is the switch band, the same as the default pair's. On CDNA4 forcing
    ROCM_AITER_MLA changes nothing: it is the default, and the AITER kernel
    still differs per dtype."""
    forced = _k25("H200", batch=32, kv_len=8192, serving={"attention_backend": "FLASHMLA"})
    assert H002.applies(forced) is None
    p = H002.predict(forced)
    # bf16 FlashMLA concatenates the query too (v0.27.1 flashmla.py:277-278),
    # so nothing is extra and the op band is the bare switch band.
    assert p.op_delta == pytest.approx(
        (1 - (0.5 / 0.55) / (1 / 0.95), 0.5, 1 - (0.5 / 0.95) / (1 / 0.55)), rel=1e-6)
    assert "different kernel" in _line(p, "backend FLASHMLA -> FLASHMLA")
    rocm_forced = _k25(serving={"VLLM_ROCM_USE_AITER": "1", "attention_backend": "ROCM_AITER_MLA"})
    assert H002.predict(rocm_forced).op_delta == pytest.approx(H002.predict(_k25()).op_delta)


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        # E36: vLLM v0.29.0 needs both flags for ROCM_AITER_MLA
        # (_aiter_ops.py:1953-1956); with AITER_MLA off the list is TRITON_MLA.
        ({"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MLA": "0"}, "VLLM_ROCM_USE_AITER_MLA is off"),
        ({"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MLA": "false"}, "VLLM_ROCM_USE_AITER_MLA is off"),
        ({"VLLM_ROCM_USE_AITER": "true", "VLLM_ROCM_USE_AITER_MLA": "no"}, "VLLM_ROCM_USE_AITER_MLA is off"),
        # vLLM reads only "1" and "true" as on (envs.py:1235-1237, :1282-1283).
        ({"VLLM_ROCM_USE_AITER": "yes"}, "without VLLM_ROCM_USE_AITER=1"),
        ({"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MLA": "yes"}, "VLLM_ROCM_USE_AITER_MLA is off"),
        # The bf16 small-head kernel follows this knob; at 8 heads auto is Gluon.
        ({"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_AITER_MLA_ASM_PADDING": "asm"}, "ASM_PADDING=asm"),
    ],
)
def test_h002_refuses_cdna4_envs_that_leave_the_priced_kernels(env, fragment):
    assert fragment in H002.applies(_k25(serving=env))


@pytest.mark.parametrize(
    "env",
    [
        {"attention_backend": "ROCM_AITER_MLA"},
        {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MLA": "0",
         "attention_backend": "ROCM_AITER_MLA"},
    ],
)
def test_h002_env_refusal_stays_true_when_the_aiter_backend_is_pinned(env):
    """A pinned backend is honoured whatever the AITER flags say (vLLM v0.29.0
    platforms/rocm.py:628-642, _aiter_ops.py:134-146), so the reason cannot
    say TRITON_MLA is selected. It is still refused: the loop env the
    derivation prices sets the flag."""
    why = H002.applies(_k25(serving=env))
    assert why is not None and "pinned" in why and "selects TRITON_MLA" not in why


def test_h002_refuses_the_padding_mode_only_where_it_moves_the_bf16_kernel():
    """use_gluon_decode (vLLM v0.29.0 rocm_aiter_mla.py:1334-1353) reads the
    mode only under 16 heads: auto takes Gluon when 16 is a multiple of the
    head count, gluon takes it whenever it exists, asm never. At 8 heads per
    rank gluon and auto are the same kernel; at 6 they are not, and asm is."""
    gluon = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_AITER_MLA_ASM_PADDING": "gluon"}
    asm = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_AITER_MLA_ASM_PADDING": "asm"}
    assert H002.applies(_k25(serving=gluon)) is None
    assert "ASM_PADDING=asm moves the bf16 arm off the gluon kernel" in H002.applies(_k25(serving=asm))
    six_heads = replace(load_spec("kimi-k2.5"), n_heads=48)
    assert "ASM_PADDING=gluon moves the bf16 arm off the asm kernel" in H002.applies(
        replace(_k25(serving=gluon), spec=six_heads))
    assert H002.applies(replace(_k25(serving=asm), spec=six_heads)) is None
    # Gauntlet G11: at 16 or more heads per rank use_gluon_decode returns
    # before reading the mode (:1334-1338), so neither value changes the kernel.
    for serving in (gluon, asm):
        assert H002.applies(replace(_k25(serving=serving), sharding=ShardingConfig(tp=4))) is None


def test_h002_refuses_prefill_context_parallelism_only_where_rocm_overrides_the_graph():
    """Only ROCm's platform override turns full graphs to PIECEWISE under PCP
    (platforms/rocm.py:902-913); vLLM v0.29.0 has no such override on CUDA."""
    pcp = {"prefill_context_parallel_size": 2}
    assert "prefill_context_parallel_size=2" in H002.applies(_k25(serving={**_AITER, **pcp}))
    assert H002.applies(_k25("H200", batch=32, kv_len=8192, serving=pcp)) is None


@pytest.mark.parametrize(
    "serving",
    [
        {"VLLM_ROCM_USE_AITER": "1", "attention_config": {"backend": "TRITON_MLA"}},
        {"VLLM_ROCM_USE_AITER": "1", "attention_config": '{"backend": "TRITON_MLA"}'},
        {"VLLM_ROCM_USE_AITER": "1",
         "attention_config": {"backend_per_kind": {"mla_attention": "TRITON_MLA"}}},
    ],
)
def test_h002_reads_the_backend_pinned_through_attention_config(serving):
    """vLLM v0.29.0 copies --attention-backend into attention_config.backend
    (engine/arg_utils.py:2489-2498) and backend_per_kind overrides it for the
    MLA kind (v1/attention/selector.py:175-183)."""
    assert "not the pair" in H002.applies(_k25(serving=serving))
    aiter = {"VLLM_ROCM_USE_AITER": "1", "attention_config": {"backend": "ROCM_AITER_MLA"}}
    assert H002.applies(_k25(serving=aiter)) is None


@pytest.mark.parametrize(
    "serving",
    [
        {"VLLM_ROCM_USE_AITER": "1", "attention_backend": "auto"},
        {"VLLM_ROCM_USE_AITER": "1", "attention_config": {"backend": "Auto"}},
        {"VLLM_ROCM_USE_AITER": "1", "attention_config": '{"backend": "AUTO"}'},
    ],
)
def test_h002_reads_auto_as_no_backend_pin(serving):
    """Gauntlet G3. vLLM v0.29.0 maps "auto" to None, automatic selection
    (config/attention.py:162-164; --attention-backend goes through the same
    validator, engine/arg_utils.py:2496-2498): the default pair the
    derivation prices."""
    assert H002.applies(_k25(serving=serving)) is None
    assert H002.predict(_k25(serving=serving)).op_delta == H002.predict(_k25()).op_delta


@pytest.mark.parametrize(
    "env",
    [
        {"VLLM_ROCM_USE_AITER": "1"},  # the loop's manifest: both others at default
        {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MLA": "1"},
        {"VLLM_ROCM_USE_AITER": "True", "VLLM_ROCM_USE_AITER_MLA": "TRUE",
         "VLLM_ROCM_AITER_MLA_ASM_PADDING": "AUTO"},
    ],
)
def test_h002_accepts_the_default_aiter_mla_env(env):
    assert H002.applies(_k25(serving=env)) is None


def test_h002_is_targeted_from_a_gluon_baseline_trace():
    """Gauntlet G6. Residual targeting classifies kernels by name
    (deviation.classify_op), so the MI355X baseline's _mla_gluon must file
    under attn_score_value, or the largest residual is the MoE and H-002 is
    never proposed on the workload it was built for."""
    from gitm.agents.autoresearch import largest_residual
    from gitm.agents.hypotheses import _executed_kv, _floor
    from gitm.optimizer.monitor import residuals

    w = _k25()
    g = _floor(replace(w.spec, kv_dtype=_executed_kv(w), kv_rope_dtype=_executed_kv(w)), w)
    floor = {op: next(n.prediction.t_pred_s for n in g.nodes if n.op == op)
             for op in ("attn_score_value", "moe_routed")}
    events, t = [], 0
    for _ in range(4):
        for name, op, over in (("_mla_gluon", "attn_score_value", 3.0), (_MOE, "moe_routed", 1.2)):
            d = int(floor[op] * over * 1e9)
            events.append(make_kernel(name, start_ns=t, end_ns=t + d))
            t += d
    target = largest_residual(residuals(make_trace(events=events, duration_ns=t, vendor="amd"), g))
    assert target is not None and target.op == "attn_score_value"
    specs = _prop(w).propose("memory_bound", target_op=target.op)
    assert [s.knob for s in specs] == ["kv_cache_dtype"]


def test_h002_scope_has_no_needle_for_aiters_triton_reducer():
    """E64 again: AITER's Triton MLA decode and its reducer share the
    "mla_decode" stem (aiter/ops/triton/_triton_kernels/attention/mla.py:335,
    :663), so no needle may match the reducer's name."""
    name = "_mla_decode_fwd_reduce_kernel"
    assert [n for n in H002.kernel_scope if n in name] == []


def test_h002_replay_does_not_credit_aiter_mla_reduce():
    """E64. AITER's split-KV reduces combine fp32 partials and never read the
    cache: kn_mla_reduce_v1_ps after the asm kernel (AITER v0.1.19
    csrc/kernels/mla/reduce.cu:687) and _mla_softmax_reducev_kernel after Gluon
    (aiter/ops/triton/gluon/mla_gluon.py:723)."""
    spec = _prop(_k25()).propose("memory_bound")[0]
    reduction_only = make_trace(events=[
        make_kernel("kn_mla_reduce_v1_ps", start_ns=0, end_ns=50),
        make_kernel("_mla_softmax_reducev_kernel", start_ns=50, end_ns=100),
        make_kernel(_MOE, start_ns=100, end_ns=1000),
    ])
    assert predict_delta(reduction_only, spec) == 0.0


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



def test_h002_rejects_forced_backend_that_cannot_follow_fp8_switch():
    workload = _k25("H200", serving={"attention_backend": "FLASH_ATTN_MLA"})
    proposer = _prop(workload, hypotheses=(H002,))
    assert proposer.propose("memory_bound") == []
    assert "explicitly pinned" in dict(proposer.skipped)["H-002"]
    assert proposer.predictions == {}


def test_a_saved_and_reloaded_h002_spec_is_refused_not_applied_ungated():
    from gitm.kernels.spec import InterventionSpec

    spec = _prop(_k25()).propose("memory_bound")[0]
    reloaded = InterventionSpec.model_validate_json(spec.model_dump_json())
    assert reloaded.safety.requires_correctness_gate and reloaded.correctness_gate is None
    config = {"kv_cache_dtype": "auto"}
    res = apply_intervention(reloaded, DictApplicator(config, measure_fn=lambda s: 0.12))
    assert not res.applied and "requires a correctness gate" in (res.error or "")
    assert config == {"kv_cache_dtype": "auto"}


def test_h002_gate_deadline_is_twice_the_docs_per_arm_gate_budget():
    doc = Path(__file__).resolve().parents[1] / "docs/hypotheses/H-002-mla-kv-fp8.md"
    budget = re.search(r"\| GSM8K and needle, 2 arms \| about ([\d.]+) h \|", doc.read_text())
    per_arm_s = float(budget.group(1)) * 3600 / 2
    spec = _prop(_k25()).propose("memory_bound")[0]
    assert spec.safety.correctness_gate_timeout_s == 2 * per_arm_s


@pytest.mark.parametrize("spelling", ["bfloat16", "float16", "BF16"])
def test_h002_reads_vllms_own_bf16_and_fp16_cache_spellings(spelling):
    # vLLM v0.29.0 config/cache.py:39-42 spells the flag "bfloat16"/"float16".
    w = _k25(serving={"VLLM_ROCM_USE_AITER": "1", "kv_cache_dtype": spelling})
    assert H002.applies(w) is None


_AITER = {"VLLM_ROCM_USE_AITER": "1"}


@pytest.mark.parametrize(
    "extra",
    [
        # ModelConfig is a pydantic dataclass (vLLM v0.29.0 config/utils.py:51-74,
        # config/model.py:123-124, :241), so a string is read as pydantic reads a bool.
        {"enforce_eager": True},
        {"enforce_eager": "true"},
        {"enforce_eager": "1"},
        {"enforce_eager": "Yes"},
        # CUDAGraphMode parses a string by name, any case (config/compilation.py:
        # 870-876); NONE and PIECEWISE leave decode attention outside a graph.
        {"compilation_config": {"cudagraph_mode": "NONE"}},
        {"compilation_config": {"cudagraph_mode": "piecewise"}},
        {"compilation_config": {"cudagraph_mode": 1}},
        {"compilation_config": '{"cudagraph_mode": "PIECEWISE"}'},
        # -O0 and -O1 default cudagraph_mode to NONE and PIECEWISE (config/vllm.py:
        # 255, :278) when the user does not set it.
        {"optimization_level": 0},
        {"optimization_level": "1"},
        # FULL_AND_PIECEWISE needs piecewise compilation, so any compilation
        # mode but VLLM_COMPILE (3) is overridden to NONE (config/vllm.py:
        # 1468-1480). TORCH_COMPILE_DISABLE=1 and -O0 set mode NONE (:1390-1395,
        # :1431-1435).
        {"compilation_config": {"mode": 0}},
        {"compilation_config": {"mode": "none"}},
        {"compilation_config": {"mode": 1}},
        {"compilation_config": '{"mode": "STOCK_TORCH_COMPILE"}'},
        {"TORCH_COMPILE_DISABLE": "1"},
        {"optimization_level": 0, "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}},
        # Breakable graphs run MLA attention eagerly between the segments
        # (mla_attention.py:1319, breakable_cudagraph.py:319-322); vLLM reads
        # the flag as "1" only (envs.py:757-759).
        {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"},
        # ROCm overrides full graphs to PIECEWISE under prefill context
        # parallelism (platforms/rocm.py:902-913).
        {"prefill_context_parallel_size": 2},
        # A layerwise KV connector overrides them too (config/vllm.py:1556-1581).
        {"kv_transfer_config": {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both",
                                "kv_connector_extra_config": {"use_layerwise": True}}},
        # A batch above the largest captured size runs without a graph
        # (v1/cudagraph_dispatcher.py:272-280).
        {"max_cudagraph_capture_size": 32},
        {"compilation_config": {"max_cudagraph_capture_size": 48}},
        {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
    ],
)
def test_h002_refuses_serving_without_graph_replay_at_decode(extra):
    """Review round 3, F3. The extra kernels are priced at the planner's
    graph-replay launch (2 us); eager is nearer 5 us, which moves the band."""
    why = H002.applies(_k25(serving={**_AITER, **extra}))
    assert why is not None and "graph" in why


def test_h002_follows_vllms_capture_ceiling_rules():
    """Gauntlet G4 and G10. vLLM v0.29.0 drops explicit sizes above
    max_num_batched_tokens instead of clamping the largest (config/vllm.py:
    2096-2104); without explicit sizes it captures a grid, 8s below 256 and
    16s from 256, and truncates an explicit max_cudagraph_capture_size to it
    (:2117-2129, :2160-2182). The dispatcher sends any larger batch to NONE
    (v1/cudagraph_dispatcher.py:274-281)."""
    gap = {**_AITER, "max_num_batched_tokens": 512, "cudagraph_capture_sizes": [8, 16, 600]}
    assert "largest captured graph size 16" in H002.applies(_k25(serving=gap))
    nested = {**_AITER, "max_num_batched_tokens": 512,
              "compilation_config": {"cudagraph_capture_sizes": [8, 16, 600]}}
    assert "largest captured graph size 16" in H002.applies(_k25(serving=nested))
    wide = {**_AITER, "max_num_seqs": 512, "max_num_batched_tokens": 8192}
    assert "largest captured graph size 288" in H002.applies(
        _k25(batch=290, serving={**wide, "compilation_config": {"max_cudagraph_capture_size": 300}}))
    assert H002.applies(_k25(batch=288, serving={**wide, "max_cudagraph_capture_size": 300})) is None
    assert "largest captured graph size 96" in H002.applies(
        _k25(batch=100, serving={**wide, "max_cudagraph_capture_size": 100}))
    # The default ceiling sits on the grid too: 2 x 65 = 130 captures up to 128.
    assert "largest captured graph size 128" in H002.applies(
        _k25(batch=130, serving={**_AITER, "max_num_seqs": 65}))
    # A token budget at or under the ceiling is captured itself, on the grid or not.
    assert H002.applies(_k25(batch=300, serving={**_AITER, "max_num_batched_tokens": 300})) is None
    assert "largest captured graph size 300" in H002.applies(
        _k25(batch=301, serving={**_AITER, "max_num_batched_tokens": 300}))
    assert H002.applies(_k25(batch=100, serving={
        **_AITER, "max_num_batched_tokens": 100, "max_cudagraph_capture_size": 100})) is None
    # Under 8 the grid is 1, 2, 4 (:2120-2122).
    assert "largest captured graph size 4" in H002.applies(
        _k25(batch=6, serving={**_AITER, "max_cudagraph_capture_size": 6}))
    assert H002.applies(_k25(batch=4, serving={**_AITER, "max_cudagraph_capture_size": 6})) is None


@pytest.mark.parametrize(
    "extra",
    [
        {"enforce_eager": False},
        {"enforce_eager": "false"},
        {"enforce_eager": 0},
        {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}},
        {"compilation_config": '{"cudagraph_mode": "full_and_piecewise"}'},
        {"optimization_level": 3},
        {"compilation_config": {"mode": 3}},
        {"compilation_config": {"mode": "vllm_compile"}},
        # A full graph needs no piecewise compilation (config/vllm.py:1468-1480).
        {"compilation_config": {"mode": 1, "cudagraph_mode": "FULL"}},
        {"TORCH_COMPILE_DISABLE": "0"},
        {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
        {"prefill_context_parallel_size": 1},
        {"max_cudagraph_capture_size": 64},
        {"cudagraph_capture_sizes": [1, 8, 64, 128]},
        # The op stays disabled, so Inductor compiles it into one kernel.
        {"compilation_config": {"custom_ops": ["+quant_fp8"]}},
        {"compilation_config": {"custom_ops": ["all", "-mla_decode_concat_quant_fp8"]}},
        {"compilation_config": {"backend": "inductor"}},
    ],
)
def test_h002_accepts_serving_that_replays_decode_in_a_full_graph(extra):
    assert H002.applies(_k25(serving={**_AITER, **extra})) is None


@pytest.mark.parametrize(
    "extra",
    [
        # Enabled, the custom op is torch.cat then AITER's quantise
        # (mla_attention.py:1419-1445, input_quant_fp8.py:137-160); custom_ops
        # enables it with all or +name (custom_op.py:285-311).
        {"compilation_config": {"custom_ops": ["all"]}},
        {"compilation_config": {"custom_ops": ["+mla_decode_concat_quant_fp8"]}},
        {"compilation_config": '{"custom_ops": ["none", "+mla_decode_concat_quant_fp8"]}'},
        # vLLM adds "all" itself when compilation is off or the backend is not
        # Inductor (config/vllm.py:1444-1451). A full graph needs no
        # compilation, but mode NONE still splits the op.
        {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL"}},
        {"compilation_config": {"mode": "none", "cudagraph_mode": "full_decode_only"}},
        {"optimization_level": 0, "compilation_config": {"cudagraph_mode": "FULL"}},
        {"TORCH_COMPILE_DISABLE": "1", "compilation_config": {"cudagraph_mode": "FULL"}},
        {"compilation_config": {"backend": "eager"}},
    ],
)
def test_h002_refuses_configs_that_split_the_query_quant_into_two_kernels(extra):
    """Greptile and gauntlet round 2. The fp8-only kernel count prices the
    query concat-and-quantise as one Inductor kernel; these configs run it as
    two, 61 more launches than priced."""
    why = H002.applies(_k25(serving={**_AITER, **extra}))
    assert why is not None and why.endswith("as one compiled kernel")


def test_h002_refuses_batches_above_vllms_default_capture_ceiling():
    """With no explicit capture size vLLM v0.29.0 captures up to
    min(2 x max_num_seqs, 512), 1024 on Blackwell (config/vllm.py:2003-2016),
    then caps at max_num_batched_tokens (:2092-2093). The loop's manifest
    (max-num-seqs 256) gives 512 on MI355X."""
    assert H002.applies(_k25(batch=512)) is None
    assert "largest captured graph size 512" in H002.applies(_k25(batch=520))
    assert "largest captured graph size 128" in H002.applies(
        _k25(batch=200, serving={**_AITER, "max_num_seqs": 64}))
    assert "largest captured graph size 256" in H002.applies(
        _k25(batch=300, serving={**_AITER, "max_num_batched_tokens": 256}))
    assert H002.applies(_k25("B200", batch=520, kv_len=8192, serving={})) is None


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        # autoresearch proposes compilation_config=3 (autoresearch.py:235, :422),
        # the old -O spelling vLLM v0.29.0 no longer takes.
        ({"compilation_config": 3}, "not the dict or JSON object"),
        ({"compilation_config": "3"}, "not the dict or JSON object"),
        ({"compilation_config": [2, 3]}, "not the dict or JSON object"),
        ({"compilation_config": "-O3"}, "is not JSON"),
        ({"compilation_config": "{'mode': 0}"}, "is not JSON"),
        ({"attention_config": '"ROCM_AITER_MLA"'}, "not the dict or JSON object"),
        ({"attention_config": "[1]"}, "not the dict or JSON object"),
        ({"attention_config": ["ROCM_AITER_MLA"]}, "not the dict or JSON object"),
        ({"attention_config": "auto"}, "is not JSON"),
        # CUDAGraphMode and CompilationMode parse a name or an int value
        # (config/compilation.py:853-876), and the composite graph modes have
        # tuple values, so 4 names neither.
        ({"compilation_config": {"cudagraph_mode": ["FULL"]}}, "not a CUDAGraphMode name"),
        ({"compilation_config": {"cudagraph_mode": "full_graph"}}, "not a CUDAGraphMode name"),
        ({"compilation_config": {"cudagraph_mode": 4}}, "not a CUDAGraphMode name"),
        ({"compilation_config": {"mode": [3]}}, "not a CompilationMode name"),
        ({"compilation_config": {"mode": 4}}, "not a CompilationMode name"),
        ({"cudagraph_capture_sizes": "64"}, "not the list of ints"),
        ({"compilation_config": {"cudagraph_capture_sizes": "[1, 64]"}}, "not the list of ints"),
        ({"compilation_config": {"cudagraph_capture_sizes": ["a"]}}, "not the list of ints"),
        ({"compilation_config": {"max_cudagraph_capture_size": "abc"}}, "not the positive int"),
        ({"max_cudagraph_capture_size": 0}, "not the positive int"),
        ({"max_num_batched_tokens": "8k"}, "not the positive int"),
        ({"data_parallel_size": "auto"}, "not the positive int"),
        ({"compilation_config": {"custom_ops": 5}}, "not the list of op names"),
        ({"attention_config": {"backend_per_kind": "ROCM_AITER_MLA"}}, "not the dict"),
    ],
)
def test_h002_refuses_rather_than_crashes_on_a_malformed_serving_config(extra, fragment):
    """Gauntlet G1. A value vLLM v0.29.0 would not take is reported in
    skipped; propose() must not raise and lose every other hypothesis."""
    workload = _k25(serving={**_AITER, **extra})
    assert fragment in H002.applies(workload)
    proposer = _prop(workload)
    assert proposer.propose("memory_bound") == []
    assert fragment in dict(proposer.skipped)["H-002"]


def test_h002_refuses_a_partly_fp8_cache():
    """Review round 3, F4. --kv-cache-dtype-skip-layers keeps the named layers
    at the model dtype (vLLM v0.29.0 engine/arg_utils.py:1299, mla_attention.py:
    452-456); the derivation prices all 61 layers in fp8."""
    why = H002.applies(_k25(serving={**_AITER, "kv_cache_dtype_skip_layers": ["0"]}))
    assert why is not None and "kv_cache_dtype_skip_layers" in why
    assert H002.applies(_k25(serving={**_AITER, "kv_cache_dtype_skip_layers": []})) is None


@pytest.mark.parametrize(
    ("n_seq", "kv_len", "tiles"),
    [
        # 64 x (272 + 16) = 18,432 blocks over 256 slots: payload 72 + 16, so a
        # 272-block sequence spans exactly 4 slots and the boundaries line up.
        (64, 4352, 256),
        # One more token: 273 blocks, the boundaries no longer line up, and 57
        # of 64 sequences span 5 slots.
        (64, 4353, 313),
        (64, 1152, 256),
        (128, 4352, 256),
        # A sequence that fits one slot leaves no partial.
        (1, 16, 0),
        # 256 x (1 + 16) blocks: payload 17 + 16. After the first, every
        # sequence starts with 16 left, exactly the overhead, so it takes one
        # fresh slot whole: one fragment, no partial (v1_2_device.cuh:200-204).
        (256, 16, 0),
    ],
)
def test_aiter_partial_tiles_follow_its_persistent_schedule(n_seq, kv_len, tiles):
    """Phase 1 of AITER v0.1.19 kn_get_mla_metadata_v1_2_parallel
    (csrc/kernels/mla/metadata/v1_2_device.cuh:121-208) at vLLM's call (page
    size 1, kv_granularity 16, rocm_aiter_mla.py:1053-1077) and 256 slots."""
    from gitm.agents.hypotheses import _aiter_partial_tiles

    assert _aiter_partial_tiles(n_seq, kv_len) == tiles


def test_h002_prices_the_partials_stage_one_writes():
    p = H002.predict(_k25())
    line = _line(p, "split-KV partials")
    assert "300.34 tiles (mean over 511 decode steps with the batch at one cached length, 4097 to 4607" in line
    assert "153472 / 511, 256 to 317 per step; 4.69 per sequence" in line
    assert f"= {_PARTIALS_RAG64:.0f} B per layer" in line
    assert ("0.443 ms floor: kernels 0.368 ms / 0.75 = 0.491 ms at lo, mean and hi; "
            "stage 1 writes 0.075 ms at the fp8 kernel's efficiency: 0.137 / 0.100 / 0.079"
            ) in _line(p, "fp8-only total")


@pytest.mark.parametrize(
    ("batch", "kv_len", "tiles"),
    [
        (64, 4353, 313),  # one token past the aligned 4,352
        (65, 4352, 308),  # one sequence more than the aligned 64
    ],
)
def test_h002_prices_the_tiles_of_the_workloads_own_batch_and_length(batch, kv_len, tiles):
    """At one cache length the tile count follows the workload's sequences and
    cached tokens, not the aligned 64 x 4,352 that gives 256."""
    p = H002.predict(_k25(batch=batch, kv_len=kv_len, lengths=None))
    assert f"{tiles}.00 tiles (at {kv_len} cached tokens only" in _line(p, "split-KV partials")


@pytest.mark.parametrize(
    ("batch", "kv_len", "lengths", "total", "steps"),
    [
        (64, 1152, "loop", 70448, 255),   # chat 1024/256: 1,025 to 1,279 cached tokens
        (128, 4352, "loop", 163088, 511),  # rag at c=128
        # An unaligned prompt: 256 tiles at 4,091 tokens and 313 at 4,105, so a
        # window shifted by one step (4,090 to 4,104) would sum to 4,296.
        (64, 4098, (4090, 16), 4353, 15),
    ],
)
def test_h002_averages_the_tiles_over_the_lockstep_window(batch, kv_len, lengths, total, steps):
    """Review round 4, F1. vllm bench serve at request rate inf queues every
    request behind --max-concurrency, and every request has the same lengths,
    so a step's sequences share one cache length. The band uses the mean tile
    count over the window's decode steps, not the aligned best case (256)."""
    p = H002.predict(_k25(batch=batch, kv_len=kv_len, lengths=lengths))
    line = _line(p, "split-KV partials")
    assert f"{total} / {steps}" in line and f"{total / steps:.2f} tiles" in line


def test_h002_charges_the_reduce_past_its_launch_at_large_batches():
    """At 64 and 128 sequences mla_reduce_v1 moves less than its 2 us launch
    (1.36 and 1.57 us). At 384 it reads the partials plus 384 x 16 x 512 x 2 B
    of output in 2.357 us, and the 0.357 us past the launch is charged per layer."""
    from gitm.agents.hypotheses import _aiter_partial_tiles

    p = H002.predict(_k25(batch=384))
    tiles = sum(_aiter_partial_tiles(384, n) for n in range(4097, 4608)) / 511
    partial_b = tiles * 16 * (512 * 4 + 4)
    reduce_s = (partial_b + 384 * 16 * 512 * 2) / 8.0e12
    past = 61 * (reduce_s - 2e-6)
    assert past > 0
    attn = 384 * 4352 * 1152 * 61 / 8.0e12
    assert p.predicted_step_s == pytest.approx(
        p.baseline_step_s - attn / 2 + _LAUNCHES_MI355X + 61 * partial_b / 8.0e12 + past,
        rel=1e-9)
    assert "in 2.357 us, charged past its 2.0 us launch only: 0.022 ms floor" in _line(
        p, "split-KV partials")


def test_h002_refuses_request_lengths_that_miss_the_cache_length():
    for lengths in ((4096, 1024), (4096, 256), (4095, 512)):  # long, short, short
        with pytest.raises(ValueError, match="prompt \\+ output // 2"):
            _k25(lengths=lengths)
    with pytest.raises(ValueError, match="at least one decode step"):
        _k25(kv_len=4096, lengths=(4096, 1))


def test_h002_states_the_baseline_kernel_times_the_band_assumes():
    """Review round 3, F2. The pre-candidate check reads arm B's per-layer
    _mla_gluon time against the bf16 core's floor per layer (2.4466 ms / 61)
    at eff 0.95 / 0.75 / 0.55."""
    line = _line(H002.predict(_k25()), "bf16 core per layer")
    floor = _ATTN_RAG64 / 61
    assert f"floor {floor * 1e6:.1f} us" in line
    assert "42.2 / 53.5 / 72.9 us" in line


def test_h002_rederives_the_band_from_a_measured_baseline_kernel_time():
    """Review round 3, F2. With the bf16 kernel's per-layer time t measured,
    the bf16 core is 61 t at every end; only the fp8 kernel stays in the band:
    saved = 61 t - fp8 floor / e - extra, ITL reduction = saved / (rest + 61 t)."""
    from gitm.agents.hypotheses import rederive_h002

    reg = H002.predict(_k25())
    at_lo = rederive_h002(_k25(), _ATTN_RAG64 / 61 / 0.55 * 1e6)
    assert at_lo.step_delta[2] == pytest.approx(reg.step_delta[2], rel=1e-9)

    t = 100e-6
    p = rederive_h002(_k25(), 100.0)
    rest = (p.baseline_step_s - _ATTN_RAG64) / 0.75
    old = 61 * t
    for got, e in zip(p.step_delta, (0.55, 0.75, 0.95), strict=True):
        saved = old - (_ATTN_RAG64 / 2 + _WRITES_RAG64) / e - _LAUNCHES_MI355X / 0.75
        assert got == pytest.approx(saved / (rest + old), rel=1e-9)
    assert "measured 100.0 us per layer = eff 0.401" in _line(p, "bf16 core per layer")
    assert _line(p, "ITL reduction") == "ITL reduction 18.53% / 22.13% / 24.20% (lo/mean/hi)"


@pytest.mark.parametrize(
    ("t_us", "fragment"),
    [
        (0.0, "not a positive number"),
        (-50.0, "not a positive number"),
        (float("nan"), "not a positive number"),
        (float("inf"), "not a positive number"),
        # The headline floor is 2.4466 ms / 61 = 40.1 us per layer.
        (30.0, "at or below its byte floor 40.1 us, an efficiency of 1.34"),
        (_ATTN_RAG64 / 61 * 1e6, "at or below its byte floor"),
        # 100 us passed in seconds.
        (100e-6, "seconds passed as us?"),
        # 100 us passed in nanoseconds: 2,494x the floor.
        (100_000.0, "ns passed as us?"),
        # Just under the 0.05 efficiency floor (floor / 0.051 is accepted below).
        (_ATTN_RAG64 / 61 * 1e6 / 0.049, "an efficiency under 0.05"),
    ],
)
def test_h002_rederivation_refuses_times_the_model_cannot_use(t_us, fragment):
    """Review round 4, F3. A measured time at or below the floor is not an
    efficiency above 1; it means the bytes or the measurement are wrong."""
    from gitm.agents.hypotheses import rederive_h002

    with pytest.raises(ValueError, match=re.escape(fragment)):
        rederive_h002(_k25(), t_us)


def test_h002_rederivation_accepts_the_band_edges_and_a_slow_kernel():
    """A measured time at eff 0.95 or 0.55 pins the bf16 core where the
    registered lo or hi end already has it, so those ends agree; one ulp above
    the floor is a time the model can use (the floor itself is not, on every
    Python version); a kernel under the band widens the saving at every end."""
    from gitm.agents.hypotheses import rederive_h002

    reg = H002.predict(_k25())
    floor_us = _ATTN_RAG64 / 61 * 1e6
    assert rederive_h002(_k25(), floor_us / 0.95).step_delta[0] == pytest.approx(
        reg.step_delta[0], rel=1e-9)
    assert rederive_h002(_k25(), floor_us / 0.55).step_delta[2] == pytest.approx(
        reg.step_delta[2], rel=1e-9)
    above = rederive_h002(_k25(), math.nextafter(floor_us, math.inf))
    assert "= eff 1.000, used at every end" in _line(above, "bf16 core per layer")
    slow = rederive_h002(_k25(), floor_us / 0.051).step_delta
    assert reg.step_delta[2] < slow[0] < slow[1] < slow[2]


@pytest.mark.parametrize(
    ("serving", "fragment"),
    [
        ({**_AITER, "kv_cache_dtype": "fp8"}, "already storing an fp8"),
        ({**_AITER, "enforce_eager": True}, "enforce_eager"),
        ({**_AITER, "decode_context_parallel_size": 2}, "decode_context_parallel_size=2"),
    ],
)
def test_h002_rederivation_refuses_a_workload_the_hypothesis_refuses(serving, fragment):
    """The doc's step 2.5 passes the pod's config: arm L's (an fp8 cache) or the
    eager correlation arm's must not come back as a band."""
    from gitm.agents.hypotheses import rederive_h002

    with pytest.raises(ValueError, match=fragment):
        rederive_h002(_k25(serving=serving), 60.0)


@pytest.mark.parametrize(
    "workload",
    [
        # 16 heads per rank: asm stage 1 plus a reduce (aiter/mla.py:684-694).
        lambda: replace(_k25(), sharding=ShardingConfig(tp=4)),
        # 12 heads: 16 is not a multiple, so the padded asm path too.
        lambda: replace(_k25(), spec=replace(load_spec("kimi-k2.5"), n_heads=48),
                        sharding=ShardingConfig(tp=4)),
        lambda: _k25("H200", batch=32, kv_len=8192),
        lambda: _k25("B200", batch=32, kv_len=8192),
    ],
)
def test_h002_rederivation_refuses_a_baseline_that_is_not_one_gluon_launch(workload):
    """The measured time is one _mla_gluon launch per layer (doc, Experiment
    step 2); a baseline whose bf16 decode is the asm pair, FA3 or FlashInfer
    has no such kernel, so a time for it is not a band."""
    from gitm.agents.hypotheses import rederive_h002

    with pytest.raises(ValueError, match="one _mla_gluon launch per layer"):
        rederive_h002(workload(), 60.0)
