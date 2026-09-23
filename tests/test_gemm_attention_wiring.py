"""GEMM and attention characterization, end to end through the loop's wiring.

Each test pins one seam where a correct component was connected to the wrong
thing: a projection priced at the wrong shape or width, an expert GEMM named
something no kernel classifies to, an NVTX identity that residuals honoured but
ranking ignored, or a measured result scaled as if it were a prior.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gitm.optimizer.monitor import residuals
from gitm.planner.graph import predict_graph
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec
from gitm.tracer.schema import KernelEvent, Trace

H100 = HardwareSpec(name="H100", peak_flops_fp16_per_s=989e12, peak_flops_bf16_per_s=989e12,
                    peak_mem_bw_bytes_per_s=3.35e12)


def _pred(model: ModelSpec, op: str, batch: int = 1):
    g = predict_graph(model, H100, BatchConfig(batch=batch))
    return next(n for n in g.nodes if n.op == op).prediction


def _trace(*events: tuple[str, int, str | None]) -> Trace:
    """``(name, duration_ns, range_op)`` laid end to end on one stream."""
    out, t = [], 0
    for i, (name, dur, range_op) in enumerate(events):
        out.append(KernelEvent(name=name, start_ns=t, end_ns=t + dur, stream_id=7,
                               device_id=0, correlation_id=i + 1, range_op=range_op))
        t += dur
    return Trace(workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
                 vendor="nvidia", captured_at_ns=0, duration_ns=t, events=out)


# ── dense graph: projection shapes ──────────────────────────────────────────


def test_out_proj_is_priced_from_the_attention_width_not_hidden():
    """Qwen3-0.6B: 16 heads x 128 = 2048 into a 1024-wide residual. The old
    ``h x h`` form priced a quarter of the real matrix."""
    m = ModelSpec(hidden=1024, n_layers=1, n_heads=16, num_kv_heads=8, head_dim=128,
                  intermediate=3072, vocab=151936)
    p = _pred(m, "attn_out_proj", batch=4)
    b, h, width, dt = 4, 1024, 16 * 128, 2
    assert p.flops == 2 * b * width * h
    assert p.bytes == dt * (b * width + b * h) + dt * width * h


def test_llama_shaped_out_proj_is_unchanged():
    """Where n_heads x head_dim == hidden the corrected form is the old one."""
    m = ModelSpec(n_layers=1)
    p = _pred(m, "attn_out_proj", batch=2)
    assert p.flops == 2 * 2 * 4096 * 4096
    assert p.bytes == 2 * (2 * 4096 + 4096 * 4096 + 2 * 4096)


def test_attention_projections_use_the_weight_width():
    """A dense fp8 checkpoint stores q/k/v/o narrow just as it does the FFN."""
    base = dict(n_layers=1)
    for op in ("qkv_proj", "attn_out_proj"):
        bf16 = _pred(ModelSpec(**base), op).bytes
        fp8 = _pred(ModelSpec(**base, weight_dtype_bytes=1), op).bytes
        assert fp8 < 0.55 * bf16, op  # the weight read is nearly all of it at b=1


def test_lm_head_stays_at_the_activation_width():
    """fp8/int4 checkpoints conventionally leave lm_head unquantized."""
    bf16 = _pred(ModelSpec(n_layers=1), "lm_head").bytes
    fp8 = _pred(ModelSpec(n_layers=1, weight_dtype_bytes=1), "lm_head").bytes
    assert fp8 == bf16


# ── dense graph: quantization is read for dense models too ──────────────────


def _hf(**kw):
    base = dict(hidden_size=4096, num_attention_heads=32, num_hidden_layers=2,
                intermediate_size=11008, vocab_size=32000)
    return SimpleNamespace(**{**base, **kw})


def test_dense_fp8_checkpoint_gets_its_weight_width():
    from gitm.scheduler.loop import _model_spec_from_hf

    spec = _model_spec_from_hf(_hf(quantization_config={"quant_method": "fp8"}))
    assert spec is not None and spec.w_bytes == 1
    assert not spec.is_moe


def test_object_quantization_config_is_read_too():
    from gitm.scheduler.loop import _model_spec_from_hf

    q = SimpleNamespace(quant_method="fp8")
    assert _model_spec_from_hf(_hf(quantization_config=q)).w_bytes == 1


@pytest.mark.parametrize(("bits", "expected"), [(8, 1), (4, 2)])
def test_compressed_tensors_is_a_container_not_a_width(bits, expected):
    """W8A8 is 1 byte; a W4A16 pack must not be priced as fp8 — it takes the
    conservative activation-width fallback, which cannot invent headroom."""
    from gitm.scheduler.loop import _model_spec_from_hf

    q = {"quant_method": "compressed-tensors",
         "config_groups": {"group_0": {"weights": {"num_bits": bits}}}}
    assert _model_spec_from_hf(_hf(quantization_config=q)).w_bytes == expected


def test_unquantized_dense_model_is_unchanged():
    from gitm.scheduler.loop import _model_spec_from_hf

    assert _model_spec_from_hf(_hf()).weight_dtype_bytes is None


# ── dense-graph MoE: expert GEMMs pair with the kernels that run them ───────

_MOE = ModelSpec(hidden=2048, n_layers=2, n_heads=32, num_kv_heads=4, head_dim=128,
                 intermediate=768, vocab=151936, num_experts=128, experts_per_token=8,
                 moe_intermediate=768)


def test_fused_moe_kernels_are_scored_not_dropped_as_unmodeled():
    """vLLM launches ``fused_moe_kernel`` once per expert GEMM. With the graph's
    MoE nodes named ``mlp_*`` neither launch had anything to pair with."""
    g = predict_graph(_MOE, H100, BatchConfig(batch=8))
    t = _trace(("fused_moe_kernel", 40_000, None), ("fused_moe_kernel", 20_000, None))
    res = residuals(t, g)
    assert [kr.op for kr in res.per_kernel] == ["moe_routed", "moe_routed"]
    # Two structural classes (gate_up, down) with no layer known: interval-scored.
    assert all(kr.n_classes == 2 for kr in res.per_kernel)


def test_deviation_floor_for_experts_is_the_sum_of_both_gemms():
    from gitm.optimizer.deviation import predicted_per_op

    g = predict_graph(_MOE, H100, BatchConfig(batch=8))
    floors = predicted_per_op(g)
    assert "mlp_gate_up" not in floors and "mlp_down" not in floors
    expert = [n.prediction.t_pred_s for n in g.nodes if n.op == "moe_routed"]
    assert len(expert) == 2 * _MOE.n_layers
    assert floors["moe_routed"] == pytest.approx(sum(expert))


# ── NVTX identity reaches ranking, not just residuals ───────────────────────


def _spec(name, kernels, *, mean=0.05):
    from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate

    return InterventionSpec(
        name=name, summary="s", knob=name, value=1,
        expected_delta_mean=mean, expected_delta_lo=0.0, expected_delta_hi=0.2,
        source="t", applies_to_kernels=kernels,
        applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="moderate"),
    )


_BARE_GEMM = "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn"


def test_a_bare_gemm_identified_by_its_nvtx_range_counts_toward_coverage():
    """``residuals()`` pairs this kernel with mlp_gate_up via its range; a lever
    scoped to mlp_gate_up must see it too, or the op it targets is invisible."""
    from gitm.optimizer.replay import predict_delta

    lever = _spec("gate_up_lever", ["mlp_gate_up"], mean=0.10)
    named_only = _trace((_BARE_GEMM, 1000, None))
    with_range = _trace((_BARE_GEMM, 1000, "mlp_gate_up"))
    assert predict_delta(named_only, lever) == 0.0
    assert predict_delta(with_range, lever) == pytest.approx(0.10)


def test_the_autoresearch_target_survives_when_only_nvtx_names_it():
    from gitm.agents.autoresearch import _op_present

    t = _trace((_BARE_GEMM, 1000, "mlp_down"))
    assert _op_present(t, "mlp_down")
    assert not _op_present(t, "mlp_gate_up")


# ── autoresearch candidates are ranked from history like the catalog ───────

SKU, FP = "NVIDIA H100 80GB HBM3", "fp-1"


def _history(name, mean):
    from gitm.optimizer.history import History, LeverRecord

    r = LeverRecord(intervention_name=name, gpu_sku=SKU, fingerprint=FP, runs=1,
                    attempts=1, wins=0 if mean < 0 else 1, losses=1 if mean < 0 else 0,
                    inconclusive=0, mean_delta=mean, best_delta=mean, worst_delta=mean,
                    last_run_id="r1")
    return History(records={(name, SKU, FP): r}, runs_read=1)


class _OneShot:
    def __init__(self, *specs):
        self._specs = list(specs)

    def propose(self, bottleneck_class, *, target_op=None):
        return list(self._specs)


def test_autoresearch_candidates_are_scored_from_what_they_measured():
    from gitm.agents.autoresearch import autoresearch
    from gitm.agents.policy import Policy
    from gitm.optimizer.apply import DryRunApplicator

    loser = _spec("autoresearch:compute_bound:compilation_config=3", ["attn_score_value"])
    t = _trace(("flash_fwd_splitkv_kernel", 1000, None))
    run = autoresearch(t, applicator=DryRunApplicator(), policy=Policy(use_history=True),
                       proposer=_OneShot(loser), history=_history(loser.name, -0.30),
                       gpu_sku=SKU, fingerprint=FP)
    assert [r.predicted_delta for r in run.results] == [pytest.approx(-0.30)]


def test_without_history_autoresearch_keeps_its_prior():
    from gitm.agents.autoresearch import autoresearch
    from gitm.agents.policy import Policy
    from gitm.optimizer.apply import DryRunApplicator

    spec = _spec("autoresearch:x", ["attn_score_value"])
    t = _trace(("flash_fwd_splitkv_kernel", 1000, None))
    run = autoresearch(t, applicator=DryRunApplicator(), policy=Policy(use_history=True),
                       proposer=_OneShot(spec))
    assert run.results[0].predicted_delta == pytest.approx(0.05)


# ── a measured delta is end-to-end, not scaled by the lever's scope ─────────


def test_a_measured_win_on_a_narrow_lever_outranks_a_broad_untested_prior():
    """Lever A covers 20% of the trace and measured +10% end to end. Lever B is
    untested, covers everything, prior 5%. Coverage x measured scored A at +2%
    and ran B first; the measurement already is the end-to-end answer."""
    from gitm.agents.policy import Policy, select_interventions

    t = _trace(("flash_fwd_splitkv_kernel", 200, None), (_BARE_GEMM, 800, "mlp_down"))
    narrow = _spec("narrow", ["attn_score_value"], mean=0.05)
    broad = _spec("broad", ["attn_score_value", "mlp_down"], mean=0.05)
    ranked = select_interventions(t, [broad, narrow], Policy(use_history=True), top_n=2,
                                  history=_history("narrow", 0.10), gpu_sku=SKU,
                                  fingerprint=FP)
    assert [c.spec.name for c in ranked] == ["narrow", "broad"]
    assert ranked[0].predicted_delta == pytest.approx(0.10)
    assert ranked[0].delta_source == "measured"


def test_a_measured_win_on_an_empty_scope_is_not_zeroed():
    from gitm.agents.policy import Policy, select_interventions

    t = _trace(("flash_fwd_splitkv_kernel", 1000, None))
    unscoped = _spec("unscoped", [], mean=0.10)
    [c] = select_interventions(t, [unscoped], Policy(use_history=True), top_n=1,
                               history=_history("unscoped", 0.12), gpu_sku=SKU,
                               fingerprint=FP)
    assert c.predicted_delta == pytest.approx(0.12)


# ── the catalog's MoE and collective levers cover the kernels they act on ──


def _lever(name):
    from gitm.kernels.library import load_library

    return next(s for s in load_library() if s.name == name)


@pytest.mark.parametrize("name", ["moe_backend_deep_gemm", "enable_expert_parallel",
                                  "enable_eplb"])
def test_moe_levers_cover_the_expert_gemms(name):
    """Coverage itself, not the ranked number: enable_expert_parallel's prior is
    0.0, so a delta check would pass whether or not it covers anything."""
    from gitm.optimizer.replay import predict_delta

    t = _trace(("fused_moe_kernel", 1000, None))
    assert predict_delta(t, _lever(name), delta_mean=1.0) == pytest.approx(1.0)


def test_custom_all_reduce_lever_covers_the_all_reduce_kernels():
    from gitm.optimizer.replay import predict_delta

    t = _trace(("void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>", 500, None),
               ("fused_moe_kernel", 500, None))
    assert predict_delta(t, _lever("disable_custom_all_reduce")) == pytest.approx(
        0.5 * _lever("disable_custom_all_reduce").expected_delta_mean)


def test_a_known_layer_with_two_expert_gemms_scores_each_launch_fairly():
    """With NVTX, both ``fused_moe_kernel`` launches carry ``moe_routed@L0``. Pairing
    took the layer's first node, so a down launch exactly on prediction scored
    -50% against the gate_up point — a systematic offset ``check_invariants``
    reads as confirmation. Within the layer's own interval both are on target."""
    g = predict_graph(_MOE, H100, BatchConfig(batch=8))
    gu, dn = [n.prediction.t_pred_s for n in g.nodes if n.op == "moe_routed" and n.layer == 0]
    ev, t = [], 0
    for i, dur in enumerate((int(gu * 1e9), int(dn * 1e9))):
        ev.append(KernelEvent(name="fused_moe_kernel", start_ns=t, end_ns=t + dur,
                              stream_id=7, device_id=0, correlation_id=i + 1,
                              range_op="moe_routed", range_layer=0))
        t += dur
    trace = Trace(workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
                  vendor="nvidia", captured_at_ns=0, duration_ns=t, events=ev)
    res = residuals(trace, g)
    assert [kr.r_kt for kr in res.per_kernel] == [pytest.approx(0.0, abs=1e-3)] * 2
    assert all(kr.layer == 0 and kr.interval_based for kr in res.per_kernel)


def test_a_known_layer_with_one_node_still_gets_a_point_residual():
    g = predict_graph(ModelSpec(n_layers=2), H100, BatchConfig(batch=1))
    pred = next(n for n in g.nodes if n.op == "qkv_proj" and n.layer == 1).prediction.t_pred_s
    ev = [KernelEvent(name=_BARE_GEMM, start_ns=0, end_ns=int(2 * pred * 1e9), stream_id=7,
                      device_id=0, correlation_id=1, range_op="qkv_proj", range_layer=1)]
    trace = Trace(workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
                  vendor="nvidia", captured_at_ns=0, duration_ns=10**6, events=ev)
    [kr] = residuals(trace, g).per_kernel
    assert kr.r_kt == pytest.approx(1.0, rel=1e-3) and not kr.interval_based
