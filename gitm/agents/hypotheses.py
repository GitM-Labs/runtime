"""Registered hypotheses: interventions whose effect is derived before they run.

Autoresearch's other proposers search a knob surface and attach a flat, unproven
``expected_delta`` band to every candidate. A registered hypothesis differs in
one respect only: its expected effect, applicability and rejection condition are
*derived from the planner* for a named workload and written down (``docs/
hypotheses/``) before any measurement. It still enters through the same
:class:`~gitm.agents.autoresearch.Proposer` seam and the same selection gate,
rollback and measured-keep rule; nothing here is a new trust path.

Two conventions the gate imposes, stated because both are easy to get wrong:

* ``expected_delta_*`` on the emitted spec is the effect on the **covered ops**
  (``applies_to_kernels``), not on the step: ``replay.predict_delta`` multiplies
  it by the trace's coverage of those ops. The step-level figure a verdict is
  judged against lives in :class:`Prediction`.
* A hypothesis may target a knob the catalogue also carries. The catalogue entry
  is a hand-authored prior; this is a derivation for one workload, and the loop
  records which of the two produced a candidate by its name prefix.

Every uncertainty band here is the planner's own efficiency band
(``HardwareSpec.eff_lo``/``eff_hi``) applied to the nodes the intervention
changes. Where the intervention swaps the kernel, before and after may land
anywhere in the band independently; where the kernel is the same, both land at
the same efficiency. Times on the *measured basis* are floors divided by that
efficiency, so a total and its recoverable part are in the same units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.planner.glm_graph import GlmMoeDsaModelSpec, kv_entry_bytes, predict_glm_graph
from gitm.planner.graph import Graph
from gitm.planner.roofline import (
    QUANT_FORMATS,
    BatchConfig,
    HardwareSpec,
    ShardingConfig,
    _canon_dtype,
    quant_format,
)


@dataclass(frozen=True)
class Workload:
    """The operating point a prediction is made for.

    ``spec`` is the catalogue entry, whose ``kv_dtype`` records what vLLM's
    ``--kv-cache-dtype auto`` resolves to for that checkpoint. ``serving`` is an
    explicit engine override in EngineArgs names (``{"kv_cache_dtype": "fp8"}``);
    absent keys mean the engine default.
    """

    model: str
    spec: GlmMoeDsaModelSpec
    hw: HardwareSpec
    batch: BatchConfig
    sharding: ShardingConfig
    serving: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """A registered prediction: metric, baseline, effect, uncertainty, arithmetic.

    ``op_delta`` is the covered-op effect the gate consumes; ``step_delta`` is the
    decode-step effect a verdict is judged against. Both are reductions: positive
    means faster. Every tuple is (lo, mean, hi).
    """

    metric: str
    baseline_step_s: float
    predicted_step_s: float
    #: The modelled loss on the measured basis: everything the mechanism touches.
    total_overhead_s: float
    #: What the intervention can plausibly take back, measured basis.
    recoverable_s: tuple[float, float, float]
    op_delta: tuple[float, float, float]
    step_delta: tuple[float, float, float]
    arithmetic: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hypothesis:
    """One registered hypothesis. ``applies`` returns ``None`` or why not."""

    id: str
    title: str
    doc: str
    bottleneck_class: str
    knob: str
    value: Any
    #: Planner op names, for residual targeting.
    target_ops: tuple[str, ...]
    applies: Callable[[Workload], str | None]
    predict: Callable[[Workload], Prediction]
    #: Kernel-name substrings the *replay* credits. Narrower than ``target_ops``
    #: where the op's taxonomy rule is wider than the mechanism: ``classify_op``
    #: files cache insertion and slot mapping under ``attn_score_value`` too, and
    #: an fp8 cache does not halve those.
    kernel_scope: tuple[str, ...] = ()
    #: Weight or topology changes are never proposed by autoresearch; they are
    #: registered here so their arithmetic is tested, and run by hand.
    proposable: bool = True
    #: The live keep gate measures throughput only (``optimizer/apply.py``). A
    #: hypothesis that can change model output is emitted only when the proposer
    #: was given a correctness gate, and the emitted spec carries that gate so
    #: ``apply_intervention`` runs it before keep with any applicator.
    requires_correctness_gate: bool = False
    requires_hardware: tuple[str, ...] = ()
    safety_notes: str = ""


def _floor(spec: GlmMoeDsaModelSpec, w: Workload) -> Graph:
    return predict_glm_graph(spec, w.hw, w.batch, w.sharding)


def _op_time(g: Graph, *ops: str) -> float:
    return sum(n.prediction.t_pred_s for n in g.nodes if n.op in ops)


def _effect(
    g_old: Graph, g_new: Graph, ops: tuple[str, ...], hw: HardwareSpec,
    *, same_kernel: bool, extra_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """(saved seconds, op delta, step delta) for the changed ``ops``, (lo, mean, hi).

    The unchanged rest of the step is priced at mid-band efficiency throughout;
    only the changed nodes move within the band. With ``same_kernel`` both sides
    share one efficiency, so lo is that efficiency at ``eff_hi`` (the saving is
    smallest in absolute time) and hi at ``eff_lo``. Otherwise the old kernel and
    the new one are independent: lo puts the old at ``eff_hi`` and the new at
    ``eff_lo``. ``extra_s`` is cost the intervention adds (negative saving).
    """
    lo_e, hi_e = hw.eff_lo, hw.eff_hi
    mid = (lo_e + hi_e) / 2.0
    old, new = _op_time(g_old, *ops), _op_time(g_new, *ops)
    rest = (g_old.total_pred_s - old) / mid
    ends = ((hi_e, hi_e), (mid, mid), (lo_e, lo_e)) if same_kernel else (
        (hi_e, lo_e), (mid, mid), (lo_e, hi_e))
    saved, op, step = [], [], []
    for (e_old, e_new), extra in zip(ends, extra_s, strict=True):
        s = old / e_old - new / e_new - extra
        saved.append(s)
        op.append(s / (old / e_old))
        step.append(s / (rest + old / e_old))
    return tuple(saved), tuple(op), tuple(step)  # type: ignore[return-value]


# ── H-001: half of the 4-bit scale traffic, recovered by MXFP4's coarser blocks ──

_MLP_OPS = ("moe_routed", "moe_shared", "mlp_gate_up", "mlp_down")


def _h001_applies(w: Workload) -> str | None:
    fmt = quant_format(w.spec.dtype_for("moe_routed", w.spec.expert_dtype))
    if fmt is None or fmt.name not in ("nvfp4", "int4_g32"):
        return "experts are not a 1/16-B-per-weight-scale 4-bit format"
    # Quark OCP-MX MoE emulates unless the platform reports MX support, and only
    # ROCm gfx95x does (platforms/interface.py:577-581 returns False; rocm.py:
    # 744-745). Emulation dequantises every local expert to bf16 each forward
    # (fused_moe/fused_moe.py:1758-1762): a regression, not a recovery.
    if w.hw.arch != "cdna4":
        return (f"MXFP4 (Quark OCP-MX) emulates on {w.hw.arch or 'unknown'}: "
                "supports_mx() is False off ROCm gfx95x")
    return None


def mxfp4_variant(s: GlmMoeDsaModelSpec) -> GlmMoeDsaModelSpec:
    """amd/Kimi-K2.x-MXFP4's precision: every MLP (routed, shared, dense layer 0)
    MXFP4; attention, router and lm_head bf16 (its quantization_config exclude
    list carries exactly the attention projections, mlp.gate and lm_head)."""
    keep = tuple(o for o in s.op_dtype_overrides
                 if o[0] not in ("moe_shared", "mlp_gate_up", "mlp_down"))
    return replace(
        s, expert_dtype="mxfp4",
        op_dtype_overrides=keep + (
            ("moe_shared", "mxfp4"), ("mlp_gate_up", "mxfp4"), ("mlp_down", "mxfp4"),
        ),
    )


def _scale_stream_s(g: Graph, s: GlmMoeDsaModelSpec) -> float:
    fmt = quant_format(s.dtype_for("moe_routed", s.expert_dtype))
    share = fmt.scale_overhead if fmt else 0.0
    return share * sum(n.prediction.t_memory_s for n in g.nodes if n.op == "moe_routed")


def _h001_predict(w: Workload) -> Prediction:
    base, cand = w.spec, mxfp4_variant(w.spec)
    g_old, g_new = _floor(base, w), _floor(cand, w)
    mid = (w.hw.eff_lo + w.hw.eff_hi) / 2.0
    # Two mechanisms ride on one checkpoint swap and are priced apart. Routed
    # experts go 0.5625 -> 0.53125 B/weight: the scale recovery this hypothesis
    # is about. Shared expert and dense layer 0 go bf16 -> MXFP4: a precision
    # change the AMD checkpoint bundles in, which is not.
    routed = _op_time(g_old, "moe_routed") - _op_time(g_new, "moe_routed")
    bundled = (_op_time(g_old, *_MLP_OPS) - _op_time(g_new, *_MLP_OPS)) - routed
    # The checkpoint quantises activations too (input_tensors fp4, dynamic). If
    # the backend runs that as separate kernels, a W4A4 MLP needs two: its input
    # and the SiLU output feeding the down projection. The lo end pays both per
    # MLP layer; the hi end assumes they fuse into the GEMMs.
    n_quant = 2 * base.n_layers
    launches = n_quant * w.hw.kernel_launch_overhead_s
    saved, op, step = _effect(
        g_old, g_new, _MLP_OPS, w.hw, same_kernel=False,
        extra_s=(launches, launches / 2.0, 0.0),
    )
    scale_stream = _scale_stream_s(g_old, base)
    nv, mx = QUANT_FORMATS["nvfp4"], QUANT_FORMATS["mxfp4"]
    arithmetic = (
        f"scale bytes per weight: 1/16 = {nv.scale_bytes_per_elem:.4f} (nvfp4, int4 g32) "
        f"vs 1/32 = {mx.scale_bytes_per_elem:.4f} (mxfp4); overhead "
        f"{nv.scale_overhead:.1%} vs {mx.scale_overhead:.1%} of streamed expert bytes",
        f"routed scale stream {scale_stream * 1e3:.3f} ms floor; MXFP4 removes "
        f"{routed * 1e3:.3f} ms of it (half), the other half stays",
        f"bundled bf16 -> MXFP4 on shared expert + dense layer 0: "
        f"{bundled * 1e3:.3f} ms floor (not scale recovery)",
        f"activation-quant launches if unfused: {n_quant} x "
        f"{w.hw.kernel_launch_overhead_s * 1e6:.0f} us = {launches * 1e3:.3f} ms",
        f"MLP floor {_op_time(g_old, *_MLP_OPS) * 1e3:.3f} -> "
        f"{_op_time(g_new, *_MLP_OPS) * 1e3:.3f} ms; step floor "
        f"{g_old.total_pred_s * 1e3:.3f} -> {g_new.total_pred_s * 1e3:.3f} ms",
        "band: two different MoE kernels, each anywhere in eff "
        f"{w.hw.eff_lo}-{w.hw.eff_hi}: saved "
        + " / ".join(f"{x * 1e3:+.3f}" for x in saved) + " ms (lo/mean/hi)",
    )
    return Prediction(
        metric="MoE kernel time and HBM read per decode step, then decode ITL p50",
        baseline_step_s=g_old.total_pred_s,
        predicted_step_s=g_new.total_pred_s,
        total_overhead_s=scale_stream / mid,
        recoverable_s=saved,
        op_delta=op,
        step_delta=step,
        arithmetic=arithmetic,
    )


H001 = Hypothesis(
    id="H-001",
    title="MXFP4's 32-element e8m0 scales recover half of NVFP4/INT4-g32 scale traffic",
    doc="docs/hypotheses/H-001-mxfp4-scale-overhead.md",
    bottleneck_class="memory_bound",
    knob="model",
    value="amd/Kimi-K2.5-MXFP4",
    target_ops=_MLP_OPS,
    applies=_h001_applies,
    predict=_h001_predict,
    proposable=False,
    requires_hardware=("MI355X",),
    safety_notes="Weight change with W4A4 activations: accuracy gate mandatory.",
)


# ── H-002: a bf16 MLA cache where the engine could store fp8 ──────────────────

#: (backend with a bf16 cache, backend with fp8, same kernel?) per arch, in
#: vLLM v0.19.1. Hopper switches: FLASH_ATTN_MLA (first on sm90, platforms/
#: cuda.py:92-97) has no fp8 path (flashattn_mla.py:45-49, :322-323), so fp8
#: lands on FLASHMLA (flashmla.py:48-53, :73-74). Blackwell keeps FLASHINFER_MLA
#: (flashinfer_mla.py:40-45, :65-66); whether its fp8 path is the same kernel
#: was not read, so it is priced as a switch, the wider band. CDNA4 keeps the
#: ROCM_AITER_MLA backend (platforms/rocm.py:317-322; rocm_aiter_mla.py:30-38)
#: but not the kernel: AITER picks a hand-written asm kernel per dtype pair
#: (aiter/csrc/py_itfs_cu/asm_mla.cu:253-287), mla_a16w16_qh16_* for bf16 and
#: mla_a8w8_qh16_qseqlen1_gqaratio16 for fp8, after vLLM quantises the query to
#: fp8 too (mla_attention.py:669-674, :2099). A different kernel, so a switch.
#: Without AITER the ROCm list is TRITON_MLA alone (:324-326), not priced.
_MLA_BACKENDS: dict[str, tuple[str, str, bool]] = {
    "hopper": ("FLASH_ATTN_MLA", "FLASHMLA", False),
    "blackwell": ("FLASHINFER_MLA", "FLASHINFER_MLA", False),
    "cdna4": ("ROCM_AITER_MLA", "ROCM_AITER_MLA", False),
}


def _aiter_enabled(w: Workload) -> bool:
    """``VLLM_ROCM_USE_AITER=1`` in the serving config, as the loop deployment
    sets it. The env var is not an EngineArg, so it is read by its own name."""
    return str(w.serving.get("VLLM_ROCM_USE_AITER", "0")).lower() in ("1", "true")


def _executed_kv(w: Workload) -> str:
    """The cache dtype the engine stores: an explicit flag, else what ``auto``
    resolves to. vLLM resolves ``auto`` from the checkpoint's quantization
    config (engine/arg_utils.py:1567-1570 -> utils/torch_utils.py:324-342): a
    modelopt ``kv_cache_scheme`` of static 8-bit float becomes fp8 (:262-296),
    and anything else falls back to the model dtype (:345-351). The catalogue's
    ``kv_dtype`` records that resolved value."""
    flag = str(w.serving.get("kv_cache_dtype", "auto")).lower()
    return w.spec.kv_dtype if flag == "auto" else flag


def _mla_backends(w: Workload) -> tuple[str, str, bool] | str:
    """(backend with the executed cache, backend with fp8, same kernel), or why not.

    The derivation prices one specific kernel pair per arch: vLLM's default
    choice. An explicit ``attention_backend`` in the serving config is honoured
    only when it is one of that pair. Forcing the fp8-capable backend on the
    bf16 side is the same-kernel case only where that backend runs one kernel
    for both dtypes (Hopper's FLASHMLA); on CDNA4 the backend class is shared
    and the asm kernel is not, so it stays a switch.
    """
    if w.hw.arch not in _MLA_BACKENDS:
        return f"no fp8 MLA decode backend pinned for arch {w.hw.arch or 'unknown'!r}"
    if w.hw.arch == "cdna4" and not _aiter_enabled(w):
        return ("CDNA4 without VLLM_ROCM_USE_AITER=1 selects TRITON_MLA "
                "(platforms/rocm.py:324-326); the prediction is for ROCM_AITER_MLA")
    default, fp8, same = _MLA_BACKENDS[w.hw.arch]
    forced = str(w.serving.get("attention_backend", "") or "").upper()
    if not forced or forced == default:
        return default, fp8, same
    if forced == fp8:
        # One backend both sides. Same kernel only if the backend is one kernel
        # family for both dtypes, which is true where the default differs from
        # it (Hopper: forcing FLASHMLA at bf16 keeps FLASHMLA at fp8).
        return fp8, fp8, default != fp8
    return (f"attention_backend={forced} is not the pair the derivation prices "
            f"on {w.hw.arch} ({default} -> {fp8})")


def _h002_applies(w: Workload) -> str | None:
    s = w.spec
    if s.kv_lora_rank <= 0:
        return "not MLA: the cache is not one shared latent per token"
    # Sparse MLA (a DSA indexer on any layer) takes the fp8_ds_mla layout on
    # its own backends: 656 B per entry with per-128 fp32 scales, not the
    # generic 576 B this derivation prices (mla_attention.py:341-362).
    if s.n_full_indexer_layers > 0 or s.index_n_heads > 0:
        return ("sparse MLA (DSA indexer present): fp8 becomes the fp8_ds_mla "
                "layout on a sparse backend, which this derivation does not price")
    executed = _executed_kv(w)
    if _canon_dtype(executed) == "fp8":
        return ("already storing an fp8 cache (an explicit flag, or a checkpoint "
                "whose kv_cache_scheme vLLM resolves 'auto' to fp8)")
    if _canon_dtype(executed) != "fp16":
        return f"executed cache dtype {executed!r} is neither bf16/fp16 nor fp8"
    pair = _mla_backends(w)
    if isinstance(pair, str):
        return pair
    if w.batch.is_prefill or w.batch.batch <= 0:
        return "decode-only prediction; prefill steps are out of scope"
    return None


def _h002_predict(w: Workload) -> Prediction:
    executed = replace(w.spec, kv_dtype=_executed_kv(w), kv_rope_dtype=_executed_kv(w))
    candidate = replace(w.spec, kv_dtype="fp8", kv_rope_dtype="fp8")
    g_old, g_new = _floor(executed, w), _floor(candidate, w)
    pair = _mla_backends(w)
    if isinstance(pair, str):
        raise ValueError(f"H-002 does not apply: {pair}")
    old_b, new_b, same = pair
    saved, op, step = _effect(g_old, g_new, ("attn_score_value",), w.hw, same_kernel=same)
    mid = (w.hw.eff_lo + w.hw.eff_hi) / 2.0
    old_attn = _op_time(g_old, "attn_score_value")
    new_attn = _op_time(g_new, "attn_score_value")
    b = w.batch
    arithmetic = (
        f"cache entry {executed.kv_entry_dim} elems: {kv_entry_bytes(executed):.0f} B "
        f"stored vs {kv_entry_bytes(candidate):.0f} B as generic fp8",
        f"attention core reads {b.batch} x {b.kv_cache_len} entries x {executed.n_layers} "
        "layers per rank (replicated latent, TP does not split it)",
        f"floor: attn_score_value {old_attn * 1e3:.3f} -> {new_attn * 1e3:.3f} ms; "
        f"step {g_old.total_pred_s * 1e3:.3f} -> {g_new.total_pred_s * 1e3:.3f} ms",
        f"backend {old_b} -> {new_b}"
        + (" (same kernel: one efficiency for both)" if same
           else " (different kernel: each end of the band independent)"),
        "saved " + " / ".join(f"{x * 1e3:.3f}" for x in saved) + " ms (lo/mean/hi)",
    )
    return Prediction(
        metric="decode ITL (ms/step), p50 over the steady-state window",
        baseline_step_s=g_old.total_pred_s,
        predicted_step_s=g_new.total_pred_s,
        total_overhead_s=(old_attn - new_attn) / mid,
        recoverable_s=saved,
        op_delta=op,
        step_delta=step,
        arithmetic=arithmetic,
    )


H002 = Hypothesis(
    id="H-002",
    title="An MLA cache stored bf16 where the engine could store fp8",
    doc="docs/hypotheses/H-002-mla-kv-fp8.md",
    bottleneck_class="memory_bound",
    knob="kv_cache_dtype",
    value="fp8",
    target_ops=("attn_score_value",),
    # MLA decode kernels only. ``reshape_and_cache`` and ``slot_mapping`` also
    # classify to attn_score_value, and a narrower cache does not halve them.
    # CUDA names, plus AITER's persistent-mode pair (aiter/aiter/mla.py:318-349:
    # mla_decode_stage1_asm_fwd then mla_reduce_v1) and its asm symbols
    # (mla_a16w16_*, mla_a8w8_*).
    kernel_scope=("flash_mla", "flashmla", "cutlass_mla", "flashinfer_mla",
                  "mla_decode", "aiter_mla", "mla_fwd", "mla_a16w16", "mla_a8w8",
                  "mla_reduce"),
    applies=_h002_applies,
    predict=_h002_predict,
    requires_correctness_gate=True,
    requires_hardware=("H100", "H200", "B200", "B300", "GB200", "GB300", "MI355X"),
    safety_notes=(
        "Kimi K2.5 ships no k_scale/v_scale, so vLLM stores with scale 1.0 "
        "(quantization/kv_cache.py:71-75): gate on the accuracy sentinel in "
        "docs/hypotheses/H-002-mla-kv-fp8.md before keep."
    ),
)

REGISTERED: tuple[Hypothesis, ...] = (H001, H002)


class HypothesisProposer:
    """Emit registered hypotheses as candidates, gated on derived applicability.

    Two gates, in order. Here: the hypothesis's own ``applies`` (a fact about
    the model, the engine config and the arch), then its predicted *mean* step
    effect against ``noise_floor``: a hypothesis whose expected effect sits
    under the measured noise cannot be decided by one A/B, so it is not worth a
    restart. Afterwards, unchanged: the emitted :class:`Applicability` goes
    through ``preconditions.applicable`` in ``select_interventions``.

    ``correctness_gate`` is the accuracy sentinel for hypotheses that can
    change model output. Without one those are skipped, because the live keep
    gate cannot see accuracy. With one, the emitted spec carries it in
    ``InterventionSpec.correctness_gate`` and ``apply_intervention`` runs it
    after measuring and before keeping, whatever applicator the caller passed.

    ``skipped`` and ``predictions`` describe the most recent ``propose`` call.
    """

    def __init__(
        self,
        workload: Workload,
        *,
        hypotheses: tuple[Hypothesis, ...] = REGISTERED,
        noise_floor: float = 0.0,
        correctness_gate: Callable[[InterventionSpec], str | None] | None = None,
    ) -> None:
        self.workload = workload
        self.hypotheses = hypotheses
        self.noise_floor = noise_floor
        self.correctness_gate = correctness_gate
        self.skipped: list[tuple[str, str]] = []
        self.predictions: dict[str, Prediction] = {}

    def propose(
        self, bottleneck_class: str, *, target_op: str | None = None
    ) -> list[InterventionSpec]:
        self.skipped, self.predictions = [], {}
        out: list[InterventionSpec] = []
        for h in self.hypotheses:
            why = self._reject(h, bottleneck_class, target_op)
            if why is not None:
                self.skipped.append((h.id, why))
                continue
            out.append(self._spec(h, self.predictions[h.id]))
        return out

    def _reject(self, h: Hypothesis, cls: str, target_op: str | None) -> str | None:
        if not h.proposable:
            return "not proposable: weight or topology change, run by hand"
        if h.requires_correctness_gate and self.correctness_gate is None:
            return ("requires a correctness gate: the live keep gate measures "
                    "throughput only (optimizer/apply.py); construct the proposer "
                    "with correctness_gate=")
        if h.bottleneck_class != cls:
            return f"targets {h.bottleneck_class}, trace is {cls}"
        if target_op is not None and target_op not in h.target_ops:
            return f"largest residual is {target_op}, not {'/'.join(h.target_ops)}"
        why = h.applies(self.workload)
        if why is not None:
            return why
        pred = h.predict(self.workload)
        self.predictions[h.id] = pred
        if pred.step_delta[1] < self.noise_floor:
            return (f"predicted {pred.step_delta[1]:.1%} step effect is under the "
                    f"{self.noise_floor:.1%} noise floor")
        return None

    def _spec(self, h: Hypothesis, p: Prediction) -> InterventionSpec:
        w = self.workload
        lo, mean, hi = p.op_delta
        return InterventionSpec(
            name=f"hypothesis:{h.id}:{h.knob}={h.value}",
            summary=h.title,
            knob=h.knob,
            value=h.value,
            applies_to_kernels=list(h.kernel_scope or h.target_ops),
            expected_delta_mean=round(mean, 4),
            expected_delta_lo=round(lo, 4),
            expected_delta_hi=round(hi, 4),
            source=h.doc,
            applicability=Applicability(
                workloads=["vllm-decode"],
                requires_hardware=list(h.requires_hardware) or None,
                # The derivation priced activations at the model dtype; a run
                # whose dtype the gate cannot read is not one it covers.
                requires_dtype=[w.spec.act_dtype],
                other=(f"{w.model}; predicted step reduction {p.step_delta[0]:.1%} / "
                       f"{p.step_delta[1]:.1%} / {p.step_delta[2]:.1%} (lo/mean/hi)"),
            ),
            safety=SafetyGate(tier="moderate", requires_rollback_window_s=120,
                              notes=h.safety_notes),
            correctness_gate=self.correctness_gate if h.requires_correctness_gate else None,
        )
