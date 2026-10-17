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

import json
import math
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
    UnsupportedExecution,
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

    ``request_lengths`` is (prompt, output) tokens when every request has those
    lengths and the load keeps the batch full of them (``vllm bench serve`` at
    request rate inf under ``--max-concurrency``, v0.29.0 benchmarks/serve.py:
    458-459, :968-976); ``batch.kv_cache_len`` is then prompt + output // 2.
    """

    model: str
    spec: GlmMoeDsaModelSpec
    hw: HardwareSpec
    batch: BatchConfig
    sharding: ShardingConfig
    serving: dict[str, Any] = field(default_factory=dict)
    request_lengths: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.request_lengths is None:
            return
        prompt, output = self.request_lengths
        if prompt < 1 or output < 2 or prompt + output // 2 != self.batch.kv_cache_len:
            raise ValueError(
                f"request_lengths {self.request_lengths} do not put kv_cache_len "
                f"{self.batch.kv_cache_len} at prompt + output // 2 with at least one "
                "decode step")


@dataclass(frozen=True)
class Prediction:
    """A registered prediction: metric, baseline, effect, uncertainty, arithmetic.

    ``op_delta`` is the covered-op effect the gate consumes; ``step_delta`` is the
    decode-step effect a verdict is judged against. Both are reductions: positive
    means faster. Every tuple is (lo, mean, hi); ``hardware`` is the SKU it was
    priced on.
    """

    metric: str
    hardware: str
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
    #: How long apply_intervention waits for the gate before restoring.
    correctness_gate_timeout_s: float = 1800.0
    requires_hardware: tuple[str, ...] = ()
    safety_notes: str = ""


def _floor(spec: GlmMoeDsaModelSpec, w: Workload) -> Graph:
    return predict_glm_graph(spec, w.hw, w.batch, w.sharding)


def _op_time(g: Graph, *ops: str) -> float:
    # fsum: sum() loses an ulp before 3.12, and rederive_h002 compares a time
    # against this / n_layers.
    return math.fsum(n.prediction.t_pred_s for n in g.nodes if n.op in ops)


def _effect(
    g_old: Graph, g_new: Graph, ops: tuple[str, ...], hw: HardwareSpec,
    *, same_kernel: bool, extra_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    old_eff: float | None = None, new_add_s: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """(saved seconds, op delta, step delta) for the changed ``ops``, (lo, mean, hi).

    The unchanged rest of the step is priced at mid-band efficiency throughout;
    only the changed nodes move within the band. With ``same_kernel`` both sides
    share one efficiency, so lo is that efficiency at ``eff_hi`` (the saving is
    smallest in absolute time) and hi at ``eff_lo``. Otherwise the old kernel and
    the new one are independent: lo puts the old at ``eff_hi`` and the new at
    ``eff_lo``. ``extra_s`` is cost the intervention adds (negative saving).
    ``old_eff`` pins the old kernel at one measured efficiency at every end;
    ``new_add_s`` is floor the new kernel moves beyond the graph's op, priced at
    its efficiency at each end.
    """
    lo_e, hi_e = hw.eff_lo, hw.eff_hi
    mid = (lo_e + hi_e) / 2.0
    old, new = _op_time(g_old, *ops), _op_time(g_new, *ops) + new_add_s
    rest = (g_old.total_pred_s - old) / mid
    ends = ((hi_e, hi_e), (mid, mid), (lo_e, lo_e)) if same_kernel else (
        (hi_e, lo_e), (mid, mid), (lo_e, hi_e))
    if old_eff is not None:
        ends = ((old_eff, lo_e), (old_eff, mid), (old_eff, hi_e))
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
        hardware=w.hw.name,
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

#: (backend with a bf16 cache, backend with fp8, same kernel?) per arch. CUDA
#: rows read at vLLM v0.27.1 (last verified on H200), CDNA4 at v0.29.0 (the
#: MI355X loop image). Hopper: FLASH_ATTN_MLA is first on sm90 but stores no
#: fp8 (flashattn_mla.py:45-49), so fp8 lands on FLASHMLA. Blackwell keeps
#: FLASHINFER_MLA; whether its fp8 path is the same kernel was not read, so it
#: is priced as a switch. CDNA4 keeps ROCM_AITER_MLA but not the kernel: under
#: 16 heads per rank bf16 runs AITER's Gluon ``_mla_gluon`` and fp8 the asm
#: ``mla_a8w8_*`` kernels (rocm_aiter_mla.py:1334-1353).
_MLA_BACKENDS: dict[str, tuple[str, str, bool]] = {
    "hopper": ("FLASH_ATTN_MLA", "FLASHMLA", False),
    "blackwell": ("FLASHINFER_MLA", "FLASHINFER_MLA", False),
    "cdna4": ("ROCM_AITER_MLA", "ROCM_AITER_MLA", False),
}


#: AITER's smallest native MLA decode head count (vLLM v0.29.0
#: v1/attention/backends/mla/rocm_aiter_mla.py:1216).
_AITER_MIN_MLA_HEADS = 16


def _env_on(w: Workload, name: str, default: str) -> bool:
    """On for "1" or "true" in any case, as vLLM reads a boolean env var (envs.py:1235-1237)."""
    return str(w.serving.get(name, default)).lower() in ("1", "true")


def _cdna4_refusal(w: Workload) -> str | None:
    """Why the CDNA4 serving env is not the loop env the derivation prices, or
    None: ROCM_AITER_MLA heads the MLA list only with both AITER flags on
    (v0.29.0 _aiter_ops.py:1953-1956, platforms/rocm.py:469-479), else it is
    TRITON_MLA unless pinned, and a pin needs neither flag (_aiter_ops.py:134-146)."""
    if not _env_on(w, "VLLM_ROCM_USE_AITER", "false"):
        return ("CDNA4 without VLLM_ROCM_USE_AITER=1 is not the loop env the derivation "
                "prices: ROCm lists TRITON_MLA first (platforms/rocm.py:469-479), and a "
                "pinned ROCM_AITER_MLA runs with AITER off everywhere else (_aiter_ops.py:134-146)")
    if not _env_on(w, "VLLM_ROCM_USE_AITER_MLA", "true"):
        return ("VLLM_ROCM_USE_AITER_MLA is off, so CDNA4 lists TRITON_MLA first "
                "(_aiter_ops.py:1953-1956, platforms/rocm.py:469-479) unless ROCM_AITER_MLA "
                "is pinned; the derivation prices the loop env, which leaves it on")
    padding = str(w.serving.get("VLLM_ROCM_AITER_MLA_ASM_PADDING", "auto")).lower()
    heads = w.spec.n_heads // w.sharding.tp
    # use_gluon_decode reads the mode only under 16 heads; auto takes Gluon
    # when 16 is a multiple of the head count (rocm_aiter_mla.py:1334-1353).
    if heads < _AITER_MIN_MLA_HEADS:
        auto = "gluon" if _AITER_MIN_MLA_HEADS % heads == 0 else "asm"
        if padding not in ("auto", auto):
            return (f"VLLM_ROCM_AITER_MLA_ASM_PADDING={padding} moves the bf16 arm off "
                    f"the {auto} kernel auto picks at {heads} heads per rank "
                    "(rocm_aiter_mla.py:1334-1353); the derivation prices auto")
    return None


def _executed_kv(w: Workload) -> str:
    """The cache dtype the engine stores: an explicit flag, else what ``auto``
    resolves to. vLLM resolves ``auto`` from the checkpoint's quantization
    config (engine/arg_utils.py:2069-2071 -> utils/torch_utils.py:506-524): a
    modelopt ``kv_cache_scheme`` of static 8-bit float becomes fp8 (:452-467),
    and anything else falls back to the model dtype (:494, :524). The catalogue's
    ``kv_dtype`` records that resolved value."""
    flag = str(w.serving.get("kv_cache_dtype", "auto")).lower()
    if flag == "bfloat16":  # vLLM's own spelling (config/cache.py:39-42)
        return "bf16"
    return w.spec.kv_dtype if flag == "auto" else flag


def _nested_config(w: Workload, key: str) -> dict[str, Any] | str:
    """A config-object EngineArg (``compilation_config``, ``attention_config``):
    a dict, or the JSON string the CLI takes (engine/arg_utils.py:815-819); else
    why vLLM v0.29.0 would not take it."""
    value = w.serving.get(key)
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return f"{key}={value!r} is not JSON (engine/arg_utils.py:815-819)"
    if not isinstance(value, dict):
        return (f"{key}={value!r} is not the dict or JSON object vLLM v0.29.0 takes "
                "(engine/arg_utils.py:815-819)")
    return value


def _forced_backend(w: Workload, ac: dict[str, Any]) -> str:
    """The MLA backend the serving config pins, or "": ``--attention-backend`` lands
    in ``attention_config.backend`` (v0.29.0 engine/arg_utils.py:2489-2498),
    ``backend_per_kind`` overrides it per kind (v1/attention/selector.py:175-183),
    and ``auto`` is vLLM's spelling for no pin (config/attention.py:162-164)."""
    per_kind = ac.get("backend_per_kind") or {}
    forced = str(per_kind.get("mla_attention") or ac.get("backend")
                 or w.serving.get("attention_backend") or "").upper()
    return "" if forced == "AUTO" else forced


def _mla_backends(w: Workload) -> tuple[str, str, bool] | str:
    """(backend with the executed cache, backend with fp8, same kernel), or why not.

    The derivation prices vLLM's default kernel pair per arch; a pinned backend
    is honoured only when it is one of the pair. Forcing the fp8-capable
    backend on the bf16 side still switches the kernel: FLASHMLA and
    ROCM_AITER_MLA each run a different kernel per cache dtype.
    """
    if w.hw.arch not in _MLA_BACKENDS:
        return f"no fp8 MLA decode backend pinned for arch {w.hw.arch or 'unknown'!r}"
    if w.hw.arch == "cdna4":
        why = _cdna4_refusal(w)
        if why is not None:
            return why
    ac = _nested_config(w, "attention_config")
    if isinstance(ac, str):
        return ac
    per_kind = ac.get("backend_per_kind")
    if per_kind is not None and not isinstance(per_kind, dict):
        return (f"attention_config.backend_per_kind={per_kind!r} is not the dict vLLM "
                "v0.29.0 takes (config/attention.py:30, :176-199)")
    default, fp8, same = _MLA_BACKENDS[w.hw.arch]
    forced = _forced_backend(w, ac)
    if forced == default and default != fp8:
        return (f"attention_backend={forced} is explicitly pinned; the fp8 intervention "
                f"does not switch it to {fp8}")
    if not forced or forced == default:
        return default, fp8, same
    if forced == fp8:
        # Pinned FLASHMLA still switches: flash_mla_with_kvcache for bf16 and
        # flash_mla_with_kvcache_fp8 for fp8 (v0.27.1 flashmla.py:317-342).
        return fp8, fp8, False
    return (f"attention_backend={forced} is not the pair the derivation prices "
            f"on {w.hw.arch} ({default} -> {fp8})")


def _h002_applies(w: Workload) -> str | None:
    s = w.spec
    if s.kv_lora_rank <= 0:
        return "not MLA: the cache is not one shared latent per token"
    # Sparse MLA (a DSA indexer on any layer) takes the fp8_ds_mla layout on
    # its own backends: 656 B per entry with per-128 fp32 scales, not the
    # generic 576 B this derivation prices (mla_attention.py:349-362, :1218-1220).
    if s.n_full_indexer_layers > 0 or s.index_n_heads > 0:
        return ("sparse MLA (DSA indexer present): fp8 becomes the fp8_ds_mla "
                "layout on a sparse backend, which this derivation does not price")
    executed = _executed_kv(w)
    if _canon_dtype(executed) == "fp8":
        return ("already storing an fp8 cache (an explicit flag, or a checkpoint "
                "whose kv_cache_scheme vLLM resolves 'auto' to fp8)")
    if _canon_dtype(executed) != "fp16":
        return f"executed cache dtype {executed!r} is neither bf16/fp16 nor fp8"
    # The named layers stay at the model dtype (engine/arg_utils.py:1299,
    # mla_attention.py:452-456); the derivation prices every layer fp8.
    if w.serving.get("kv_cache_dtype_skip_layers"):
        return ("kv_cache_dtype_skip_layers is set: a partly fp8 cache is a different "
                "prediction and must be re-derived")
    pair = _mla_backends(w)
    if isinstance(pair, str):
        return pair
    if w.batch.is_prefill or w.batch.batch <= 0:
        return "decode-only prediction; prefill steps are out of scope"
    cc = _nested_config(w, "compilation_config")
    if isinstance(cc, str):
        return cc
    return (_count_arg_refusal(w) or _decode_shape_refusal(w) or _graph_refusal(w, cc)
            or _concat_quant_refusal(w, cc))


#: Strings pydantic reads as True for a bool field; vLLM's ModelConfig is a
#: pydantic dataclass (v0.29.0 config/utils.py:51-74, config/model.py:123-124).
_PYDANTIC_TRUE = ("1", "on", "t", "true", "y", "yes")


def _opt_level(w: Workload) -> str:
    return str(w.serving.get("optimization_level", 2)).upper().removeprefix("O")


#: vLLM v0.29.0's CompilationMode and CUDAGraphMode (config/compilation.py:37-63),
#: taken as its config does: a name in any case, or the int value (:853-876).
#: FULL_DECODE_ONLY and FULL_AND_PIECEWISE have tuple values, so no int names them.
_COMPILATION_MODES = ("NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE")
_CUDAGRAPH_MODES = ("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")


def _decode_cudagraph_mode(w: Workload, cc: dict[str, Any]) -> str | None:
    """vLLM v0.29.0's cudagraph mode name: an explicit ``cudagraph_mode``, else
    the one ``optimization_level`` supplies (config/vllm.py:255, :278, :301);
    None for a value that names no mode."""
    mode = cc.get("cudagraph_mode")
    if mode is None:
        return {"0": "NONE", "1": "PIECEWISE"}.get(_opt_level(w), "FULL_AND_PIECEWISE")
    if isinstance(mode, int):
        name = {0: "NONE", 1: "PIECEWISE", 2: "FULL"}.get(mode)
    else:
        name = str(mode).upper()
    return name if name in _CUDAGRAPH_MODES else None


def _compile_mode(w: Workload, cc: dict[str, Any]) -> str | None:
    """vLLM v0.29.0's CompilationMode name. TORCH_COMPILE_DISABLE=1 forces NONE
    (config/vllm.py:1390-1395); with no explicit ``mode``, -O0 gives NONE and
    any other level VLLM_COMPILE (:1431-1435). None for a value that names
    no mode."""
    if str(w.serving.get("TORCH_COMPILE_DISABLE", "")) == "1":
        return "NONE"
    mode = cc.get("mode")
    if mode is None:
        return "NONE" if _opt_level(w) == "0" else "VLLM_COMPILE"
    if isinstance(mode, int):
        name = _COMPILATION_MODES[mode] if 0 <= mode < 4 else None
    else:
        name = str(mode).upper()
    return name if name in _COMPILATION_MODES else None


def _graph_refusal(w: Workload, cc: dict[str, Any]) -> str | None:
    """Why decode would not replay one full graph at this batch, or None. The
    extra kernels are priced at the graph-replay launch (2 us); eager is nearer
    5 us, which moves the whole band. PIECEWISE runs attention outside its
    graphs, and a batch above the largest captured size dispatches to NONE
    (v1/cudagraph_dispatcher.py:274-281)."""
    why = "the derivation prices graph-replay launches (2 us), not eager ones"
    if str(w.serving.get("enforce_eager", False)).lower() in _PYDANTIC_TRUE:
        return f"enforce_eager runs decode without a graph: {why}"
    if str(w.serving.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "")) == "1":
        return ("VLLM_USE_BREAKABLE_CUDAGRAPH=1 runs MLA attention eagerly between graph "
                "segments (mla_attention.py:1319, compilation/breakable_cudagraph.py:319-322): "
                f"{why}")
    mode = _decode_cudagraph_mode(w, cc)
    if mode is None:
        return (f"compilation_config.cudagraph_mode={cc['cudagraph_mode']!r} is not a "
                "CUDAGraphMode name or value (config/compilation.py:53-63, :870-876)")
    if mode in ("NONE", "PIECEWISE"):
        return f"cudagraph_mode {mode} runs decode attention outside a graph: {why}"
    compile_mode = _compile_mode(w, cc)
    if compile_mode is None:
        return (f"compilation_config.mode={cc['mode']!r} is not a CompilationMode name or "
                "value (config/compilation.py:37-50, :853-868)")
    if mode == "FULL_AND_PIECEWISE" and compile_mode != "VLLM_COMPILE":
        return (f"compilation mode {compile_mode} does no piecewise compilation, so vLLM "
                f"overrides cudagraph_mode {mode} to NONE (config/vllm.py:1468-1480): {why}")
    pcp = int(w.serving.get("prefill_context_parallel_size") or 1)
    if pcp > 1 and w.hw.arch == "cdna4":
        return (f"prefill_context_parallel_size={pcp}: ROCm overrides full graphs to "
                f"PIECEWISE (platforms/rocm.py:902-913), so decode attention runs outside "
                f"a graph: {why}")
    if w.serving.get("kv_transfer_config"):
        return ("kv_transfer_config is set: a KV connector may override full graphs to "
                "PIECEWISE (config/vllm.py:1556-1581), and its transfers are not priced")
    cap = _capture_ceiling(w, cc)
    if isinstance(cap, str):
        return cap
    if w.batch.batch > cap:
        return (f"batch {w.batch.batch} is above the largest captured graph size {cap}, "
                f"so decode runs without a graph: {why}")
    return None


def _capture_ceiling(w: Workload, cc: dict[str, Any]) -> int | str:
    """The largest decode batch vLLM v0.29.0 replays a graph for (config/vllm.py:
    1996-2182), or why the config gives none. Explicit ``cudagraph_capture_sizes``
    keep those at or under ``max_num_batched_tokens``. Otherwise the ceiling,
    ``max_cudagraph_capture_size`` or min(2 x max_num_seqs, 512; 1024 on
    Blackwell), is capped at that budget and, unless the budget itself is
    captured, rounded down to the capture grid: 1, 2, 4, then 8s below 256 and
    16s from 256. ``max_num_seqs`` defaults to 1024 on every priced SKU
    (engine/arg_utils.py:2698-2717)."""
    budget = int(w.serving.get("max_num_batched_tokens") or 0)
    sizes = cc.get("cudagraph_capture_sizes") or w.serving.get("cudagraph_capture_sizes")
    if sizes:
        if not isinstance(sizes, list) or not all(_is_count(s) for s in sizes):
            return (f"cudagraph_capture_sizes={sizes!r} is not the list of ints vLLM v0.29.0 "
                    "takes (config/compilation.py:648)")
        return max((int(s) for s in sizes if not budget or int(s) <= budget), default=0)
    cap = cc.get("max_cudagraph_capture_size", w.serving.get("max_cudagraph_capture_size"))
    if cap is None:
        cap = min(1024 if w.hw.arch == "blackwell" else 512,
                  2 * int(w.serving.get("max_num_seqs") or 1024))
    elif not _is_count(cap):
        return (f"max_cudagraph_capture_size={cap!r} is not the positive int vLLM v0.29.0 "
                "takes (config/compilation.py:692, config/vllm.py:2095-2098)")
    cap = int(cap)
    if budget and budget <= cap:
        return budget
    if cap < 8:
        return max(s for s in (1, 2, 4) if s <= cap)
    return cap - cap % (8 if cap < 256 else 16)


def _concat_quant_refusal(w: Workload, cc: dict[str, Any]) -> str | None:
    """Why the fp8 arm's query concat-and-quantise would not be the one Inductor
    kernel priced, or None. Enabled, the custom op runs ``torch.cat`` then
    AITER's quantise, two kernels (mla_attention.py:1419-1445,
    quantization/input_quant_fp8.py:137-160). ``custom_ops`` enables it with
    ``all`` or ``+mla_decode_concat_quant_fp8`` (custom_op.py:285-311), and vLLM
    adds ``all`` itself when compilation is off or the backend is not Inductor
    (config/vllm.py:1444-1451); off, a disabled op is not compiled either
    (custom_op.py:222-225)."""
    why = "the derivation prices the query concat-and-quantise as one compiled kernel"
    name = "mla_decode_concat_quant_fp8"
    ops = cc.get("custom_ops") or []
    if not isinstance(ops, list):
        return (f"custom_ops={ops!r} is not the list of op names vLLM v0.29.0 takes "
                "(config/compilation.py:495)")
    if ("all" in ops or f"+{name}" in ops) and f"-{name}" not in ops:
        return f"custom_ops={ops} enables {name}, torch.cat plus AITER's quantise: {why}"
    if _compile_mode(w, cc) == "NONE":
        return ("compilation mode NONE runs the query concat-and-quantise as its custom op "
                f"or as uncompiled torch (config/vllm.py:1444-1451, custom_op.py:222-225): {why}")
    backend = cc.get("backend") or "inductor"
    if backend != "inductor":
        return (f"compilation backend {backend!r} turns every custom op on "
                f"(config/vllm.py:1444-1451), so {name} runs torch.cat plus AITER's "
                f"quantise: {why}")
    return None


#: Serving keys the checks read as counts; vLLM v0.29.0 types each as an int of
#: at least 1 (engine/arg_utils.py:493, :518-524, :583-586; config/parallel.py:
#: 126-129, :349; config/scheduler.py:49, :63).
_COUNT_ARGS = ("max_num_batched_tokens", "max_num_seqs", "max_cudagraph_capture_size",
               "data_parallel_size", "decode_context_parallel_size",
               "prefill_context_parallel_size")


def _is_count(value: Any) -> bool:
    try:
        return int(value) >= 1
    except (TypeError, ValueError):
        return False


def _count_arg_refusal(w: Workload) -> str | None:
    for key in _COUNT_ARGS:
        value = w.serving.get(key)
        if value not in (None, "") and not _is_count(value):
            return (f"{key}={value!r} is not the positive int vLLM v0.29.0 takes "
                    "(engine/arg_utils.py:493, :518-524, :583-586)")
    return None


#: Serving keys that turn on speculative decoding: EngineArgs (v0.29.0
#: engine/arg_utils.py:692-695) and gitm's draft-length knob (optimizer/vllm_knobs.py:100-102).
_SPEC_DECODE_KEYS = ("speculative_config", "spec_method", "spec_model", "spec_tokens",
                     "num_speculative_tokens")


def _decode_shape_refusal(w: Workload) -> str | None:
    """Why a decode step is not one query token per sequence on
    ``n_heads / tp`` heads per rank, the shape the derivation prices, or None."""
    if w.batch.speculative_tokens > 0 or any(w.serving.get(k) for k in _SPEC_DECODE_KEYS):
        return ("speculative decoding: verify steps carry more than one query token "
                "per sequence, which turns AITER's Gluon decode off "
                "(rocm_aiter_mla.py:1339-1340) and is not priced")
    dcp = int(w.serving.get("decode_context_parallel_size") or 1)
    if dcp > 1:
        return (f"decode_context_parallel_size={dcp}: each rank decodes {dcp}x the "
                "query heads (rocm_aiter_mla.py:1426-1428) over its shard of the "
                "cache (v1/attention/backends/utils.py:1091-1100), which is not priced")
    dp = max(int(w.serving.get("data_parallel_size") or 1), w.sharding.dp)
    if dp > 1:
        return (f"data_parallel_size={dp}: the DP groups step together on their combined "
                "token count (forward_context.py:80-99, :116-125), which the one TP group "
                "priced here does not do")
    return None


def _gluon_decode(w: Workload) -> bool:
    """Whether the bf16 arm decodes in AITER's Gluon ``_mla_gluon``: CDNA4 under
    16 heads per rank with 16 a multiple of the count (rocm_aiter_mla.py:1334-1353)."""
    heads = w.spec.n_heads // w.sharding.tp
    return (w.hw.arch == "cdna4" and heads < _AITER_MIN_MLA_HEADS
            and _AITER_MIN_MLA_HEADS % heads == 0)


def _fp8_extra_kernels(
    w: Workload, bf16_backend: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Kernels the fp8 arm runs that the bf16 arm does not: (per layer, per step).

    With an fp8 cache vLLM concatenates and quantises the query in one op
    (v0.29.0 mla_attention.py:945-950, :1415-1445), extra only where the bf16
    backend takes the (q_nope, q_pe) tuple as is: FLASH_ATTN_MLA
    (flashattn_mla.py:328-329) and AITER's Gluon path (rocm_aiter_mla.py:1770-1771).
    Gluon serves bf16 under 16 heads per rank (16 a multiple of the count) with one
    KV split and no reduce or schedule (:1090, :1798; mla_gluon.py:943-945); the
    fp8 arm pads the query to 16 heads, runs asm stage 1 then ``mla_reduce_v1``
    every layer (aiter/mla.py:656-694) and builds the schedule once per step
    (:1053-1077). At 16 or more heads both arms take the padded asm path.
    """
    heads = w.spec.n_heads // w.sharding.tp
    if _gluon_decode(w):
        return (("query concat+quant", f"{heads}->{_AITER_MIN_MLA_HEADS} head pad",
                 "mla_reduce_v1"), ("get_mla_metadata_v1",))
    if bf16_backend == "FLASH_ATTN_MLA":
        return ("query concat+quant",), ()
    return (), ()


#: AITER's persistent work slots: get_mla_metadata_v1 sizes its schedule by
#: multiProcessorCount (v1_2_device.cuh:847, :901-903), 256 CUs on gfx950.
_GFX950_CUS = 256


def _aiter_partial_tiles(n_seq: int, kv_len: int, num_splits: int = _GFX950_CUS) -> int:
    """Split-KV partial tiles AITER's persistent decode schedule makes for
    ``n_seq`` sequences of ``kv_len`` cached tokens, one query token each: phase 1
    of kn_get_mla_metadata_v1_2_parallel (AITER v0.1.19 v1_2_device.cuh:121-208)
    at vLLM's call (page size 1, granularity 16). Each sequence costs its 16-token
    blocks plus 16 of overhead, the total is cut into ``num_splits`` equal
    payloads, and a sequence spanning k > 1 slots leaves k partials (:200-204)."""
    gran = overhead = 16
    blocks = -(-kv_len // gran)
    payload = -(-(n_seq * (blocks + overhead)) // num_splits) + overhead
    per_slot = payload - overhead
    remain, tiles = payload, 0
    for _ in range(n_seq):
        if blocks + overhead <= remain:
            remain -= blocks + overhead
            continue
        if remain > overhead:
            rest = blocks - (remain - overhead)
            fresh = -(-rest // per_slot)
            frags = fresh + 1
        else:
            rest = blocks
            fresh = -(-rest // per_slot)
            frags = fresh
        remain = payload - ((rest - (fresh - 1) * per_slot) + overhead)
        tiles += frags if frags > 1 else 0
    return tiles


def _h002_mean_tiles(w: Workload) -> tuple[float, str]:
    """AITER's partial tiles per decode step, and how they were counted: with
    ``request_lengths`` (P, O) the mean over a request's O - 1 decode steps at
    P + 1 .. P + O - 1 cached tokens, the batch taken at one length per step
    (vLLM writes the step's own token before attention, mla_attention.py:703-713),
    else the count at ``kv_cache_len``."""
    b = w.batch
    if w.request_lengths is None:
        return (float(_aiter_partial_tiles(b.batch, b.kv_cache_len)),
                f"at {b.kv_cache_len} cached tokens only")
    prompt, output = w.request_lengths
    counts = [_aiter_partial_tiles(b.batch, n) for n in range(prompt + 1, prompt + output)]
    return (sum(counts) / len(counts),
            f"mean over {len(counts)} decode steps with the batch at one cached length, "
            f"{prompt + 1} to {prompt + output - 1}: {sum(counts)} / {len(counts)}, "
            f"{min(counts)} to {max(counts)} per step")


def _h002_predict(w: Workload, *, bf16_layer_s: float | None = None) -> Prediction:
    """The registered derivation; with ``bf16_layer_s`` (the baseline decode
    kernel's measured time per layer) the bf16 core is that time, not the band."""
    executed = replace(w.spec, kv_dtype=_executed_kv(w), kv_rope_dtype=_executed_kv(w))
    candidate = replace(w.spec, kv_dtype="fp8", kv_rope_dtype="fp8")
    g_old, g_new = _floor(executed, w), _floor(candidate, w)
    pair = _mla_backends(w)
    if isinstance(pair, str):
        raise ValueError(f"H-002 does not apply: {pair}")
    old_b, new_b, same = pair
    # Each extra kernel moves well under a microsecond of bytes, so it costs
    # its launch, and nothing hides it, so lo, mean and hi all pay it.
    per_layer, per_step = _fp8_extra_kernels(w, old_b)
    launch = w.hw.kernel_launch_overhead_s
    bw = w.hw.peak_mem_bw_bytes_per_s
    mid = (w.hw.eff_lo + w.hw.eff_hi) / 2.0
    launches = len(per_layer) * executed.n_layers + len(per_step)
    b = w.batch
    # Where only the fp8 arm reduces, stage 1 writes split-KV partials of 16 heads
    # x (fp32 output + fp32 LSE) (aiter/mla.py:607-616), priced with the fp8
    # kernel; the reduce reads them plus the bf16 output (:684-694) and, like any
    # node, costs max(launch, bytes / bandwidth), so only the part past its launch is new.
    m, lora = _AITER_MIN_MLA_HEADS, executed.kv_lora_rank
    tiles, counted = _h002_mean_tiles(w) if "mla_reduce_v1" in per_layer else (0.0, "none")
    partial_b = tiles * m * (lora * 4 + 4)
    writes_s = executed.n_layers * partial_b / bw
    reduce_s = (partial_b + b.batch * m * lora * 2) / bw if tiles else 0.0
    reduce_past_s = executed.n_layers * max(0.0, reduce_s - launch)
    kernels_floor = launches * launch + reduce_past_s
    extra_floor = kernels_floor + writes_s
    # Same basis as the unchanged rest of the step in _effect: floor / mid.
    extra = kernels_floor / mid
    old_attn = _op_time(g_old, "attn_score_value")
    new_attn = _op_time(g_new, "attn_score_value")
    layer_floor = old_attn / executed.n_layers
    old_eff = None if bf16_layer_s is None else layer_floor / bf16_layer_s
    saved, op, step = _effect(g_old, g_new, ("attn_score_value",), w.hw,
                              same_kernel=same, extra_s=(extra, extra, extra),
                              old_eff=old_eff, new_add_s=writes_s)
    new_ends = ((w.hw.eff_hi, mid, w.hw.eff_lo) if same and old_eff is None
                else (w.hw.eff_lo, mid, w.hw.eff_hi))
    ends = (w.hw.eff_hi, mid, w.hw.eff_lo)
    bf16_core = (
        f"bf16 core per layer: floor {layer_floor * 1e6:.1f} us; at eff "
        + " / ".join(f"{e:.2f}" for e in ends) + " = "
        + " / ".join(f"{layer_floor / e * 1e6:.1f}" for e in ends) + " us"
        + ("" if bf16_layer_s is None else
           f"; measured {bf16_layer_s * 1e6:.1f} us per layer = eff {old_eff:.3f}, "
           "used at every end")
    )
    arithmetic = (
        f"{w.hw.name}, TP{w.sharding.tp}: {b.batch} sequences x {b.kv_cache_len} cached "
        f"tokens, {executed.n_heads // w.sharding.tp} query heads per rank",
        f"cache entry {executed.kv_entry_dim} elems: {kv_entry_bytes(executed):.0f} B "
        f"stored vs {kv_entry_bytes(candidate):.0f} B as generic fp8",
        f"attention core reads {b.batch} x {b.kv_cache_len} entries x {executed.n_layers} "
        "layers per rank (replicated latent, TP does not split it)",
        f"floor: attn_score_value {old_attn * 1e3:.3f} -> {new_attn * 1e3:.3f} ms; "
        f"step {g_old.total_pred_s * 1e3:.3f} -> {g_new.total_pred_s * 1e3:.3f} ms, "
        f"{(g_new.total_pred_s + extra_floor) * 1e3:.3f} ms with the fp8-only kernels",
        f"fp8-only kernels: {len(per_layer)} per layer ({', '.join(per_layer) or 'none'}) "
        f"x {executed.n_layers} layers + {len(per_step)} per step "
        f"({', '.join(per_step) or 'none'}) = {launches} launches x {launch * 1e6:.1f} us "
        f"= {launches * launch * 1e3:.3f} ms floor",
        f"split-KV partials: {tiles:.2f} tiles ({counted}; {tiles / b.batch:.2f} per "
        f"sequence) x {m} heads x ({lora} x 4 + 4) B = {partial_b:.0f} B per layer, "
        "written by stage 1 "
        f"({partial_b / bw * 1e6:.3f} us, {executed.n_layers} layers = "
        f"{writes_s * 1e3:.3f} ms floor); the reduce reads them and writes the output "
        f"in {reduce_s * 1e6:.3f} us, charged past its {launch * 1e6:.1f} us launch only: "
        f"{reduce_past_s * 1e3:.3f} ms floor",
        f"fp8-only total {extra_floor * 1e3:.3f} ms floor: kernels "
        f"{kernels_floor * 1e3:.3f} ms / {mid:.2f} = {extra * 1e3:.3f} ms at lo, mean "
        f"and hi; stage 1 writes {writes_s * 1e3:.3f} ms at the fp8 kernel's efficiency: "
        + " / ".join(f"{writes_s / e * 1e3:.3f}" for e in new_ends) + " ms (lo/mean/hi)",
        bf16_core,
        f"backend {old_b} -> {new_b}"
        + (" (same kernel: one efficiency for both)" if same
           else " (different kernel: each end of the band independent)"),
        "saved " + " / ".join(f"{x * 1e3:.3f}" for x in saved) + " ms (lo/mean/hi)",
        "ITL reduction " + " / ".join(f"{x:.2%}" for x in step) + " (lo/mean/hi)",
    )
    return Prediction(
        metric="decode ITL (ms/step), p50 over the steady-state window",
        hardware=w.hw.name,
        baseline_step_s=g_old.total_pred_s,
        predicted_step_s=g_new.total_pred_s + extra_floor,
        total_overhead_s=(old_attn - new_attn) / mid,
        recoverable_s=saved,
        op_delta=op,
        step_delta=step,
        arithmetic=arithmetic,
    )


#: The lowest bf16 kernel efficiency ``rederive_h002`` accepts: 64 of MI355X's
#: 256 CUs at the band's low end is 0.55 x 64 / 256 = 0.14, and well under that
#: the number is more likely a unit error (ns read as us) or another kernel's time.
_H002_MIN_MEASURED_EFF = 0.05


def rederive_h002(w: Workload, bf16_layer_us: float) -> Prediction:
    """H-002's band with the bf16 core at the baseline arm's measured decode kernel
    time per layer (``_mla_gluon`` on MI355X), in microseconds; the fp8 kernel, the
    extra kernels and the rest of the step stay as registered. Refuses a workload
    H-002 refuses, a baseline whose decode is not one ``_mla_gluon`` launch per
    layer, and a time the model cannot use."""
    why = _h002_applies(w)
    if why is not None:
        raise ValueError(f"H-002 does not apply: {why}")
    if not _gluon_decode(w):
        raise ValueError(
            f"the bf16 arm at {w.spec.n_heads // w.sharding.tp} heads per rank on {w.hw.name} "
            "does not decode in one _mla_gluon launch per layer, the kernel time the "
            "re-derivation reads (doc, Experiment step 2)")
    if not math.isfinite(bf16_layer_us) or bf16_layer_us <= 0:
        raise ValueError(f"bf16 kernel time {bf16_layer_us!r} us is not a positive number")
    executed = replace(w.spec, kv_dtype=_executed_kv(w), kv_rope_dtype=_executed_kv(w))
    floor_us = _op_time(_floor(executed, w), "attn_score_value") / executed.n_layers * 1e6
    if bf16_layer_us <= floor_us:
        raise ValueError(
            f"bf16 kernel time {bf16_layer_us:g} us per layer is at or below its byte "
            f"floor {floor_us:.1f} us, an efficiency of {floor_us / bf16_layer_us:.2f}: "
            "the measurement (kernel, attribution, seconds passed as us?) or the floor "
            "(bytes per entry, cached tokens) is wrong")
    if floor_us / bf16_layer_us < _H002_MIN_MEASURED_EFF:
        raise ValueError(
            f"bf16 kernel time {bf16_layer_us:g} us per layer is "
            f"{bf16_layer_us / floor_us:.0f}x its byte floor {floor_us:.1f} us, an "
            f"efficiency under {_H002_MIN_MEASURED_EFF}: check the unit (ns passed as "
            "us?) and that the time is one _mla_gluon launch")
    return _h002_predict(w, bf16_layer_s=bf16_layer_us * 1e-6)


H002 = Hypothesis(
    id="H-002",
    title="An MLA cache stored bf16 where the engine could store fp8",
    doc="docs/hypotheses/H-002-mla-kv-fp8.md",
    bottleneck_class="memory_bound",
    knob="kv_cache_dtype",
    value="fp8",
    target_ops=("attn_score_value",),
    # Replay runs on the baseline (bf16) trace, so these are substrings of the
    # cache-reading decode kernels in source: AITER's _mla_gluon (MI355X bf16
    # under 16 heads per rank) and its bf16/fp8 asm decode kernels (mla_asm.csv),
    # FlashMLA's sm90 mains (a8f794d), CUTLASS_MLA's main kernel (v0.27.1).
    # Cache inserts and split-KV reduces also classify to attn_score_value but
    # do not read the bytes fp8 halves, so no needle matches them ("mla_decode"
    # would catch AITER's Triton _mla_decode_fwd_reduce_kernel). Not covered:
    # H200's default bf16 kernel, FA3's FlashAttnFwdSm90, which prefill runs
    # too, and B200's FlashInfer MLA kernels, named only in downloaded cubins.
    kernel_scope=("mla_gluon", "mla_a16w16", "mla_dec_stage1", "mla_a8w8",
                  "splitkv_mla_kernel", "Sm100FmhaMlaKernel"),
    applies=_h002_applies,
    predict=_h002_predict,
    requires_correctness_gate=True,
    # 2x the ~45 min one arm's GSM8K + needle takes (doc budget: 1.5 h, 2 arms).
    correctness_gate_timeout_s=5400.0,
    requires_hardware=("H100", "H200", "B200", "B300", "GB200", "GB300", "MI355X"),
    safety_notes=(
        "Kimi K2.5 ships no k_scale/v_scale, so vLLM v0.29.0 stores the cache and "
        "quantises the query at scale 1.0 (quantization/kv_cache.py:107-111, "
        ":129-136), and on MI355X first-chunk prefill runs in fp8 too: gate on the "
        "accuracy sentinel in docs/hypotheses/H-002-mla-kv-fp8.md before keep. On "
        "the pinned image that prefill drops heads 4-7 at 8 heads per rank "
        "(vllm-project/vllm#60907, fixed on vLLM main 2026-10-14, in no release) and "
        "AITER v0.1.19's a8w8 decode under-weights diffuse attention (fixed on AITER "
        "main by ROCm/aiter#6422 on 2026-10-14, in no release), so a gate failure "
        "there is an image defect, not a scale result: record it as blocked."
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
        try:
            pred = h.predict(self.workload)
        except UnsupportedExecution as e:
            return f"cannot price this workload: {e}"
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
                other=(f"{w.model} on {p.hardware}; predicted step reduction "
                       f"{p.step_delta[0]:.1%} / {p.step_delta[1]:.1%} / "
                       f"{p.step_delta[2]:.1%} (lo/mean/hi)"),
            ),
            safety=SafetyGate(tier="moderate", requires_rollback_window_s=120,
                              requires_correctness_gate=h.requires_correctness_gate,
                              correctness_gate_timeout_s=h.correctness_gate_timeout_s,
                              notes=h.safety_notes),
            correctness_gate=self.correctness_gate if h.requires_correctness_gate else None,
        )
