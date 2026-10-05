"""The 24-hour autonomous loop.

This is the orchestration glue — it composes tracer, planner, optimizer,
kernels, and agents in the 5 phases below. Each phase writes its artifact
to local scratch under ``<scratch>/runs/<run_id>/`` (see ``gitm._paths``) so a
partial run is still useful; the durable copy is synced to S3 afterwards.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gitm._paths import runs_dir, traces_dir
from gitm.agents.autoresearch import (
    AutoresearchRun,
    EngineArgsProposer,
    FallbackProposer,
    TableProposer,
    autoresearch,
    classify_bottleneck,
)
from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.library import load_library, parse_skips, skipped_by
from gitm.optimizer.apply import (
    Applicator,
    DryRunApplicator,
    LiveEngineApplicator,
    apply_intervention,
    resolve_restart_mode,
)
from gitm.optimizer.attribution import attribute
from gitm.optimizer.collective_signal import collective_causes, worst_device_comm
from gitm.optimizer.degradation import (
    AB_PROBE,
    AB_UNIT,
    AB_UNITS,
    AFFECTS_AB,
    AFFECTS_CLAIMS,
    AFFECTS_RESIDUALS,
    APPROXIMATE,
    AR_SKIPPED,
    ENGINE_LOST,
    GRAPH_BATCH,
    GRAPH_HARDWARE,
    GRAPH_MODEL,
    TOKENS,
    UNRELIABLE,
    WORKLOAD_RUNNER,
    Degradation,
    DegradationLog,
    ab_unit,
    unreliable_ab,
)
from gitm.optimizer.deviation import deviation_summary, deviation_trace, write_deviation_jsonl
from gitm.optimizer.dr import attribute_dr
from gitm.optimizer.history import load_history
from gitm.optimizer.measure import measure_trace, measurement_claims, measurement_summary
from gitm.optimizer.monitor import check_invariants, recoverable_by_op, residuals
from gitm.optimizer.qualification import qualify
from gitm.optimizer.report import Claim, build_provenance, write_report
from gitm.optimizer.scheduler_attribution import scheduler_causes
from gitm.optimizer.verification_export import (
    VerificationRecord,
    build_record,
    write_verification,
)
from gitm.optimizer.vllm_knobs import (
    KNOB_PREREQUISITES,
    expand_relative_candidates,
    knob_kind,
    unmet_prerequisite,
)
from gitm.planner.context import build_planner_context, hardware_spec_for
from gitm.planner.graph import predict_graph
from gitm.planner.moe_graph import (
    predict_moe_graph,
    spec_from_hf_config,
)
from gitm.safety.audit import AuditLog, _write_report
from gitm.tracer.capture import capture
from gitm.tracer.vllm_stats import sample_scheduler_stats, summarize_requests
from gitm.workloads import WorkloadRunner, get_factory, sync_device

_BUDGET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$")

# Workloads the predicted graph + intervention library actually model. Anything
# else gets a measurement-only report (see _measurement_result) rather than
# vLLM-specific intervention claims that wouldn't apply.
_LIBRARY_WORKLOADS = {"vllm-decode"}

# Workloads with a real, output-verified intervention applied through the
# rollback gate (not the vLLM library). Their runner carries an ``.applicator``
# (see gitm.workloads) so the loop can observe → attribute → select → apply →
# prove with a *measured* delta instead of a measurement-only report.
_HFT_INTERVENTION_WORKLOADS = {"hft", "hft-lob"}

# OpenFold/AF2 has a real, plDDT-gated intervention (bf16 inference) applied
# through the same rollback gate. Its runner carries an ``.applicator``.
_OPENFOLD_INTERVENTION_WORKLOADS = {"openfold", "alphafold", "af2"}

# Edge (3D LiDAR detection) has a real, detection-equivalence-gated intervention
# (fp16 autocast inference) applied through the same rollback gate. Its runner
# carries an ``.applicator``.
_EDGE_INTERVENTION_WORKLOADS = {"edge", "kitti", "nuscenes"}


def _parse_budget_s(budget: str) -> float:
    m = _BUDGET_RE.match(budget.lower())
    if not m:
        raise ValueError(f"unparseable budget: {budget!r} (use 24h, 90m, 3600s, 1d)")
    value, unit = float(m.group(1)), m.group(2)
    return value * {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]

def _note_ab(
    degradations: DegradationLog | None, stage: str, used: str, reason: str, severity: str
) -> None:
    """Record a throughput-probe fallback, when the caller passed a log."""
    if degradations is not None:
        degradations.record(stage, used=used, reason=reason, severity=severity,
                            affects=(AFFECTS_AB, AFFECTS_CLAIMS))


def _engine_throughput_fn(
    engine: Any, runner: Any, degradations: DegradationLog | None = None
) -> Any:
    """Resolve a decode-throughput probe for the live A/B.

    Prefers an explicit ``engine.gitm_throughput_fn`` (the engine owns what "a
    decode" means); otherwise times the workload ``runner`` and divides generated
    tokens by elapsed seconds. Re-running the runner re-runs the decode under the
    engine's current config, so an in-place hot-swap is reflected in the measurement.

    Contract: this default probe re-runs the (potentially expensive) full workload
    and is bound to the *original* engine, so it is only valid for in-place
    hot-swap A/Bs. A deployment that supplies ``gitm_restart_fn`` (structural-knob
    restart-apply, which swaps in a *new* engine) MUST also supply an engine-aware
    ``gitm_throughput_fn`` — the default cannot measure the restarted engine.

    Both halves of that contract are now enforced rather than documented, and
    each refusal is a raise: :func:`apply_intervention` turns a failed measure
    into a restore with the error on the record, so the candidate is neither kept
    nor written to ``verification.json`` as a measured loss.

    * No runner: there is nothing to time. The old probe timed an empty call and
      divided 1 by ~1e-9 s, so keep-or-rollback followed timer noise.
    * A restarted engine: the runner drives the engine it was built with, so
      timing it after a restart measures the old engine and credits the new one.
    * No token count in the runner's output: the probe still works, as workload
      runs per second, and the unit is recorded so no report calls it tok/s.
    * The unit is fixed by an A/B's first call and held for that A/B. A speedup
      is a ratio of two probe calls, so a runner that reports ``generated_tokens``
      on one call and nothing (or ``decode_steps``) on the next would divide one
      unit by another; that call raises and the A/B ends as an error. The lock is
      keyed by :attr:`DegradationLog.current_scope`, so the next candidate starts
      fresh and a consistent A/B in another unit is still measured.
    * Any count other than ``generated_tokens`` is recorded as an
      :data:`AB_UNIT` degradation naming its unit, so nothing reports steps or
      events as tok/s.
    """
    explicit = getattr(engine, "gitm_throughput_fn", None)
    if callable(explicit):
        return explicit

    if runner is None:
        why = "no workload runner and no engine.gitm_throughput_fn: nothing to time"
        _note_ab(degradations, AB_PROBE, "no throughput probe", why, UNRELIABLE)

        def _no_probe(_engine: Any) -> float:
            raise RuntimeError(why)

        return _no_probe

    # Per A/B: the unit of its first call, held for the rest of it. Keyed by the
    # candidate being measured; one key for everything when there is no log.
    locked: dict[str | None, str] = {}

    def _tps(_engine: Any) -> float:
        if _engine is not engine:
            why = ("the default probe drives the engine the runner was built with, "
                   "not the restarted one; supply engine.gitm_throughput_fn")
            _note_ab(degradations, AB_PROBE, "no measurement of the restarted engine", why,
                     UNRELIABLE)
            raise RuntimeError(why)
        t0 = time.perf_counter()
        out = runner()
        dt = max(time.perf_counter() - t0, 1e-9)
        # First key that is actually present wins — `or` would treat a legitimate
        # 0 (a window that produced no tokens) as missing and fabricate a count.
        unit, count = "runs", 1.0
        if isinstance(out, dict):
            for key in ("generated_tokens", "decode_steps", "events"):
                if out.get(key) is not None:
                    unit, count = key, float(out[key])
                    break
        ab = degradations.current_scope if degradations is not None else None
        first = locked.setdefault(ab, unit)
        if unit != first:
            why = (f"the runner reported {unit!r} after reporting {first!r} in the same A/B; "
                   "a speedup across the two would divide one unit by another")
            _note_ab(degradations, AB_PROBE, "no measurement in a mixed unit", why, UNRELIABLE)
            raise RuntimeError(why)
        if unit != TOKENS:
            counted = "" if unit == "runs" else f"; counted {unit} instead"
            _note_ab(degradations, AB_UNIT, AB_UNITS[unit][2],
                     "the runner reported no generated_tokens" + counted, APPROXIMATE)
        return count / dt

    return _tps


def _scheduler_note(s: Any) -> str | None:
    """One-line scheduler-stats sentence for the report, or None if no samples.

    ``s`` is a :class:`gitm.tracer.vllm_stats.SchedulerStatsSummary`; read
    duck-typed so an empty/absent summary degrades to no note rather than a crash.
    """
    if s is None or getattr(s, "n_samples", 0) == 0:
        return None
    parts: list[str] = []
    if s.peak_queue_depth is not None:
        parts.append(f"peak queue depth {s.peak_queue_depth}")
    if s.mean_batch_occupancy is not None:
        parts.append(f"mean batch occupancy {s.mean_batch_occupancy:.0%}")
    if s.total_preemptions is not None:
        parts.append(f"{s.total_preemptions} preemption(s)")
    if s.peak_gpu_cache_usage is not None:
        parts.append(f"peak KV-cache {s.peak_gpu_cache_usage:.0%}")
    if not parts:
        return None
    return "Engine scheduler: " + ", ".join(parts) + f" (over {s.n_samples} samples)."


@dataclass
class LoopConfig:
    engine: Any | None = None
    workload: str | None = None
    budget: str = "24h"
    target: float = 0.15
    scratch: str | None = None
    top_n_interventions: int = 5
    #: Rank levers from what previous runs measured on this GPU, instead of from
    #: the library's hand-authored estimates alone. ``None`` means nobody said,
    #: which reads as off: the loop never asks. ``gitm run`` puts the question to
    #: the operator and passes an explicit answer down.
    use_history: bool | None = None
    #: What to do with what this run learns, between one candidate and the next.
    #: ``"off"`` keeps the opening order to the end, which is how the loop has
    #: always behaved. ``"recapture"`` traces the workload again after each
    #: applied candidate and re-ranks what is left against it.
    rerank: str = "off"
    #: Levers this run must not try, as names or knobs with shell globs (see
    #: :func:`gitm.kernels.library.skipped_by`). Merged with ``GITM_SKIP_LEVERS``
    #: so a Kubernetes manifest can set it without a new flag in the pod spec.
    #:
    #: Exists because a lever can hang the model rather than merely regress: on
    #: Kimi at TP=8 on ROCm, n-gram speculative decoding stalls in an RCCL
    #: all-gather, and it ranks first in the catalogue, so every run met it
    #: before anything else. Without this the only way past was to edit the
    #: library.
    skip_levers: tuple[str, ...] = ()
    # Optional explicit driver for the embedded/engine path. When unset, the
    # loop looks up ``workload`` in the workload registry (gitm.workloads).
    workload_runner: WorkloadRunner | None = None


def _hf_config_from_engine(engine: Any) -> Any:
    """The live vLLM engine's HF config object, or ``None``. Duck-typed so it
    survives vLLM version drift; the config is where both graph-selection and the
    model shape are read from."""
    if engine is None:
        return None
    for path in (
        "llm_engine.model_config.hf_config",
        "llm_engine.vllm_config.model_config.hf_config",
        "model_config.hf_config",
    ):
        obj: Any = engine
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _hf_config_dict(hf: Any) -> dict[str, Any]:
    """A plain dict of the HF config, with ``quantization_config`` flattened.

    ``spec_from_hf_config`` reads the config as a dict and expects
    ``quantization_config`` to be one too; a live config carries it as a nested
    object, so normalise it here rather than teaching the dict-based builder about
    vLLM's object shapes."""
    raw = hf.to_dict() if hasattr(hf, "to_dict") else dict(vars(hf))
    q = raw.get("quantization_config")
    if q is not None and not isinstance(q, dict):
        raw["quantization_config"] = q.to_dict() if hasattr(q, "to_dict") else dict(vars(q))
    # Same treatment for the multimodal wrapper's text sub-config, and for the
    # same reason. ``vars()`` is shallow, so on a config with no ``to_dict`` the
    # inner config stays an *object*; ``registry.text_config`` unwraps dicts, so
    # a wrapped MoE would still read as dense on exactly that engine shape.
    inner = raw.get("text_config")
    if inner is not None and not isinstance(inner, dict):
        raw["text_config"] = inner.to_dict() if hasattr(inner, "to_dict") else dict(vars(inner))
    return raw


def _model_spec_from_hf(hf: Any):
    """Build a dense ``ModelSpec`` from an HF config object, or ``None``.

    ``predict_graph()`` with no model defaults to Llama-2-7B (32 layers). A run
    of a *different* model (e.g. opt-125m, 12 layers) is then scored against the
    wrong predicted graph, which makes residuals and deviation meaningless — so
    read the real architecture off the config. Duck-typed across vLLM version
    drift; any failure returns ``None`` and the caller falls back to the default
    graph rather than crashing.
    """
    if hf is None:
        return None
    # Same wrapper, read off an object rather than a dict: a multimodal config
    # carries the decode shapes on ``hf.text_config``, and reading ``hidden_size``
    # from the wrapper raises straight into the except below.
    hf = getattr(hf, "text_config", None) or hf
    try:
        from gitm.planner.roofline import ModelSpec

        hidden = int(hf.hidden_size)
        n_heads = int(hf.num_attention_heads)
        n_kv = int(getattr(hf, "num_key_value_heads", n_heads) or n_heads)
        head_dim = int(getattr(hf, "head_dim", 0) or (hidden // n_heads))
        moe = _moe_fields_from_hf(hf)
        return ModelSpec(
            hidden=hidden,
            n_layers=int(hf.num_hidden_layers),
            n_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            intermediate=int(getattr(hf, "intermediate_size", 4 * hidden)),
            vocab=int(hf.vocab_size),
            **moe,
        )
    except Exception:
        return None


def _model_spec_from_hf_explained(hf: Any):
    """``(spec, None)``, or ``(None, why)`` when the default graph will be used.

    :func:`_model_spec_from_hf` is kept for callers that only want the spec; this
    is what the loop records, because "no engine" and "a config field this
    reader could not parse" are different fixes for whoever reads the run.
    """
    if hf is None:
        return None, "no HF config to read"
    for need in ("hidden_size", "num_attention_heads", "num_hidden_layers", "vocab_size"):
        if getattr(hf, need, None) is None:
            return None, f"HF config has no {need!r}"
    spec = _model_spec_from_hf(hf)
    return (spec, None) if spec is not None else (None, "HF config fields did not parse")


def _model_spec_from_engine(engine: Any):
    """Dense ``ModelSpec`` for a live engine's model — the config, then the spec."""
    return _model_spec_from_hf(_hf_config_from_engine(engine))


#: HF config field aliases per MoE family — the same quantity is spelled
#: differently by Qwen / Mixtral / DeepSeek, so try each in order.
_MOE_ALIASES: dict[str, tuple[str, ...]] = {
    "num_experts": ("num_experts", "num_local_experts", "n_routed_experts"),
    "experts_per_token": ("num_experts_per_tok", "moe_top_k", "num_selected_experts"),
    "moe_intermediate": ("moe_intermediate_size", "expert_intermediate_size"),
    "shared_experts": ("n_shared_experts", "num_shared_experts"),
    "shared_expert_intermediate": ("shared_expert_intermediate_size",),
    # Per-layer placement: MoE checkpoints are not uniformly sparse.
    "first_dense_layers": ("first_k_dense_replace",),
    "moe_layer_step": ("decoder_sparse_step", "moe_layer_freq"),
}

#: Attention-shape aliases. Deliberately separate from the MoE table: hybrid
#: attention and a mixture FFN are independent choices, and a hybrid model with a
#: dense FFN (or a plain MoE transformer) must still get the right one.
_ATTN_ALIASES: dict[str, tuple[str, ...]] = {
    "full_attn_layer_step": ("full_attention_interval", "attn_layer_freq"),
}

#: quant_method -> bytes per weight element. MoE decode is weight-fetch bound,
#: so using the activation width for a quantized checkpoint would overstate the
#: dominant term by 2x (fp8) or 4x (4-bit).
_QUANT_WEIGHT_BYTES: dict[str, int] = {"fp8": 1, "compressed-tensors": 1, "modelopt_fp8": 1}


def _batch_config_from_stats(sched: Any) -> tuple[Any, str | None]:
    """``(BatchConfig, source)`` carrying the *observed* decode batch, or ``(None, None)``.

    ``predict_graph``'s default is ``batch=1``. That is wrong for any real serving
    window and especially wrong for a mixture-of-experts model, where weight
    traffic scales with the distinct experts a batch activates: at top-8 of 256,
    a batch of 1 touches 8 experts but a batch of 16 touches ~100, so scoring a
    batch-16 step against the batch-1 ceiling understates expert traffic by more
    than 10x.

    Two sources, in order of directness:

    ``mean_running`` is vLLM's own count of concurrently running sequences, which
    *is* the decode batch. It is also the one most often missing: it comes off the
    scheduler, and for the offline ``LLM`` engine the scheduler lives in a separate
    process (``VLLM_ENABLE_V1_MULTIPROCESSING`` defaults on), so nothing in this
    process can reach it.

    ``mean_bounded_inflight`` comes from ``get_num_unfinished_requests()``, a
    method on the engine handle itself and therefore readable whatever the engine
    core does. It counts requests in flight rather than requests decoding, so
    each sample is an upper bound: anything the scheduler has admitted is
    decoding, anything it has not is queued. Bounding each sample by
    ``max_num_seqs`` — the most the engine will ever decode at once — turns that
    into the batch wherever the queue is what the surplus is, which is the shape
    of every workload gitm submits (all prompts at once, then drain). The
    bounding happens per sample in :func:`~gitm.tracer.vllm_stats.summarize`,
    which is not the same as bounding the average and is the reason this reads a
    separate field rather than clamping ``mean_unfinished`` here. Without a
    capacity to bound against it stays unused rather than becoming a guess.

    Returns ``(None, None)`` when neither is available — a CPU box, a dry run, or
    an engine that exposes no stats. Better a documented default than a fabricated
    batch, and the caller records which of the two it got.

    ``kv_cache_len`` is deliberately left at its default: nothing in the sampled
    stats gives a token count (``peak_gpu_cache_usage`` is a fraction of blocks,
    not a length), and inventing one would move the full-attention ceiling on a
    guess. Sourcing it is tracked separately.
    """
    from gitm.planner.roofline import BatchConfig

    if sched is None or getattr(sched, "n_samples", 0) == 0:
        return None, None

    def _batch(value: Any) -> Any:
        return BatchConfig(batch=max(int(round(float(value))), 1))

    running = getattr(sched, "mean_running", None)
    if running is not None and running >= 1:
        return _batch(running), "running"

    bounded = getattr(sched, "mean_bounded_inflight", None)
    if bounded is not None and bounded >= 1:
        return _batch(bounded), "unfinished"
    return None, None


def _read_int_aliases(hf: Any, table: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """First positive int found for each field across its aliases. Duck-typed."""
    out: dict[str, Any] = {}
    for field, aliases in table.items():
        for alias in aliases:
            raw = getattr(hf, alias, None)
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                out[field] = value
                break
    return out


def _moe_fields_from_hf(hf: Any) -> dict[str, Any]:
    """Shape ``ModelSpec`` kwargs read off an HF config.

    Covers two *independent* axes — whether the FFN is a mixture, and whether
    attention is hybrid — so a hybrid-attention dense model still gets its
    attention shape, and a plain MoE transformer still gets its experts. Every
    field is optional; anything absent falls back to the dense/conventional
    default rather than being guessed. Tolerant of partial configs.
    """
    # Attention shape is independent of the FFN, so it survives the MoE gate.
    out: dict[str, Any] = _read_int_aliases(hf, _ATTN_ALIASES)
    # Weight width is independent of the FFN too. It used to be read only past
    # the MoE gate below, so a dense fp8 checkpoint priced every projection at
    # the bf16 width — a decode floor ~2x too slow, i.e. hidden headroom.
    wb = _quant_weight_bytes(getattr(hf, "quantization_config", None))
    if wb:
        out["weight_dtype_bytes"] = wb
    moe = _read_int_aliases(hf, _MOE_ALIASES)

    # Only a routed-expert count *and* a top-k make the FFN a mixture; without
    # both, leave the FFN dense rather than half-configured.
    if not (moe.get("num_experts") and moe.get("experts_per_token")):
        return out
    out.update(moe)
    return out


def _quant_weight_bytes(quant: Any) -> int | None:
    """Bytes per stored weight a ``quantization_config`` declares, or ``None``.

    ``None`` falls back to the activation width, which over-counts weight
    traffic and so predicts a slower floor — the direction that can't invent
    headroom. ``compressed-tensors`` is a container, not a width: it is 1 byte
    only when every weight group declares 8 bits. A W4A16 pack (Kimi's experts)
    is half a byte, which an integer ``weight_dtype_bytes`` cannot say, so it
    takes the conservative fallback rather than being priced as fp8.
    """
    if quant is None:
        return None
    q = quant if isinstance(quant, dict) else (
        quant.to_dict() if hasattr(quant, "to_dict") else dict(vars(quant)))
    method = q.get("quant_method")
    if not isinstance(method, str):
        return None
    wb = _QUANT_WEIGHT_BYTES.get(method.lower())
    if wb and method.lower() == "compressed-tensors":
        groups = q.get("config_groups")
        bits = [
            (g.get("weights") or {}).get("num_bits")
            for g in (groups.values() if isinstance(groups, dict) else ())
            if isinstance(g, dict)
        ]
        if not bits or any(b != 8 for b in bits):
            return None
    return wb


def _execution_graph_family(engine: Any, hw: Any, batch: Any):
    """``(graph, family)`` — :func:`_execution_graph_basis` without the reason."""
    graph, family, _why = _execution_graph_basis(engine, hw, batch)
    return graph, family


def _execution_graph_basis(engine: Any, hw: Any, batch: Any):
    """The predicted graph for the model that actually ran, its family, and —
    when it could not be read — why the default dense graph stands in.

    The third element is ``None`` when the graph describes the engine's own
    model. Otherwise every residual scored against it is scored against
    Llama-2-7B, and the caller must say so rather than write it beside real ones.

    The family comes from :func:`gitm.planner.registry.detect_family` — the same
    dispatch ``gitm plan`` and ``gitm deviate`` use — so the live loop can never
    price a checkpoint with a different graph than the offline tools do. That
    order matters: a GLM-5.2 (``glm_moe_dsa``) config carries ``index_topk`` and
    ``n_routed_experts`` exactly like a DeepSeek-V4 one, so testing
    :func:`is_sparse_moe_config` alone sent GLM to the V4 graph (compressed-KV,
    fp4 experts, no MLA absorb) and scored every GLM kernel against the wrong
    model. A Mixtral still falls through to the dense graph, whose FFN already
    prices a mixture (:func:`_moe_fields_from_hf`).

    Sharding stays whole-model so every family shares the dense path's
    comparison basis against the in-process trace, rather than predicting one
    rank against an all-rank capture.
    """
    from gitm.planner.registry import detect_family, text_config
    from gitm.planner.registry import spec_from_hf_config as family_spec
    from gitm.planner.roofline import ShardingConfig

    hf = _hf_config_from_engine(engine)
    if hf is not None:
        # The wrapper is stripped once here rather than inside each reader: a
        # multimodal checkpoint keeps every decode shape under ``text_config``,
        # and a predicate reading the top level finds nothing and resolves to
        # ``dense``.
        cfg = text_config(_hf_config_dict(hf))
        family = detect_family(cfg)
        name = str(cfg.get("model_type") or family)
        if family == "sparse_moe":
            spec = spec_from_hf_config(cfg, name=name)
            return predict_moe_graph(spec, hw, batch, ShardingConfig()), family, None
        if family == "glm_moe_dsa":
            from gitm.planner.glm_graph import predict_glm_graph

            return predict_glm_graph(family_spec(cfg, name=name), hw, batch,
                                     ShardingConfig()), family, None
        if family == "hybrid":
            from gitm.planner.hybrid_graph import predict_hybrid_graph

            return predict_hybrid_graph(family_spec(cfg, name=name), hw, batch,
                                        ShardingConfig()), family, None
    if hf is None:
        # More specific than the helper's "no HF config": which of the two it is
        # decides the fix (attach an engine vs. teach the reader a config path).
        spec, why = None, ("no engine attached" if engine is None
                           else "the engine exposes no HF config at any known path")
    else:
        spec, why = _model_spec_from_hf_explained(hf)
    return predict_graph(model=spec, hw=hw, batch=batch), "dense", why


def _execution_graph(engine: Any, hw: Any, batch: Any):
    """``(graph, is_sparse)`` — :func:`_execution_graph_family` with the family
    collapsed to "anything but the dense graph"."""
    graph, family = _execution_graph_family(engine, hw, batch)
    return graph, family != "dense"


def _clamp_pct(value: float) -> float:
    """Bound a residual ratio to +/-100% so a bad/misaligned prediction (or a
    small-sample outlier) can't blow up a report row into an absurd 18x."""
    return max(-1.0, min(1.0, value))


def _agg_kt_residual(res: Any) -> float:
    """Run-level kernel-time residual for the report: duration-weighted
    ``sum(obs - pred) / sum(pred)`` when timings are available, else the
    median per-kernel ratio. Same value for every catalog claim in a run."""
    rows = list(getattr(res, "per_kernel", []))
    if not rows:
        return 0.0

    total_obs = sum(float(kr.t_obs_s) for kr in rows if getattr(kr, "t_obs_s", None) is not None)
    total_pred = sum(float(kr.t_pred_s) for kr in rows if getattr(kr, "t_pred_s", None) is not None)
    if total_pred > 0.0:
        value = (total_obs - total_pred) / total_pred
    else:
        kts = sorted(float(kr.r_kt) for kr in rows)
        mid = len(kts) // 2
        value = kts[mid] if len(kts) % 2 else (kts[mid - 1] + kts[mid]) / 2.0
    return _clamp_pct(value)


def _ar_target_residual(ar_run: AutoresearchRun, fallback: float = 0.0) -> float:
    """Residual for autoresearch claims.

    Prefer the largest-residual op that autoresearch targeted; when there is no
    target, fall back to the run-level kernel-time residual so generated claims
    do not all display a misleading +0.0% gap.
    """
    return _clamp_pct(ar_run.target.residual) if ar_run.target is not None else fallback


def _ab_evidence(
    ab: Any, rolled_back: bool, measured_under: Iterable[Any], *, restore_failed: bool = False,
) -> str:
    """The claim sentence for a live A/B, in the unit the probe actually measured.

    ``measured_under`` is this candidate's own list
    (:meth:`DegradationLog.measured_under`), never the run's: a runs/s fallback
    in a later A/B says nothing about the unit of this one.
    """
    what, unit, _export = AB_UNITS[ab_unit(measured_under)]
    outcome = ("not kept, baseline not restored" if restore_failed
               else "rolled back" if rolled_back else "kept")
    return (
        f"live A/B: {outcome} ({ab.speedup - 1.0:+.1%} {what}, via {ab.via}); "
        f"baseline {ab.baseline_tps:.1f} → candidate {ab.candidate_tps:.1f} {unit}"
    )


def _record_graph_basis(
    log: DegradationLog, *, pctx: Any, batch: Any, batch_source: str | None,
    sched: Any, graph_default_why: str | None,
) -> None:
    """Record each default the predicted graph was built on.

    The model is ``unreliable``: a default dense graph is another model, and
    residuals against it describe nothing about this run. Hardware is
    ``approximate``: the graph is still this model, priced under a stated default.

    Having to default the *batch* is ``unreliable`` too, and used not to be.
    ``approximate`` was the first reading, on the same grounds as hardware —
    same model, stated default — but the measured gap on an MoE decode is ~30x
    (5.1 ms/step predicted against ~157 ms observed, 74% of kernels matching no
    graph op, 99.96% of residuals violating). A ceiling an order of magnitude
    under the floor describes this run no better than the wrong model does: every
    deviation computed against it is noise rather than a measurement with error
    bars. So it belongs with the default dense graph, and it ``AFFECTS_CLAIMS``
    as well as residuals, because the ranking reads the table it poisons.

    A batch that was *observed* but bounded (see ``_batch_config_from_stats``)
    stays ``approximate`` — that is a real measurement with a stated caveat.
    """
    if graph_default_why is not None:
        log.record(GRAPH_MODEL, used="default dense graph (Llama-2-7B shape)",
                   reason=graph_default_why, severity=UNRELIABLE,
                   affects=(AFFECTS_RESIDUALS, AFFECTS_CLAIMS))
    if getattr(pctx, "peak", None) is None:
        sku = getattr(pctx, "sku", None)
        log.record(GRAPH_HARDWARE, used="A100-SXM4-80GB peaks",
                   reason=(f"GPU SKU {sku!r} has no entry in the peak table" if sku
                           else "no GPU SKU (GITM_GPU_SKU unset and NVML gave no name)"),
                   severity=APPROXIMATE, affects=(AFFECTS_RESIDUALS,))
    if batch is None:
        n = getattr(sched, "n_samples", 0) if sched is not None else 0
        log.record(GRAPH_BATCH, used="batch=1",
                   reason=("no scheduler samples (no engine stats in the window)" if not n
                           else "no running-sequence count and no in-flight count to "
                                "fall back to (or no max_num_seqs to bound it)"),
                   severity=UNRELIABLE, affects=(AFFECTS_RESIDUALS, AFFECTS_CLAIMS))
    elif batch_source == "unfinished":
        log.record(GRAPH_BATCH, used=f"batch={batch.batch} from in-flight requests",
                   reason="the scheduler exposed no running count, so the batch is the "
                          "mean in-flight request count clamped to max_num_seqs — an "
                          "upper bound, exact only while nothing is queued",
                   severity=APPROXIMATE, affects=(AFFECTS_RESIDUALS,))
    log.record(GRAPH_BATCH, used="kv_cache_len=128 (BatchConfig default)",
               reason="the sampled scheduler stats carry no context length",
               severity=APPROXIMATE, affects=(AFFECTS_RESIDUALS,))


RERANK_MODES = ("off", "recapture")
"""What a run may do with what it learns between candidates.

Named here so the CLI's ``choices=`` and the embedded entry point agree. An
unrecognised value used to read as "off", so a typo in a script silently ran
without the thing it asked for.
"""


def _recapture(
    path, *, workload: str, run_id: str, runner
) -> tuple[Any, str | None]:
    """Trace the workload again, as it stands after what has been applied.

    The deviation profile moves as candidates land: a region the last one fixed
    is no longer where time is going, and ranking the rest against the opening
    trace asks where it *was* going. Coverage is measured per trace, so a fresh
    one re-ranks on its own without any new scoring rule.

    Affordable because ``measure()`` already runs this workload several times
    for the A/B — the extra cost is another run of something already running,
    not a new phase. It is deliberately taken *after* the gate has decided, so
    the tracing overhead never lands on the numbers that decide keep or
    rollback.

    Returns ``(trace, None)``, or ``(None, reason)`` when it could not be taken.
    A run that has already paid for its trace and its A/Bs must not be lost to a
    failed re-measurement, so the caller keeps the order it had — but the reason
    is carried out rather than swallowed. The workload failing during the extra
    run and the tracer being unavailable are different events for whoever reads
    the run afterwards, and recording both as "no trace" hides the first.
    """
    try:
        with capture(path, workload_id=workload, run_id=run_id) as trace:
            if runner is not None:
                try:
                    runner()
                except Exception as exc:
                    return None, f"workload run failed: {exc}"
                sync_device()
        return trace, None
    except Exception as exc:
        return None, f"capture failed: {exc}"


def run_loop(cfg: LoopConfig) -> dict[str, Any]:
    """Execute the 24-hour loop and return ``{summary, report_md, ...}``.

    Every fallback the run takes is recorded in one
    :class:`~gitm.optimizer.degradation.DegradationLog`: it travels on the
    provenance (report, ``verification.json``), is written to
    ``degradations.json`` on every path, and is summarised in the summary as
    ``degraded`` / ``degradations``.
    """
    degradations = DegradationLog()
    out = _run_loop(cfg, degradations)
    degradations.write(out["run_dir"])
    out["summary"]["degraded"] = bool(degradations.unreliable)
    out["summary"]["degradations"] = degradations.summary()
    return out


def _run_loop(cfg: LoopConfig, degradations: DegradationLog) -> dict[str, Any]:
    workload = cfg.workload or (getattr(cfg.engine, "workload_id", None) or "vllm-decode")
    if cfg.rerank not in RERANK_MODES:
        # Silently reading as "off" would let a scripted run spend its whole
        # budget without the thing it asked for, and say nothing.
        raise ValueError(
            f"rerank must be one of {RERANK_MODES}, got {cfg.rerank!r}")

    # Never prompts. Deciding that is the CLI's job, because a library entry
    # point that can block on stdin is one an embedded caller cannot use.
    use_history = bool(cfg.use_history)
    run_id = uuid.uuid4().hex
    budget_s = _parse_budget_s(cfg.budget)
    started_ns = time.time_ns()

    run_dir = runs_dir(cfg.scratch) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir(cfg.scratch) / f"{run_id}.jsonl"

    # Phase 1 — capture, fingerprint, predict graph
    # Resolve a workload runner: an explicit one wins, else the registry. The
    # runner launches GPU work *inside* the capture window so the trace reflects
    # the real workload instead of an empty no-op. Resolution happens outside
    # capture (data loading / warmup shouldn't be traced).
    runner = cfg.workload_runner
    runner_error: str | None = None
    if runner is None:
        factory = get_factory(workload)
        if factory is not None:
            try:
                runner = factory(cfg)
            except Exception as exc:  # missing deps/data on this box — degrade, don't crash
                runner_error = f"workload runner unavailable for {workload!r}: {exc}"
        else:
            runner_error = f"no workload runner registered for {workload!r}"

    # If the workload built a live engine (e.g. vLLM), expose it so the
    # scheduler-stats sampler AND the Phase-4 live A/B can drive it. The runner
    # carries it as ``.engine`` (see the vllm-decode factory). Without this the
    # loop stays predict-only (DryRunApplicator, live=False) — the engine is
    # built but never handed to the applicator.
    if cfg.engine is None and runner is not None:
        cfg.engine = getattr(runner, "engine", None)

    # A factory-built runner may know its own workload id better than the
    # guessed/default one above (e.g. a caller passed ``workload_runner``
    # directly with no ``cfg.workload``, so ``workload`` fell through to the
    # "vllm-decode" default regardless of what the runner actually is). It
    # must be re-checked here, after the runner is resolved, not folded into
    # the initial guess a few lines up — at that point neither the runner nor
    # ``cfg.engine`` (populated from it just above) exist yet.
    #
    # ``cfg.workload`` (the field) is never reassigned anywhere in this
    # function — it stays exactly the caller's original input for the whole
    # call, unlike the local ``workload`` var this line progressively
    # resolves. So this is an unambiguous "did the caller pin one explicitly"
    # check, not a proxy for it. Deliberately not falling back to
    # ``cfg.engine.workload_id`` here too: no current runner sets both an
    # engine and a top-level workload_id, so there's no real precedence
    # question yet — the runner's own attribute is preferred as the most
    # specific source when it exists.
    if cfg.workload is None and runner is not None:
        workload = getattr(runner, "workload_id", None) or workload

    # Sample the engine scheduler (queue depth, batch occupancy, preemptions)
    # over the same window as the CUPTI capture — engine-level telemetry the GPU
    # trace can't see. A no-op when no engine is attached (empty series).
    run_out: Any = None
    with (
        capture(trace_path, workload_id=workload, run_id=run_id) as trace,
        sample_scheduler_stats(cfg.engine) as sched_stats,
    ):
        if runner is not None:
            try:
                run_out = runner()
                sync_device()  # ensure all kernels land in the trace before stop
            except Exception as exc:
                runner_error = f"workload run failed: {exc}"
    if runner_error is not None:
        # Carried on every path, not only the no-data one: a runner that failed
        # part-way still leaves kernels, and residuals, deviations and an A/B
        # re-running the same runner all describe that fragment, not the workload.
        degradations.record(
            WORKLOAD_RUNNER,
            used="a partial workload run" if runner is not None else "no workload run",
            reason=runner_error, severity=UNRELIABLE,
            affects=(AFFECTS_RESIDUALS, AFFECTS_AB, AFFECTS_CLAIMS))

    # Persist the scheduler series + summary when an engine actually produced one.
    # Turn the summary into ranked causal hypotheses (feeds attribution / claim
    # evidence below) — empty when no engine produced samples.
    sched_summary = sched_stats.summary()
    sched_causes = scheduler_causes(sched_summary)
    # Per-request serving latency, when the runner reported request records
    # (vllm-decode does; synthetic runners don't). Same window as the scheduler
    # series and the trace — joined via SchedulerStatsSummary.t0_wall_ns.
    req_records = list(run_out.get("requests") or []) if isinstance(run_out, dict) else []
    serving_summary = summarize_requests(req_records) if req_records else None
    # Collective-communication causes from the same trace — ranked beside the
    # scheduler causes below. Empty when the trace holds no collective kernels.
    coll_causes = collective_causes(worst_device_comm(trace))
    if sched_stats.samples or serving_summary is not None:
        (run_dir / "scheduler_stats.json").write_text(
            json.dumps(
                {
                    "summary": asdict(sched_summary),
                    "samples": sched_stats.to_records(),
                    "serving": asdict(serving_summary) if serving_summary else None,
                    "requests": [asdict(r) for r in req_records],
                },
                indent=2,
            )
        )

    qual = qualify(trace, target_floor=cfg.target)
    (run_dir / "qualification.json").write_text(
        json.dumps(
            {
                "commit": qual.commit,
                "floor": qual.floor,
                "fingerprint": qual.fingerprint,
                "diagnostic": qual.diagnostic,
            },
            indent=2,
        )
    )

    # Parsed here, above the curated-intervention branches below, because each of
    # those *returns*. Parsing it at the catalogue instead left hft, openfold and
    # edge applying the very lever the operator excluded, which is worse than the
    # flag not existing: it was asked for and silently ignored.
    _skips = parse_skips([*cfg.skip_levers, os.environ.get("GITM_SKIP_LEVERS") or ""])
    _excluded: list[dict[str, str]] = []

    def _record_skip(spec: Any, source: str) -> str | None:
        """Note the exclusion of ``spec`` and return the pattern, or ``None``.

        Deduplicated on lever and source: autoresearch proposes the same knob
        repeatedly, and a count that grew with each proposal would describe the
        proposer rather than what the run held back.
        """
        if not _skips:
            return None
        why = skipped_by(spec, _skips)
        if why is None:
            return None
        entry = {"lever": spec.name, "knob": spec.knob, "pattern": why, "source": source}
        if entry not in _excluded:
            _excluded.append(entry)
        return why

    def _write_skips() -> None:
        """Written whether or not anything was excluded, so an empty list is the
        evidence that nothing was held back quietly."""
        (run_dir / "skipped_levers.json").write_text(
            json.dumps({"patterns": list(_skips), "excluded": _excluded}, indent=2))

    def _with_skips(result: dict[str, Any]) -> dict[str, Any]:
        """Put the exclusion count in a curated path's summary too.

        The curated results are built by their own functions and returned
        straight out, so without this an operator comparing two run summaries
        sees the count on a catalogue run and nothing on an hft one — and has to
        open the run directory to find out whether anything was held back.
        """
        summary = result.get("summary")
        if isinstance(summary, dict):
            summary["n_skipped_levers"] = len(_excluded)
        return result

    def _curated_skipped(spec_fn: Any, source: str) -> bool:
        """Whether the one curated lever on this workload was excluded.

        Read through a factory because each lives behind a lazy import in its own
        result function, and importing all three eagerly would pull three
        benchmark stacks into every run.
        """
        if not _skips:
            return False
        try:
            spec = spec_fn()
        except Exception:  # noqa: BLE001 - an unimportable benchmark is not a skip
            return False
        if _record_skip(spec, source) is None:
            return False
        _write_skips()
        return True

    # Written before the curated branches, each of which returns: the file is
    # promised whether or not anything was excluded, and an empty list is the
    # evidence that nothing was held back. Rewritten later as exclusions are
    # recorded.
    _write_skips()

    # HFT carries a real, output-verified intervention on its runner. Apply+prove
    # it through the rollback gate — the A/B runs on the active backend, so the
    # delta is measured even on a box without CUPTI. (Runs before the empty-trace
    # guard for that reason; attribution below is included only if kernels exist.)
    if workload in _HFT_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)

        def _hft_spec():
            from gitm.benchmarks.hft.optimize import hft_intervention_spec

            return hft_intervention_spec()

        if applicator is not None and not _curated_skipped(_hft_spec, "hft"):
            return _with_skips(_hft_intervention_result(
                degradations=degradations,
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            ))

    # OpenFold/AF2 carries the bf16 intervention on its runner. Same pattern as
    # HFT: apply+prove through the rollback gate (measure() runs the fp32-vs-bf16
    # A/B, gated on plDDT-equivalence). Before the empty-trace guard so the A/B
    # still runs on a box without CUPTI; attribution is included if kernels exist.
    if workload in _OPENFOLD_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)

        def _openfold_spec():
            from benchmarks.biotech.optimize import openfold_intervention_spec

            return openfold_intervention_spec()

        if applicator is not None and not _curated_skipped(_openfold_spec, "openfold"):
            return _with_skips(_openfold_intervention_result(
                degradations=degradations,
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            ))

    # Edge (kitti/nuscenes) carries the fp16 intervention on its runner. Same
    # pattern as HFT/AF2: apply+prove through the rollback gate (measure() runs
    # the fp32-vs-fp16 A/B, gated on detection-equivalence). Before the empty-
    # trace guard so the A/B still runs on a box without CUPTI.
    if workload in _EDGE_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)

        def _edge_spec():
            from gitm.benchmarks.edge.optimize import edge_intervention_spec

            # The same resolution _edge_intervention_result makes. Checking the
            # module default instead meant the guard and the run disagreed about
            # which lever this is: excluding the applicator's own lever did not
            # stop it, and excluding the default one stopped a lever nobody named.
            return getattr(applicator, "spec", None) or edge_intervention_spec()

        if applicator is not None and not _curated_skipped(_edge_spec, "edge"):
            return _with_skips(_edge_intervention_result(
                degradations=degradations,
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            ))

    # Guard: if the tracer captured nothing (no GPU/shim, or the workload never
    # ran), do NOT proceed to attribution + emit claims — that fabricates a
    # result from an empty trace. Report no-data honestly instead.
    if trace.vendor == "none" or not trace.kernels():
        diagnostic = runner_error or qual.diagnostic or (
            "Tracer captured no GPU kernels. Either no GPU/CUPTI shim is present, "
            "or the workload did not run under the runtime."
        )
        return _no_data_result(
            degradations=degradations,
            run_dir=run_dir,
            run_id=run_id,
            workload=workload,
            qual=qual,
            started_ns=started_ns,
            trace_path=trace_path,
            diagnostic=diagnostic,
        )

    # The predicted graph + intervention library model vLLM decode specifically.
    # For any other workload, pairing the real trace with that transformer graph
    # produces vLLM serving-knob "claims" that don't apply. Instead, emit an
    # honest measurement report computed from the actual captured kernels.
    if workload not in _LIBRARY_WORKLOADS:
        return _measurement_result(
            degradations=degradations,
            run_dir=run_dir,
            run_id=run_id,
            workload=workload,
            trace=trace,
            qual=qual,
            started_ns=started_ns,
            trace_path=trace_path,
        )

    # Predict against the model that ACTUALLY ran (read from the live engine) with
    # the graph its architecture needs — the sparse-MoE graph for a V4-class
    # checkpoint, the dense graph otherwise (_execution_graph). Falls back to the
    # default dense graph when there is no engine or its config can't be read (CPU
    # boxes, tests, dry-run), so residuals never score kernels against Llama-2-7B.
    #
    # Same for hardware: predict_graph's own default is A100-SXM4-80GB peaks,
    # which silently over-predicts on anything weaker (T4/L4/...) and
    # produces a run-level kernel-time residual that saturates the report's
    # +/-100% clamp on every claim. pctx is built here (moved up from Phase 3)
    # so its NVML-detected SKU peak feeds the graph before residuals are ever
    # computed against it.
    pctx = build_planner_context(cfg.engine, workload=workload)
    _hw = hardware_spec_for(pctx.peak)
    # Batch matters for the same reason the model does — and more so on a
    # mixture, where weight traffic follows the *distinct* experts a batch
    # activates: distinct(1)=top_k but distinct(16) is an order of magnitude
    # larger, so predicting a batch-16 step at the batch-1 default understates
    # expert traffic ~12x. Read the real concurrency off the sampled scheduler
    # rather than defaulting.
    _batch, _batch_source = _batch_config_from_stats(sched_summary)
    graph, family, graph_default_why = _execution_graph_basis(cfg.engine, _hw, _batch)
    _record_graph_basis(degradations, pctx=pctx, batch=_batch, batch_source=_batch_source,
                        sched=sched_summary, graph_default_why=graph_default_why)
    is_moe = family != "dense"
    _graph_summary: dict[str, Any] = {
        "graph": "moe" if is_moe else "dense",
        "family": family,
        "nodes": len(graph.nodes),
        "total_pred_s": graph.total_pred_s,
        "hardware": _hw.name,
    }
    if is_moe:
        m = graph.model
        # Read duck-typed: the GLM and hybrid specs spell their dtypes their own
        # way, and a summary field must never be the thing that crashes a run.
        _graph_summary.update(
            model=getattr(m, "name", family),
            sharding="whole-model",
            dtypes={k: getattr(m, attr, None) for k, attr in (
                ("weight", "weight_dtype"), ("expert", "expert_dtype"), ("kv", "kv_dtype"))},
            has_unpriced_collectives=graph.has_unpriced_collectives,
        )
    # What the graph was built from, beside what it is: a residual is only as
    # good as the model, hardware and batch it was predicted for.
    #
    # The batch is stated outright rather than left to be read out of ``basis``.
    # ``basis`` is built from degradations, and a batch read straight off the
    # engine's running count is not a degradation — so on the one path where the
    # batch is fully trustworthy the artifact said nothing about it at all, and a
    # reader could not tell which batch priced the graph or where it came from.
    # That is the path whose number you most want recorded when comparing two
    # runs.
    # Read off the graph rather than rebuilt from ``_batch``: every predict_*
    # entry point does ``batch = batch or BatchConfig()`` and stores the result,
    # so the graph holds the config it was actually priced with, defaults
    # included. Reconstructing it here would be a second copy of that fallback,
    # free to drift from the one that did the work.
    _eff_batch = getattr(graph, "batch", None)
    _graph_summary["batch"] = {
        "batch": getattr(_eff_batch, "batch", None),
        "kv_cache_len": getattr(_eff_batch, "kv_cache_len", None),
        "source": _batch_source or "default",
    }
    _graph_summary["basis"] = [d.to_dict() for d in degradations
                               if d.stage in (GRAPH_MODEL, GRAPH_HARDWARE, GRAPH_BATCH)]
    (run_dir / "predicted_graph.json").write_text(json.dumps(_graph_summary, indent=2))

    # Phase 2 — residuals + attribution
    res = residuals(trace, graph)
    violations = check_invariants(res)  # multi-basis confirmed
    hypotheses = attribute(res, graph)  # Granger
    dr_hypotheses = attribute_dr(res, graph)  # doubly-robust, corroborating

    (run_dir / "violations.json").write_text(
        json.dumps(
            [
                {
                    "invariant": v.invariant,
                    "node_op": v.node_op,
                    "layer": v.layer,
                    "residual": v.residual,
                    "severity": v.severity,
                }
                for v in violations
            ],
            indent=2,
        )
    )
    (run_dir / "residuals.json").write_text(
        json.dumps(
            {
                "n_kernel_residuals": len(res.per_kernel),
                "n_violations": len(violations),
                "serialized_concurrency_fraction": res.serialized_concurrency_fraction,
                "top_hypotheses_granger": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value}
                    for h in hypotheses.top(5)
                ],
                "top_hypotheses_doubly_robust": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value,
                     "notes": h.notes}
                    for h in dr_hypotheses.top(5)
                ],
                # Engine-scheduler causes (from the vLLM stats adapter) ranked
                # alongside the kernel-level hypotheses (the engine-signal causal link).
                "scheduler_causes": [
                    {"signal": c.signal, "effect": c.effect, "severity": c.severity,
                     "note": c.note, "motivates_knobs": c.motivates_knobs}
                    for c in sched_causes
                ],
                # Collective (NCCL) causes — communication time the kernel-time
                # residuals can't distinguish from compute. Empty on single-GPU
                # runs and on any trace with no collective kernels.
                "collective_causes": [
                    {"signal": c.signal, "effect": c.effect, "severity": c.severity,
                     "note": c.note, "motivates_knobs": c.motivates_knobs}
                    for c in coll_causes
                ],
            },
            indent=2,
        )
    )

    # Deviation-only tracing: record only the kernels that *departed* from the
    # predicted graph — trace storage scales with deviation, not duration. We
    # always write the compact summary (n_kept, reduction, which ops departed);
    # the full reduced JSONL is written only under GITM_DEVIATION_ONLY=1 (it is
    # the storage-saving artifact, off by default while capture-time integration
    # is still on the roadmap).
    (run_dir / "deviations.json").write_text(
        json.dumps(deviation_summary(trace, graph), indent=2)
    )
    if os.environ.get("GITM_DEVIATION_ONLY") == "1":
        write_deviation_jsonl(deviation_trace(trace, graph), run_dir / "deviation_trace.jsonl")

    # Phase 3 — library + counterfactual replay ranking
    # pctx was built earlier (Phase 1) so its hardware peak could feed predict_graph.
    # Relative/swept levers resolve against the live engine here, once, before
    # ranking. See expand_relative_candidates.
    # Levers the operator excluded, applied before anything resolves them against
    # the engine, and outside load_library so the exclusion covers this run only
    # and the catalogue on disk stays the catalogue.
    _catalogue = [s for s in load_library(workload=workload)
                  if _record_skip(s, "catalogue") is None]
    library = [
        resolved
        for s in _catalogue
        for resolved in expand_relative_candidates(s, cfg.engine)
    ]
    _write_skips()
    policy = Policy(require_qualification_commit=qual.commit, skip_high_risk=not qual.commit,
                    use_history=use_history)
    # Read once per run, filtered to this box. A lever measured on another GPU is
    # not evidence about this one, and load_history counts what it filtered out
    # rather than letting a thin record look like a weak lever.
    prior_runs = (load_history(runs_dir(cfg.scratch), gpu_sku=pctx.sku,
                               fingerprint=qual.fingerprint) if use_history else None)
    if prior_runs is not None:
        (run_dir / "history_read.json").write_text(json.dumps({
            "runs_read": prior_runs.runs_read,
            "filtered": prior_runs.filtered,
            "skipped": prior_runs.skipped,
            "excluded": prior_runs.excluded,
            "levers": len(prior_runs.records),
            "gpu_sku": pctx.sku,
            "fingerprint": qual.fingerprint,
        }, indent=2))
    # Where time is actually recoverable, per op, from the residuals already
    # computed above. This is the first thing in selection that depends on the
    # trace rather than on the catalogue: a lever whose gain comes from speeding
    # up a region measured at its predicted floor is not a candidate, however
    # well it scores.
    #
    # A floor is only usable here if it was priced for this run, which takes two
    # things and not one:
    #
    # * the graph must be this model's. Against a default dense graph the floors
    #   describe another model entirely.
    # * the hardware peaks must have resolved. This one is the trap: the A100
    #   fallback on a *faster* device predicts floors slower than the device
    #   actually achieves, so observed comes in under predicted, every gap reads
    #   zero, and the gate rejects the whole op-scoped catalogue on what looks
    #   like a clean measurement. The hardware fallback is already recorded as
    #   affecting residuals, and this gate consumes residuals.
    #
    # A defaulted *batch* is not in that list because it fails the other way: at
    # batch 1 the floors are far too low, so everything reads as over its floor
    # and nothing is filtered. Useless, but not wrong. Same for the kv_cache_len
    # default, which understates the attention floor and so keeps attention
    # levers rather than dropping them.
    _floors_priced_for_this_run = (graph_default_why is None
                                   and getattr(pctx, "peak", None) is not None)
    _recoverable = recoverable_by_op(res) if _floors_priced_for_this_run else None
    ranked = select_interventions(trace, library, policy, top_n=cfg.top_n_interventions,
                                  ctx=pctx.gate, history=prior_runs, gpu_sku=pctx.sku,
                                  fingerprint=qual.fingerprint, recoverable=_recoverable)
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [
                {
                    "name": c.spec.name,
                    "predicted_delta": c.predicted_delta,
                    "rejected_reason": c.rejected_reason,
                }
                for c in ranked
            ],
            indent=2,
        )
    )
    # The gate's own input, saved beside what it decided. A lever dropped for
    # "no_recoverable_time" is a claim about the trace, and the claim is only
    # checkable if the measured gaps are written down next to it.
    (run_dir / "recoverable_ops.json").write_text(
        json.dumps(
            {
                "basis": ("per-kernel residuals against this model's graph"
                          if _recoverable is not None else None),
                "not_gated_because": (
                    None if _recoverable is not None
                    else "the predicted graph is a default, so its per-op floors "
                         "describe another model" if graph_default_why is not None
                    else "no GPU peaks resolved, so the floors were priced on the "
                         "A100 fallback and a faster device reads as at its floor"),
                "recoverable_s": _recoverable,
            },
            indent=2,
        )
    )

    # Phase 4 — apply with rollback gates.
    # With a live engine attached, each candidate runs the rollback-gated decode-
    # throughput A/B (LiveEngineApplicator): snapshot baseline tps, apply the
    # candidate, measure candidate tps, keep only on a non-negative delta, else
    # restore. vLLM EngineArgs are routed through ``engine.gitm_restart_fn``
    # (if the deployment provides one) because the real engine reads them at
    # construction time. With no engine it is predict-only (DryRunApplicator):
    # candidates land in the report as unverified (measured_delta=None), never
    # claimed as won.
    live_restart_fn = getattr(cfg.engine, "gitm_restart_fn", None) if cfg.engine else None
    if cfg.engine is not None:
        # Serial wherever a baseline rebuild is available, rather than parallel
        # by default. Parallel holds the baseline and the candidate at once, and
        # at any realistic gpu_memory_utilization the second one cannot fit: it
        # is what cost the MI355X run 27 of its 29 candidates.
        _restart_mode, _restart_why = resolve_restart_mode(
            cfg.engine, os.environ.get("GITM_RESTART_MODE"))
        applicator: Applicator = LiveEngineApplicator(
            cfg.engine,
            throughput_fn=_engine_throughput_fn(cfg.engine, runner, degradations),
            restart_fn=live_restart_fn,
            baseline_restart_fn=getattr(cfg.engine, "gitm_baseline_restart_fn", None),
            restart_mode=_restart_mode,
            reps=int(os.environ.get("GITM_AB_REPS", "1")),
            # Compatibility escape hatch for custom scheduling-classified knobs
            # that should still be measured through engine rebuild.
            force_restart=os.environ.get("GITM_KNOBS_VIA_RESTART") == "1",
        )
        (run_dir / "restart_mode.json").write_text(json.dumps({
            "mode": _restart_mode,
            "why": _restart_why,
            "cannot_fit": applicator.restart_mode_warning,
        }, indent=2))
        if applicator.restart_mode_warning:
            # Warned, not recorded as a degradation. A run-wide AFFECTS_AB entry
            # would mark every candidate unreliable, including the hot-swapped
            # ones that never reach a rebuild — excluding their perfectly good
            # measurements from history and from the report's verified count.
            # The condition only harms candidates that need a restart, and those
            # already carry their own error when the rebuild is refused.
            #
            # Said once, at the start, because the alternative is learning it
            # from an OOM traceback per candidate: how the last run spent 93% of
            # its budget.
            warnings.warn(
                "gitm: most structural candidates will be refused in this run — "
                + applicator.restart_mode_warning,
                RuntimeWarning, stacklevel=2)
    else:
        applicator = DryRunApplicator()

    def _find_motivating_cause(spec: Any) -> tuple[str, Any] | None:
        for knob in spec.knob_values:
            hit = next((sc for sc in sched_causes if knob in sc.motivates_knobs), None)
            if hit is not None:
                return ("scheduler", hit)
            hit = next((cc for cc in coll_causes if knob in cc.motivates_knobs), None)
            if hit is not None:
                return ("collective", hit)
        return None

    def _has_structural_knob(spec: Any) -> bool:
        return any(knob_kind(k) == "structural" for k in spec.knob_values)

    claims: list[Claim] = []
    rolled_back: list[str] = []
    rejected: list[str] = []
    # Customer-verification records: the full A/B behind each claim, captured as
    # it happens. EngineABResult lives on applicator.last_result and is
    # overwritten by the next candidate, so it has to be taken per-iteration.
    verification: list[VerificationRecord] = []
    # Aggregate kernel-time residual for the report (was hardcoded 0.0). Same for
    # every claim in a run — it describes the run's gap vs the predicted graph.
    kt_residual = _agg_kt_residual(res)
    # A queue rather than a fixed list. `for c in ranked` walked an order decided
    # before a single candidate had been measured, so nothing the run learned
    # could reach the next choice until the following run read the export back.
    # Popping from the front leaves `queue` as exactly what is still unattempted,
    # which is what a re-rank has to re-order — and a candidate already popped,
    # applied or rejected, cannot return.
    queue = list(ranked)
    reranks: list[dict[str, Any]] = []
    # Set when a candidate's rollback failed. From then on there is no baseline
    # to measure against, so the run stops trying candidates and goes straight
    # to writing up what it has.
    engine_lost: str | None = None
    n_untried = 0
    while queue:
        c = queue.pop(0)
        if c.rejected_reason is not None:
            rejected.append(f"{c.spec.name} ({c.rejected_reason})")
            continue
        # Live + structural knob + no restart hook → it *cannot* be enacted on the
        # running engine, so it's "not evaluable here", not a regression. Mark it
        # rejected (honest) instead of attempting an apply that would roll back and
        # read as "tried and lost" — and skip the wasted baseline benchmark.
        if cfg.engine is not None and live_restart_fn is None and _has_structural_knob(c.spec):
            rejected.append(f"{c.spec.name} (structural knob: needs engine restart, no restart_fn)")
            continue
        # Snapshot the engine config BEFORE the apply: a hot-swap mutates these
        # kwargs in place and a restart replaces the engine outright, so reading
        # them afterwards would report the candidate on both sides of the diff.
        baseline_cfg = dict(getattr(getattr(applicator, "engine", cfg.engine), "gitm_llm_kwargs", None) or {})
        with degradations.scope(c.spec.name):
            result = apply_intervention(c.spec, applicator, min_keep_delta=0.0)
        measured_under = degradations.measured_under(c.spec.name)
        ab = (
            getattr(applicator, "last_result", None)
            if result.measured_delta is not None
            else None
        )
        if result.rolled_back:
            rolled_back.append(c.spec.name)
        if ab is not None:
            candidate_cfg = {**baseline_cfg, **(c.spec.knobs or {c.spec.knob: c.spec.value})}
            verification.append(
                build_record(
                    c.spec, ab, result,
                    baseline_config=baseline_cfg,
                    candidate_config=candidate_cfg,
                    degradations=measured_under,
                )
            )
        # Causal evidence: the measured A/B verdict when live, else the Granger
        # signal that motivated the candidate. The kept/rolled-back wording comes
        # from the authoritative ApplyResult (the real gate decision), not from
        # EngineABResult.kept (a measure-time delta>=0 indicator).
        if ab is not None:
            causal_evidence = _ab_evidence(ab, result.rolled_back, measured_under,
                                           restore_failed=result.restore_failed)
        else:
            causal_evidence = ", ".join(
                f"{h.cause_op}→{h.effect_op} (p={h.p_value:.2g})" for h in hypotheses.top(2)
            ) or "no strong causal signal"
        if result.error is not None and result.measured_delta is None:
            causal_evidence += f"; apply failed: {result.error}"
        motivating = _find_motivating_cause(c.spec)
        if motivating is not None:
            channel, cause = motivating
            causal_evidence += f"; {channel}[{cause.signal}]: {cause.note}"
        claims.append(
            Claim(
                summary=c.spec.summary,
                residual_invariant="kernel_time",
                residual_value=kt_residual,
                causal_evidence=causal_evidence,
                intervention_name=c.spec.name,
                predicted_delta=c.predicted_delta,
                # Display the TRUE measured delta (speedup-1); the gate uses the noise-adjusted
                # return, so a within-noise gain reads as rolled back
                # with its real (small) number, not a distorted one.
                measured_delta=((ab.speedup - 1.0) if ab is not None else result.measured_delta),
                rolled_back=result.rolled_back,
                restore_failed=result.restore_failed,
                unreliable_ab=unreliable_ab(measured_under) if ab is not None else [],
            )
        )
        if result.restore_failed:
            engine_lost = result.error or f"restore failed after {c.spec.name}"
            n_untried = sum(1 for x in queue if x.rejected_reason is None)
            # What the gate rejected is a verdict that needed no engine, so it is
            # still recorded. Breaking here without it left those candidates in
            # neither the rejected list nor the untried count.
            rejected.extend(f"{x.spec.name} ({x.rejected_reason})"
                            for x in queue if x.rejected_reason is not None)
            # Approximate, not unreliable. The A/Bs measured before this one
            # were taken against a sound baseline, and an unreliable mark here
            # would exclude them from history along with everything else.
            degradations.record(
                ENGINE_LOST, used=f"a run that stopped after {c.spec.name}",
                reason=f"{engine_lost}; {n_untried} ranked candidate(s) not tried",
                severity=APPROXIMATE, affects=(AFFECTS_CLAIMS,))
            break
        if time.time_ns() - started_ns >= int(budget_s * 1e9):
            break

        if cfg.rerank == "recapture" and queue:
            # After the gate has decided, so the tracing overhead never lands on
            # the A/B that decides keep or rollback.
            step = len(reranks) + 1
            fresh, why = _recapture(
                traces_dir(cfg.scratch) / f"{run_id}-rerank{step}.jsonl",
                workload=workload, run_id=run_id, runner=runner)
            was = [x.spec.name for x in queue]
            if fresh is not None and fresh.kernels():
                # Deliberately not gated on recoverable time. The fresh trace
                # would have to be compared against ``graph``, which was priced
                # for the engine as it opened — and by this point a kept
                # whole-step candidate may have changed the batch shape the
                # floors assume, so an op could read as at a floor that no
                # longer describes the running workload. Re-pricing the graph
                # needs the engine re-sampled, which this path does not do.
                #
                # Little is lost: the queue was already filtered at selection,
                # so re-gating could only add rejections, and those are exactly
                # the ones resting on the stale floors. Coverage is still
                # recomputed per trace, which is what re-ranking is for.
                queue = select_interventions(
                    fresh, [x.spec for x in queue], policy, top_n=len(queue),
                    ctx=pctx.gate, history=prior_runs, gpu_sku=pctx.sku,
                    fingerprint=qual.fingerprint)
            now = [x.spec.name for x in queue]
            reranks.append({
                "after": c.spec.name,
                "measured_delta": result.measured_delta,
                "recaptured": fresh is not None,
                "error": why,
                "order_before": was,
                "order_after": now,
                "changed": was != now,
            })
            # The re-capture runs the workload, so it spends budget. Checked
            # again here because the check above ran before that spend: a
            # candidate cycle started on its strength could overrun by a whole
            # A/B on top of the trace.
            if time.time_ns() - started_ns >= int(budget_s * 1e9):
                break

    if reranks:
        # What the run re-decided, and on what. Without it a report says which
        # candidates were tried but not that the order moved, nor why.
        (run_dir / "rerank.json").write_text(json.dumps({
            "mode": cfg.rerank, "steps": reranks,
        }, indent=2))

    # Phase 4b - agentic autoresearch through the catalog gate/rollback path.
    if engine_lost is None and time.time_ns() - started_ns < int(budget_s * 1e9):
        proposer = FallbackProposer(EngineArgsProposer(), TableProposer())

        def _unenactable(spec: Any) -> str | None:
            # The exclusion has to reach here too. Autoresearch proposes its own
            # candidates rather than drawing from the catalogue, so a knob the
            # operator excluded because it hangs the model would otherwise come
            # straight back as a proposal.
            why = _record_skip(spec, "autoresearch")
            if why is not None:
                return f"excluded by --skip-lever {why!r}"
            if (
                cfg.engine is not None
                and live_restart_fn is None
                and _has_structural_knob(spec)
            ):
                return "structural knob: needs engine restart, no restart_fn"
            values = spec.knob_values
            # The engine running *now*: a Phase-4 restart that was kept has
            # replaced cfg.engine, and a prerequisite it turned on (or off) is
            # visible only on the engine the applicator holds.
            engine_now = getattr(applicator, "engine", None) or cfg.engine
            for k in values:
                reason = unmet_prerequisite(engine_now, k)
                if reason is None:
                    continue
                prereq = next(
                    (p for needle, p in KNOB_PREREQUISITES if needle in k.lower()),
                    None,
                )
                if prereq in values:
                    continue
                return reason
            return None

        ar_run = autoresearch(
            trace,
            applicator=applicator,
            policy=policy,
            residuals=res,
            proposer=proposer,
            ctx=pctx.gate,
            reject=_unenactable,
            history=prior_runs,
            gpu_sku=pctx.sku,
            fingerprint=qual.fingerprint,
            degradations=degradations,
        )
    else:
        # An empty result list reads the same as "searched and found nothing";
        # say that the pass never ran, and why.
        ar_run = AutoresearchRun(bottleneck_class=classify_bottleneck(trace, res), results=[])
        ar_run.degradations.append(Degradation(
            AR_SKIPPED, used="no autoresearch pass",
            reason=(f"engine lost in Phase 4: {engine_lost}" if engine_lost is not None
                    else f"budget {cfg.budget} exhausted by Phase 4"),
            severity=APPROXIMATE, affects=(AFFECTS_CLAIMS,)))
    degradations.extend(ar_run.degradations)
    lost_in_ar = next((r for r in ar_run.results
                       if r.apply_result is not None and r.apply_result.restore_failed), None)
    if engine_lost is None and lost_in_ar is not None:
        engine_lost = lost_in_ar.apply_error or f"restore failed after {lost_in_ar.spec.name}"
        n_untried = ar_run.n_untried
        degradations.record(
            ENGINE_LOST, used=f"an autoresearch pass that stopped after {lost_in_ar.spec.name}",
            reason=f"{engine_lost}; {n_untried} ranked candidate(s) not tried",
            severity=APPROXIMATE, affects=(AFFECTS_CLAIMS,))
    # Again, now that autoresearch has had its proposals vetoed: an exclusion
    # that only stopped a proposal is still something the run held back, and a
    # file written before that pass would report none of them.
    _write_skips()

    # A kept restart replaced the engine and released the one cfg.engine still
    # points at: hand the caller the live engine, not a shut-down handle.
    if cfg.engine is not None:
        cfg.engine = getattr(applicator, "engine", cfg.engine)

    ar_granger_evidence = ", ".join(
        f"{h.cause_op}→{h.effect_op} (p={h.p_value:.2g})" for h in hypotheses.top(2)
    ) or "no strong causal signal"
    ar_residual = _ar_target_residual(ar_run, kt_residual)
    for r in ar_run.results:
        if not r.applicable:
            rejected.append(f"{r.spec.name} ({r.rejected_reason})")
            continue
        if r.rolled_back:
            rolled_back.append(r.spec.name)
        ar_ab = r.ab_result
        ar_lost = r.apply_result is not None and r.apply_result.restore_failed
        if ar_ab is not None:
            evidence = _ab_evidence(ar_ab, r.rolled_back, r.degradations,
                                    restore_failed=ar_lost)
        else:
            evidence = ar_granger_evidence
        if r.measured_delta is None and r.apply_error:
            evidence += f"; apply failed: {r.apply_error}"
        motivating = _find_motivating_cause(r.spec)
        if motivating is not None:
            channel, cause = motivating
            evidence += f"; {channel}[{cause.signal}]: {cause.note}"
        true_delta = (ar_ab.speedup - 1.0) if ar_ab is not None else r.measured_delta
        claims.append(
            Claim(
                summary=r.spec.summary,
                residual_invariant="kernel_time",
                residual_value=ar_residual,
                causal_evidence=evidence,
                intervention_name=r.spec.name,
                predicted_delta=r.predicted_delta,
                measured_delta=true_delta,
                rolled_back=r.rolled_back,
                restore_failed=ar_lost,
                unreliable_ab=unreliable_ab(r.degradations) if ar_ab is not None else [],
            )
        )
        if ar_ab is not None and r.apply_result is not None:
            verification.append(
                build_record(
                    r.spec, ar_ab, r.apply_result,
                    baseline_config=r.baseline_config or {},
                    candidate_config=r.candidate_config or {},
                    degradations=r.degradations,
                )
            )
    (run_dir / "autoresearch.json").write_text(
        json.dumps(
            {
                "bottleneck_class": ar_run.bottleneck_class,
                "target": (
                    {
                        "op": ar_run.target.op,
                        "residual": ar_run.target.residual,
                        "n_kernels": ar_run.target.n_kernels,
                    }
                    if ar_run.target is not None
                    else None
                ),
                "results": [
                    {
                        "name": r.spec.name,
                        "knob": r.spec.knob,
                        "value": r.spec.value,
                        "applicable": r.applicable,
                        "rejected_reason": r.rejected_reason,
                        "predicted_delta": r.predicted_delta,
                        "measured_delta": r.measured_delta,
                        "rolled_back": r.rolled_back,
                        "target_op": r.target_op,
                        "apply_error": r.apply_error,
                    }
                    for r in ar_run.results
                ],
                "degradations": [d.to_dict() for d in ar_run.degradations],
            },
            indent=2,
        )
    )

    # Phase 5 — stabilize + write report
    provenance = build_provenance(
        degradations=degradations,
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rejected_candidates = rejected
    provenance.rolled_back = rolled_back

    # Customer-verification export: the A/B numbers the markdown report states as
    # a percentage, emitted as structured data with the config each side ran
    # under. Written only when a live A/B actually produced results.
    if verification:
        write_verification(
            verification, provenance, run_dir / "verification.json", gpu_sku=pctx.sku
        )

    sched_note = _scheduler_note(sched_summary)
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"vLLM decode on {pctx.sku or 'unknown SKU'}: {len(claims)} candidate(s) "
            f"evaluated, {len(rolled_back)} rolled back. {sched_note}"
            if sched_note
            else None
        ),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": len(rejected),
        "bottleneck_class": ar_run.bottleneck_class,
        "n_autoresearch": len(ar_run.results),
        # In the summary, not only the run dir: a reader comparing two runs needs
        # to know one of them was not asked to try everything.
        "n_skipped_levers": len(_excluded),
        # Set when a rollback failed and the run stopped trying candidates. The
        # report is still written; this is what says it is a partial one.
        "engine_lost": engine_lost,
        "n_untried": n_untried,
        "scheduler_stats": asdict(sched_summary) if sched_stats.samples else None,
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _measurement_result(
    *,
    degradations: DegradationLog | None = None,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Honest measurement report for a workload with no intervention library.

    Computes residuals/attribution from the *actual* captured kernels and emits
    observations (not optimization claims) — so an HFT or edge run describes its
    real cuDF/CUB kernels instead of fabricating vLLM serving-knob claims.
    """
    result = measure_trace(trace)
    claims = measurement_claims(result)

    (run_dir / "measurement.json").write_text(
        json.dumps(
            {
                "n_kernels": result.n_kernels,
                "n_memcpy": result.n_memcpy,
                "serialized_concurrency_fraction": result.serialized_fraction,
                "n_violations": len(result.violations),
                "families": result.families,
                "top_hypotheses": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value}
                    for h in result.top_hypotheses
                ],
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        degradations=degradations,
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=(
            "Measurement-only run: the runtime observed the workload and reports "
            "its real kernels. No intervention library applies to this workload."
        ),
        summary=measurement_summary(workload, result),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "measurement",
        "fingerprint": qual.fingerprint,
        "commit": False,
        "floor": qual.floor,
        "n_observations": len(claims),
        "n_claims": 0,
        "n_rolled_back": 0,
        "n_rejected": 0,
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _hft_intervention_result(
    *,
    degradations: DegradationLog | None = None,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    applicator: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Full observe → attribute → select → apply → prove for HFT.

    Reuses the real pieces: attribution from the captured kernels
    (:func:`measure_trace`), the curated lever (:func:`hft_intervention_spec`)
    ranked by counterfactual replay (:func:`predict_delta`), and the rollback
    gate (:func:`apply_intervention`) whose measure runs the output-verified A/B.
    The claim's ``measured_delta`` is the A/B speedup — a real number, gated on
    byte-identical output, so a wrong or slower candidate is rolled back.
    """
    from gitm.benchmarks.hft.optimize import hft_intervention_spec
    from gitm.optimizer.apply import apply_intervention
    from gitm.optimizer.replay import predict_delta

    # Attribute: residuals → invariants → Granger over the actual kernels. Empty
    # when no CUPTI trace was captured (CPU box) — the apply+prove still runs.
    mres = measure_trace(trace)

    # Select: the one curated HFT lever, ranked by predicted delta on this trace.
    spec = hft_intervention_spec()
    predicted = predict_delta(trace, spec) if trace.kernels() else spec.expected_delta_mean
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [{"name": spec.name, "predicted_delta": predicted, "rejected_reason": None}],
            indent=2,
        )
    )

    # Apply behind the rollback gate — measure() runs the verified baseline-vs-
    # candidate A/B and returns the signed speedup (raises → rollback if output
    # diverges; negative delta → rollback if slower).
    apply_res = apply_intervention(
        spec, applicator, min_keep_delta=0.0, audit=AuditLog(run_dir / "audit.jsonl")
    )
    ab = applicator.last_result

    # Prove: one claim carrying the measured delta, gated on identical output.
    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); serialized-concurrency={mres.serialized_fraction:.3f}"
        )
    elif mres.n_kernels:
        evidence = (
            f"serialized-concurrency={mres.serialized_fraction:.3f} over "
            f"{mres.n_kernels} kernels"
        )
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend baseline-vs-candidate A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if ab is not None:
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant="stream_concurrency",
                residual_value=float(mres.serialized_fraction),
                causal_evidence=evidence,
                intervention_name=spec.name,
                predicted_delta=predicted,
                measured_delta=(ab.speedup - 1.0) if ab.identical else None,
                rolled_back=apply_res.rolled_back,
            )
        )
        if apply_res.rolled_back:
            rolled_back.append(spec.name)

    (run_dir / "apply_result.json").write_text(
        json.dumps(
            {
                "intervention": spec.name,
                "applied": apply_res.applied,
                "rolled_back": apply_res.rolled_back,
                "measured_delta": apply_res.measured_delta,
                "error": apply_res.error,
                "identical_output": getattr(ab, "identical", None),
                "kept": getattr(ab, "kept", None),
                "verdict": getattr(ab, "verdict", None),
                "baseline_events_per_second": getattr(ab, "baseline_eps", None),
                "candidate_events_per_second": getattr(ab, "candidate_eps", None),
                "speedup": getattr(ab, "speedup", None),
                "serialized_concurrency_fraction": mres.serialized_fraction,
                "families": mres.families,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        degradations=degradations,
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rolled_back = rolled_back
    verdict = getattr(ab, "verdict", "no A/B result")
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"HFT intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, serialized-concurrency="
            f"{mres.serialized_fraction:.3f}."
        ),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": 0,
        "speedup": getattr(ab, "speedup", None),
        "kept": getattr(ab, "kept", None),
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _openfold_intervention_result(
    *,
    degradations: DegradationLog | None = None,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    applicator: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Full observe → attribute → select → apply → prove for AF2 (OpenFold).

    Mirrors :func:`_hft_intervention_result` but the gate is plDDT-equivalence,
    not byte-identical output: the applicator's measure() runs the fp32-vs-bf16
    A/B and keeps bf16 only if median plDDT stays within tolerance AND it is
    faster, else rolls back to fp32. The claim's ``measured_delta`` is the
    measured speedup, so a quality regression is never reported as a win.
    """
    from benchmarks.biotech.optimize import openfold_intervention_spec
    from gitm.optimizer.apply import apply_intervention
    from gitm.optimizer.replay import predict_delta

    mres = measure_trace(trace)

    spec = openfold_intervention_spec()
    predicted = predict_delta(trace, spec) if trace.kernels() else spec.expected_delta_mean
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [{"name": spec.name, "predicted_delta": predicted, "rejected_reason": None}],
            indent=2,
        )
    )

    apply_res = apply_intervention(
        spec, applicator, min_keep_delta=0.0, audit=AuditLog(run_dir / "audit.jsonl")
    )
    ab = applicator.last_result  # AF2ABResult

    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); serialized-concurrency={mres.serialized_fraction:.3f}"
        )
    elif mres.n_kernels:
        evidence = (
            f"serialized-concurrency={mres.serialized_fraction:.3f} over "
            f"{mres.n_kernels} kernels"
        )
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend fp32-vs-bf16 A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if ab is not None:
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant="stream_concurrency",
                residual_value=float(mres.serialized_fraction),
                causal_evidence=evidence,
                intervention_name=spec.name,
                # plDDT-equivalence is the AF2 correctness gate (vs byte-identical).
                measured_delta=(ab.speedup - 1.0) if ab.equivalent else None,
                predicted_delta=predicted,
                rolled_back=apply_res.rolled_back,
            )
        )
        if apply_res.rolled_back:
            rolled_back.append(spec.name)

    (run_dir / "apply_result.json").write_text(
        json.dumps(
            {
                "intervention": spec.name,
                "applied": apply_res.applied,
                "rolled_back": apply_res.rolled_back,
                "measured_delta": apply_res.measured_delta,
                "error": apply_res.error,
                "plddt_equivalent": getattr(ab, "equivalent", None),
                "plddt_delta": getattr(ab, "plddt_delta", None),
                "plddt_tol": getattr(ab, "plddt_tol", None),
                "kept": getattr(ab, "kept", None),
                "verdict": getattr(ab, "verdict", None),
                "baseline_structures_per_hour": getattr(ab, "baseline_sph", None),
                "candidate_structures_per_hour": getattr(ab, "candidate_sph", None),
                "speedup": getattr(ab, "speedup", None),
                "serialized_concurrency_fraction": mres.serialized_fraction,
                "families": mres.families,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        degradations=degradations,
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rolled_back = rolled_back
    verdict = getattr(ab, "verdict", "no A/B result")
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"AF2 intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, serialized-concurrency="
            f"{mres.serialized_fraction:.3f}."
        ),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": 0,
        "speedup": getattr(ab, "speedup", None),
        "kept": getattr(ab, "kept", None),
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _edge_intervention_result(
    *,
    degradations: DegradationLog | None = None,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    applicator: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Full observe → attribute → select → apply → prove for edge (kitti/nuscenes).

    Mirrors :func:`_hft_intervention_result`/:func:`_openfold_intervention_result`
    but the gate is detection-equivalence (count + sorted scores within
    tolerance), not byte-identical output: the applicator's measure() runs the
    fp32-vs-fp16 A/B and keeps fp16 only if detections stay equivalent AND it is
    faster, else rolls back to fp32. The claim's ``measured_delta`` is the
    measured speedup, so a detection regression is never reported as a win.
    """
    from gitm.benchmarks.edge.optimize import edge_intervention_spec
    from gitm.optimizer.apply import apply_intervention
    from gitm.optimizer.replay import predict_delta

    mres = measure_trace(trace)

    # The applicator carries its own spec; fall back to the module factory.
    spec = getattr(applicator, "spec", None) or edge_intervention_spec()
    predicted = predict_delta(trace, spec) if trace.kernels() else spec.expected_delta_mean
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [{"name": spec.name, "predicted_delta": predicted, "rejected_reason": None}],
            indent=2,
        )
    )

    apply_res = apply_intervention(
        spec, applicator, min_keep_delta=0.0, audit=AuditLog(run_dir / "audit.jsonl")
    )
    ab = applicator.last_result  # EdgeABResult

    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); serialized-concurrency={mres.serialized_fraction:.3f}"
        )
    elif mres.n_kernels:
        evidence = (
            f"serialized-concurrency={mres.serialized_fraction:.3f} over "
            f"{mres.n_kernels} kernels"
        )
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend fp32-vs-fp16 A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if ab is not None:
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant="stream_concurrency",
                residual_value=float(mres.serialized_fraction),
                causal_evidence=evidence,
                intervention_name=spec.name,
                # detection-equivalence is the edge correctness gate.
                measured_delta=(ab.speedup - 1.0) if ab.identical else None,
                predicted_delta=predicted,
                rolled_back=apply_res.rolled_back,
            )
        )
        if apply_res.rolled_back:
            rolled_back.append(spec.name)

    (run_dir / "apply_result.json").write_text(
        json.dumps(
            {
                "intervention": spec.name,
                "applied": apply_res.applied,
                "rolled_back": apply_res.rolled_back,
                "measured_delta": apply_res.measured_delta,
                "error": apply_res.error,
                "detections_equivalent": getattr(ab, "identical", None),
                "kept": getattr(ab, "kept", None),
                "verdict": getattr(ab, "verdict", None),
                "baseline_frames_per_second": getattr(ab, "baseline_eps", None),
                "candidate_frames_per_second": getattr(ab, "candidate_eps", None),
                "speedup": getattr(ab, "speedup", None),
                "serialized_concurrency_fraction": mres.serialized_fraction,
                "families": mres.families,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        degradations=degradations,
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rolled_back = rolled_back
    verdict = getattr(ab, "verdict", "no A/B result")
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"edge intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, serialized-concurrency="
            f"{mres.serialized_fraction:.3f}."
        ),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": 0,
        "speedup": getattr(ab, "speedup", None),
        "kept": getattr(ab, "kept", None),
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _no_data_result(
    *,
    degradations: DegradationLog | None = None,
    run_dir: Path,
    run_id: str,
    workload: str,
    qual: Any,
    started_ns: int,
    trace_path: Path,
    diagnostic: str,
) -> dict[str, Any]:
    """Write an honest no-data report and return its summary (status=no_data).

    Used when the trace has no kernels — a misconfigured box or a workload that
    never ran. We emit zero claims rather than fabricating results from nothing.
    """
    provenance = build_provenance(
        degradations=degradations,
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    report_md = write_report(
        claims=[],
        provenance=provenance,
        qualification_diagnostic=diagnostic,
        summary="NO DATA — tracer captured no GPU kernels; nothing was measured.",
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "no_data",
        "fingerprint": qual.fingerprint,
        "commit": False,
        "floor": qual.floor,
        "n_claims": 0,
        "n_rolled_back": 0,
        "n_rejected": 0,
        "diagnostic": diagnostic,
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}
