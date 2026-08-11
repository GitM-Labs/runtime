"""The 24-hour autonomous loop.

This is the orchestration glue — it composes tracer, planner, optimizer,
kernels, and agents in the 5 phases below. Each phase writes its artifact
to local scratch under ``<scratch>/runs/<run_id>/`` (see ``gitm._paths``) so a
partial run is still useful; the durable copy is synced to S3 afterwards.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gitm._paths import runs_dir, traces_dir
from gitm._timing import require_positive_duration, require_positive_work
from gitm.agents.autoresearch import (
    AutoresearchRun,
    EngineArgsProposer,
    FallbackProposer,
    TableProposer,
    autoresearch,
    classify_bottleneck,
)
from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.library import load_library
from gitm.optimizer.apply import (
    Applicator,
    DryRunApplicator,
    LiveEngineApplicator,
    apply_intervention,
)
from gitm.optimizer.attribution import attribute
from gitm.optimizer.collective_signal import collective_causes, worst_device_comm
from gitm.optimizer.deviation import deviation_summary, deviation_trace, write_deviation_jsonl
from gitm.optimizer.dr import attribute_dr
from gitm.optimizer.measure import measure_trace, measurement_claims, measurement_summary
from gitm.optimizer.monitor import check_invariants, residuals
from gitm.optimizer.qualification import qualify
from gitm.optimizer.report import Claim, build_provenance, write_report
from gitm.optimizer.scheduler_attribution import scheduler_causes
from gitm.optimizer.verification_export import (
    VerificationRecord,
    build_record,
    write_verification,
)
from gitm.optimizer.vllm_knobs import (
    expand_relative_candidates,
    knob_kind,
    unmet_prerequisite,
)
from gitm.planner.context import build_planner_context, hardware_spec_for
from gitm.planner.graph import Graph, predict_graph
from gitm.planner.moe_graph import predict_moe_graph, spec_from_hf_config
from gitm.planner.roofline import (
    BatchConfig,
    ModelSpec,
    ShardingConfig,
    weight_bytes,
    weight_bytes_is_fallback,
)
from gitm.safety.audit import AuditLog, _write_report
from gitm.serve.model_config import (
    is_sparse_moe_config,
    normalize_moe_config,
    validate_moe_config,
    validate_priceable_dtypes,
)
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
    seconds = value * {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]
    if seconds <= 0.0:
        raise ValueError(f"budget must be positive, got {budget!r}")
    return seconds

def _engine_throughput_fn(engine: Any, runner: Any) -> Any:
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
    """
    explicit = getattr(engine, "gitm_throughput_fn", None)
    if callable(explicit):
        return explicit

    def _tps(_engine: Any) -> float:
        t0 = time.perf_counter()
        out = runner() if runner is not None else {}
        dt = require_positive_duration(
            time.perf_counter() - t0, context="live-engine throughput probe"
        )
        # First key that is actually present wins — `or` would treat a legitimate
        # 0 (a window that produced no tokens) as missing and fabricate a count.
        toks: float | None = None
        if isinstance(out, dict):
            for key in ("generated_tokens", "decode_steps", "events"):
                if out.get(key) is not None:
                    toks = float(out[key])
                    break
        if toks is None:
            raise RuntimeError(
                "live-engine throughput work coverage unavailable: runner returned none of "
                "generated_tokens, decode_steps, or events"
            )
        return float(require_positive_work(toks, context="live-engine throughput probe")) / dt

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
    # Optional explicit driver for the embedded/engine path. When unset, the
    # loop looks up ``workload`` in the workload registry (gitm.workloads).
    workload_runner: WorkloadRunner | None = None


#: Attention-shape aliases. Deliberately separate from the MoE table: hybrid
#: attention and a mixture FFN are independent choices, and a hybrid model with a
#: dense FFN (or a plain MoE transformer) must still get the right one.
_FULL_ATTN_ALIASES = ("full_attention_interval", "attn_layer_freq")

#: quant_method -> bytes per weight element. MoE decode is weight-fetch bound,
#: so using the activation width for a quantized checkpoint would overstate the
#: dominant term by 2x (fp8) or 4x (4-bit).
_QUANT_WEIGHT_BYTES: dict[str, int] = {"fp8": 1, "compressed-tensors": 1, "modelopt_fp8": 1}


def _batch_config_from_stats(sched: Any):
    """A :class:`BatchConfig` carrying the *observed* decode batch, or ``None``.

    ``predict_graph``'s default is ``batch=1``. That is wrong for any real serving
    window and especially wrong for a mixture-of-experts model, where weight
    traffic scales with the distinct experts a batch activates: at top-8 of 256,
    a batch of 1 touches 8 experts but a batch of 16 touches ~100, so scoring a
    batch-16 step against the batch-1 ceiling understates expert traffic by more
    than 10x. ``mean_running`` is vLLM's own count of concurrently running
    sequences, which is exactly the decode batch.

    Returns ``None`` (caller falls back to the default) when no scheduler samples
    were taken — a CPU box, a dry run, or an engine that exposes no stats. Better
    a documented default than a fabricated batch.

    ``kv_cache_len`` is deliberately left at its default: nothing in the sampled
    stats gives a token count (``peak_gpu_cache_usage`` is a fraction of blocks,
    not a length), and inventing one would move the full-attention ceiling on a
    guess. Sourcing it is tracked separately.
    """
    from gitm.planner.roofline import BatchConfig

    if sched is None or getattr(sched, "n_samples", 0) == 0:
        return None
    running = getattr(sched, "mean_running", None)
    if running is None or running < 1:
        return None
    return BatchConfig(batch=max(int(round(float(running))), 1))


@dataclass
class ExecutionGraphResolution:
    """A trustworthy loop graph, or the named reason graph-based claims refuse."""

    graph: Graph | None
    diagnostics: list[str]
    refusal_reason: str = ""
    model_source: str = ""

    @property
    def ok(self) -> bool:
        return self.graph is not None and not self.refusal_reason


def _engine_hf_config(engine: Any) -> tuple[Any | None, str]:
    if engine is None:
        return None, "no live engine was supplied"
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
            return obj, path
    return None, "the live engine exposes no HuggingFace config"


def _config_dict(hf: Any) -> tuple[dict[str, Any] | None, str]:
    try:
        if isinstance(hf, dict):
            return dict(hf), ""
        to_dict = getattr(hf, "to_dict", None)
        raw = to_dict() if callable(to_dict) else vars(hf)
        if not isinstance(raw, dict):
            return None, f"config conversion returned {type(raw).__name__}, not dict"
        return dict(raw), ""
    except Exception as exc:
        # Fail open at the runtime boundary, but preserve the parser failure as
        # the refusal reason instead of turning it into a default model.
        return None, f"config conversion failed ({type(exc).__name__}: {exc})"


def _engine_value(engine: Any, paths: tuple[str, ...]) -> Any:
    for path in paths:
        obj: Any = engine
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _loop_batch(engine: Any, pctx: Any, sched: Any) -> tuple[BatchConfig, list[str]]:
    diagnostics: list[str] = []
    observed = _batch_config_from_stats(sched)
    batch = observed.batch if observed is not None else 1
    if observed is None:
        diagnostics.append("decode concurrency was not observed; using batch=1")

    kv_len = getattr(getattr(pctx, "gate", None), "kv_cache_len", None)
    if not isinstance(kv_len, int) or kv_len <= 0:
        kv_len = 4096
        diagnostics.append("KV cache length was not exposed; using kv_cache_len=4096")

    speculative = _engine_value(
        engine,
        (
            "speculative_config.num_speculative_tokens",
            "vllm_config.speculative_config.num_speculative_tokens",
        ),
    )
    speculative_tokens = int(speculative) if isinstance(speculative, int) and speculative > 0 else 0
    return BatchConfig(
        batch=batch,
        kv_cache_len=kv_len,
        speculative_tokens=speculative_tokens,
    ), diagnostics


def _loop_sharding(engine: Any) -> tuple[ShardingConfig, list[str]]:
    tp = _engine_value(
        engine,
        ("parallel_config.tensor_parallel_size", "vllm_config.parallel_config.tensor_parallel_size"),
    )
    dp = _engine_value(
        engine,
        ("parallel_config.data_parallel_size", "vllm_config.parallel_config.data_parallel_size"),
    )
    ep = _engine_value(
        engine,
        ("parallel_config.enable_expert_parallel", "vllm_config.parallel_config.enable_expert_parallel"),
    )
    if tp is not None and (isinstance(tp, bool) or not isinstance(tp, int) or tp < 1):
        raise ValueError(f"tensor_parallel_size must be a positive integer, got {tp!r}")
    if tp is None:
        return ShardingConfig(), [
            "sharding topology was not exposed; using whole-model tp=1 ep=1 dp=1"
        ]
    diagnostics: list[str] = []
    if dp is None:
        dp_i = 1
        diagnostics.append("data_parallel_size was not exposed; using dp=1")
    elif isinstance(dp, bool) or not isinstance(dp, int) or dp < 1:
        raise ValueError(f"data_parallel_size must be a positive integer, got {dp!r}")
    else:
        dp_i = dp
    if ep is None:
        ep_enabled = False
        diagnostics.append("enable_expert_parallel was not exposed; using ep=1")
    elif not isinstance(ep, bool):
        raise ValueError(f"enable_expert_parallel must be boolean, got {ep!r}")
    else:
        ep_enabled = ep
    return ShardingConfig(tp=tp, ep=tp if ep_enabled else 1, dp=dp_i), diagnostics


def _dense_spec_from_config(cfg: dict[str, Any]) -> tuple[ModelSpec | None, str]:
    required = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "torch_dtype",
    )
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        return None, "dense model config is missing answer-deciding fields: " + ", ".join(missing)
    try:
        hidden = int(cfg["hidden_size"])
        n_heads = int(cfg["num_attention_heads"])
        n_kv = int(cfg.get("num_key_value_heads", n_heads) or n_heads)
        head_dim = int(cfg.get("head_dim", 0) or (hidden // n_heads))
        act_raw = str(cfg.get("torch_dtype", "bf16")).lower().removeprefix("torch.")
        act = {
            "bfloat16": "bf16",
            "float16": "fp16",
            "half": "fp16",
            "float32": "fp32",
        }.get(act_raw, act_raw)
        if weight_bytes_is_fallback(act):
            return None, f"dense activation dtype {act!r} is not priceable"
        dtype_bytes = int(weight_bytes(act))
        quant = cfg.get("quantization_config") or {}
        if not isinstance(quant, dict):
            return None, "dense quantization_config must be an object when declared"
        method = quant.get("quant_method")
        if method is not None and str(method).lower() not in _QUANT_WEIGHT_BYTES:
            return None, f"dense quantization method {method!r} is not priceable"
        full_attn_layer_step = 1
        for key in _FULL_ATTN_ALIASES:
            raw = cfg.get(key)
            if raw is not None:
                full_attn_layer_step = int(raw)
                if full_attn_layer_step <= 0:
                    return None, f"dense attention interval {key} must be positive"
                break
        return ModelSpec(
            name=str(cfg.get("_name_or_path") or cfg.get("model_type") or "live-dense"),
            hidden=hidden,
            n_layers=int(cfg["num_hidden_layers"]),
            n_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            intermediate=int(cfg["intermediate_size"]),
            compute_dtype=act,
            dtype_bytes=dtype_bytes,
            weight_dtype_bytes=_QUANT_WEIGHT_BYTES.get(str(method).lower()) if method else None,
            vocab=int(cfg["vocab_size"]),
            full_attn_layer_step=full_attn_layer_step,
        ), ""
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return None, f"dense model config could not be parsed ({type(exc).__name__}: {exc})"


def _execution_graph(engine: Any, pctx: Any, sched: Any) -> ExecutionGraphResolution:
    """Resolve the live engine to the matching graph, or refuse namedly."""
    hf, source = _engine_hf_config(engine)
    if hf is None:
        return ExecutionGraphResolution(None, [], source)
    cfg, error = _config_dict(hf)
    if cfg is None:
        return ExecutionGraphResolution(None, [], error)

    if getattr(pctx, "peak", None) is None:
        sku = getattr(pctx, "sku", None) or "unknown"
        return ExecutionGraphResolution(
            None,
            [],
            f"GPU SKU {sku!r} is not in the hardware catalogue; refusing A100 substitution",
        )
    hw = hardware_spec_for(pctx.peak)
    batch, diagnostics = _loop_batch(engine, pctx, sched)
    try:
        sharding, sharding_diagnostics = _loop_sharding(engine)
    except ValueError as exc:
        return ExecutionGraphResolution(None, diagnostics, f"invalid live sharding topology: {exc}")
    diagnostics.extend(sharding_diagnostics)

    if is_sparse_moe_config(cfg):
        invalid = validate_moe_config(cfg)
        if invalid:
            return ExecutionGraphResolution(
                None,
                diagnostics,
                "sparse-MoE config cannot be priced without guessing: " + "; ".join(invalid),
            )
        try:
            spec = spec_from_hf_config(normalize_moe_config(cfg), name=str(cfg.get("_name_or_path") or "live-moe"))
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return ExecutionGraphResolution(
                None,
                diagnostics,
                f"sparse-MoE config could not be parsed ({type(exc).__name__}: {exc})",
            )
        act_dtype = getattr(getattr(pctx, "gate", None), "dtype", None)
        kv_dtype = _engine_value(
            engine,
            ("cache_config.cache_dtype", "cache_config.kv_cache_dtype", "vllm_config.cache_config.cache_dtype"),
        )
        changes: dict[str, Any] = {}
        if act_dtype:
            changes["act_dtype"] = str(act_dtype).lower()
        if kv_dtype:
            changes["kv_dtype"] = str(kv_dtype).lower().replace("fp8_e4m3", "fp8")
        else:
            diagnostics.append("KV cache dtype was not exposed; using planner default kv_dtype='fp8'")
        if cfg.get("expert_dtype") is None:
            diagnostics.append(
                f"expert_dtype absent; inherited weight_dtype={spec.weight_dtype!r} for expert bytes"
            )
        if changes:
            from dataclasses import replace

            spec = replace(spec, **changes)
        unpriceable = validate_priceable_dtypes(spec)
        if unpriceable:
            return ExecutionGraphResolution(None, diagnostics, "; ".join(unpriceable))
        graph = predict_moe_graph(spec, hw, batch, sharding)
    else:
        if cfg.get("num_key_value_heads") is None:
            diagnostics.append(
                "dense num_key_value_heads was not declared; assuming standard MHA "
                "with one KV head per query head"
            )
        if cfg.get("head_dim") is None:
            diagnostics.append(
                "dense head_dim was not declared; derived hidden_size / num_attention_heads"
            )
        spec, error = _dense_spec_from_config(cfg)
        if spec is None:
            return ExecutionGraphResolution(None, diagnostics, error)
        try:
            graph = predict_graph(model=spec, hw=hw, batch=batch, sharding=sharding)
        except ValueError as exc:
            return ExecutionGraphResolution(
                None, diagnostics, f"dense sharding cannot be priced: {exc}"
            )

    if graph.has_fallback_peaks:
        diagnostics.append("one or more nodes use fallback compute peaks")
    if graph.has_fallback_bytes:
        diagnostics.append("one or more nodes use fallback byte widths")
    if graph.has_unpriced_nodes:
        missing = []
        if graph.has_unpriced_compute:
            missing.append("compute throughput")
        if graph.has_unpriced_memory:
            missing.append("memory bandwidth")
        diagnostics.append(f"one or more predicted nodes have unpriced {' and '.join(missing)}")
    n_estimated = sum(1 for node in graph.nodes if node.prediction.estimated)
    if n_estimated:
        diagnostics.append(f"{n_estimated} predicted node(s) use estimated cost models")
    return ExecutionGraphResolution(
        graph,
        diagnostics,
        model_source="live_hf_config",
    )


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
    return value


def _ar_target_residual(ar_run: AutoresearchRun, fallback: float = 0.0) -> float:
    """Residual for autoresearch claims.

    Prefer the largest-residual op that autoresearch targeted; when there is no
    target, fall back to the run-level kernel-time residual so generated claims
    do not all display a misleading +0.0% gap.
    """
    return ar_run.target.residual if ar_run.target is not None else fallback


def run_loop(cfg: LoopConfig) -> dict[str, Any]:
    """Execute the 24-hour loop and return ``{summary, report_md, ...}``."""
    workload = cfg.workload or (getattr(cfg.engine, "workload_id", None) or "vllm-decode")
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
    if sched_stats.samples or sched_summary.diagnostics or serving_summary is not None:
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

    # HFT carries a real, output-verified intervention on its runner. Apply+prove
    # it through the rollback gate — the A/B runs on the active backend, so the
    # delta is measured even on a box without CUPTI. (Runs before the empty-trace
    # guard for that reason; attribution below is included only if kernels exist.)
    if workload in _HFT_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)
        if applicator is not None:
            return _hft_intervention_result(
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            )

    # OpenFold/AF2 carries the bf16 intervention on its runner. Same pattern as
    # HFT: apply+prove through the rollback gate (measure() runs the fp32-vs-bf16
    # A/B, gated on plDDT-equivalence). Before the empty-trace guard so the A/B
    # still runs on a box without CUPTI; attribution is included if kernels exist.
    if workload in _OPENFOLD_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)
        if applicator is not None:
            return _openfold_intervention_result(
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            )

    # Edge (kitti/nuscenes) carries the fp16 intervention on its runner. Same
    # pattern as HFT/AF2: apply+prove through the rollback gate (measure() runs
    # the fp32-vs-fp16 A/B, gated on detection-equivalence). Before the empty-
    # trace guard so the A/B still runs on a box without CUPTI.
    if workload in _EDGE_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)
        if applicator is not None:
            return _edge_intervention_result(
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            )

    # Guard: a kernel launch with no positive duration is not measurement
    # coverage. Do not classify it or emit claims from a fabricated denominator.
    valid_kernels = [k for k in trace.kernels() if k.end_ns > k.start_ns]
    if trace.vendor == "none" or not valid_kernels:
        diagnostic = runner_error or (
            "Tracer captured no positive-duration GPU kernels. Either no GPU/CUPTI "
            "shim is present, the workload did not run, or kernel timestamps are invalid."
        )
        return _no_data_result(
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
            run_dir=run_dir,
            run_id=run_id,
            workload=workload,
            trace=trace,
            qual=qual,
            started_ns=started_ns,
            trace_path=trace_path,
        )

    # Resolve model, hardware, serving batch, and sharding as one trust gate. A
    # missing/partial live config or unknown SKU refuses graph-based claims and
    # falls through to an honest measurement report; it never becomes a plausible
    # Llama/A100 default prediction.
    pctx = build_planner_context(cfg.engine, workload=workload)
    graph_resolution = _execution_graph(cfg.engine, pctx, sched_summary)
    if pctx.num_gpus_is_fallback:
        graph_resolution.diagnostics.append(
            "GPU count was unavailable; using 1 for intervention applicability only"
        )
    graph_resolution.diagnostics.extend(sched_summary.diagnostics)
    if not graph_resolution.ok:
        (run_dir / "prediction_refusal.json").write_text(
            json.dumps(
                {
                    "reason": graph_resolution.refusal_reason,
                    "diagnostics": graph_resolution.diagnostics,
                    "hardware": pctx.sku,
                },
                indent=2,
            )
        )
        return _measurement_result(
            run_dir=run_dir,
            run_id=run_id,
            workload=workload,
            trace=trace,
            qual=qual,
            started_ns=started_ns,
            trace_path=trace_path,
            diagnostic=(
                "Prediction gate refused graph-based optimization claims: "
                f"{graph_resolution.refusal_reason}"
            ),
            runtime_diagnostics=graph_resolution.diagnostics,
            status="prediction_refused",
        )
    graph = graph_resolution.graph
    assert graph is not None
    if graph.resident_weight_bytes_is_lower_bound:
        graph_resolution.diagnostics.append(
            "resident weight footprint is a lower bound because DSpark parameter shapes are private"
        )
    (run_dir / "predicted_graph.json").write_text(
        json.dumps(
            {
                "model": graph.model.name,
                "model_source": graph_resolution.model_source,
                "nodes": len(graph.nodes),
                "total_pred_s": graph.total_pred_s,
                "resident_weight_bytes_per_rank": graph.resident_weight_bytes_per_rank,
                "resident_weight_bytes_is_lower_bound": (
                    graph.resident_weight_bytes_is_lower_bound
                ),
                "kv_bytes_per_token_per_sequence": graph.kv_bytes_per_token_per_sequence,
                "kv_fixed_bytes_per_sequence": graph.kv_fixed_bytes_per_sequence,
                "hardware": pctx.sku,
                "hardware_pricing": graph.hw.name,
                "hardware_is_fallback": graph.hardware_is_fallback,
                "batch": {
                    "batch": graph.batch.batch,
                    "kv_cache_len": graph.batch.kv_cache_len,
                    "speculative_tokens": graph.batch.speculative_tokens,
                },
                "sharding": {
                    "tp": graph.sharding.tp,
                    "ep": graph.sharding.ep,
                    "dp": graph.sharding.dp,
                },
                "has_unpriced_collectives": graph.has_unpriced_collectives,
                "has_unpriced_nodes": graph.has_unpriced_nodes,
                "has_unpriced_compute": graph.has_unpriced_compute,
                "has_unpriced_memory": graph.has_unpriced_memory,
                "has_fallback_peaks": graph.has_fallback_peaks,
                "has_fallback_bytes": graph.has_fallback_bytes,
                "diagnostics": graph_resolution.diagnostics,
                "predictions": [
                    {
                        "op": node.op,
                        "layer": node.layer,
                        "estimated": node.prediction.estimated,
                        "peak_is_fallback": node.prediction.peak_is_fallback,
                        "bytes_are_fallback": node.prediction.bytes_are_fallback,
                        "compute_is_unpriced": node.prediction.compute_is_unpriced,
                        "memory_is_unpriced": node.prediction.memory_is_unpriced,
                    }
                    for node in graph.nodes
                ],
            },
            indent=2,
        )
    )

    # Phase 2 — residuals + attribution
    res = residuals(trace, graph)
    violations = check_invariants(res)  # multi-basis confirmed
    hypotheses = attribute(res, graph)  # Granger
    dr_hypotheses = attribute_dr(res, graph)  # doubly-robust, corroborating
    coverage = {
        "total_kernels": res.total_kernels,
        "classified_kernels": res.classified_kernels,
        "matched_kernels": res.matched_kernels,
        "classification_coverage": res.classification_coverage,
        "match_coverage": res.match_coverage,
        "classified_time_coverage": res.classified_time_coverage,
        "matched_time_coverage": res.matched_time_coverage,
        "warnings": res.coverage_warnings,
    }

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
                "coverage": coverage,
                "top_hypotheses_granger": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value}
                    for h in hypotheses.top(5)
                ],
                "top_hypotheses_doubly_robust": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value,
                     "notes": h.notes}
                    for h in dr_hypotheses.top(5)
                ],
                "attribution_diagnostics": (
                    hypotheses.diagnostics + dr_hypotheses.diagnostics
                ),
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
    try:
        raw_library = load_library(workload=workload)
    except (FileNotFoundError, ValueError) as exc:
        diagnostic = f"intervention candidate coverage unavailable: {type(exc).__name__}: {exc}"
        return _measurement_result(
            run_dir=run_dir,
            run_id=run_id,
            workload=workload,
            trace=trace,
            qual=qual,
            started_ns=started_ns,
            trace_path=trace_path,
            diagnostic=diagnostic,
            runtime_diagnostics=graph_resolution.diagnostics + [diagnostic],
            status="candidate_coverage_unavailable",
        )
    library = [
        resolved
        for s in raw_library
        for resolved in expand_relative_candidates(s, cfg.engine)
    ]
    policy = Policy(require_qualification_commit=qual.commit, skip_high_risk=not qual.commit)
    ranked = select_interventions(trace, library, policy, top_n=cfg.top_n_interventions, ctx=pctx.gate)
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
        applicator: Applicator = LiveEngineApplicator(
            cfg.engine,
            throughput_fn=_engine_throughput_fn(cfg.engine, runner),
            restart_fn=live_restart_fn,
            baseline_restart_fn=getattr(cfg.engine, "gitm_baseline_restart_fn", None),
            restart_mode=os.environ.get("GITM_RESTART_MODE", "parallel"),
            reps=int(os.environ.get("GITM_AB_REPS", "1")),
            # Compatibility escape hatch for custom scheduling-classified knobs
            # that should still be measured through engine rebuild.
            force_restart=os.environ.get("GITM_KNOBS_VIA_RESTART") == "1",
        )
    else:
        applicator = DryRunApplicator()

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
    for c in ranked:
        if c.rejected_reason is not None:
            rejected.append(f"{c.spec.name} ({c.rejected_reason})")
            continue
        # Live + structural knob + no restart hook → it *cannot* be enacted on the
        # running engine, so it's "not evaluable here", not a regression. Mark it
        # rejected (honest) instead of attempting an apply that would roll back and
        # read as "tried and lost" — and skip the wasted baseline benchmark.
        if cfg.engine is not None and live_restart_fn is None and knob_kind(c.spec.knob) == "structural":
            rejected.append(f"{c.spec.name} (structural knob: needs engine restart, no restart_fn)")
            continue
        # Snapshot the engine config BEFORE the apply: a hot-swap mutates these
        # kwargs in place and a restart replaces the engine outright, so reading
        # them afterwards would report the candidate on both sides of the diff.
        baseline_cfg = dict(getattr(cfg.engine, "gitm_llm_kwargs", None) or {})
        result = apply_intervention(c.spec, applicator, min_keep_delta=0.0)
        ab = getattr(applicator, "last_result", None)
        if result.rolled_back:
            rolled_back.append(c.spec.name)
        if ab is not None:
            # Read the candidate config off whichever engine the applicator is
            # holding now — for a restart A/B that is the rebuilt engine, not
            # the original. Falls back to the knob delta over the baseline.
            live_engine = getattr(applicator, "engine", cfg.engine)
            candidate_cfg = dict(getattr(live_engine, "gitm_llm_kwargs", None) or {})
            if not candidate_cfg and baseline_cfg:
                candidate_cfg = {**baseline_cfg, **(c.spec.knobs or {c.spec.knob: c.spec.value})}
            verification.append(
                build_record(
                    c.spec, ab, result,
                    baseline_config=baseline_cfg,
                    candidate_config=candidate_cfg,
                )
            )
        # Causal evidence: the measured A/B verdict when live, else the Granger
        # signal that motivated the candidate. The kept/rolled-back wording comes
        # from the authoritative ApplyResult (the real gate decision), not from
        # EngineABResult.kept (a measure-time delta>=0 indicator).
        if ab is not None:
            outcome = "rolled back" if result.rolled_back else "kept"
            causal_evidence = (
                f"live A/B: {outcome} ({ab.speedup - 1.0:+.1%} decode throughput, via {ab.via}); "
                f"baseline {ab.baseline_tps:.1f} → candidate {ab.candidate_tps:.1f} tok/s"
            )
        else:
            causal_evidence = ", ".join(
                f"{h.cause_op}→{h.effect_op} (p={h.p_value:.2g})" for h in hypotheses.top(2)
            ) or "no strong causal signal"
        # Attach the top scheduler cause that argues for *this* knob — the
        # engine-level signal (C) tied to the specific lever it motivates (B).
        motivating = next((sc for sc in sched_causes if c.spec.knob in sc.motivates_knobs), None)
        if motivating is not None:
            causal_evidence += f"; scheduler[{motivating.signal}]: {motivating.note}"
        claims.append(
            Claim(
                summary=c.spec.summary,
                residual_invariant="kernel_time",
                residual_value=kt_residual,
                residual_scope="run",
                causal_evidence=causal_evidence,
                intervention_name=c.spec.name,
                predicted_delta=c.predicted_delta,
                # Display the TRUE measured delta (speedup-1); the gate uses the noise-adjusted
                # return, so a within-noise gain reads as rolled back
                # with its real (small) number, not a distorted one.
                measured_delta=((ab.speedup - 1.0) if ab is not None else result.measured_delta),
                rolled_back=result.rolled_back,
            )
        )
        if time.time_ns() - started_ns >= int(budget_s * 1e9):
            break


    # Phase 4b - agentic autoresearch through the catalog gate/rollback path.
    if time.time_ns() - started_ns < int(budget_s * 1e9):
        proposer = FallbackProposer(
            EngineArgsProposer(gpu_count=pctx.num_gpus),
            TableProposer(),
        )

        def _unenactable(spec: Any) -> str | None:
            if (
                cfg.engine is not None
                and live_restart_fn is None
                and knob_kind(spec.knob) == "structural"
            ):
                return "structural knob: needs engine restart, no restart_fn"
            return unmet_prerequisite(cfg.engine, spec.knob)

        ar_run = autoresearch(
            trace,
            applicator=applicator,
            policy=policy,
            residuals=res,
            proposer=proposer,
            ctx=pctx.gate,
            reject=_unenactable,
        )
    else:
        ar_run = AutoresearchRun(bottleneck_class=classify_bottleneck(trace, res), results=[])

    ar_causal_evidence = ", ".join(
        f"{h.cause_op}→{h.effect_op} (p={h.p_value:.2g})" for h in hypotheses.top(2)
    ) or "no strong causal signal"
    ar_residual = _ar_target_residual(ar_run, kt_residual)
    for r in ar_run.results:
        if not r.applicable:
            rejected.append(f"{r.spec.name} ({r.rejected_reason})")
            continue
        if r.rolled_back:
            rolled_back.append(r.spec.name)
        # A rolled-back candidate with no measured_delta means the live apply
        # itself raised (engine build/restart failure), not "measured and lost" —
        # surface why directly in the report so that distinction isn't buried in
        # autoresearch.json.
        evidence = ar_causal_evidence
        if r.measured_delta is None and r.apply_error:
            evidence += f"; apply failed: {r.apply_error}"
        claims.append(
            Claim(
                summary=r.spec.summary,
                residual_invariant="kernel_time",
                residual_value=ar_residual,
                residual_scope=("target_op" if ar_run.target is not None else "run"),
                causal_evidence=evidence,
                intervention_name=r.spec.name,
                predicted_delta=r.predicted_delta,
                measured_delta=r.measured_delta,
                rolled_back=r.rolled_back,
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
            },
            indent=2,
        )
    )

    # Phase 5 — stabilize + write report
    provenance = build_provenance(
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
        runtime_diagnostics=(
            graph_resolution.diagnostics
            + res.coverage_warnings
            + (serving_summary.warnings if serving_summary is not None else [])
            + hypotheses.diagnostics
            + dr_hypotheses.diagnostics
        ),
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
        "scheduler_stats": asdict(sched_summary) if sched_stats.samples else None,
        "residual_coverage": coverage,
        "prediction_diagnostics": graph_resolution.diagnostics,
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _measurement_result(
    *,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    started_ns: int,
    trace_path: Path,
    diagnostic: str | None = None,
    runtime_diagnostics: list[str] | None = None,
    status: str = "ok",
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
                "n_invalid_duration": result.n_invalid_duration,
                "diagnostics": result.diagnostics,
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
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=diagnostic or (
            "Measurement-only run: the runtime observed the workload and reports "
            "its real kernels. No intervention library applies to this workload."
        ),
        runtime_diagnostics=(runtime_diagnostics or []) + result.diagnostics,
        summary=measurement_summary(workload, result),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": status,
        "mode": "measurement",
        "fingerprint": qual.fingerprint,
        "commit": False,
        "floor": qual.floor,
        "n_observations": len(claims),
        "n_claims": 0,
        "n_rolled_back": 0,
        "n_rejected": 0,
        "prediction_refusal": diagnostic,
        "runtime_diagnostics": runtime_diagnostics or [],
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _specialized_claim_basis(
    mres: Any, ab: Any
) -> tuple[tuple[str, float] | None, float | None, list[str]]:
    """Choose an observed residual basis without fabricating trace coverage."""
    diagnostics = list(mres.diagnostics)
    if ab is None:
        diagnostics.append("intervention A/B produced no result; no performance claim emitted")
        return None, None, diagnostics
    try:
        measured_delta = float(ab.speedup) - 1.0
    except (AttributeError, TypeError, ValueError):
        diagnostics.append("intervention A/B result has no numeric speedup; no claim emitted")
        return None, None, diagnostics
    if not math.isfinite(measured_delta):
        diagnostics.append("intervention A/B speedup is non-finite; no claim emitted")
        return None, None, diagnostics
    if mres.serialized_fraction is not None:
        return ("stream_concurrency", float(mres.serialized_fraction)), measured_delta, diagnostics
    if mres.n_kernels > mres.n_invalid_duration:
        diagnostics.append(
            "no adjacent cross-stream CUPTI kernel pairs; claim residual uses the "
            "measured A/B throughput delta instead of fabricated concurrency evidence"
        )
        return ("throughput_delta", measured_delta), measured_delta, diagnostics
    diagnostics.append(
        "no positive-duration CUPTI kernels; claim residual uses the measured A/B "
        "throughput delta instead of a fabricated stream-concurrency value"
    )
    return ("throughput_delta", measured_delta), measured_delta, diagnostics


def _serialized_evidence(mres: Any) -> str:
    if mres.serialized_fraction is None:
        return "serialized-concurrency unavailable (no adjacent cross-stream pairs)"
    return f"serialized-concurrency={mres.serialized_fraction:.3f}"


def _hft_intervention_result(
    *,
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
    basis, measured_delta, runtime_diagnostics = _specialized_claim_basis(mres, ab)
    if apply_res.error:
        runtime_diagnostics.append(f"intervention apply failed: {apply_res.error}")

    # Prove: one claim carrying the measured delta, gated on identical output.
    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); {_serialized_evidence(mres)}"
        )
    elif mres.n_kernels:
        evidence = f"{_serialized_evidence(mres)} over {mres.n_kernels} kernels"
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend baseline-vs-candidate A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if basis is not None and ab is not None:
        residual_invariant, residual_value = basis
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant=residual_invariant,
                residual_value=residual_value,
                residual_scope="run",
                causal_evidence=evidence,
                intervention_name=spec.name,
                predicted_delta=predicted,
                measured_delta=measured_delta if ab.identical else None,
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
                "residual_basis": basis[0] if basis is not None else None,
                "diagnostics": runtime_diagnostics,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
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
        runtime_diagnostics=runtime_diagnostics,
        summary=(
            f"HFT intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, {_serialized_evidence(mres)}."
        ),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok" if basis is not None else "intervention_failed",
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
    basis, measured_delta, runtime_diagnostics = _specialized_claim_basis(mres, ab)
    if apply_res.error:
        runtime_diagnostics.append(f"intervention apply failed: {apply_res.error}")

    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); {_serialized_evidence(mres)}"
        )
    elif mres.n_kernels:
        evidence = f"{_serialized_evidence(mres)} over {mres.n_kernels} kernels"
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend fp32-vs-bf16 A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if basis is not None and ab is not None:
        residual_invariant, residual_value = basis
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant=residual_invariant,
                residual_value=residual_value,
                residual_scope="run",
                causal_evidence=evidence,
                intervention_name=spec.name,
                # plDDT-equivalence is the AF2 correctness gate (vs byte-identical).
                measured_delta=measured_delta if ab.equivalent else None,
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
                "residual_basis": basis[0] if basis is not None else None,
                "diagnostics": runtime_diagnostics,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
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
        runtime_diagnostics=runtime_diagnostics,
        summary=(
            f"AF2 intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, {_serialized_evidence(mres)}."
        ),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok" if basis is not None else "intervention_failed",
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
    basis, measured_delta, runtime_diagnostics = _specialized_claim_basis(mres, ab)
    if apply_res.error:
        runtime_diagnostics.append(f"intervention apply failed: {apply_res.error}")

    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); {_serialized_evidence(mres)}"
        )
    elif mres.n_kernels:
        evidence = f"{_serialized_evidence(mres)} over {mres.n_kernels} kernels"
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend fp32-vs-fp16 A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if basis is not None and ab is not None:
        residual_invariant, residual_value = basis
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant=residual_invariant,
                residual_value=residual_value,
                residual_scope="run",
                causal_evidence=evidence,
                intervention_name=spec.name,
                # detection-equivalence is the edge correctness gate.
                measured_delta=measured_delta if ab.identical else None,
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
                "residual_basis": basis[0] if basis is not None else None,
                "diagnostics": runtime_diagnostics,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
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
        runtime_diagnostics=runtime_diagnostics,
        summary=(
            f"edge intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, {_serialized_evidence(mres)}."
        ),
    )
    _write_report(run_dir, report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok" if basis is not None else "intervention_failed",
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
    run_dir: Path,
    run_id: str,
    workload: str,
    qual: Any,
    started_ns: int,
    trace_path: Path,
    diagnostic: str,
) -> dict[str, Any]:
    """Write an honest no-data report and return its summary (status=no_data).

    Used when the trace has no positive-duration kernels — a misconfigured box,
    a workload that never ran, or invalid timestamps. We emit zero claims.
    """
    provenance = build_provenance(
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
        summary="NO DATA — tracer captured no positive-duration GPU kernels; nothing was measured.",
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
