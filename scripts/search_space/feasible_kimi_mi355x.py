"""Feasible intervention search space for Kimi-K2.5 on 8x MI355X, as the loop sees it.

Every rule below is the loop's own code, called directly:

* library loader + workload scope ...... gitm.kernels.library.load_library
* sweep expansion ...................... gitm.optimizer.vllm_knobs.expand_relative_candidates
* prerequisite check ................... gitm.optimizer.vllm_knobs.unmet_prerequisite
                                          (+ the joint-exemption at gitm/scheduler/loop.py:895-900)
* deployment (restart) veto ............ gitm.optimizer.vllm_knobs.knob_kind
                                          (condition from gitm/scheduler/loop.py:816)
* hardware / applicability gate ........ gitm.optimizer.preconditions.applicable
* applicability + safety + ranking ..... gitm.agents.policy.select_interventions
* coverage x prior ..................... gitm.optimizer.replay.predict_delta
* autoresearch proposers ............... gitm.agents.autoresearch.autoresearch with
                                          FallbackProposer(EngineArgsProposer(), TableProposer())
* policy ............................... gitm.optimizer.qualification.qualify -> Policy (loop.py:727)
* gate context ......................... gitm.planner.context.build_planner_context

What is *not* the loop's code, and is pinned here instead of measured:

* The engine. There is no live vLLM on this box, so ``stub_engine()`` exposes the
  baseline launch flags (plus vLLM defaults, each marked) on the attribute paths
  the knob taxonomy reads. Knobs whose baseline value is unknown are left off so
  ``get_knob`` raises, exactly as it would against an engine that lacks them.
* The trace. No measured Kimi/MI355X profile is committed, so the ranked stages
  run on a stand-in: one kernel per node of the catalogue's predicted graph at
  the headline point (``standin_trace``). Coverage numbers are therefore
  coverage of *predicted* time. This needs sign-off (open question 1 in
  docs/search_space_kimi_mi355x.md).

Usage::

    python scripts/search_space/feasible_kimi_mi355x.py            # print report
    python scripts/search_space/feasible_kimi_mi355x.py --write    # + evidence JSON
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import gitm.agents.autoresearch as _ar  # noqa: E402
from gitm.agents.autoresearch import (  # noqa: E402
    EngineArgsProposer,
    FallbackProposer,
    TableProposer,
    autoresearch,
)
from gitm.agents.policy import Policy, select_interventions  # noqa: E402
from gitm.kernels.library import load_library  # noqa: E402
from gitm.kernels.spec import Applicability, InterventionSpec  # noqa: E402
from gitm.optimizer.apply import DryRunApplicator  # noqa: E402
from gitm.optimizer.deviation import classify_op  # noqa: E402
from gitm.optimizer.monitor import residuals  # noqa: E402
from gitm.optimizer.preconditions import applicable  # noqa: E402
from gitm.optimizer.qualification import qualify  # noqa: E402
from gitm.optimizer.replay import predict_delta  # noqa: E402
from gitm.optimizer.vllm_knobs import (  # noqa: E402
    KNOB_PREREQUISITES,
    expand_relative_candidates,
    get_knob,
    knob_kind,
    resolve_knob,
    unmet_prerequisite,
)
from gitm.planner.context import (  # noqa: E402
    build_planner_context,
    hardware_spec_for,
    peak_for_sku,
)
from gitm.planner.glm_graph import predict_glm_graph  # noqa: E402
from gitm.planner.model_catalogue import available, load_spec  # noqa: E402
from gitm.planner.roofline import BatchConfig, ShardingConfig  # noqa: E402
from gitm.scheduler.loop import LoopConfig  # noqa: E402
from gitm.tracer.schema import KernelEvent, Trace  # noqa: E402

OUT = REPO / "evidence" / "kimi-mi355x" / "search_space" / "feasible.json"
WORKLOAD = "vllm-decode"
SKU = "AMD Instinct MI355X"

# --------------------------------------------------------------------------- #
# Pinned baseline. Every value names where it comes from; ``None`` means the
# repo does not record it and it is listed as an open question in the doc.
# --------------------------------------------------------------------------- #
BASELINE: dict[str, Any] = {
    "model": {
        "catalogue_entry": "kimi-k2.5",
        "catalogue_file": "gitm/planner/models/kimi-k2.5.yaml",
        "hf_repo": "moonshotai/Kimi-K2.5",
        "revision": None,
        "revision_source": "not pinned: deploy/k8s/mi355x-kimi-loop.yaml serves the hub "
                           "default at launch; no MANIFEST from run_loop.sh is committed",
    },
    "engine": {
        "name": "vLLM (ROCm)",
        "image": "vllm/vllm-openai-rocm@sha256:"
                 "e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1",
        "version": None,
        "version_source": "image digest pinned in deploy/k8s/mi355x-kimi-loop.yaml; the "
                          "vllm.__version__ is only written to MANIFEST at run time "
                          "(scripts/kimi_loop/run_loop.sh) and none is committed",
    },
    "topology": {
        "gpu_sku": SKU,
        "gpus": 8,
        "nodes": 1,
        "peaks_source": "gitm/planner/context.py _PEAKS/_QUANT_PEAKS/_INTERCONNECT['MI355X']",
    },
    "parallelism": {
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "source": "--tensor-parallel-size 8, no PP/DP/EP flags (deploy/k8s/mi355x-kimi-loop.yaml)",
    },
    "launch": {
        "command": [
            "vllm", "serve", "moonshotai/Kimi-K2.5", "--trust-remote-code",
            "--tensor-parallel-size", "8", "--gpu-memory-utilization", "0.92",
            "--max-num-batched-tokens", "8192", "--max-num-seqs", "256",
            "--tool-call-parser", "kimi_k2", "--enable-auto-tool-choice",
            "--reasoning-parser", "kimi_k2",
        ],
        "env": {"VLLM_ROCM_USE_AITER": "1"},
        "source": "deploy/k8s/mi355x-kimi-loop.yaml",
    },
    "traffic": {
        "regime": "rag",
        "prompt_tokens": 4096,
        "output_tokens": 512,
        "source": "scripts/kimi_loop/run_loop.sh LENGTH_CONFIGS / predict_sweep.py HEADLINE_C",
    },
    "batch": {
        "concurrency": 64,
        "decode_batch_modeled": 64,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
    },
    "context": {
        "kv_len_modeled": 4096 + 512 // 2,
        "kv_len_rule": "prompt + output/2 (scripts/kimi_loop/predict_sweep.py kv_mid)",
        "max_model_len": 262144,
        "max_model_len_source": "no --max-model-len flag; vLLM defaults to "
                                "max_position_embeddings (kimi-k2.5.yaml)",
    },
    "precision": {
        "weights": "bf16",
        "routed_experts": "int4 (compressed-tensors W4A16)",
        "kv_cache": "bf16 (kv_cache_dtype auto)",
        "activations": "bf16",
        "source": "gitm/planner/models/kimi-k2.5.yaml; no --kv-cache-dtype flag",
    },
    "loop": {
        "workload": WORKLOAD,
        "top_n_interventions": LoopConfig().top_n_interventions,
        "restart_fn_available": True,
        "restart_fn_source": "gitm/workloads.py vllm-decode factory sets llm.gitm_restart_fn",
        "history": None,
        "history_source": "no committed run history for this box; select_interventions "
                          "is called with history=None",
    },
}

# (value, provenance). "flag" values come from the launch command above;
# everything else is a vLLM default the repo does not verify for this image.
ENGINE_STATE: dict[str, tuple[Any, str]] = {
    "scheduler_config.max_num_seqs": (256, "flag --max-num-seqs"),
    "scheduler_config.max_num_batched_tokens": (8192, "flag --max-num-batched-tokens"),
    "scheduler_config.chunked_prefill_enabled": (True, "vLLM V1 default (unverified for image)"),
    "scheduler_config.async_scheduling": (False, "vLLM default (unverified for image)"),
    "scheduler_config.policy": ("fcfs", "vLLM default"),
    "scheduler_config.enable_dbo": (False, "vLLM default: no --enable-dbo flag"),
    "cache_config.gpu_memory_utilization": (0.92, "flag --gpu-memory-utilization"),
    "cache_config.cache_dtype": ("auto", "vLLM default: no --kv-cache-dtype flag"),
    "cache_config.enable_prefix_caching": (True, "vLLM V1 default (unverified for image)"),
    "cache_config.swap_space_bytes": (4 * 2**30, "vLLM default swap_space=4 GiB (unverified)"),
    "model_config.enforce_eager": (False, "vLLM default: no --enforce-eager flag"),
    "model_config.max_model_len": (262144, "max_position_embeddings (catalogue)"),
    "model_config.dtype": ("bfloat16", "checkpoint torch_dtype (catalogue)"),
    "model_config.quantization": ("compressed-tensors", "checkpoint quantization_config"),
    "parallel_config.tensor_parallel_size": (8, "flag --tensor-parallel-size"),
    "parallel_config.pipeline_parallel_size": (1, "vLLM default"),
    "parallel_config.data_parallel_size": (1, "vLLM default"),
    "parallel_config.enable_expert_parallel": (False, "vLLM default: no --enable-expert-parallel"),
    "parallel_config.enable_eplb": (False, "vLLM default"),
    "parallel_config.disable_custom_all_reduce": (False, "vLLM default"),
    "parallel_config.distributed_executor_backend": ("mp", "vLLM default for single node"),
}
UNKNOWN_ENGINE_STATE: dict[str, str] = {
    "cache_config.block_size": "ROCm MLA backends pick their own block size; not recorded",
    "model_config.max_seq_len_to_capture": "renamed/removed across vLLM versions; version unpinned",
    "model_config.moe_backend": "not an EngineArgs field on every version; AITER MoE via env",
    "speculative_config": "absent: no --speculative-config flag",
}


def stub_engine() -> SimpleNamespace:
    """Engine double carrying ENGINE_STATE on the paths the knob taxonomy reads."""
    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for path, (value, _) in ENGINE_STATE.items():
        head, leaf = path.split(".", 1)
        groups[head][leaf] = value
    eng = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in groups.items()})
    eng.gitm_restart_fn = object()
    return eng


# --------------------------------------------------------------------------- #
# Stand-in trace from the predicted graph.
# --------------------------------------------------------------------------- #
def predicted_graph():
    spec = load_spec("kimi-k2.5")
    hw = hardware_spec_for(peak_for_sku(SKU))
    batch = BatchConfig(batch=BASELINE["batch"]["decode_batch_modeled"],
                        kv_cache_len=BASELINE["context"]["kv_len_modeled"])
    return predict_glm_graph(spec, hw, batch,
                             ShardingConfig(tp=BASELINE["parallelism"]["tensor_parallel_size"]))


def standin_trace(graph) -> Trace:
    events, t = [], 0
    for i, node in enumerate(graph.nodes):
        dur = max(1, int(round(node.prediction.t_pred_s * 1e9)))
        events.append(KernelEvent(name=node.op, start_ns=t, end_ns=t + dur,
                                  stream_id=node.expected_stream_id, device_id=0,
                                  correlation_id=i))
        t += dur
    return Trace(workload_id=WORKLOAD, fingerprint="kimi-k2.5@mi355x/predicted-standin",
                 run_id="predicted-standin", device_count=8, vendor="amd",
                 captured_at_ns=0, duration_ns=t, events=events, source="none")


# --------------------------------------------------------------------------- #
# Candidates and rules.
# --------------------------------------------------------------------------- #
@dataclass
class Cand:
    spec: InterventionSpec
    source: str          # "library" | "autoresearch"
    entry: str           # library entry name / proposer target
    workloads: list[str]
    reasons: dict[str, str | None] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def knob_key(self) -> str:
        return "+".join(sorted(self.spec.knob_values))

    @property
    def point(self) -> tuple[str, str]:
        return self.knob_key, json.dumps(self.spec.knob_values, sort_keys=True, default=str)


def _prereq_of(knob: str) -> str | None:
    # Same lookup unmet_prerequisite and loop.py:895 use.
    return next((p for needle, p in KNOB_PREREQUISITES if needle in knob.lower()), None)


def rule_validity(c: Cand, scoped: set[str]) -> str | None:
    if c.source == "library" and c.entry not in scoped:
        return f"load_library(workload={WORKLOAD!r}) excludes it (workloads={c.workloads})"
    return None


def rule_prerequisites(c: Cand, engine: Any) -> str | None:
    values = c.spec.knob_values
    for k in values:
        reason = unmet_prerequisite(engine, k)
        if reason is None:
            continue
        if _prereq_of(k) in values:  # loop.py:899: the lever supplies its own prerequisite
            continue
        return reason
    return None


def raw_unmet_prerequisite(c: Cand, engine: Any) -> dict[str, str]:
    return {k: r for k in c.spec.knob_values if (r := unmet_prerequisite(engine, k))}


def rule_hardware(c: Cand, ctx) -> str | None:
    app = c.spec.applicability
    hw_only = c.spec.model_copy(update={"applicability": Applicability(
        workloads=[ctx.workload],
        requires_hardware=app.requires_hardware,
        min_gpus=app.min_gpus,
        requires_collective=app.requires_collective,
        requires_interconnect=app.requires_interconnect,
    )})
    ok, why = applicable(hw_only, ctx)
    return None if ok else why


def suppliers_of(prereq: str, pool: list[Cand]) -> list[Cand]:
    return [c for c in pool if c.spec.knob_values.get(prereq)]


def rule_mutual_dependency(c: Cand, engine: Any, feasible_suppliers: list[Cand]) -> str | None:
    """Dependent lever whose prerequisite is unmet and no feasible lever can supply it."""
    values = c.spec.knob_values
    for k in values:
        prereq = _prereq_of(k)
        if prereq is None or prereq == k or prereq in values:
            continue
        if unmet_prerequisite(engine, k) is None:
            continue
        if not suppliers_of(prereq, feasible_suppliers):
            return f"needs {prereq!r}; no feasible lever in the pool sets it"
    return None


def rule_deployment(c: Cand, engine: Any, restart_fn: Any) -> str | None:
    if engine is not None and restart_fn is None and any(
        knob_kind(k) == "structural" for k in c.spec.knob_values
    ):
        return "structural knob: needs engine restart, no restart_fn"
    return None


def baseline_equal(c: Cand, engine: Any) -> str | None:
    """Informational on main: the lever sets only values the engine already runs.

    Not a loop stage on main. Same rule as ``baseline_noop`` on branch
    loop/skip-baseline-noops: an unreadable knob never counts as equal.
    """
    values = c.spec.knob_values
    if not values:
        return None
    for k, v in values.items():
        try:
            cur = get_knob(engine, k)
        except AttributeError:
            return None
        if isinstance(cur, bool) != isinstance(v, bool) or cur != v:
            return None
    return "baseline already runs " + ", ".join(f"{k}={v!r}" for k, v in values.items())


# --------------------------------------------------------------------------- #
# Bits and joint configs.
# --------------------------------------------------------------------------- #
def knob_points(pool: list[Cand]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for c in pool:
        key, point = c.point
        out[key].add(point)
    return out


def bits_for(n: int) -> int:
    return math.ceil(math.log2(n + 1)) if n > 0 else 0


def total_bits(pool: list[Cand]) -> int:
    return sum(bits_for(len(v)) for v in knob_points(pool).values())


def joint_configs(pool: list[Cand], engine: Any) -> dict[str, int]:
    """Product of (values + unchanged) per knob, minus prerequisite-violating combos.

    A dependent knob may only be set when its prerequisite holds on the engine or
    a lever in the same config sets it truthy.
    """
    pts = knob_points(pool)
    by_knob = {k: [json.loads(p) for p in v] for k, v in pts.items()}
    groups: dict[str, list[str]] = defaultdict(list)
    for key in by_knob:
        for k in key.split("+"):
            p = _prereq_of(k)
            if p and p != k and p not in key.split("+") and unmet_prerequisite(engine, k):
                groups[p].append(key)
    unconstrained = math.prod(len(vals) + 1 for vals in by_knob.values())
    constrained_keys = {k for deps in groups.values() for k in deps} | set(groups)
    feasible = 1
    for key, vals in by_knob.items():
        if key not in constrained_keys:
            feasible *= len(vals) + 1
    for prereq, deps in groups.items():
        dep_free = math.prod(len(by_knob[d]) + 1 for d in deps)
        sup_vals = by_knob.get(prereq, [])
        truthy = sum(1 for v in sup_vals if v.get(prereq))
        falsy = len(sup_vals) + 1 - truthy
        feasible *= truthy * dep_free + falsy
    return {"unconstrained": unconstrained, "feasible": feasible,
            "dependency_violating": unconstrained - feasible}


def stage_counts(pool: list[Cand], engine: Any) -> dict[str, Any]:
    pts = knob_points(pool)
    return {
        "candidates": len(pool),
        "distinct_points": sum(len(v) for v in pts.values()),
        "knobs": len(pts),
        "bits": total_bits(pool),
        "joint_configs": joint_configs(pool, engine),
        "by_source": dict(Counter(c.source for c in pool)),
    }


def consistency(before: list[Cand], after: list[Cand], stage: str,
                partial_ok: bool) -> dict[str, Any]:
    """Candidate-gap vs bit-gap alignment between two consecutive stages.

    Per knob, removing all n points costs n candidates but only bits(n) bits, so
    the gaps legitimately diverge by n - bits(n) for each fully removed swept
    knob. Anything else means a stage removed *some* points of a swept knob; that
    is only expected where the stage ranks (partial_ok).
    """
    pb, pa = knob_points(before), knob_points(after)
    d_points = sum(len(v) for v in pb.values()) - sum(len(v) for v in pa.values())
    d_bits = total_bits(before) - total_bits(after)
    explained = 0
    partial: list[dict[str, Any]] = []
    fully_removed_swept: list[str] = []
    for key, vals in pb.items():
        n_b, n_a = len(vals), len(pa.get(key, ()))
        if n_a == 0 and n_b > 1:
            fully_removed_swept.append(key)
            explained += n_b - bits_for(n_b)
        elif 0 < n_a < n_b:
            partial.append({"knob": key, "before": n_b, "after": n_a})
            explained += (n_b - n_a) - (bits_for(n_b) - bits_for(n_a))
    aligned = (d_points - d_bits) == explained
    ok = aligned and (partial_ok or not partial)
    return {
        "stage": stage,
        "candidate_gap": len(before) - len(after),
        "point_gap": d_points,
        "bit_gap": d_bits,
        "gap_difference": d_points - d_bits,
        "explained_by_swept_knobs": explained,
        "fully_removed_swept_knobs": fully_removed_swept,
        "partial_swept_removals": partial,
        "partial_removals_allowed": partial_ok,
        "ok": ok,
    }


# --------------------------------------------------------------------------- #
# Main analysis.
# --------------------------------------------------------------------------- #
def _vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def build_pool(engine: Any, trace: Trace, res, policy: Policy, ctx) -> dict[str, Any]:
    all_entries = load_library()
    scoped_entries = load_library(workload=WORKLOAD)
    scoped = {s.name for s in scoped_entries}

    lib: list[Cand] = []
    sweep = []
    for s in all_entries:
        expanded = expand_relative_candidates(s, engine)
        if s.value_multiplier_grid:
            sweep.append({
                "entry": s.name,
                "grid": list(s.value_multiplier_grid),
                "grid_points": len(s.value_multiplier_grid),
                "emitted": [e.name for e in expanded],
                "values": [e.value for e in expanded],
                "collapsed": len(s.value_multiplier_grid) - len(expanded),
            })
        lib.extend(Cand(e, "library", s.name, list(s.applicability.workloads)) for e in expanded)

    ar_run = autoresearch(
        trace,
        applicator=DryRunApplicator(),
        policy=policy,
        residuals=res,
        proposer=FallbackProposer(EngineArgsProposer(), TableProposer()),
        ctx=ctx,
        reject=None,
    )
    ar = [Cand(r.spec, "autoresearch", f"{ar_run.bottleneck_class}:{r.target_op}",
               list(r.spec.applicability.workloads)) for r in ar_run.results]

    return {
        "library_entries_total": len(all_entries),
        "library_entries_scoped": len(scoped_entries),
        "scoped": scoped,
        "library": lib,
        "autoresearch": ar,
        "sweep": sweep,
        "ar_run": ar_run,
    }


def gate_reasons(pool: list[Cand], trace: Trace, policy: Policy, ctx) -> dict[str, str | None]:
    if not pool:
        return {}
    ranked = select_interventions(trace, [c.spec for c in pool], policy,
                                  top_n=len(pool), ctx=ctx)
    return {r.spec.name: r.rejected_reason for r in ranked}


def coverage(trace: Trace, spec: InterventionSpec) -> float:
    return predict_delta(trace, spec, delta_mean=1.0)


def analyse() -> dict[str, Any]:
    assert "kimi-k2.5" in available(), "kimi-k2.5 missing from gitm/planner/model_catalogue.py"
    os.environ["GITM_GPU_SKU"] = SKU
    env_leak = os.environ.get("VLLM_ATTENTION_BACKEND")
    assert env_leak is None, "VLLM_ATTENTION_BACKEND is set in this shell; the baseline has none"

    engine = stub_engine()
    restart_fn = getattr(engine, "gitm_restart_fn", None)
    pctx = build_planner_context(engine, workload=WORKLOAD, num_gpus=BASELINE["topology"]["gpus"])
    ctx = pctx.gate
    graph = predicted_graph()
    trace = standin_trace(graph)
    res = residuals(trace, graph)
    qual = qualify(trace)
    policy = Policy(require_qualification_commit=qual.commit, skip_high_risk=not qual.commit)
    alt_policy = Policy(require_qualification_commit=not qual.commit, skip_high_risk=qual.commit)

    pool_info = build_pool(engine, trace, res, policy, ctx)
    lib, ar = pool_info["library"], pool_info["autoresearch"]
    raw = lib + ar
    scoped = pool_info["scoped"]

    # ---- independent view: every rule against every raw candidate ----------
    non_prereq_feasible: list[Cand] = []
    gate_all = gate_reasons(raw, trace, Policy(require_qualification_commit=True,
                                               skip_high_risk=False), ctx)
    safety_all = gate_reasons(raw, trace, policy, None)
    for c in raw:
        c.reasons["validity"] = rule_validity(c, scoped)
        c.reasons["prerequisites"] = rule_prerequisites(c, engine)
        c.reasons["hardware"] = rule_hardware(c, ctx)
        c.reasons["deployment"] = rule_deployment(c, engine, restart_fn)
        c.reasons["applicability"] = gate_all.get(c.name)
        c.reasons["safety"] = safety_all.get(c.name)
    for c in raw:
        if not any(c.reasons[r] for r in ("validity", "hardware", "deployment",
                                          "applicability", "safety")):
            non_prereq_feasible.append(c)
    for c in raw:
        c.reasons["mutual_dependency"] = rule_mutual_dependency(c, engine, non_prereq_feasible)

    rules = ["validity", "prerequisites", "hardware", "mutual_dependency", "deployment",
             "applicability", "safety"]
    independent = {}
    for r in rules:
        hit = [c for c in raw if c.reasons[r]]
        only = [c for c in hit if not any(c.reasons[o] for o in rules if o != r)]
        independent[r] = {
            "rejected": len(hit),
            "rejected_only_by_this_rule": len(only),
            "names": sorted(c.name for c in hit),
        }
    overlap = {
        f"{a}&{b}": sum(1 for c in raw if c.reasons[a] and c.reasons[b])
        for a, b in itertools.combinations(rules, 2)
    }
    overlap = {k: v for k, v in overlap.items() if v}

    # ---- sequential funnel ---------------------------------------------------
    stages: list[tuple[str, list[Cand]]] = [("raw", raw)]
    cur = raw
    for r in ["validity", "prerequisites", "hardware", "mutual_dependency", "deployment"]:
        cur = [c for c in cur if not c.reasons[r]]
        stages.append((f"after_{r}", cur))
    seq_gate = gate_reasons(cur, trace, policy, ctx)
    for c in cur:
        c.reasons["gate_sequential"] = seq_gate.get(c.name)
    cur = [c for c in cur if not seq_gate.get(c.name)]
    stages.append(("after_applicability_safety", cur))
    gate_survivors = cur

    top_n = BASELINE["loop"]["top_n_interventions"]
    lib_surv = [c for c in gate_survivors if c.source == "library"]
    ar_surv = [c for c in gate_survivors if c.source == "autoresearch"]
    lib_ranked = select_interventions(trace, [c.spec for c in lib_surv], policy,
                                      top_n=top_n, ctx=ctx) if lib_surv else []
    ranked_names = {r.spec.name for r in lib_ranked}
    # autoresearch_v0 ranks with top_n=len(proposals): every survivor is attempted.
    after_ranking = [c for c in lib_surv if c.name in ranked_names] + ar_surv
    stages.append(("after_ranking", after_ranking))

    funnel = []
    for name, pool in stages:
        funnel.append({"stage": name, **stage_counts(pool, engine)})
    checks = []
    for (_, p0), (n1, p1) in zip(stages, stages[1:], strict=False):
        checks.append(consistency(p0, p1, n1, partial_ok=(n1 == "after_ranking")))

    # ---- dedup: separate numbers, separate pools ----------------------------
    point_seen: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in raw:
        point_seen[c.point].append(f"{c.source}:{c.name}")
    cross_source = {f"{k[0]}={k[1]}": v for k, v in point_seen.items()
                    if len({s.split(':')[0] for s in v}) > 1}
    ea = EngineArgsProposer()
    surface = [k.name for k in ea._source.knobs()]
    searchable = [k.name for k in ea._searchable()]
    catalog_excluded = sorted(set(surface) & ea._catalog)
    ar_knobs = {k for c in ar for k in c.spec.knob_values}
    bl_equal = {c.name: baseline_equal(c, engine) for c in gate_survivors}
    bl_equal = {k: v for k, v in bl_equal.items() if v}
    dedup = {
        "sweep_point_collapse": {
            "pool": "grid points of swept library entries (value_multiplier_grid), "
                    "before expansion",
            "pool_size": sum(s["grid_points"] for s in pool_info["sweep"]),
            "collapsed": sum(s["collapsed"] for s in pool_info["sweep"]),
            "code": "gitm/optimizer/vllm_knobs.py expand_relative_candidates (seen_values)",
            "by_entry": {s["entry"]: s["collapsed"] for s in pool_info["sweep"] if s["collapsed"]},
        },
        "proposer_catalog_exclusion": {
            "pool": "EngineArgsProposer knob surface on this box (all classes)",
            "vllm_importable": _vllm_importable(),
            "pool_size": len(surface),
            "surface": surface,
            "excluded_as_catalog": len(catalog_excluded),
            "excluded_knobs": catalog_excluded,
            "searchable_after_exclusion": searchable,
            "emitted_for_this_class": sorted(ar_knobs),
            "code": "gitm/agents/autoresearch.py _ProposerBase._searchable "
                    "(k.name not in self._catalog)",
        },
        "cross_source_duplicates": {
            "pool": "raw library candidates + autoresearch proposals, keyed by (knob, value)",
            "pool_size": len(raw),
            "duplicates": len(cross_source),
            "points": cross_source,
        },
        "baseline_equal": {
            "pool": "applicability+safety gate survivors",
            "pool_size": len(gate_survivors),
            "equal_to_baseline": len(bl_equal),
            "levers": bl_equal,
            "note": "informational on main: the loop does not skip these; they would be "
                    "applied and measured as a no-op A/B",
        },
    }

    # ---- loop-faithful Phase 3 -> Phase 4 slots on main ----------------------
    phase3_pool = [c.spec for c in lib if c.entry in scoped]
    phase3 = select_interventions(trace, phase3_pool, policy, top_n=top_n, ctx=ctx)
    by_name = {c.name: c for c in raw}
    slots = []
    for r in phase3:
        c = by_name[r.spec.name]
        slots.append({
            "name": r.spec.name,
            "predicted_delta": r.predicted_delta,
            "coverage": coverage(trace, r.spec),
            "gate_rejected": r.rejected_reason,
            "phase4_veto_on_main": rule_deployment(c, engine, restart_fn),
            "prerequisite_veto_not_applied_to_library_on_main": c.reasons["prerequisites"],
            "raw_unmet_prerequisite": raw_unmet_prerequisite(c, engine) or None,
            "baseline_equal": baseline_equal(c, engine),
            "zero_or_negative_delta": r.predicted_delta <= 0.0,
        })

    # ---- coverage: where the predicted step time goes, and who can touch it ----
    op_time: Counter[str] = Counter()
    for k in trace.kernels():
        op_time[k.name] += k.end_ns - k.start_ns
    total = sum(op_time.values())
    survivors_specs = [c.spec for c in gate_survivors]
    op_rows = []
    for op, t in op_time.most_common():
        covering = [s.name for s in survivors_specs if s.applies_to_kernels and (
            (classify_op(op) or "") in s.applies_to_kernels
            or any(p in op for p in s.applies_to_kernels))]
        op_rows.append({"op": op, "classify_op": classify_op(op), "share": t / total,
                        "covered_by_gate_survivors": covering})
    uncovered = sum(r["share"] for r in op_rows if not r["covered_by_gate_survivors"])
    moe_ops = [r for r in op_rows if r["op"].startswith("moe_")]
    moe_levers = [
        {"name": c.name, "applies_to_kernels": list(c.spec.applies_to_kernels),
         "coverage": coverage(trace, c.spec),
         "status": next((f"{r}: {c.reasons[r]}" for r in rules if c.reasons.get(r)), "feasible")}
        for c in raw if c.source == "library" and any(
            k in ("enable_expert_parallel", "enable_eplb", "moe_backend")
            for k in c.spec.knob_values)
    ]
    all_ops_cov = {c.name: coverage(trace, c.spec) for c in gate_survivors}

    # ---- sensitivity ---------------------------------------------------------
    alt_gate = gate_reasons([c for c in stages[5][1]], trace, alt_policy, ctx)
    no_restart = [c.name for c in raw
                  if not rule_deployment(c, engine, restart_fn)
                  and rule_deployment(c, engine, None)]

    per_candidate = []
    for c in raw:
        per_candidate.append({
            "name": c.name,
            "source": c.source,
            "entry": c.entry,
            "knob_values": c.spec.knob_values,
            "knob_kind": {k: knob_kind(k) for k in c.spec.knob_values},
            "in_knob_taxonomy": {k: resolve_knob(k) is not None for k in c.spec.knob_values},
            "applies_to_kernels": list(c.spec.applies_to_kernels),
            "expected_delta_mean": c.spec.expected_delta_mean,
            "coverage": coverage(trace, c.spec),
            "predicted_delta": predict_delta(trace, c.spec),
            "safety_tier": c.spec.safety.tier,
            "requires_qualification_commit": c.spec.safety.requires_qualification_commit,
            "reasons": {r: c.reasons.get(r) for r in rules},
            "gate_sequential": c.reasons.get("gate_sequential"),
            "raw_unmet_prerequisite": raw_unmet_prerequisite(c, engine) or None,
            "baseline_equal": baseline_equal(c, engine),
            "ranked_top_n": c.name in ranked_names,
            "survives": c in after_ranking,
        })

    ar_run = pool_info["ar_run"]
    return {
        "generated_by": "scripts/search_space/feasible_kimi_mi355x.py",
        "git_commit": git_commit(),
        "baseline": BASELINE,
        "engine_state": {k: {"value": v, "source": s} for k, (v, s) in ENGINE_STATE.items()},
        "engine_state_unknown": UNKNOWN_ENGINE_STATE,
        "gate_context": {k: getattr(ctx, k) for k in ctx.__dataclass_fields__},
        "standin_trace": {
            "source": "predict_glm_graph(kimi-k2.5, MI355X, batch=64, kv=4352, tp=8)",
            "kernels": len(trace.kernels()),
            "step_ms": trace.duration_ns / 1e6,
            "trace_source_field": trace.source,
        },
        "qualification": {"commit": qual.commit, "diagnostic": qual.diagnostic},
        "policy": {"require_qualification_commit": policy.require_qualification_commit,
                   "skip_high_risk": policy.skip_high_risk,
                   "code": "gitm/scheduler/loop.py:727"},
        "autoresearch": {
            "bottleneck_class": ar_run.bottleneck_class,
            "target_op": ar_run.target.op if ar_run.target else None,
            "target_note": "stand-in residuals are zero by construction (trace == prediction), "
                           "so the largest-residual target op carries no signal",
            "max_abs_residual_r_kt": max((abs(r.r_kt) for r in res.per_kernel), default=0.0),
            "classify_scores": {
                "memory": (_ar._roofline_memory_fraction(res) or 0.0) / _ar._MEMCPY_THRESHOLD,
                "serialized_concurrency": _ar._serialized_fraction(trace.kernels())
                                          / _ar._SC_THRESHOLD,
                "threshold": 1.0,
                "code": "gitm/agents/autoresearch.py classify_bottleneck",
            },
            "proposals": len(ar),
            "proposal_names": [c.name for c in ar],
            "proposals_per_class": {
                cls: [s.name for s in FallbackProposer(EngineArgsProposer(), TableProposer())
                      .propose(cls, target_op=None)]
                for cls in ("idle_stall", "memory_bound", "compute_bound")
            },
            "proposer": "FallbackProposer(EngineArgsProposer(), TableProposer())",
        },
        "sweep_expansion": pool_info["sweep"],
        "counts": {
            "library_entries_total": pool_info["library_entries_total"],
            "library_entries_scoped": pool_info["library_entries_scoped"],
            "library_candidates_raw": len(lib),
            "autoresearch_candidates_raw": len(ar),
        },
        "funnel": funnel,
        "consistency_checks": checks,
        "consistency_ok": all(c["ok"] for c in checks),
        "independent": independent,
        "independent_overlaps": overlap,
        "rejection_ranking_dedup": {
            "rejected_before_ranking": len(raw) - len(gate_survivors),
            "cut_by_ranking": len(gate_survivors) - len(after_ranking),
            "dedup": dedup,
            "note": "three separate numbers from separate pools; they are not additive",
        },
        "phase3_top_n_on_main": slots,
        "coverage": {
            "ops": op_rows,
            "uncovered_share_by_gate_survivors": uncovered,
            "moe_share": sum(r["share"] for r in moe_ops),
            "moe_levers": moe_levers,
            "gate_survivor_coverage": all_ops_cov,
        },
        "sensitivity": {
            "other_qualification_value": {
                "policy": {"require_qualification_commit": alt_policy.require_qualification_commit,
                           "skip_high_risk": alt_policy.skip_high_risk},
                "gate_rejected_after_deployment_stage": sorted(
                    k for k, v in alt_gate.items() if v),
                "phase3_top_n": [
                    {"name": r.spec.name, "predicted_delta": r.predicted_delta,
                     "baseline_equal": baseline_equal(by_name[r.spec.name], engine)}
                    for r in select_interventions(trace, phase3_pool, alt_policy,
                                                  top_n=top_n, ctx=ctx)
                ],
            },
            "no_restart_fn_additionally_vetoed": len(no_restart),
        },
        "candidates": per_candidate,
    }


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def _fmt_joint(j: dict[str, int]) -> str:
    return f"{j['feasible']:,} (of {j['unconstrained']:,}; {j['dependency_violating']:,} dep-violating)"


def report(a: dict[str, Any]) -> None:
    p = print
    p("=" * 78)
    p("PINNED BASELINE")
    p("=" * 78)
    b = a["baseline"]
    p(f"model        {b['model']['catalogue_entry']} ({b['model']['hf_repo']}), "
      f"revision={b['model']['revision']}")
    p(f"engine       {b['engine']['image']}  version={b['engine']['version']}")
    p(f"topology     {b['topology']['gpus']}x {b['topology']['gpu_sku']}, "
      f"{b['topology']['nodes']} node")
    par = b["parallelism"]
    p(f"parallelism  TP={par['tensor_parallel_size']} PP={par['pipeline_parallel_size']} "
      f"DP={par['data_parallel_size']} EP={par['enable_expert_parallel']}")
    p(f"launch       {' '.join(b['launch']['command'])}  env={b['launch']['env']}")
    t = b["traffic"]
    p(f"traffic      {t['regime']} {t['prompt_tokens']}/{t['output_tokens']} "
      f"at concurrency {b['batch']['concurrency']}")
    p(f"context      kv_len_modeled={b['context']['kv_len_modeled']} "
      f"max_model_len={b['context']['max_model_len']}")
    pr = b["precision"]
    p(f"precision    weights={pr['weights']} experts={pr['routed_experts']} "
      f"kv={pr['kv_cache']} act={pr['activations']}")
    p(f"loop         workload={b['loop']['workload']} top_n={b['loop']['top_n_interventions']} "
      f"restart_fn={b['loop']['restart_fn_available']}")
    p(f"gate ctx     {a['gate_context']}")
    p(f"stand-in     {a['standin_trace']}")
    p(f"qualify      commit={a['qualification']['commit']} -> policy {a['policy']}")
    p(f"autoresearch class={a['autoresearch']['bottleneck_class']} "
      f"target={a['autoresearch']['target_op']} proposals={a['autoresearch']['proposal_names']}")
    p(f"             per class: {a['autoresearch']['proposals_per_class']}")

    p("\nSEQUENTIAL FUNNEL")
    p(f"{'stage':32s} {'cand':>5s} {'pts':>5s} {'knobs':>5s} {'bits':>5s}  joint configs")
    for row in a["funnel"]:
        p(f"{row['stage']:32s} {row['candidates']:5d} {row['distinct_points']:5d} "
          f"{row['knobs']:5d} {row['bits']:5d}  {_fmt_joint(row['joint_configs'])}")

    p("\nCONSISTENCY (point gap - bit gap must equal sum over fully removed swept knobs)")
    for c in a["consistency_checks"]:
        p(f"  {c['stage']:32s} pts-{c['point_gap']:<3d} bits-{c['bit_gap']:<3d} "
          f"diff={c['gap_difference']:<3d} explained={c['explained_by_swept_knobs']:<3d} "
          f"partial={c['partial_swept_removals'] or '-'} ok={c['ok']}")
    p(f"  all ok: {a['consistency_ok']}")

    p("\nINDEPENDENT VIEW (each rule vs all raw candidates)")
    for r, v in a["independent"].items():
        p(f"  {r:18s} rejects {v['rejected']:3d}  (only this rule: "
          f"{v['rejected_only_by_this_rule']:3d})")
    p(f"  overlaps: {a['independent_overlaps']}")

    rrd = a["rejection_ranking_dedup"]
    p("\nREJECTION / RANKING / DEDUP (separate, not additive)")
    p(f"  rejected before ranking: {rrd['rejected_before_ranking']}")
    p(f"  cut by ranking:          {rrd['cut_by_ranking']}")
    for k, v in rrd["dedup"].items():
        n = next(v[f] for f in ("collapsed", "excluded_as_catalog", "duplicates",
                                "equal_to_baseline") if f in v)
        p(f"  dedup {k:28s} {n:3d} of {v['pool_size']:3d}   pool: {v['pool']}")

    p("\nPHASE 3 TOP-N ON MAIN (what the loop would spend its slots on)")
    for s in a["phase3_top_n_on_main"]:
        flags = [f for f in ("gate_rejected", "phase4_veto_on_main",
                             "prerequisite_veto_not_applied_to_library_on_main",
                             "baseline_equal") if s[f]]
        if s["zero_or_negative_delta"]:
            flags.append("delta<=0")
        p(f"  {s['name']:40s} delta={s['predicted_delta']:+.4f} cov={s['coverage']:.3f} "
          f"{flags or ''}")

    cov = a["coverage"]
    p("\nCOVERAGE OF PREDICTED STEP TIME")
    for r in cov["ops"][:12]:
        p(f"  {r['op']:26s} {r['share']:6.1%} classify={r['classify_op']!s:22s} "
          f"levers={len(r['covered_by_gate_survivors'])}")
    p(f"  uncovered by any gate survivor: {cov['uncovered_share_by_gate_survivors']:.1%}; "
      f"moe_* share: {cov['moe_share']:.1%}")
    for m in cov["moe_levers"]:
        p(f"  moe lever {m['name']:24s} targets={m['applies_to_kernels']} "
          f"cov={m['coverage']:.4f} {m['status']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--write", action="store_true", help=f"write {OUT.relative_to(REPO)}")
    args = ap.parse_args()
    a = analyse()
    report(a)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(a, indent=2, default=str) + "\n")
        print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0 if a["consistency_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
