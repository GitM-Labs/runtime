"""Traffic replay library — production traces in, harness workloads out.

Deliverable 1 of the validation infrastructure. Real production traces normalized
into one canonical format, fired through the harness by a tool that already
exists (vLLM's ``bench serve``), and tagged with the workload regime every result
row is keyed on.

    from gitm.traffic import read_burstgpt, Regime, SourceKind, write_timed_trace

    trace = read_burstgpt("BurstGPT_1.csv")
    print(trace.meta.summary())                        # provenance and drops
    print(Regime.from_trace(trace).label())            # the result-row key
    plan = write_timed_trace(trace, "replay.jsonl")    # what bench serve consumes
    print(" ".join(plan.bench_serve_argv(model="...")))

CPU-only: nothing here needs a GPU, and only the final firing needs a server.
Run ``python -m gitm.traffic --selftest`` for the check that fails if any of it
regresses.
"""

from gitm.traffic.adapters import ADAPTERS, read_burstgpt, read_mooncake
from gitm.traffic.parameterize import RegimeFit, fit, grid, sample_trace
from gitm.traffic.regime import Regime, SourceKind, index_of_dispersion
from gitm.traffic.replay import (
    VLLM_MIN_VERSION,
    ReplayPlan,
    read_timed_trace,
    write_timed_trace,
)
from gitm.traffic.results import BenchRun, join_result, unjoined_keys
from gitm.traffic.runner import RunResult, VllmUnavailable, check_vllm, run_replay
from gitm.traffic.schema import SCHEMA, CanonicalRequest, DropReason, Trace, TraceMeta
from gitm.traffic.validate import (
    REPLAY_THRESHOLDS,
    SAMPLED_THRESHOLDS,
    ValidationReport,
    compare,
    ks_statistic,
)

__all__ = [
    "ADAPTERS",
    "REPLAY_THRESHOLDS",
    "SAMPLED_THRESHOLDS",
    "SCHEMA",
    "BenchRun",
    "CanonicalRequest",
    "DropReason",
    "Regime",
    "RegimeFit",
    "ReplayPlan",
    "RunResult",
    "VLLM_MIN_VERSION",
    "VllmUnavailable",
    "SourceKind",
    "Trace",
    "TraceMeta",
    "ValidationReport",
    "compare",
    "fit",
    "grid",
    "index_of_dispersion",
    "join_result",
    "check_vllm",
    "ks_statistic",
    "read_burstgpt",
    "read_mooncake",
    "read_timed_trace",
    "run_replay",
    "sample_trace",
    "unjoined_keys",
    "write_timed_trace",
]
