"""Named slowdown causes and the constants their witness checks use.

A duration cannot name a cause: many causes change timing the same way. Each entry
here names one cause in the terms a profiler or an engine's docs use, the second
signal that only it produces (its witness), and what would settle it when the
captured data cannot. :mod:`gitm.optimizer.witness` evaluates the entries.

Every constant below is registered under :data:`THRESHOLDS_VERSION`. They were fixed
before any fixture was run and are not tuned against synthetic data; a change is a
re-registration on real-capture evidence and bumps the version by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

THRESHOLDS_VERSION = "2026-10-12.2"

#: A partition window shorter than this is ``too_short``.
MIN_WINDOW_NS = 1_000_000_000
#: Rows 10, 11, 22, 25 and G1 fire at this share of the window.
FIRE_AT_WINDOW_SHARE = 0.01

#: A priced row says where its time sat across slices of this length, a final partial
#: slice left out. Past the maximum (about 28 h) no slices are built: no capture is
#: that long, and one stray timestamp must not cost hours or gigabytes.
SLICE_NS = 1_000_000_000
MAX_SLICES = 100_000

#: The brief's three statuses; :mod:`gitm.optimizer.pricing` says how a row gets one.
STATUSES = ("estimated", "not_identifiable_from_this_data", "absent")
FORMS = (
    "region slowdown",
    "serialization",
    "additive",
    "regime gate",
    "observation distortion",
    "baseline",
)


@dataclass(frozen=True)
class Primitive:
    """One cause, or one undivided group of causes (``members`` non-empty)."""

    id: str
    #: What the output calls the row: what the record shows, not what caused it.
    name: str
    form: str
    #: ``partition`` (pid, device), ``device`` (pids pooled) or ``run``.
    scope: str
    witness: str
    #: What a ``cannot_tell`` reader does next.
    remedy: str
    group: str | None = None
    members: tuple[str, ...] = ()
    fire_at: float | None = None
    fire_unit: str | None = None
    #: Names of ``gitm/kernels/library.yaml`` entries; empty means no lever in the library.
    levers: tuple[str, ...] = ()
    #: Fixed ``cannot_tell`` reason in this build, for rows no check can evaluate.
    fixed_reason: str | None = None
    #: The channel or input a fixed reason names.
    fixed_detail: str | None = None
    #: A baseline is listed and never checked.
    baseline: bool = False

    @property
    def is_group(self) -> bool:
        return bool(self.members)


_WINDOW = {"fire_at": FIRE_AT_WINDOW_SHARE, "fire_unit": "share of window"}

_ENTRIES: tuple[Primitive, ...] = (
    # ── serving and scheduler ────────────────────────────────────────────────
    Primitive(
        "1", "prefill-decode interference", "regime gate", "run",
        "inter-token latency tail against vllm:iteration_tokens_total; prefill-phase kernels "
        "between decode steps",
        "per-step metadata (step id, batch size) on trace events",
        levers=("enable_chunked_prefill", "max_num_batched_tokens_dynamic"),
        fixed_reason="no_passive_witness", fixed_detail="no trace event carries a step id",
    ),
    Primitive(
        "2", "request queue time",
        "additive", "run", "mean vllm:request_queue_time_seconds as a share of mean "
        "vllm:time_to_first_token_seconds",
        "the server's /metrics counters around the window",
        fixed_reason="not_built",
        fixed_detail="the /metrics counters are not read by this build; "
        "scheduler_attribution reports admission_backlog from the server's queue depth",
        levers=("max_num_seqs_dynamic", "max_num_batched_tokens_dynamic"),
    ),
    Primitive(
        "3", "KV-cache preemption (recompute)", "regime gate", "run",
        "vllm:num_preemptions_total delta",
        "the server's /metrics counters around the window",
        fixed_reason="not_built",
        fixed_detail="the /metrics counters are not read by this build; "
        "scheduler_attribution reports kv_cache_preemption from the preemption count",
        levers=("gpu_memory_utilization_dynamic", "max_num_seqs_dynamic"),
    ),
    Primitive(
        "4", "prefix-cache miss fraction", "additive", "run",
        "1 - vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total, on deltas",
        "the server's /metrics counters around the window",
        fixed_reason="not_built",
        fixed_detail="the /metrics counters are not read by this build",
        levers=("enable_prefix_caching",),
    ),
    Primitive(
        "5", "speculative-draft rejection fraction",
        "region slowdown", "run", "1 - vllm:spec_decode_num_accepted_tokens_total / "
        "vllm:spec_decode_num_draft_tokens_total, on deltas",
        "the server's /metrics counters around the window",
        fixed_reason="not_built",
        fixed_detail="the /metrics counters are not read by this build",
    ),
    Primitive(
        "6", "structured-output mask overhead", "additive", "partition", "none in engine metrics",
        "host CPU counters and thread state; request mix", group="G2",
    ),
    Primitive(
        "7", "batch-size and context-length step scaling", "baseline", "run", "num_requests_running, KV usage",
        "per-step metadata", baseline=True,
    ),
    # ── host and runtime ─────────────────────────────────────────────────────
    Primitive(
        "8", "framework-bound launch",
        "additive", "partition", "launch-scale gaps; not separable from row 23",
        "an untraced arm; runtime API records", group="G1",
        levers=("cuda_graphs_enable", "max_seq_len_to_capture_dynamic"),
    ),
    Primitive(
        "9", "GPU starvation",
        "additive", "partition",
        "long idle gap with no sync or memcpy over it; not separable from row 6 or no demand",
        "host CPU counters and thread state; request mix", group="G2",
        levers=("async_scheduling",),
    ),
    Primitive(
        "10", "gap time under a synchronization record",
        "serialization", "partition",
        "gap time covered by a sync record and not by a memcpy record",
        "a capture of at least 1 s in which every record carries its pid", **_WINDOW,
    ),
    Primitive(
        "11", "gap time under a memcpy record",
        "additive", "partition", "gap time covered by a memcpy record",
        "a capture of at least 1 s in which every record carries its pid", **_WINDOW,
    ),
    Primitive(
        "12", "allocator miss", "regime gate", "partition",
        "runtime API records", "runtime API records", group="G3",
    ),
    # ── kernel ───────────────────────────────────────────────────────────────
    Primitive(
        "13", "fused-path fallback, missed fusion", "region slowdown",
        "device", "kernel set differs from a reference",
        "NVTX op identity, which gives kernel count per op and the op's dtype", group="G4",
        levers=("attention_backend_flashinfer",),
    ),
    Primitive(
        "14", "precision / Tensor Core fallback", "region slowdown", "device",
        "kernel set differs from a reference",
        "NVTX op identity, which gives kernel count per op and the op's dtype", group="G4",
    ),
    Primitive(
        "15", "algorithm mis-selection", "region slowdown",
        "device", "kernel set differs from a reference",
        "NVTX op identity, which gives kernel count per op and the op's dtype", group="G4",
    ),
    Primitive(
        "16", "tile / wave quantization", "baseline", "run",
        "grid dims against duration", "a calibration capture", baseline=True,
    ),
    Primitive(
        "17", "warmup", "additive", "partition", "first-occurrence outlier; not separable from row 12",
        "runtime API records", group="G3",
    ),
    # ── device and multi-GPU ─────────────────────────────────────────────────
    Primitive(
        "18", "context time-slicing", "serialization",
        "run", "queue time against another pid's activity",
        "launch timestamps and the other context's activity on the device",
        fixed_reason="channel_missing",
        fixed_detail="context_id is kept, but nothing in the trace says when another "
        "context held the device",
    ),
    Primitive(
        "19", "SM contention", "region slowdown", "run",
        "kernel duration inflates while another pid runs",
        "per-process device utilization; trace every tenant on the device",
        fixed_reason="channel_missing",
        fixed_detail="per-process utilization is not collected and an untraced co-tenant "
        "leaves no record",
    ),
    Primitive(
        "20", "thermal throttling, power capping", "regime gate", "run",
        "any NVML sample in the window with SW_POWER_CAP, HW_SLOWDOWN, SW_THERMAL_SLOWDOWN, "
        "HW_THERMAL_SLOWDOWN or HW_POWER_BRAKE_SLOWDOWN set; the observation lists the bits "
        "seen. GPU_IDLE, APPLICATIONS_CLOCKS_SETTING, SYNC_BOOST and DISPLAY_CLOCK_SETTING "
        "never fire it",
        "an NVML reader in the witness checks; this build reads no NVML samples",
        fixed_reason="channel_missing", fixed_detail="no NVML samples are read by this build",
    ),
    Primitive(
        "21", "memory-bound regime against efficiency loss", "baseline", "run", "measured bytes per kernel",
        "a calibration capture with measured memory traffic",
        fixed_reason="no_passive_witness", fixed_detail="measured memory traffic is never set",
    ),
    Primitive(
        "22", "exposed collective communication",
        "additive", "device",
        "time in kernels classed tp_all_reduce that no compute kernel on the device overlaps",
        "kernel names the op classifier recognises as collectives", **_WINDOW,
    ),
    Primitive(
        "25", "exposed expert-parallel dispatch (all-to-all)", "additive",
        "device",
        "time in kernels classed moe_all_to_all that no compute kernel on the device overlaps",
        "kernel names the op classifier recognises as collectives",
        levers=("enable_dbo",), **_WINDOW,
    ),
    # ── measurement ──────────────────────────────────────────────────────────
    Primitive(
        "23", "tracer overhead", "observation distortion", "partition",
        "untraced TPOT", "an untraced arm", group="G1",
    ),
    Primitive(
        "24", "kernel misroute, event dropping", "observation distortion", "run",
        "the trace header's dropped-record count, above zero; a misrouted kernel is not "
        "checked",
        "NVTX op identity, to check for misrouted kernels",
        fire_at=1, fire_unit="dropped records",
    ),
    # ── undivided groups: the group is the reported unit ─────────────────────
    Primitive(
        "G1", "launch-scale gaps (framework launch or tracer overhead)", "additive",
        "partition", "gap time left in gaps of 20 us or less after memcpy and sync cover",
        "an untraced arm; runtime API records", members=("8", "23"), **_WINDOW,
    ),
    Primitive(
        "G2", "long idle (GPU waiting on the host, or no work)", "additive", "partition",
        "the join to the running-requests gauge needs a trace-to-wall clock mapping that is "
        "unproven without hardware",
        "metrics_samples.jsonl for the window and a proven trace-to-wall clock mapping",
        members=("6", "9"),
    ),
    Primitive(
        "G3", "first-occurrence cost (warmup or allocator)", "additive", "partition",
        "two rules on kernel durations were built and withdrawn after review; where warmup "
        "time lands needs a real capture",
        "runtime API records, and a real capture showing where warmup time lands",
        members=("12", "17"),
    ),
    Primitive(
        "G4", "kernel names differ from the reference",
        "region slowdown",
        "device",
        "kernel names holding at least 0.5% of launches in one of capture and reference and "
        "missing from the other",
        "NVTX op identity, which gives kernel count per op and the op's dtype",
        members=("13", "14", "15"), fire_at=1, fire_unit="kernel names",
    ),
)

#: Checks this build does not have. Their rows are ``cannot_tell``, ``not_built``.
NOT_BUILT = ("G2", "G3", "G4")

#: Ordered and read-only. Ids are strings: row numbers ``"1"``..``"25"`` and ``"G1"``..``"G4"``.
REGISTRY: Mapping[str, Primitive] = MappingProxyType({p.id: p for p in _ENTRIES})


def get(primitive_id: str | int) -> Primitive:
    """The entry for a row number or group id; ``KeyError`` names an unknown id."""
    return REGISTRY[str(primitive_id)]
