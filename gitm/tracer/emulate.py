"""Render one ground-truth execution as each vendor's collector would record it.

Kernel identity is decided by records no test box can produce: CUPTI's
RUNTIME/MARKER activity, rocprofiler-sdk's external-correlation stamps, the
capture-time graph-node map. Unit tests that hand-write those dicts check the
decoder against the decoder's own assumptions. This module instead starts from
what *happened* — which kernels the host launched, in which order, on which
thread, under which ranges, captured into which graph, replayed how — and
derives the records from the documented semantics of each collector:

* **CUPTI** (``cupti_core.c``): a kernel record carries the ``correlationId`` of
  its launch API call; the RUNTIME record carries the host window and thread;
  NVTX ranges arrive as MARKER start/end halves. A graph replay is one
  ``cudaGraphLaunch`` — every kernel in it carries that one id — plus
  ``graphId``/``graphNodeId`` of the *instantiated* node. The capture-time node
  map (``graph_node`` records) is the planned Nsight-style collector; it is
  rendered only when ``cupti_node_map`` asks for it, so tests can measure both
  today's NVIDIA behaviour and the planned one.
* **rocprofiler-sdk** (``rocm_inject.c``): every dispatch, copy and HIP API
  record is stamped, at enqueue on the issuing thread, with the innermost rocTX
  range open *on that thread at that instant* — or, inside ``hipGraphLaunch``,
  with ``(exec, ordinal)`` in dispatch order. Captured launches become
  ``graph_node`` records with the range open at capture time and a signature;
  instantiation links capture to executable.

Ground truth stays separate from the records (:attr:`Emulation.truth`) and is
keyed by device start, so the scorer never reads an identity the decoder could
have written. Every hazard the decoder claims to handle has a switch here —
helper-thread launches, an async device clock offset, replays dispatched in a
different order than captured, ROCclr blit nodes, a stamp that does not advance
per dispatch, launches of an executable never seen instantiated — so the claim
is tested against an execution that exhibits it, not against a record shaped
to pass.

Pure Python; nothing here needs a GPU.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from gitm.distributed.correlate import (
    GRAPH_UNTRACKED,
    capture_node_id,
    exec_node_id,
)
from gitm.tracer.schema import KernelEvent

Vendor = Literal["nvidia", "amd"]

#: Opaque, real-dialect kernel names: what each vendor's libraries call the
#: kernel, which by design says nothing about the op that launched it. A
#: GEMM's name cannot tell qkv_proj from mlp_down on either vendor — the point
#: of correlation — while attention and collectives are recognisable by name.
#: Sources: hipBLASLt/Tensile solution naming (``Cijk_<layout>_<types>_MT..``),
#: AITER's paged-attention kernels, RCCL (which keeps NCCL's device-kernel
#: names), ROCclr's blit kernels; cuBLAS's nvjet JIT family, FlashAttention-2's
#: split-KV kernel, NCCL.
DIALECT: dict[str, dict[str, str]] = {
    "amd": {
        "gemm": "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x64_MI16x16x1_SN_"
                "LDSB1_GRPM1_GSU1_ISA950_K1_WG32_8_1",
        "attn": "_ZN5aiter33paged_attention_ll4mi_QKV_mfma16_kernelI13__hip_bfloat16EEvv",
        "collective": "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)",
        "memset": "__amd_rocclr_fillBufferAligned",
    },
    "nvidia": {
        "gemm": "nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT",
        "attn": "flash_fwd_splitkv_kernel",
        "collective": "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
        "memset": "memset_kernel",
    },
}


def kernel_class(op: str | None, name: str) -> str:
    if op == "attn_score_value":
        return "attn"
    if op in (None, "tp_all_reduce") and "nccl" in name.lower():
        return "collective"
    return "gemm"


@dataclass(frozen=True)
class TruthLaunch:
    """What the host launched: identity, order and the device interval it ran."""

    step: int
    op: str | None
    layer: int | None
    stream: int
    event: KernelEvent

    @property
    def range_name(self) -> str | None:
        """The range the model's instrumentation pushes around this launch."""
        if self.op is None:
            return None
        return f"L{self.layer}/{self.op}" if self.layer is not None else self.op


@dataclass(frozen=True)
class EmulationConfig:
    vendor: Vendor
    #: Replay every step from one captured graph (vLLM's default) instead of
    #: launching eagerly (``--enforce-eager``).
    graphs: bool = False
    #: Emit the CUPTI capture-time node map (the planned NVTX/RESOURCE
    #: callback collector). Ignored for AMD, whose collector always emits it.
    cupti_node_map: bool = False
    #: Per-step H2D input upload and D2H sampled-token download, issued
    #: outside any graph under the step range.
    step_copies: bool = True
    #: Annotations appended to an op's range name (``#k=v,...``).
    annotate: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Per-launch annotations (dynamic: e.g. an expert-parallel wave index);
    #: merged over ``annotate``.
    annotate_launch: Callable[[TruthLaunch], dict[str, str] | None] | None = None
    #: Ops whose launch API is called from a second host thread (eager only):
    #: their ranges are pushed on that thread too.
    helper_thread_ops: frozenset[str] = frozenset()
    #: Device clock offset vs host clock. Any comparison of a kernel's device
    #: window against a host range would break under a large offset.
    device_offset_ns: int = 0
    api_ns: int = 2_000
    gap_ns: int = 500
    #: AMD graph hazards.
    stamp_per_dispatch: bool = True     # False: every replay dispatch gets ordinal 0
    swap_replay_pair: tuple[int, int] | None = None  # dispatch order differs from capture
    blit_memset_node: bool = False      # a captured hipMemsetAsync runs as a ROCclr blit
    exec_untracked: bool = False        # the instantiate was never seen
    #: The likeliest 7.2.3 failure: no per-dispatch request inside a launch, so
    #: each dispatch inherits the hipGraphLaunch call's stamp — the live launch
    #: range's id — and carries no graph identity at all.
    stamp_inherits_launch: bool = False
    #: Capture without ranges (instrumentation traced away under compile).
    capture_unranged: bool = False
    #: Range around each graph launch: what a naive decoder would take as every
    #: replayed kernel's op. A layer-op range here (vLLM piecewise graphs launch
    #: inside module ranges) is what makes that mistake expensive.
    graph_launch_range: str = "decode_step"
    pid: int = 4242
    seed: int = 0


@dataclass
class Emulation:
    records: list[dict]
    #: (device start_ns as recorded, stream) -> (true op, true layer, name).
    #: Start alone is not a key: a side-stream kernel can start in the same
    #: nanosecond as the compute kernel it overlaps.
    truth: dict[tuple[int, int], tuple[str | None, int | None, str]]
    config: EmulationConfig


def launches_from_fixture(fx) -> list[TruthLaunch]:
    """Ground-truth launches from a :class:`~gitm.optimizer.mechanism_fixtures.Fixture`."""
    from gitm.optimizer.mechanism_fixtures import SIDE_OP

    out = []
    for la in fx.launches:
        op = None if la.op == SIDE_OP else la.op
        out.append(TruthLaunch(la.step, op, la.layer if op else None, la.stream,
                               fx.trace.events[la.event]))
    return out


def _kernel_id(name: str) -> int:
    return int.from_bytes(hashlib.sha1(name.encode()).digest()[:6], "big") | 1


def _geometry(op: str | None, cls: str) -> tuple[list[int], list[int]]:
    """Launch geometry by op: same GEMM kernel, different shapes per projection."""
    h = int.from_bytes(hashlib.sha1(f"{op}/{cls}".encode()).digest()[:2], "big")
    return [8 + h % 504, 1, 1], [256, 1, 1]


class _Host:
    """Host timeline: per-thread range stacks, API records, marker halves."""

    def __init__(self, cfg: EmulationConfig, armed: bool = True):
        self.cfg = cfg
        self.t = 1_000_000_000
        self.corr = 0
        self.marker_seq = 0
        self.stacks: dict[int, list[tuple[int, str]]] = {}
        self.records: list[dict] = []
        self.armed = armed

    def tick(self, ns: int) -> int:
        self.t += ns
        return self.t

    def push(self, thread: int, name: str) -> None:
        self.marker_seq += 1
        mid = self.marker_seq
        self.stacks.setdefault(thread, []).append((mid, name))
        if self.armed:
            self.records.append({"kind": "marker", "name": name, "timestamp_ns": self.tick(10),
                                 "marker_id": mid, "marker_flags": 0, "thread_id": thread})
        else:
            self.tick(10)

    def pop(self, thread: int) -> None:
        mid, _ = self.stacks[thread].pop()
        if self.armed:
            self.records.append({"kind": "marker", "name": None, "timestamp_ns": self.tick(10),
                                 "marker_id": mid, "marker_flags": 1, "thread_id": thread})
        else:
            self.tick(10)

    def top(self, thread: int) -> tuple[int, str] | None:
        s = self.stacks.get(thread)
        return s[-1] if s else None

    def api(self, thread: int, *, graph_launch: bool = False) -> dict:
        """A launch-API call: returns its runtime record (emitted when armed)."""
        self.corr += 1
        start = self.tick(self.cfg.gap_ns)
        end = self.tick(self.cfg.api_ns)
        top = self.top(thread)
        rec = {"kind": "runtime", "start_ns": start, "end_ns": end,
               "correlation_id": self.corr, "thread_id": thread}
        if self.cfg.vendor == "amd":
            rec["range_id"] = top[0] if top else 0
            rec["graph_launch"] = int(graph_launch)
        if self.armed:
            self.records.append(rec)
        return rec


def _range_name(cfg: EmulationConfig, la: TruthLaunch) -> str | None:
    base = la.range_name
    if base is None:
        return None
    attrs = dict(cfg.annotate.get(la.op or "", {}))
    if cfg.annotate_launch is not None:
        attrs.update(cfg.annotate_launch(la) or {})
    if attrs:
        return base + "#" + ",".join(f"{k}={v}" for k, v in attrs.items())
    return base


def emulate(launches: Sequence[TruthLaunch], cfg: EmulationConfig) -> Emulation:
    """Render ``launches`` (host emission order) as ``cfg.vendor``'s collector would."""
    names = DIALECT[cfg.vendor]
    host = _Host(cfg)
    records = host.records
    truth: dict[tuple[int, int], tuple[str | None, int | None, str]] = {}
    main, helper = 1001, 1002
    off = cfg.device_offset_ns

    def kernel_rec(la: TruthLaunch, corr: int, *, name: str | None = None) -> dict:
        cls = kernel_class(la.op, la.event.name)
        nm = name or names[cls]
        grid, block = _geometry(la.op, cls)
        rec = {"kind": "kernel", "name": nm, "start_ns": la.event.start_ns + off,
               "end_ns": la.event.end_ns + off, "device_id": 0, "context_id": 0,
               "stream_id": la.stream, "correlation_id": corr, "grid": grid,
               "block": block, "static_shared_mem": 0, "dynamic_shared_mem": 0,
               "registers_per_thread": 64}
        if cfg.vendor == "amd":
            rec["kernel_id"] = _kernel_id(nm)
        key = (rec["start_ns"], rec["stream_id"])
        assert key not in truth, f"two launches share device start and stream: {key}"
        truth[key] = (la.op, la.layer, nm)
        return rec

    def copy_rec(corr: int, kind: int, t: int, *, thread_top: tuple | None) -> dict:
        rec = {"kind": "memcpy", "copy_kind": kind, "bytes": 4096, "start_ns": t + off,
               "end_ns": t + off + 800, "device_id": 0, "context_id": 0, "stream_id": 0,
               "correlation_id": corr}
        if cfg.vendor == "amd":
            rec["range_id"] = thread_top[0] if thread_top else 0
        return rec

    steps: dict[int, list[TruthLaunch]] = {}
    for la in launches:
        steps.setdefault(la.step, []).append(la)

    if not cfg.graphs:
        for _s, step in sorted(steps.items()):
            host.push(main, "decode_step")
            first, last = step[0].event, step[-1].event
            if cfg.step_copies:
                rt = host.api(main)
                records.append(copy_rec(rt["correlation_id"], 1, first.start_ns - 2_000,
                                        thread_top=host.top(main)))
            for la in step:
                thread = helper if la.op in cfg.helper_thread_ops else main
                rn = _range_name(cfg, la)
                if rn:
                    host.push(thread, rn)
                top = host.top(thread)
                rt = host.api(thread)
                k = kernel_rec(la, rt["correlation_id"])
                if cfg.vendor == "amd":
                    k["range_id"] = top[0] if top else 0
                records.append(k)
                if rn:
                    host.pop(thread)
            if cfg.step_copies:
                rt = host.api(main)
                records.append(copy_rec(rt["correlation_id"], 2, last.end_ns + 500,
                                        thread_top=host.top(main)))
            host.pop(main)
        return _finish(records, truth, cfg)

    # ── graph mode ──────────────────────────────────────────────────────────
    template = sorted(steps.items())[0][1]
    node_kinds = ["kernel"] * len(template)
    if cfg.blit_memset_node:
        node_kinds.append("memset")

    # Capture: at engine start, before any window arms — markers and API
    # records of this phase never reach the shard; structural records do.
    capture = _Host(cfg, armed=False)
    capture.t = 10_000_000
    capture_id, exec_id = 1, 1
    cupti_graph, cupti_base, cupti_clone = 7, 10_000, 20_000
    for k, kind in enumerate(node_kinds):
        la = template[k] if kind == "kernel" else None
        rn = _range_name(cfg, la) if la is not None and not cfg.capture_unranged else None
        if rn:
            capture.push(main, rn)
        top = capture.top(main)
        if cfg.vendor == "amd":
            node = {"kind": "graph_node", "graph_node_id": capture_node_id(capture_id, k),
                    "capture_id": capture_id, "node_kind": kind,
                    "name": top[1] if top else ""}
            if kind == "kernel":
                cls = kernel_class(la.op, la.event.name)
                node["kernel_id"] = _kernel_id(names[cls])
                node["grid"], node["block"] = _geometry(la.op, cls)
            else:
                node["kernel_id"] = 0
            records.append(node)
        elif cfg.cupti_node_map:
            records.append({"kind": "graph_node", "graph_node_id": cupti_base + k,
                            "name": top[1] if top else "", "cloned_from": None})
            records.append({"kind": "graph_node", "graph_node_id": cupti_clone + k,
                            "name": "", "cloned_from": cupti_base + k})
        if rn:
            capture.pop(main)
    if cfg.vendor == "amd" and not cfg.exec_untracked:
        records.append({"kind": "graph_exec", "graph_id": exec_id, "capture_id": capture_id,
                        "n_nodes": len(node_kinds)})

    for _s, step in sorted(steps.items()):
        host.push(main, "decode_step")
        if cfg.step_copies:
            rt = host.api(main)
            records.append(copy_rec(rt["correlation_id"], 1, step[0].event.start_ns - 2_000,
                                    thread_top=host.top(main)))
        if cfg.graph_launch_range != "decode_step":
            host.push(main, cfg.graph_launch_range)
        rt = host.api(main, graph_launch=True)
        if cfg.graph_launch_range != "decode_step":
            host.pop(main)
        # Dispatch order: capture order, unless the replay is forked and the
        # runtime reorders it; the ordinal stamp follows dispatch order.
        dispatch = list(range(len(step)))
        if cfg.swap_replay_pair:
            i, j = cfg.swap_replay_pair
            dispatch[i], dispatch[j] = dispatch[j], dispatch[i]
        for ordinal, k in enumerate(dispatch):
            la = step[k]
            rec = kernel_rec(la, rt["correlation_id"])
            if cfg.vendor == "amd" and cfg.stamp_inherits_launch:
                rec["range_id"] = rt["range_id"]
            elif cfg.vendor == "amd":
                exec_seq = GRAPH_UNTRACKED if cfg.exec_untracked else exec_id
                stamp_ord = ordinal if cfg.stamp_per_dispatch else 0
                rec["graph_id"] = exec_seq
                rec["graph_node_id"] = exec_node_id(exec_seq, stamp_ord)
                rec["range_id"] = 0
            else:
                rec["graph_id"] = cupti_graph
                rec["graph_node_id"] = cupti_clone + k
            records.append(rec)
        if cfg.blit_memset_node:
            last = step[-1].event
            blit = TruthLaunch(step[-1].step, None, None, 0, KernelEvent(
                name=names["memset"], start_ns=last.end_ns + 100, end_ns=last.end_ns + 400,
                stream_id=0, device_id=0))
            rec = kernel_rec(blit, rt["correlation_id"], name=names["memset"])
            if cfg.vendor == "amd" and cfg.stamp_inherits_launch:
                rec["range_id"] = rt["range_id"]
            elif cfg.vendor == "amd":
                exec_seq = GRAPH_UNTRACKED if cfg.exec_untracked else exec_id
                rec["graph_id"] = exec_seq
                rec["graph_node_id"] = exec_node_id(
                    exec_seq, len(step) if cfg.stamp_per_dispatch else 0)
                rec["range_id"] = 0
            else:
                rec["graph_id"] = cupti_graph
                rec["graph_node_id"] = cupti_clone + len(step)
            records.append(rec)
        if cfg.step_copies:
            rt2 = host.api(main)
            records.append(copy_rec(rt2["correlation_id"], 2, step[-1].event.end_ns + 500,
                                    thread_top=host.top(main)))
        host.pop(main)
    return _finish(records, truth, cfg)


def _finish(records, truth, cfg) -> Emulation:
    if cfg.vendor == "amd":
        records.insert(0, {"kind": "meta", "collector": "rocprofiler-sdk",
                           "sdk_version": "1.1.0", "identity": 1,
                           "agents": [{"ordinal": 0, "name": "gfx950",
                                       "product": "AMD Instinct MI355X"}]})
    for r in records:
        r["pid"] = cfg.pid
    # Shards interleave buffers in no promised order; the decoder must not care.
    random.Random(cfg.seed).shuffle(records)
    return Emulation(records, truth, cfg)


@dataclass
class IdentityScore:
    """Per-kernel identity against ground truth, on what downstream consumes."""

    n: int = 0
    correct: int = 0
    #: resolved to a *different* op or layer than the one launched — the
    #: silent failure; must be zero for any mechanism worth shipping.
    wrong: int = 0
    #: resolved to nothing where truth had an op (falls back to the name).
    missing: int = 0
    wrong_examples: list[tuple] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 1.0


def score(events, truth: dict[tuple[int, int], tuple[str | None, int | None, str]]
          ) -> IdentityScore:
    """Compare decoded kernels to the launch truth.

    Identity is judged as the monitor consumes it — ``observed_op(name,
    range_op)`` and ``range_layer`` — so a step-level range that downstream
    already ignores is not penalised, and a wrong op that downstream would
    trust is.
    """
    from gitm.optimizer.deviation import classify_op, observed_op

    out = IdentityScore()
    for e in events:
        key = (e.start_ns, e.stream_id)
        if getattr(e, "kind", None) != "kernel" or key not in truth:
            continue
        t_op, t_layer, name = truth[key]
        want = (t_op or classify_op(name), t_layer)
        got = (observed_op(e.name, e.range_op), e.range_layer)
        out.n += 1
        if got == want:
            out.correct += 1
        elif e.range_op is None or (e.range_layer is None and t_layer is not None
                                    and got[0] == want[0]):
            out.missing += 1
        else:
            out.wrong += 1
            if len(out.wrong_examples) < 5:
                out.wrong_examples.append((name, want, got))
    return out
