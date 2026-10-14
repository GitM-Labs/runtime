"""Render one ground-truth execution as each vendor's collector would record it.

Tests that hand-write collector dicts check the decoder against its own
assumptions. This starts from what *happened* — launches in host order, on which
thread, under which ranges, captured into which graph, replayed how — and
derives the records from each collector's semantics:

* CUPTI: a kernel carries its launch API's ``correlation_id``; runtime records
  carry the host window and thread; NVTX ranges arrive as marker halves. A graph
  replay is one launch whose kernels report ``graphId``/``graphNodeId``; the
  capture-time node map (NVTX + RESOURCE callbacks) is rendered with
  ``cupti_node_map``, off to model a collector built without it.
* rocprofiler-sdk (``rocm_inject.c``): records are stamped at enqueue with the
  innermost range on the issuing thread, or inside a graph launch with
  ``(exec, ordinal)``; captured launches become signed ``graph_node`` records.

Truth is kept apart from the records (:attr:`Emulation.truth`), and every hazard
the decoder claims to handle has a switch in :class:`EmulationConfig`.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from gitm.distributed.correlate import GRAPH_UNTRACKED, capture_node_id, exec_node_id
from gitm.tracer.schema import KernelEvent

Vendor = Literal["nvidia", "amd"]

#: Real-dialect kernel names that say nothing about the launching op: a GEMM's
#: name can't tell qkv_proj from mlp_down on either vendor.
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
MAIN, HELPER = 1001, 1002
#: The range vLLM's TorchCompileWrapper pushes around the compiled model.
COMPILED_WRAPPER_RANGE = "Torch Compiled Module (input):LlamaForCausalLM"


def kernel_class(op: str | None, name: str) -> str:
    if op == "attn_score_value":
        return "attn"
    if op in (None, "tp_all_reduce") and "nccl" in name.lower():
        return "collective"
    return "gemm"


@dataclass(frozen=True)
class TruthLaunch:
    step: int
    op: str | None
    layer: int | None
    stream: int
    event: KernelEvent

    @property
    def range_name(self) -> str | None:
        if self.op is None:
            return None
        return f"L{self.layer}/{self.op}" if self.layer is not None else self.op


@dataclass(frozen=True)
class EmulationConfig:
    vendor: Vendor
    graphs: bool = False
    cupti_node_map: bool = False
    step_copies: bool = True
    annotate: dict[str, dict[str, str]] = field(default_factory=dict)
    annotate_launch: Callable[[TruthLaunch], dict[str, str] | None] | None = None
    helper_thread_ops: frozenset[str] = frozenset()
    device_offset_ns: int = 0
    api_ns: int = 2_000
    gap_ns: int = 500
    #: range open around each graph launch — what a naive decoder takes as the op
    graph_launch_range: str = "decode_step"
    # AMD graph hazards
    stamp_per_dispatch: bool = True     # False: every dispatch gets ordinal 0
    stamp_inherits_launch: bool = False  # dispatches inherit the launch call's range stamp
    swap_replay_pair: tuple[int, int] | None = None
    blit_memset_node: bool = False
    exec_untracked: bool = False
    #: AMD_DIRECT_DISPATCH=0: a worker thread submits replays, outside the
    #: launch's correlation scope — fresh correlation ids, no stamps
    worker_dispatch: bool = False
    capture_unranged: bool = False
    #: vLLM's compiled path: only the whole-model wrapper range is open while
    #: graphs are captured (layerwise hooks never fire there)
    capture_compiled: bool = False
    pid: int = 4242
    seed: int = 0


@dataclass
class Emulation:
    records: list[dict]
    #: (device start_ns, stream) -> (true op, true layer, name)
    truth: dict[tuple[int, int], tuple[str | None, int | None, str]]
    config: EmulationConfig


def launches_from_fixture(fx) -> list[TruthLaunch]:
    from gitm.optimizer.mechanism_fixtures import SIDE_OP

    return [TruthLaunch(la.step, None if la.op == SIDE_OP else la.op,
                        None if la.op == SIDE_OP else la.layer, la.stream,
                        fx.trace.events[la.event]) for la in fx.launches]


def _kernel_id(name: str) -> int:
    return int.from_bytes(hashlib.sha1(name.encode()).digest()[:6], "big") | 1


def _geometry(op: str | None, cls: str) -> tuple[list[int], list[int]]:
    """Same GEMM kernel, a different shape per projection."""
    h = int.from_bytes(hashlib.sha1(f"{op}/{cls}".encode()).digest()[:2], "big")
    return [8 + h % 504, 1, 1], [256, 1, 1]


class _Host:
    """Host timeline: per-thread range stacks, API and marker records."""

    def __init__(self, cfg: EmulationConfig, records: list[dict], armed: bool = True):
        self.cfg, self.records, self.armed = cfg, records, armed
        self.t = 1_000_000_000 if armed else 10_000_000
        self.corr = self.marker_seq = 0
        self.stacks: dict[int, list[tuple[int, str]]] = {}

    def _emit(self, rec: dict) -> dict:
        if self.armed:
            self.records.append(rec)
        return rec

    def push(self, thread: int, name: str) -> None:
        self.marker_seq += 1
        self.stacks.setdefault(thread, []).append((self.marker_seq, name))
        self.t += 10
        self._emit({"kind": "marker", "name": name, "timestamp_ns": self.t,
                    "marker_id": self.marker_seq, "marker_flags": 0, "thread_id": thread})

    def pop(self, thread: int) -> None:
        mid, _ = self.stacks[thread].pop()
        self.t += 10
        self._emit({"kind": "marker", "name": None, "timestamp_ns": self.t,
                    "marker_id": mid, "marker_flags": 1, "thread_id": thread})

    def top(self, thread: int) -> tuple[int, str] | None:
        s = self.stacks.get(thread)
        return s[-1] if s else None

    def api(self, thread: int, *, graph_launch: bool = False) -> dict:
        self.corr += 1
        self.t += self.cfg.gap_ns
        start, self.t = self.t, self.t + self.cfg.api_ns
        rec = {"kind": "runtime", "start_ns": start, "end_ns": self.t,
               "correlation_id": self.corr, "thread_id": thread}
        if self.cfg.vendor == "amd":
            top = self.top(thread)
            rec.update(range_id=top[0] if top else 0, graph_launch=int(graph_launch))
        return self._emit(rec)


def _range_name(cfg: EmulationConfig, la: TruthLaunch | None) -> str | None:
    base = la.range_name if la is not None else None
    if base is None:
        return None
    attrs = {**cfg.annotate.get(la.op or "", {}),
             **((cfg.annotate_launch(la) or {}) if cfg.annotate_launch else {})}
    return base + "#" + ",".join(f"{k}={v}" for k, v in attrs.items()) if attrs else base


def emulate(launches: Sequence[TruthLaunch], cfg: EmulationConfig) -> Emulation:
    """Render ``launches`` (host emission order) as ``cfg.vendor``'s collector would."""
    amd = cfg.vendor == "amd"
    names = DIALECT[cfg.vendor]
    records: list[dict] = []
    truth: dict[tuple[int, int], tuple[str | None, int | None, str]] = {}
    host = _Host(cfg, records)
    off = cfg.device_offset_ns

    def kernel_rec(la: TruthLaunch, corr: int) -> dict:
        cls = kernel_class(la.op, la.event.name)
        nm = names["memset"] if la.event.name == names["memset"] else names[cls]
        grid, block = _geometry(la.op, cls)
        rec = {"kind": "kernel", "name": nm, "start_ns": la.event.start_ns + off,
               "end_ns": la.event.end_ns + off, "device_id": 0, "context_id": 0,
               "stream_id": la.stream, "correlation_id": corr, "grid": grid, "block": block}
        if amd:
            rec["kernel_id"] = _kernel_id(nm)
        key = (rec["start_ns"], rec["stream_id"])
        assert key not in truth, f"two launches share device start and stream: {key}"
        truth[key] = (la.op, la.layer, nm)
        return rec

    def copy(kind: int, t: int) -> None:
        rt = host.api(MAIN)
        rec = {"kind": "memcpy", "copy_kind": kind, "bytes": 4096, "start_ns": t + off,
               "end_ns": t + off + 800, "device_id": 0, "context_id": 0, "stream_id": 0,
               "correlation_id": rt["correlation_id"]}
        if amd:
            rec["range_id"] = rt["range_id"]
        records.append(rec)

    steps: dict[int, list[TruthLaunch]] = {}
    for la in launches:
        steps.setdefault(la.step, []).append(la)
    if cfg.graphs:
        _capture(cfg, records, sorted(steps.items())[0][1], names)

    for _s, step in sorted(steps.items()):
        host.push(MAIN, "decode_step")
        if cfg.step_copies:
            copy(1, step[0].event.start_ns - 2_000)
        if not cfg.graphs:
            for la in step:
                thread = HELPER if la.op in cfg.helper_thread_ops else MAIN
                rn = _range_name(cfg, la)
                if rn:
                    host.push(thread, rn)
                rt = host.api(thread)
                rec = kernel_rec(la, rt["correlation_id"])
                if amd:
                    rec["range_id"] = rt["range_id"]
                records.append(rec)
                if rn:
                    host.pop(thread)
        else:
            wrap = cfg.graph_launch_range != "decode_step"
            if wrap:
                host.push(MAIN, cfg.graph_launch_range)
            rt = host.api(MAIN, graph_launch=True)
            if wrap:
                host.pop(MAIN)
            order = list(step)
            if cfg.swap_replay_pair:
                i, j = cfg.swap_replay_pair
                order[i], order[j] = order[j], order[i]
            if cfg.blit_memset_node:
                end = step[-1].event.end_ns
                order.append(TruthLaunch(step[-1].step, None, None, 0, KernelEvent(
                    name=names["memset"], start_ns=end + 100, end_ns=end + 400,
                    stream_id=0, device_id=0)))
            for ordinal, la in enumerate(order):
                rec = kernel_rec(la, rt["correlation_id"])
                # CUPTI reports the node the kernel was captured as (by template
                # position), under its instantiated clone id.
                node = step.index(la) if la in step else len(step)
                records.append(_replay_stamp(cfg, rec, ordinal, node, rt))
        if cfg.step_copies:
            copy(2, step[-1].event.end_ns + 500)
        host.pop(MAIN)

    if amd:
        records.insert(0, {"kind": "meta", "collector": "rocprofiler-sdk", "sdk_version": "1.1.0",
                           "identity": 1, "direct_dispatch": int(not cfg.worker_dispatch),
                           "agents": [{"ordinal": 0, "name": "gfx950",
                                                      "product": "AMD Instinct MI355X"}]})
    for r in records:
        r["pid"] = cfg.pid
    random.Random(cfg.seed).shuffle(records)  # shards promise no record order
    return Emulation(records, truth, cfg)


_CUPTI_GRAPH, _CUPTI_NODE, _CUPTI_CLONE = 7, 10_000, 20_000
_EXEC, _CAPTURE = 1, 1


def _replay_stamp(cfg: EmulationConfig, rec: dict, ordinal: int, node: int, rt: dict) -> dict:
    if cfg.vendor == "amd" and cfg.worker_dispatch:
        rec.update(correlation_id=10**9 + rec["start_ns"] % 10**9, range_id=0)
    elif cfg.vendor != "amd":
        rec.update(graph_id=_CUPTI_GRAPH, graph_node_id=_CUPTI_CLONE + node)
    elif cfg.stamp_inherits_launch:
        rec["range_id"] = rt["range_id"]
    else:
        exec_seq = GRAPH_UNTRACKED if cfg.exec_untracked else _EXEC
        rec.update(graph_id=exec_seq, range_id=0, graph_node_id=exec_node_id(
            exec_seq, ordinal if cfg.stamp_per_dispatch else 0))
    return rec


def _capture(cfg: EmulationConfig, records: list[dict], template: list[TruthLaunch],
             names: dict[str, str]) -> None:
    """Graph capture at engine start: only structural records reach the shard."""
    host = _Host(cfg, records, armed=False)
    kinds = ["kernel"] * len(template) + (["memset"] if cfg.blit_memset_node else [])
    for k, kind in enumerate(kinds):
        la = template[k] if kind == "kernel" else None
        rn = (None if cfg.capture_unranged else
              COMPILED_WRAPPER_RANGE if cfg.capture_compiled else _range_name(cfg, la))
        if rn:
            host.push(MAIN, rn)
        top = host.top(MAIN)
        name = top[1] if top else ""
        if cfg.vendor == "amd":
            node = {"kind": "graph_node", "graph_node_id": capture_node_id(_CAPTURE, k),
                    "capture_id": _CAPTURE, "node_kind": kind, "name": name, "kernel_id": 0}
            if la is not None:
                cls = kernel_class(la.op, la.event.name)
                node["kernel_id"] = _kernel_id(names[cls])
                node["grid"], node["block"] = _geometry(la.op, cls)
            records.append(node)
        elif cfg.cupti_node_map:
            records += [{"kind": "graph_node", "graph_node_id": _CUPTI_NODE + k, "name": name,
                         "node_kind": kind, "cloned_from": 0},
                        {"kind": "graph_node", "graph_node_id": _CUPTI_CLONE + k, "name": "",
                         "node_kind": kind, "cloned_from": _CUPTI_NODE + k}]
        if rn:
            host.pop(MAIN)
    if cfg.vendor == "amd" and not cfg.exec_untracked:
        records.append({"kind": "graph_exec", "graph_id": _EXEC, "capture_id": _CAPTURE,
                        "n_nodes": len(kinds)})


@dataclass
class IdentityScore:
    n: int = 0
    correct: int = 0
    #: named as a different op or layer than launched — must be zero
    wrong: int = 0
    #: unnamed where truth had an op (falls back to the kernel name)
    missing: int = 0
    wrong_examples: list[tuple] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 1.0


def score(events, truth: dict[tuple[int, int], tuple[str | None, int | None, str]]
          ) -> IdentityScore:
    """Judge identity as downstream consumes it: ``observed_op`` and ``range_layer``.

    Wrong means a *modeled* op or a layer other than the one launched: a label
    no rule knows (a whole-model range) reaches no predicted node, so it is
    unnamed for every consumer, not misnamed.
    """
    from gitm.optimizer.deviation import _OP_RULES, classify_op, observed_op

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
        elif e.range_op is None or got[0] not in _OP_RULES or (
                e.range_layer is None and t_layer is not None and got[0] == want[0]):
            out.missing += 1
        else:
            out.wrong += 1
            if len(out.wrong_examples) < 5:
                out.wrong_examples.append((name, want, got))
    return out
