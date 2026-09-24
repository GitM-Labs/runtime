"""Synthetic traces with a known cause — the five deviation mechanisms, injected.

    fixtures = generate(Scenario(points=[BatchConfig(batch=8, kv_cache_len=2048)],
                                 mechanisms=(RegionSlowdown(0.5, ops={"attn_score_value"}),)))
    obs = observe(fixtures[0])          # through monitor.residuals / check_invariants

Every claim in ``docs/mechanism_model.md`` about which mechanisms the monitor can
and cannot tell apart is checked by building both sides here and comparing what
the *real* observation path makes of them. So the generator produces exactly what
a capture produces — a :class:`Trace` of kernel events — and nothing is compared
at the level of the generator's own bookkeeping.

Three stages, kept apart because the mechanisms split along them:

1. **Truth.** Each kernel's true duration comes from :class:`TruthModel`, which
   reuses the planner's work counts (FLOPs and bytes) but never its timing. A
   truth built from ``t_pred`` would agree with the prediction by construction,
   and the separating conditions that depend on how cost scales with the
   operating point could then not fail.
2. **Execution.** Region slowdown, serialization, additive cost and the regime
   gate change the execution; a list scheduler turns it into start/end times.
3. **Observation.** Observation distortion changes only what the trace records.
   The untraced system is the true timeline, so ``true_tpot_s`` is what the
   ``off`` arm would report.

Pure Python and NumPy. Nothing here needs a GPU.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from gitm.importers.node_rollup import device_comm_stats
from gitm.optimizer.monitor import Residuals, check_invariants, residuals
from gitm.planner.graph import Graph, predict_graph
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec
from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, Trace

COMPUTE_STREAM = 7
SIDE_STREAM = 13
#: A real NCCL name, so ``is_comm_kernel`` counts it as communication. Its op is
#: not in the dense graph, so ``monitor.residuals`` drops it — as it would a real one.
SIDE_KERNEL_NAME = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"
SIDE_OP = "tp_all_reduce"

# ── truth model ─────────────────────────────────────────────────────────────

#: ``(op, point, bytes) -> efficiency`` in (0, 1].
Efficiency = Callable[[str, BatchConfig, float], float]


def const(value: float) -> Efficiency:
    """The same efficiency at every operating point."""
    return lambda op, point, nbytes: value


def step(lo_kv: float, hi_kv: float, kv_threshold: int) -> Efficiency:
    """``lo_kv`` up to ``kv_threshold`` tokens, ``hi_kv`` above — a kernel variant switch."""
    return lambda op, point, nbytes: hi_kv if point.kv_cache_len > kv_threshold else lo_kv


def saturating(eta_max: float, bytes_half: float) -> Efficiency:
    """``eta_max * B / (B + bytes_half)``: small transfers never reach full bandwidth."""
    return lambda op, point, nbytes: eta_max * nbytes / (nbytes + bytes_half)


def _for_op(value, op: str, default):
    if isinstance(value, dict):
        return value.get(op, default)
    return value


@dataclass(frozen=True)
class TruthModel:
    """``d = c0 + max(F/(P·η_c), B·(1+τ)/(BW·η_m), n_launch·t_launch)``.

    Every field is a scalar or a per-op dict. ``eta_*`` may also be an
    :data:`Efficiency` callable. ``eta_c`` defaults to ``eta_m``. P, BW and the
    launch overhead are the hardware's; F and B are the planner's work counts.
    """

    eta_m: float | Efficiency | dict = 0.8
    eta_c: float | Efficiency | dict | None = None
    c0_ns: float | dict = 0.0
    tau: float | dict = 0.0

    @classmethod
    def matched(cls, eta: float = 0.8) -> TruthModel:
        """Truth equal to the prediction divided by one constant efficiency."""
        return cls(eta_m=eta)

    def _eta(self, which, op: str, point: BatchConfig, nbytes: float) -> float:
        v = _for_op(which, op, 1.0)
        return float(v(op, point, nbytes) if callable(v) else v)

    def duration_ns(self, node, hw: HardwareSpec, point: BatchConfig) -> float:
        p = node.prediction
        op = node.op
        eta_m = self._eta(self.eta_m, op, point, p.bytes)
        eta_c = self._eta(self.eta_c if self.eta_c is not None else self.eta_m, op, point, p.bytes)
        tau = float(_for_op(self.tau, op, 0.0))
        c0 = float(_for_op(self.c0_ns, op, 0.0))
        t_c = p.flops / (p.peak_flops_per_s * eta_c) if p.peak_flops_per_s > 0 else 0.0
        t_m = (p.bytes * (1.0 + tau) / (hw.peak_mem_bw_bytes_per_s * eta_m)
               if hw.peak_mem_bw_bytes_per_s > 0 else 0.0)
        t_l = p.serial_launches * hw.kernel_launch_overhead_s
        return c0 + max(t_c, t_m, t_l) * 1e9


# ── execution ───────────────────────────────────────────────────────────────


@dataclass
class _Inst:
    """One kernel instance in the execution, before scheduling."""

    step: int
    op: str
    layer: int | None
    stream: int
    dur_ns: float
    gap_ns: float
    deps: list[int] = field(default_factory=list)
    #: For a side kernel: the index of its layer's ``mlp_down``, where
    #: :class:`Serialization` moves its dependency.
    serialize_after: int | None = None
    start_ns: int = 0
    end_ns: int = 0


def _selected(inst: _Inst, ops, layers, steps) -> bool:
    return ((ops is None or inst.op in ops)
            and (layers is None or inst.layer in layers)
            and (steps is None or inst.step in steps))


@dataclass(frozen=True)
class RegionSlowdown:
    """M1: ``d ← (1 + α)·d`` on the selected kernels."""

    alpha: float
    ops: frozenset[str] | set[str] | None = None
    layers: frozenset[int] | set[int] | None = None
    steps: frozenset[int] | set[int] | range | None = None

    def apply(self, insts: list[_Inst], scenario: Scenario) -> None:
        for i in insts:
            if _selected(i, self.ops, self.layers, self.steps):
                i.dur_ns *= 1.0 + self.alpha


@dataclass(frozen=True)
class AdditiveCost:
    """M3: a fixed ``δ`` inside each selected kernel, or as idle time before it."""

    delta_ns: float
    where: Literal["kernel", "gap"] = "kernel"
    ops: frozenset[str] | set[str] | None = None
    layers: frozenset[int] | set[int] | None = None
    steps: frozenset[int] | set[int] | range | None = None

    def apply(self, insts: list[_Inst], scenario: Scenario) -> None:
        for i in insts:
            if _selected(i, self.ops, self.layers, self.steps):
                if self.where == "kernel":
                    i.dur_ns += self.delta_ns
                else:
                    i.gap_ns += self.delta_ns


@dataclass(frozen=True)
class Serialization:
    """M2: each selected side kernel waits for its layer's compute to finish."""

    layers: frozenset[int] | set[int] | None = None
    steps: frozenset[int] | set[int] | range | None = None

    def apply(self, insts: list[_Inst], scenario: Scenario) -> None:
        for i in insts:
            if i.serialize_after is not None and _selected(i, None, self.layers, self.steps):
                i.deps = [i.serialize_after]


@dataclass(frozen=True)
class RegimeGate:
    """M4: ``inner`` applies only on steps where ``step_z[s] > threshold``."""

    inner: RegionSlowdown | AdditiveCost | Serialization
    threshold: float

    def gated_steps(self, scenario: Scenario) -> frozenset[int]:
        if scenario.step_z is None:
            raise ValueError("RegimeGate needs Scenario.step_z")
        return frozenset(s for s, z in enumerate(scenario.step_z) if z > self.threshold)

    def apply(self, insts: list[_Inst], scenario: Scenario) -> None:
        gated = self.gated_steps(scenario)
        steps = gated if self.inner.steps is None else gated & frozenset(self.inner.steps)
        dataclasses.replace(self.inner, steps=steps).apply(insts, scenario)


Mechanism = RegionSlowdown | AdditiveCost | Serialization | RegimeGate

# ── observation ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TracerOverhead:
    """M5: the traced run pays ``ε`` per kernel; the untraced system does not."""

    eps_ns: float

    def apply_traced(self, insts: list[_Inst]) -> None:
        for i in insts:
            i.dur_ns += self.eps_ns


@dataclass(frozen=True)
class Misroute:
    """M5: kernels of ``from_op`` are recorded with the identity of ``to_op``."""

    from_op: str
    to_op: str


Distortion = TracerOverhead | Misroute


# ── scenario → fixtures ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario:
    model: ModelSpec = field(default_factory=ModelSpec)
    hw: HardwareSpec = field(default_factory=HardwareSpec)
    points: Sequence[BatchConfig] = (BatchConfig(batch=8, kv_cache_len=2048),)
    truth: TruthModel = field(default_factory=TruthModel.matched)
    mechanisms: Sequence[Mechanism] = ()
    observation: Sequence[Distortion] = ()
    n_steps: int = 4
    launch_gap_ns: int = 1_000
    side_stream: bool = False
    #: Per-step condition a :class:`RegimeGate` reads (kv_len, batch, anything).
    step_z: Sequence[float] | None = None
    #: Multiplicative noise on baseline durations, drawn before any mechanism.
    noise_cv: float = 0.0
    seed: int = 0


@dataclass
class Fixture:
    scenario: Scenario
    point: BatchConfig
    graph: Graph
    trace: Trace
    #: Per-step wall time of the untraced system (the ``off`` arm).
    true_step_ns: list[int]
    #: Per-step wall time the trace shows.
    traced_step_ns: list[int]

    @property
    def true_tpot_s(self) -> float:
        return statistics.fmean(self.true_step_ns) / 1e9

    def write(self, out_dir: str | Path) -> Path:
        """Write the traced arm and the ``off`` arm in the shape a capture leaves."""
        out = Path(out_dir)
        write_trace_jsonl(out / "trace.jsonl", self.trace)
        manifest = {"load": {"input_tokens": self.point.kv_cache_len,
                             "concurrency": self.point.batch}}
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        traced = statistics.fmean(self.traced_step_ns) / 1e9
        (out / "serving_summary.json").write_text(json.dumps(
            {"tracing": "cupti", "server": {"tpot_mean_s": traced}}, indent=2))
        off = out / "off"
        off.mkdir(parents=True, exist_ok=True)
        (off / "serving_summary.json").write_text(json.dumps(
            {"tracing": "off", "server": {"tpot_mean_s": self.true_tpot_s}}, indent=2))
        return out


def _template(scn: Scenario, graph: Graph) -> list[_Inst]:
    """One decode step per ``n_steps``, in emission order, with baseline durations."""
    dur = {(n.op, n.layer): scn.truth.duration_ns(n, scn.hw, graph.batch) for n in graph.nodes}
    rng = np.random.default_rng(scn.seed)
    g = float(scn.launch_gap_ns)
    insts: list[_Inst] = []

    def add(**kw) -> int:
        d = kw.pop("dur_ns")
        if scn.noise_cv > 0:
            d *= max(1e-3, 1.0 + scn.noise_cv * rng.standard_normal())
        insts.append(_Inst(dur_ns=d, gap_ns=g, **kw))
        return len(insts) - 1

    layers = sorted({n.layer for n in graph.nodes if n.layer is not None})
    for s in range(scn.n_steps):
        prev_side: int | None = None
        for layer in layers:
            def compute(op, deps=(), _s=s, _layer=layer):
                return add(step=_s, op=op, layer=_layer, stream=COMPUTE_STREAM,
                           dur_ns=dur[(op, _layer)], deps=list(deps))

            compute("qkv_proj", deps=[prev_side] if prev_side is not None else [])
            compute("attn_score_value")
            out = compute("attn_out_proj")
            gate_up = compute("mlp_gate_up")
            side = None
            if scn.side_stream:
                # Hidden inside gate_up: starts with it (same launch gap after
                # attn_out_proj) and lasts half as long.
                side = add(step=s, op=SIDE_OP, layer=layer, stream=SIDE_STREAM,
                           dur_ns=0.5 * insts[gate_up].dur_ns, deps=[out])
            down = compute("mlp_down")
            if side is not None:
                insts[side].serialize_after = down
            prev_side = side
        add(step=s, op="lm_head", layer=None, stream=COMPUTE_STREAM,
            dur_ns=dur[("lm_head", None)], deps=[prev_side] if prev_side is not None else [])
    return insts


def _schedule(insts: list[_Inst]) -> None:
    """Start when the stream is free and every dependency has ended, plus the launch gap.

    Each stream runs its kernels in emission order. A kernel is placed only once
    everything it depends on is placed, because :class:`Serialization` can point a
    side kernel at a compute kernel emitted after it.
    """
    free: dict[int, int] = {}
    queues: dict[int, list[int]] = {}
    for idx, i in enumerate(insts):
        queues.setdefault(i.stream, []).append(idx)
    done = [False] * len(insts)
    heads = dict.fromkeys(queues, 0)
    remaining = len(insts)
    while remaining:
        progressed = False
        for stream, q in queues.items():
            while heads[stream] < len(q):
                i = insts[q[heads[stream]]]
                if not all(done[d] for d in i.deps):
                    break
                ready = max([free.get(stream, 0), *(insts[d].end_ns for d in i.deps)])
                i.start_ns = ready + int(round(i.gap_ns))
                i.end_ns = i.start_ns + max(1, int(round(i.dur_ns)))
                free[stream] = i.end_ns
                done[q[heads[stream]]] = True
                heads[stream] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("dependency cycle in the fixture execution")


def _step_spans(insts: list[_Inst], n_steps: int) -> list[int]:
    spans = []
    for s in range(n_steps):
        mine = [i for i in insts if i.step == s]
        spans.append(max(i.end_ns for i in mine) - min(i.start_ns for i in mine))
    return spans


def _event(i: _Inst, misroutes: dict[str, str]) -> KernelEvent:
    if i.op == SIDE_OP:
        return KernelEvent(name=SIDE_KERNEL_NAME, start_ns=i.start_ns, end_ns=i.end_ns,
                           stream_id=i.stream, device_id=0)
    op = misroutes.get(i.op, i.op)
    return KernelEvent(name=f"fixture_{op}_kernel", start_ns=i.start_ns, end_ns=i.end_ns,
                       stream_id=i.stream, device_id=0, range_op=op, range_layer=i.layer)


def generate(scenario: Scenario) -> list[Fixture]:
    """One fixture per operating point, each with its own prediction graph."""
    out = []
    for point in scenario.points:
        graph = predict_graph(scenario.model, scenario.hw, point)
        true = _template(scenario, graph)
        for m in scenario.mechanisms:
            m.apply(true, scenario)
        traced = [dataclasses.replace(i, deps=list(i.deps)) for i in true]
        for d in scenario.observation:
            if isinstance(d, TracerOverhead):
                d.apply_traced(traced)
        _schedule(true)
        _schedule(traced)
        misroutes = {d.from_op: d.to_op for d in scenario.observation if isinstance(d, Misroute)}
        events = sorted((_event(i, misroutes) for i in traced), key=lambda e: e.start_ns)
        trace = Trace(
            workload_id="mechanism-fixture", fingerprint="synthetic",
            run_id=f"kv{point.kv_cache_len}-b{point.batch}", device_count=1, vendor="nvidia",
            captured_at_ns=0, duration_ns=max(e.end_ns for e in events), events=events,
        )
        out.append(Fixture(scenario, point, graph, trace,
                           _step_spans(true, scenario.n_steps),
                           _step_spans(traced, scenario.n_steps)))
    return out


# ── observing a fixture through the real monitor ────────────────────────────


@dataclass
class Observation:
    residuals: Residuals
    violations: list
    #: Per-step wall time the trace shows.
    step_ns: list[int]
    exposed_comm_ns: int
    true_tpot_s: float

    def residual_signature(self) -> tuple:
        rows = tuple((k.op, k.layer, k.r_kt, k.r_mt) for k in self.residuals.per_kernel)
        return rows, self.residuals.serialized_concurrency_fraction

    def violation_signature(self) -> tuple:
        return tuple((v.invariant, v.node_op, v.layer, round(v.residual, 9), v.severity)
                     for v in self.violations)

    def op_series(self, op: str) -> list[float]:
        """``r_kt`` of every row of ``op``, in trace order."""
        return [k.r_kt for k in self.residuals.per_kernel if k.op == op]

    def op_duration_s(self, op: str) -> float:
        """Median observed duration of ``op``, read back from the residual rows."""
        return statistics.median(k.t_obs_s for k in self.residuals.per_kernel if k.op == op)


def observe(fx: Fixture) -> Observation:
    res = residuals(fx.trace, fx.graph)
    return Observation(
        residuals=res,
        violations=check_invariants(res),
        step_ns=fx.traced_step_ns,
        exposed_comm_ns=device_comm_stats(fx.trace).exposed_comm_ns,
        true_tpot_s=fx.true_tpot_s,
    )


def same_residuals(a: Observation, b: Observation, atol: float = 1e-6) -> bool:
    """Equal residual signatures, to within the 1 ns timestamp quantization."""
    (ra, sa), (rb, sb) = a.residual_signature(), b.residual_signature()
    if len(ra) != len(rb) or abs(sa - sb) > 1e-12:
        return False
    for (oa, la, ka, ma), (ob, lb, kb, mb) in zip(ra, rb, strict=True):
        if oa != ob or la != lb or ma != mb or abs(ka - kb) > atol:
            return False
    return True


# ── the affine fit behind P1's separating condition ─────────────────────────


@dataclass(frozen=True)
class AffineFit:
    a: float
    b: float
    #: Quadratic term's swing across the range, relative to the mean of ``d``.
    #: ``None`` with fewer than three points: any two points fit a line.
    curvature: float | None


def fit_affine(t: Iterable[float], d: Iterable[float]) -> AffineFit:
    """Least-squares ``d = a·t + b``, plus how far a quadratic departs from it."""
    t_arr = np.asarray(list(t), dtype=float)
    d_arr = np.asarray(list(d), dtype=float)
    (a, b), *_ = np.linalg.lstsq(np.column_stack([t_arr, np.ones_like(t_arr)]), d_arr, rcond=None)
    curvature = None
    if len(np.unique(t_arr)) >= 3:
        # On t rescaled to [0, 1], the quadratic coefficient is directly its swing
        # across the range.
        u = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min())
        (c2, _c1, _c0), *_ = np.linalg.lstsq(
            np.column_stack([u**2, u, np.ones_like(u)]), d_arr, rcond=None)
        curvature = float(abs(c2) / d_arr.mean())
    return AffineFit(a=float(a), b=float(b), curvature=curvature)
