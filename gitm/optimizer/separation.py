"""P1 separating experiment: is a config change multiplicative or additive on one op?

    python -m gitm.optimizer.separation inspect <trace.jsonl>
    python -m gitm.optimizer.separation analyze <run_dir> [--n-layers 61] [--concurrency 16]

The design is in ``docs/experiments/p1_fp8_kv_separation.md``; this module is its
decision rule, fixed before any hardware run. At one operating point a region
slowdown (``d → (1+α)·d``) and an in-kernel additive cost (``d → d + δ``) produce
the same trace (``docs/mechanism_model.md``, P1). Across operating points they do
not: regress the candidate's per-launch time on the baseline's,

    d_cand(x) = k · d_base(x) + m

and a pure slowdown has ``m = 0`` while a pure additive cost has ``k = 1``. Using
the baseline itself as the regressor, rather than the roofline ``t(x)``, means the
rule does not need the planner to be right about how the op scales (A4); it
needs only that the mechanism's parameters do not change with ``x`` (A5).

Everything here is pure Python, NumPy and SciPy. The tests run it on synthetic
traces from :mod:`gitm.optimizer.mechanism_fixtures` and on synthetic sweeps at
the hardware's scale.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from gitm.optimizer.deviation import classify_op
from gitm.optimizer.mechanism_fixtures import fit_affine

# ── pre-registered constants (docs/experiments/p1_fp8_kv_separation.md §4) ──

#: A multiplicative change smaller than this counts as none.
KAPPA = 0.03
#: An additive change smaller than this counts as none: the largest of an
#: absolute floor, a fraction of the baseline launch at the *shortest* point
#: (where a fixed cost is the largest share), and the tracer leak below.
MU_FLOOR_S = 0.25e-6
MU_FRACTION = 0.03
#: Assumed upper bound on how much the tracer inflates one kernel's device-side
#: duration. Under a pure slowdown it leaks into the intercept as
#: ``(1 − k)·ε`` per kernel in the launch, so the margin never goes below twice
#: that. An assumption, stated in the report next to every ``m``.
EPS_MAX_S = 0.25e-6
#: Lack of fit fails only when it is both material (the quadratic term's swing,
#: relative to the mean candidate time) and significant against the noise.
CURVATURE_MAX = 0.05
LACK_OF_FIT_ALPHA = 0.01
#: Rep-to-rep coefficient of variation allowed at any point, per arm.
CV_MAX = 0.05
#: Baseline after vs before the candidate (drift) and the control op across arms:
#: the pooled mean change must stay inside this, and no single point may move by
#: more than this *and* by more than 3 standard errors.
DRIFT_MAX = 0.02
CONTROL_MAX = 0.02
#: Minimum launches of the anchor kernel, at the modal grid, per window.
MIN_LAUNCHES = 1000
#: Running requests, inside the capture window, must stay at or above this
#: fraction of the offered concurrency.
MIN_RUNNING_FRACTION = 0.9
#: Latency gate: ITL p95 over p50 above this means serving was stalling.
ITL_TAIL_MAX = 3.0
#: Monte Carlo draws for the intervals.
DRAWS = 4000
SEED = 0

#: Which kernels are the op under test. Applied to each arm on its own, because
#: the candidate may run different kernels. Resolved names are reported for audit.
TARGET_NEEDLES = ("mla", "attn", "attention", "flash", "paged", "decode")
#: Cache insert, rotary, norms and quantization are separate work, not attention
#: core; fp8 KV changes the insert, so it must not be folded in (P6). Their time
#: is still compared across arms and reported (``kernel_diff``).
TARGET_EXCLUDE = ("reshape_and_cache", "cache", "rope", "rotary", "norm", "slot_mapping",
                  "concat", "quant", "prefill", "varlen", "proj")
#: Control op: depends on batch and hidden size only, so a KV dtype change must
#: not move it. A difference is node or clock drift, not the mechanism.
CONTROL_OP = "rms_norm"

OUTCOMES = ("multiplicative", "additive", "mixed", "no_effect", "inconclusive")

#: What would resolve each kind of abstention (docs §4).
REMEDY = {
    "run": "fix the failed run-level check (see reasons) and rerun the affected phase",
    "points": "rerun the missing operating points; the line needs ≥ 3",
    "reps": "rerun the dropped windows so every point has ≥ 2 reps in both arms",
    "cv": "add reps or lengthen the capture window at the noisy points",
    "drift": "rerun on a quiet node, baseline–candidate–baseline again",
    "control": "check node clocks and power caps, then rerun both arms",
    "fit": "densify the sweep around the bend and analyse each regime separately",
    "straddle": "add reps: interval width falls as 1/√n",
}


# ── per-window statistic ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Kernel:
    name: str
    start_ns: int
    end_ns: int
    stream: int
    device: int
    grid: tuple[int, int, int]


def read_kernels(path: str | Path) -> Iterator[Kernel]:
    """Stream kernel events out of a ``trace.jsonl`` without loading it whole."""
    with open(path, encoding="utf-8") as fh:
        next(fh, None)  # header
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("kind", "kernel") != "kernel":
                continue
            yield Kernel(e["name"], int(e["start_ns"]), int(e["end_ns"]),
                         int(e.get("stream_id", 0)), int(e.get("device_id", 0)),
                         (int(e.get("grid_x", 1)), int(e.get("grid_y", 1)),
                          int(e.get("grid_z", 1))))


def kernels_from_trace(trace) -> list[Kernel]:
    """The same records from an in-memory :class:`~gitm.tracer.schema.Trace`."""
    return [Kernel(e.name, e.start_ns, e.end_ns, e.stream_id, e.device_id,
                   (e.grid_x, e.grid_y, e.grid_z)) for e in trace.kernels()]


def is_target(name: str) -> bool:
    low = name.lower()
    return (any(n in low for n in TARGET_NEEDLES)
            and not any(x in low for x in TARGET_EXCLUDE))


@dataclass
class WindowStat:
    """One capture window, reduced to what the decision rule reads."""

    #: Median per-launch time of the op under test (anchor + the target kernels
    #: that follow it on its stream), at the anchor's modal grid.
    launch_s: float | None
    launches: int
    control_s: float | None
    anchor: str | None
    targets: list[str] = field(default_factory=list)
    grid: tuple[int, int, int] | None = None
    #: Mean number of kernels in a kept launch (the tracer charges each one).
    kernels_per_launch: float = 1.0
    #: Every kernel name's total time divided by the number of anchor launches.
    per_launch_s: dict[str, float] = field(default_factory=dict)


def window_stat(kernels: Iterable[Kernel]) -> WindowStat:
    """Per-launch op time for one window.

    The anchor is the target kernel with the most total time: stage-1 of a
    split-KV decode, not its reduce, whose launch count ties it. A launch is the
    anchor plus the target kernels that immediately follow it on the same stream
    and device. Only launches at the anchor's most common grid are kept: on a
    serving run the grid tracks the decode batch, so this drops launches made
    while some requests were still prefilling.
    """
    ks = sorted(kernels, key=lambda k: (k.device, k.stream, k.start_ns))
    total: Counter[str] = Counter()
    for k in ks:
        total[k.name] += k.end_ns - k.start_ns
    control = [k.end_ns - k.start_ns for k in ks if classify_op(k.name) == CONTROL_OP]
    control_s = statistics.median(control) / 1e9 if control else None
    targets = {n: t for n, t in total.items() if is_target(n)}
    if not targets:
        return WindowStat(None, 0, control_s, None)
    anchor = max(targets, key=lambda n: (targets[n], n))

    launches: list[list] = []  # [grid, duration_ns, kernels]
    cur: list | None = None
    key = None
    for k in ks:
        if (k.device, k.stream) != key:
            cur, key = None, (k.device, k.stream)
        if k.name == anchor:
            cur = [k.grid, k.end_ns - k.start_ns, 1]
            launches.append(cur)
        elif cur is not None and is_target(k.name):
            cur[1] += k.end_ns - k.start_ns
            cur[2] += 1
        else:
            cur = None
    grid = Counter(g for g, _, _ in launches).most_common(1)[0][0]
    kept = [(d, n) for g, d, n in launches if g == grid]
    return WindowStat(
        statistics.median(d for d, _ in kept) / 1e9, len(kept), control_s, anchor,
        sorted(targets), grid, statistics.fmean(n for _, n in kept),
        {n: t / 1e9 / len(launches) for n, t in total.items()},
    )


# ── the decision ────────────────────────────────────────────────────────────

#: point label → per-rep per-launch times (seconds). The label carries the
#: operating point; the rule only needs points to be matched across arms.
Sweep = Mapping[str, Sequence[float]]


@dataclass
class Decision:
    outcome: str
    k: float | None = None
    m_s: float | None = None
    k_ci: tuple[float, float] | None = None
    m_ci_s: tuple[float, float] | None = None
    mu_s: float | None = None
    #: The most an ε-per-kernel tracer inflation could add to ``m`` under a pure
    #: slowdown, at the assumed ``EPS_MAX_S``: ``|n_c − k·n_b|·ε``.
    tracer_leak_s: float | None = None
    curvature: float | None = None
    lack_of_fit_p: float | None = None
    cv: dict[str, float] = field(default_factory=dict)
    points: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    remedies: list[str] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    (k, m), *_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)
    return float(k), float(m)


def _wfit(xs: np.ndarray, ys: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares slope and intercept, one per row of ``xs``/``ys``."""
    sw = w.sum()
    xm, ym = (xs * w).sum(axis=1) / sw, (ys * w).sum(axis=1) / sw
    xc = xs - xm[:, None]
    k = (w * xc * (ys - ym[:, None])).sum(axis=1) / (w * xc**2).sum(axis=1)
    return k, ym - k * xm


def _pooled_cv(s: Sweep, pts: Sequence[str]) -> tuple[float, int]:
    """Coefficient of variation pooled over points, and its degrees of freedom.

    Noise is taken as proportional to the mean (a per-launch median over a
    window scales with the launch), so every point contributes to one estimate.
    """
    ss = sum(sum((v / statistics.fmean(s[p]) - 1) ** 2 for v in s[p]) for p in pts)
    df = sum(len(s[p]) - 1 for p in pts)
    return (ss / df) ** 0.5 if df else float("inf"), df


def _noise(s: Sweep, pts: Sequence[str]) -> tuple[np.ndarray, int]:
    """Per-point standard deviation from ``sd² = a² + (c·d)²`` fitted over points.

    An absolute floor ``a`` (timer resolution, launch jitter) plus noise
    proportional to time ``c``, fitted by non-negative least squares on the
    per-point variances relative to the mean. Pure proportional noise gives
    ``a ≈ 0``; a floor keeps the short points from being over-trusted.
    Degrees of freedom: the pooled rep dof less the two fitted terms.
    """
    d = np.array([statistics.fmean(s[p]) for p in pts])
    var = np.array([statistics.variance(s[p]) for p in pts])
    (a2, c2), _ = optimize.nnls(np.column_stack([1 / d**2, np.ones_like(d)]), var / d**2)
    df = sum(len(s[p]) - 1 for p in pts) - 2
    return np.sqrt(a2 + c2 * d**2), max(df, 1)


def _outside(ci: tuple[float, float], lo: float, hi: float) -> bool:
    return ci[1] < lo or ci[0] > hi


def _inside(ci: tuple[float, float], lo: float, hi: float) -> bool:
    return lo <= ci[0] and ci[1] <= hi


def _ratio_gate(d: Decision, name: str, ratios: Mapping[str, float],
                se: Mapping[str, float], limit: float, pts: Sequence[str]) -> None:
    """Pooled change within ``limit``; no point beyond ``limit`` and 3 SE."""
    have = [p for p in pts if p in ratios]
    pooled = statistics.fmean(ratios[p] - 1 for p in have) if have else 0.0
    bad = {p: ratios[p] - 1 for p in have
           if abs(ratios[p] - 1) > limit and abs(ratios[p] - 1) > 3 * se.get(p, 0.0)}
    ok = len(have) >= 3 and abs(pooled) <= limit and not bad
    d.gates[name] = ok
    if not ok:
        parts = [f"pooled {pooled:+.1%}"] + [f"{p} {v:+.1%}" for p, v in bad.items()]
        if len(have) < 3:
            parts.append(f"measured at only {len(have)} point(s)")
        d.reasons.append(f"{name} beyond ±{limit:.0%}: " + ", ".join(parts))


def decide(base: Sweep, cand: Sweep, *, base_repeat: Sweep | None = None,
           control_ratio: Mapping[str, float] | None = None,
           control_se: Mapping[str, float] | None = None,
           gate_failures: Sequence[str] = (), kernels_per_launch: tuple[float, float] = (1, 1),
           draws: int = DRAWS, seed: int = SEED) -> Decision:
    """Classify the candidate's effect on the op, or abstain.

    ``base_repeat`` is the baseline run again after the candidate (drift check);
    ``control_ratio`` is candidate ÷ baseline for the control op at each point,
    with ``control_se`` its relative standard error; ``kernels_per_launch`` is
    ``(baseline, candidate)``;
    ``gate_failures`` are run-level checks already failed upstream.
    """
    d = Decision("inconclusive", reasons=list(gate_failures))
    d.gates["run"] = not gate_failures
    pts = sorted((p for p in set(base) & set(cand) if base[p] and cand[p]),
                 key=lambda p: statistics.fmean(base[p]))
    d.points = pts

    d.gates["points"] = len(pts) >= 3
    if not d.gates["points"]:
        d.reasons.append(f"{len(pts)} matched operating points; need ≥ 3 to test the line")
    d.gates["reps"] = all(len(s[p]) >= 2 for s in (base, cand) for p in pts)
    if not d.gates["reps"]:
        d.reasons.append("fewer than 2 reps at some point; no noise estimate there")
    if not (d.gates["points"] and d.gates["reps"]):
        return _finish(d)

    cv_b, df_b = _pooled_cv(base, pts)
    cv_c, df_c = _pooled_cv(cand, pts)
    d.cv = {"base": cv_b, "cand": cv_c}
    cv_r = _pooled_cv(base_repeat, [p for p in pts if len(base_repeat.get(p) or ()) >= 2])[0] \
        if base_repeat else cv_b
    cv_bad = [f"{name}@{p}" for name, s in (("base", base), ("cand", cand)) for p in pts
              if statistics.stdev(s[p]) / statistics.fmean(s[p]) > CV_MAX]
    d.gates["cv"] = not cv_bad
    if cv_bad:
        d.reasons.append(f"rep-to-rep CV above {CV_MAX:.0%} at {', '.join(cv_bad)}")

    cv_r = cv_r if np.isfinite(cv_r) else cv_b
    se_rel = {p: (cv_b**2 / len(base[p]) + cv_r**2 / len(base_repeat[p])) ** 0.5
              for p in pts if base_repeat and base_repeat.get(p)}
    if base_repeat is not None:
        _ratio_gate(d, "drift", {p: statistics.fmean(base_repeat[p]) / statistics.fmean(base[p])
                                 for p in pts if base_repeat.get(p)}, se_rel, DRIFT_MAX, pts)
    else:
        d.gates["drift"] = False
        d.reasons.append("no repeated baseline; drift is unchecked")
    if control_ratio is not None:
        _ratio_gate(d, "control", control_ratio, control_se or {}, CONTROL_MAX, pts)
    else:
        d.gates["control"] = False
        d.reasons.append("no control-op measurement")

    x = np.array([statistics.fmean(base[p]) for p in pts])
    y = np.array([statistics.fmean(cand[p]) for p in pts])
    nb = np.array([len(base[p]) for p in pts])
    nc = np.array([len(cand[p]) for p in pts])
    sd_x, df_b = _noise(base, pts)
    sd_y, df_c = _noise(cand, pts)
    se_x, se_y = sd_x / np.sqrt(nb), sd_y / np.sqrt(nc)
    # Weighted least squares: each point weighted by its inverse variance
    # (errors in both arms, through k).
    k0, _ = _fit(x, y)
    w = 1 / (se_y**2 + (k0 * se_x) ** 2)
    d.k, d.m_s = (float(v[0]) for v in _wfit(x[None], y[None], w))

    # Intervals: redraw every point mean from a t distribution around it, with
    # the pooled noise and its degrees of freedom, and refit.
    rng = np.random.default_rng(seed)
    tx = rng.standard_t(df_b, size=(draws, len(pts)))
    ty = rng.standard_t(df_c, size=(draws, len(pts)))
    ks, ms = _wfit(x + se_x * tx, y + se_y * ty, w)
    d.k_ci = (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)))
    d.m_ci_s = (float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))
    # Base reads d + n_b·ε, candidate k·d + n_c·ε, so m picks up (n_c − k·n_b)·ε.
    n_b, n_c = kernels_per_launch
    d.tracer_leak_s = abs(n_c - d.k * n_b) * EPS_MAX_S
    d.mu_s = max(MU_FLOOR_S, MU_FRACTION * float(np.min(x)), 2 * d.tracer_leak_s)

    # Lack of fit: material (curvature) and significant (F against the noise).
    d.curvature = fit_affine(x, y).curvature
    resid = y - (d.k * x + d.m_s)
    f_stat = float(np.sum(w * resid**2)) / (len(pts) - 2)
    d.lack_of_fit_p = float(stats.f.sf(f_stat, len(pts) - 2, min(df_b, df_c)))
    d.gates["fit"] = not (d.curvature > CURVATURE_MAX and d.lack_of_fit_p < LACK_OF_FIT_ALPHA)
    if not d.gates["fit"]:
        d.reasons.append(f"lack of fit: curvature {d.curvature:.3f} > {CURVATURE_MAX} "
                         f"(p = {d.lack_of_fit_p:.1e}); the effect is not one (k, m) "
                         "across points (A5 broken)")

    if not all(d.gates.values()):
        return _finish(d)

    k_lo, k_hi, mu = 1 - KAPPA, 1 + KAPPA, d.mu_s
    k_eff, k_none = _outside(d.k_ci, k_lo, k_hi), _inside(d.k_ci, k_lo, k_hi)
    m_eff, m_none = _outside(d.m_ci_s, -mu, mu), _inside(d.m_ci_s, -mu, mu)
    if k_eff and m_none:
        d.outcome = "multiplicative"
    elif k_none and m_eff:
        d.outcome = "additive"
    elif k_eff and m_eff:
        d.outcome = "mixed"
    elif k_none and m_none:
        d.outcome = "no_effect"
    else:
        d.gates["straddle"] = False
        d.reasons.append("an interval straddles its margin: the data cannot place the "
                         f"effect (k CI {d.k_ci[0]:.3f}–{d.k_ci[1]:.3f} vs ±{KAPPA}; "
                         f"m CI {d.m_ci_s[0] * 1e6:.2f}–{d.m_ci_s[1] * 1e6:.2f} µs "
                         f"vs ±{mu * 1e6:.2f} µs)")
    return _finish(d)


def _finish(d: Decision) -> Decision:
    d.remedies = [REMEDY[g] for g, ok in d.gates.items() if not ok and g in REMEDY]
    return d


# ── reading a run directory (layout written by scripts/kimi_loop/p1_separation.sh) ──

_POINT = re.compile(r"^L(\d+)_r(\d+)$")


def _prom_series(path: Path, metric: str) -> list[float]:
    vals = []
    for line in path.read_text().splitlines():
        if line.startswith(f"vllm:{metric}"):
            try:
                vals.append(float(line.rsplit(" ", 1)[1]))
            except (IndexError, ValueError):
                pass
    return vals


def _window_running(d: Path) -> list[float]:
    """Running requests sampled inside the capture window, else over the point."""
    samples = sorted(d.rglob("metrics_samples.jsonl"))
    if samples:
        vals = []
        for line in samples[-1].read_text().splitlines():
            v = json.loads(line).get("running") if line.strip() else None
            if isinstance(v, int | float):
                vals.append(float(v))
        if vals:
            return vals
    prom = d / "metrics.prom"
    return _prom_series(prom, "num_requests_running") if prom.exists() else []


@dataclass
class _Phase:
    launch: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    control: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    names: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    anchors: set[str] = field(default_factory=set)
    grids: dict[str, set[tuple]] = field(default_factory=lambda: defaultdict(set))
    per_launch: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: defaultdict(list))
    kernels_per_launch: float = 1.0


def _phase(run: Path, arm: str, phase: str, concurrency: int,
           dropped: list[str]) -> _Phase:
    """Reduce every window of one phase; a window that fails a check is dropped."""
    out = _Phase()
    root = run / "p1" / arm / phase
    for d in sorted(root.glob("L*_r*")):
        m = _POINT.match(d.name)
        if not m:
            continue
        label, where = f"L{m.group(1)}", f"{phase}/{d.name}"
        traces = sorted(d.rglob("trace.jsonl")) or sorted(d.rglob("*.jsonl"))
        traces = [t for t in traces if t.name != "metrics_samples.jsonl"]
        if not traces:
            dropped.append(f"{where}: no trace")
            continue
        w = window_stat(read_kernels(traces[-1]))
        if w.launch_s is None or w.launches < MIN_LAUNCHES:
            dropped.append(f"{where}: {w.launches} anchor launches < {MIN_LAUNCHES}")
            continue
        problems = []
        g = d / "guidellm.json"
        if not g.exists():
            problems.append("no guidellm.json")
        else:
            p50, p95 = _itl_ms(g), _itl_ms(g, "p95")
            if not (p50 and p95):
                problems.append("ITL p50/p95 not found in guidellm.json")
            elif p95 / p50 > ITL_TAIL_MAX:
                problems.append(f"ITL p95/p50 {p95 / p50:.1f} > {ITL_TAIL_MAX}")
        running = _window_running(d)
        if not running:
            problems.append("no load metrics")
        elif statistics.median(running) < MIN_RUNNING_FRACTION * concurrency:
            problems.append(f"median running {statistics.median(running):.0f} < "
                            f"{MIN_RUNNING_FRACTION:.0%} of c={concurrency}")
        prom = d / "metrics.prom"
        if prom.exists():
            pre = _prom_series(prom, "num_preemptions_total") or _prom_series(
                prom, "num_preemptions")
            if len(pre) >= 2 and pre[-1] > pre[0]:
                problems.append(f"{pre[-1] - pre[0]:.0f} preemptions")
        if problems:
            dropped.append(f"{where}: " + "; ".join(problems))
            continue
        out.launch[label].append(w.launch_s)
        if w.control_s is not None:
            out.control[label].append(w.control_s)
        out.names[label].update(w.targets)
        out.anchors.add(w.anchor)  # type: ignore[arg-type]
        out.grids[label].add(w.grid)
        out.per_launch[label].append(w.per_launch_s)
        out.kernels_per_launch = max(out.kernels_per_launch, w.kernels_per_launch)
    return out


def _kernel_diff(base: _Phase, cand: _Phase, top: int = 10) -> dict:
    """Per-name time per attention launch, candidate minus baseline, at each point.

    The op's own kernels are excluded from the verdict by design (cache insert,
    quantization); this is where an additive cost *outside* them shows up.
    """
    out = {}
    for p in sorted(set(base.per_launch) & set(cand.per_launch)):
        def mean_of(rows):
            names = set().union(*rows)
            return {n: statistics.fmean(r.get(n, 0.0) for r in rows) for n in names}

        b, c = mean_of(base.per_launch[p]), mean_of(cand.per_launch[p])
        delta = {n: c.get(n, 0.0) - b.get(n, 0.0) for n in set(b) | set(c)}
        ranked = sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:top]
        out[p] = [{"kernel": n, "delta_us": v * 1e6, "only_in": "cand" if n not in b
                   else "base" if n not in c else None} for n, v in ranked]
    return out


def analyze(run: Path, *, n_layers: int, concurrency: int) -> dict:
    failures: list[str] = []
    dropped: list[str] = []
    for phase in ("base1", "cand", "base2"):
        f = run / "p1" / "traced" / phase / "sanity.json"
        if not f.exists():
            failures.append(f"{phase}: no correctness check (sanity.json)")
        elif json.loads(f.read_text())["failures"]:
            failures.append(f"{phase}: correctness check failed")
        failed = run / "p1" / "traced" / phase / "FAILED"
        if failed.exists():
            dropped += [f"runner: {line}" for line in failed.read_text().splitlines() if line]
    base = _phase(run, "traced", "base1", concurrency, dropped)
    cand = _phase(run, "traced", "cand", concurrency, dropped)
    rep = _phase(run, "traced", "base2", concurrency, dropped)

    for arm, ph in (("base", base), ("cand", cand)):
        if len({frozenset(v) for v in ph.names.values()}) > 1:
            failures.append(f"{arm}: target kernel set differs across points "
                            "(a variant switch; see targets in the report)")
        if len(ph.anchors) > 1:
            failures.append(f"{arm}: anchor kernel differs across windows "
                            f"({', '.join(sorted(ph.anchors))})")
    if base.anchors | rep.anchors and len(base.anchors | rep.anchors) > 1:
        failures.append("baseline anchor changed between base1 and base2")

    warnings = [f"{p}: modal grid differs between arms ({sorted(base.grids[p])} vs "
                f"{sorted(cand.grids[p])}); check the decode batch matched"
                for p in sorted(set(base.grids) & set(cand.grids))
                if base.grids[p] != cand.grids[p]]

    cpts = sorted(p for p in set(base.control) & set(cand.control)
                  if len(base.control[p]) >= 2 and len(cand.control[p]) >= 2)
    ctl = {p: statistics.fmean(cand.control[p]) / statistics.fmean(base.control[p])
           for p in cpts}
    cvb, cvc = _pooled_cv(base.control, cpts)[0], _pooled_cv(cand.control, cpts)[0]
    ctl_se = {p: (cvb**2 / len(base.control[p]) + cvc**2 / len(cand.control[p])) ** 0.5
              for p in cpts}
    dec = decide(base.launch, cand.launch, base_repeat=rep.launch or None,
                 control_ratio=ctl or None, control_se=ctl_se, gate_failures=failures,
                 kernels_per_launch=(base.kernels_per_launch, cand.kernels_per_launch))

    return {
        "decision": asdict(dec),
        "dropped_windows": dropped,
        "warnings": warnings,
        "anchors": {"base": sorted(base.anchors), "cand": sorted(cand.anchors)},
        "targets": {"base": sorted(set().union(*base.names.values())) if base.names else [],
                    "cand": sorted(set().union(*cand.names.values())) if cand.names else []},
        "grids": {"base": {p: sorted(g) for p, g in base.grids.items()},
                  "cand": {p: sorted(g) for p, g in cand.grids.items()}},
        "per_point_us": {p: {"base": statistics.fmean(base.launch[p]) * 1e6,
                             "cand": statistics.fmean(cand.launch[p]) * 1e6}
                         for p in dec.points},
        "kernel_diff": _kernel_diff(base, cand),
        "off_arm_corroboration": _corroborate(run, dec, base.launch, n_layers),
        "constants": {k: v for k, v in globals().items() if k.isupper()
                      and isinstance(v, int | float | str | tuple) and not isinstance(v, bool)},
    }


def _corroborate(run: Path, dec: Decision, base: Sweep, n_layers: int) -> dict | None:
    """Untraced check, reported not gating: the multiplicative part must show in
    the off arms' ITL as a slope of ``n_layers·(k − 1)`` against ``d_base``."""
    off: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for phase in ("base1", "cand"):
        for d in sorted((run / "p1" / "off" / phase).glob("L*_r*")):
            g, m = d / "guidellm.json", _POINT.match(d.name)
            if m and g.exists() and (itl := _itl_ms(g)) is not None:
                off[phase][f"L{m.group(1)}"].append(itl)
    if dec.k is None:
        return None
    pts = [p for p in dec.points if off["base1"].get(p) and off["cand"].get(p)]
    if len(pts) < 3:
        return {"note": f"{len(pts)} points with both off arms; need ≥ 3"}
    xb = np.array([statistics.fmean(base[p]) for p in pts])
    dy = np.array([(statistics.fmean(off["cand"][p]) - statistics.fmean(off["base1"][p])) / 1e3
                   for p in pts])
    res = stats.linregress(xb, dy)
    return {"measured_slope": float(res.slope), "slope_se": float(res.stderr),
            "predicted_slope": n_layers * (dec.k - 1), "intercept_ms": float(res.intercept) * 1e3,
            "points": pts}


def _itl_ms(path: Path, stat: str = "median") -> float | None:
    """ITL out of a GuideLLM result; ``stat`` is ``median`` or ``p95``."""
    data = json.loads(path.read_text())
    b = (data.get("benchmarks") or [data])[0]
    for key in ("inter_token_latency_ms", "itl_ms", "inter_token_latency"):
        cur = b.get("metrics", {}).get(key, {}).get("successful", {})
        v = cur.get("median") if stat == "median" else (cur.get("percentiles") or {}).get("p95")
        if isinstance(v, int | float):
            return float(v)
    return None


# ── correctness gate: run after every arm switch, against the live server ───

SANITY_PROMPTS = 32
SANITY_MIN_TOKENS = 16


def sanity(base_url: str, out: Path, model: str) -> list[str]:
    """Greedy completions on fixed prompts; returns the gate failures.

    This is a validity gate, not a quality claim: the arm must answer every
    prompt, at length, without collapsing into one repeated token. Outputs are
    saved so baseline and candidate can be compared afterwards.
    """
    import urllib.request

    failures, rows = [], []
    for i in range(SANITY_PROMPTS):
        prompt = f"Q{i}: What is {i + 3} times {17 + i}? Show the steps, then the answer.\nA:"
        body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 64,
                           "temperature": 0, "logprobs": 1}).encode()
        req = urllib.request.Request(f"{base_url}/v1/completions", body,
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - any failure fails the gate
            failures.append(f"prompt {i}: {e}")
            continue
        ch = resp["choices"][0]
        toks = (ch.get("logprobs") or {}).get("tokens") or []
        n = resp.get("usage", {}).get("completion_tokens", len(toks))
        rows.append({"i": i, "text": ch.get("text", ""), "tokens": n})
        if n < SANITY_MIN_TOKENS:
            failures.append(f"prompt {i}: {n} tokens < {SANITY_MIN_TOKENS}")
        elif toks and len(set(toks)) <= 2:
            failures.append(f"prompt {i}: degenerate output ({len(set(toks))} distinct tokens)")
    out.write_text(json.dumps({"failures": failures, "outputs": rows}, indent=2))
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gitm.optimizer.separation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("inspect", help="Show which kernels the rule resolves in one trace.")
    ins.add_argument("trace")
    sa = sub.add_parser("sanity", help="Correctness gate against a live server.")
    sa.add_argument("--base-url", default="http://localhost:8000")
    sa.add_argument("--model", default="moonshotai/Kimi-K2.5")
    sa.add_argument("--out", required=True)
    an = sub.add_parser("analyze", help="Run the pre-registered decision on a run directory.")
    an.add_argument("run")
    an.add_argument("--n-layers", type=int, default=61)
    an.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args(argv)

    if args.cmd == "sanity":
        fails = sanity(args.base_url, Path(args.out), args.model)
        print("sanity: PASS" if not fails else "sanity: FAIL\n  " + "\n  ".join(fails))
        return 1 if fails else 0

    if args.cmd == "inspect":
        w = window_stat(read_kernels(args.trace))
        print(json.dumps({"anchor": w.anchor, "targets": w.targets, "launches": w.launches,
                          "grid": w.grid, "kernels_per_launch": w.kernels_per_launch,
                          "launch_us": None if w.launch_s is None else w.launch_s * 1e6,
                          "control_us": None if w.control_s is None else w.control_s * 1e6},
                         indent=2))
        return 0

    report = analyze(Path(args.run), n_layers=args.n_layers, concurrency=args.concurrency)
    out = Path(args.run) / "p1" / "decision.json"
    out.write_text(json.dumps(report, indent=2, default=list))
    dec = report["decision"]
    print(f"outcome: {dec['outcome']}")
    if dec["k_ci"] is not None:
        print(f"k = {dec['k']:.4f}  CI {dec['k_ci']}   m = {dec['m_s'] * 1e6:.3f} µs  "
              f"CI {[round(v * 1e6, 3) for v in dec['m_ci_s']]}  margin ±{dec['mu_s'] * 1e6:.2f} µs"
              f"  (tracer leak ≤ {dec['tracer_leak_s'] * 1e6:.2f} µs at ε_max)")
    for r in dec["reasons"]:
        print(f"  - {r}")
    for r in dec["remedies"]:
        print(f"  → {r}")
    if report["dropped_windows"]:
        print(f"  dropped {len(report['dropped_windows'])} window(s); see decision.json")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
