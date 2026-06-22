"""The 'act' half for edge (3D LiDAR detection) — an output-gated, rollback-gated
optimization, mirroring gitm.benchmarks.hft.optimize.

The baseline runs PointPillars/CenterPoint inference in fp32. The candidate runs
the same model under fp16 autocast: the conv backbone + BEV head do far less
memory traffic, the realizable gain on a memory-bound perception net. We prove
it the GITM way:

    measure fp32 baseline → apply fp16 candidate → verify detections EQUIVALENT
    → compare speed → keep the candidate only if it is BOTH equivalent and
    faster, else roll back.

Unlike HFT (integer/float reductions → byte-identical), NN inference across a
precision change is never bit-exact, so "equivalent" is a tolerance gate: same
detection count and the same sorted confidence scores within ``score_atol``. If
fp16 drops/adds a detection or shifts scores beyond tolerance, the correctness
gate keeps fp32 — no speedup is ever reported on top of degraded detections.

The A/B is injectable (``run_mode``) so the wiring/gate is testable on a laptop
with a fake; the real run_mode (built in gitm.workloads) drives the OpenPCDet
WorkUnit on the GPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate

# A run_mode runs N frames in a given precision and returns a summary dict:
#   {"n_frames": int, "n_detections": int, "scores": sorted-desc list[float]}
RunMode = Callable[[str], dict]


def detections_equivalent(a: dict, b: dict, *, score_atol: float = 0.02) -> bool:
    """True iff two detection summaries match within tolerance.

    Gate: identical detection count, the score list length matches that count
    (catches a malformed summary), and each confidence score agrees within
    ``score_atol``. Scores are sorted here defensively, so callers need not
    pre-sort. This is the perception analogue of HFT's byte-identical signature —
    strict enough that a real regression (dropped object, shifted confidence)
    trips it, loose enough to tolerate fp16 rounding.
    """
    na, nb = int(a.get("n_detections", -1)), int(b.get("n_detections", -2))
    if na != nb:
        return False
    sa = sorted((float(x) for x in a.get("scores", [])), reverse=True)
    sb = sorted((float(x) for x in b.get("scores", [])), reverse=True)
    # A well-formed summary carries one score per detection; mismatch = malformed.
    if len(sa) != na or len(sb) != nb:
        return False
    # lengths are equal here (both == na); strict=True surfaces any future drift.
    return all(abs(x - y) <= score_atol for x, y in zip(sa, sb, strict=True))


@dataclass
class EdgeABResult:
    baseline_eps: float       # frames/sec, fp32
    candidate_eps: float      # frames/sec, fp16
    speedup: float            # candidate / baseline (e.g. 1.6 = 60% faster)
    identical: bool           # detections equivalent within tolerance
    kept: str                 # "candidate" | "baseline"
    baseline_summary: dict = field(default_factory=dict)
    candidate_summary: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if not self.identical:
            return "rolled back — candidate detections differ from baseline beyond tolerance"
        if self.kept == "candidate":
            return f"kept candidate — verified +{(self.speedup - 1) * 100:.1f}% faster, detections equivalent"
        return "kept baseline — candidate not faster"


def optimize_edge(
    run_mode: RunMode,
    *,
    baseline_mode: str = "fp32",
    candidate_mode: str = "fp16",
    reps: int = 2,
    sync: Callable[[], None] | None = None,
    score_atol: float = 0.02,
) -> EdgeABResult:
    """Run the measure→apply→prove A/B and return a gated verdict.

    ``run_mode(mode)`` runs the frames in the given mode and returns a detection
    summary — a single callable dispatched by mode (e.g. ``"fp32"``/``"fp16"`` for
    the precision lever, ``"serial"``/``"batched"`` for the batching lever). Each
    mode is run ``reps`` times; the best (lowest) wall time is used to reduce
    launch jitter. ``sync`` is invoked after each run so GPU timing is honest
    (pass a device sync; default no-op for CPU/fake).
    """
    sync = sync or (lambda: None)

    def _timed(mode: str) -> tuple[dict, float]:
        best = float("inf")
        summary: dict = {}
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            summary = run_mode(mode)
            sync()
            best = min(best, time.perf_counter() - t0)
        n = max(int(summary.get("n_frames", 0)), 0)
        return summary, n / max(best, 1e-9)

    base_summary, base_eps = _timed(baseline_mode)
    cand_summary, cand_eps = _timed(candidate_mode)

    identical = detections_equivalent(base_summary, cand_summary, score_atol=score_atol)
    speedup = cand_eps / base_eps if base_eps else 0.0
    kept = "candidate" if (identical and cand_eps > base_eps) else "baseline"

    return EdgeABResult(
        baseline_eps=base_eps,
        candidate_eps=cand_eps,
        speedup=speedup,
        identical=identical,
        kept=kept,
        baseline_summary=base_summary,
        candidate_summary=cand_summary,
    )


# --- the intervention, wired for the autonomous loop -------------------------


class DetectionDivergenceError(RuntimeError):
    """fp16 detections diverged from fp32 beyond tolerance.

    Raised inside :meth:`EdgeFp16Applicator.measure` so the apply gate rolls
    back: a speedup is *never* kept on top of degraded detections.
    """


def edge_intervention_spec() -> InterventionSpec:
    """The curated edge lever: fp16 autocast inference for 3D detection.

    Detection-equivalence is enforced at apply time (inside the A/B), so the
    expected-delta range here is only used for *ranking* — the real number comes
    from the rollback-gated measure.
    """
    return InterventionSpec(
        name="edge_fp16_autocast",
        summary="Run PointPillars/CenterPoint inference under fp16 autocast "
        "instead of fp32 — halves memory traffic on the conv backbone + BEV "
        "head. Kept only if detections stay equivalent (count + sorted scores "
        "within tolerance) and it is faster.",
        knob="edge.inference_dtype",
        value="fp16",
        applies_to_kernels=[
            "conv", "bev", "backbone", "pillar", "voxel", "scatter", "nms", "gemm",
        ],
        expected_delta_mean=0.40,
        expected_delta_lo=0.0,
        expected_delta_hi=1.00,
        source="gitm/benchmarks/edge/optimize.py — fp16 autocast A/B, "
        "detection-equivalence gated against the fp32 baseline.",
        applicability=Applicability(workloads=["edge", "kitti", "nuscenes"]),
        safety=SafetyGate(
            tier="moderate",
            requires_rollback_window_s=0,
            forbid_if_oom_history=False,
            notes="Precision change; detection-equivalence is gated before any "
            "speedup is kept, so a degraded run rolls back to fp32.",
        ),
        review=None,
    )


class EdgeFp16Applicator:
    """Apply fp16 autocast inference through the standard rollback gate.

    The 'live state' is the active inference precision. :meth:`measure` runs the
    real fp32-vs-fp16 A/B (:func:`optimize_edge`): it raises
    :class:`DetectionDivergenceError` when the candidate's detections are not
    equivalent (forcing a rollback), otherwise returns the signed speedup delta
    so the gate keeps fp16 only when it is genuinely faster. The full
    :class:`EdgeABResult` is stashed on :attr:`last_result` for the report.

    ``run_mode(mode)`` runs the workload's frames in ``"fp32"`` or ``"fp16"`` and
    returns a summary dict — injected so this is testable without a GPU.

    Implements the :class:`gitm.optimizer.apply.Applicator` protocol structurally,
    and carries its own :attr:`spec` so the generalized loop can read it.
    """

    def __init__(
        self,
        run_mode: RunMode,
        *,
        reps: int = 2,
        sync: Callable[[], None] | None = None,
        score_atol: float = 0.02,
        spec: InterventionSpec | None = None,
    ):
        self._run_mode = run_mode
        self._reps = reps
        self._sync = sync
        self._score_atol = score_atol
        self.active = "fp32"
        self.last_result: EdgeABResult | None = None
        self.spec = spec or edge_intervention_spec()

    def snapshot(self) -> str:
        return self.active

    def apply(self, spec: InterventionSpec) -> None:
        self.active = "fp16"

    def restore(self, snapshot: str) -> None:
        self.active = snapshot

    def measure(self, spec: InterventionSpec) -> float:
        r = optimize_edge(
            self._run_mode,
            baseline_mode="fp32",
            candidate_mode="fp16",
            reps=self._reps,
            sync=self._sync,
            score_atol=self._score_atol,
        )
        self.last_result = r
        if not r.identical:
            raise DetectionDivergenceError(
                "fp16 detections differ from fp32 beyond tolerance — rolling back"
            )
        return r.speedup - 1.0


def edge_batching_spec(batch_size: int = 4) -> InterventionSpec:
    """The curated edge lever for a launch-bound profile: frame batching.

    The KITTI/nuScenes trace shows serialized-concurrency ~1.0 over ~11k tiny
    kernels — the run is launch-bound, so precision (fp16) doesn't help; cutting
    *launches* does. Batching B frames into one forward pass amortizes per-kernel
    launch overhead. Per-frame detections are equivalent in eval mode, gated at
    apply time, so the expected-delta range here is only for ranking.

    ``batch_size`` is recorded as the spec's ``value`` so the provenance reflects
    the batch size actually run, not a hardcoded constant.
    """
    return InterventionSpec(
        name="edge_frame_batching",
        summary=f"Run inference on batches of {batch_size} frames in one forward "
        "pass instead of one frame at a time — amortizes per-launch overhead on a "
        "launch-bound workload (serialized-concurrency ~1.0). Kept only if per-frame "
        "detections stay equivalent and throughput improves.",
        knob="edge.batch_size",
        value=batch_size,
        applies_to_kernels=[
            "conv", "bev", "backbone", "pillar", "voxel", "scatter", "nms",
            "gemm", "elementwise",
        ],
        expected_delta_mean=0.30,
        expected_delta_lo=0.0,
        expected_delta_hi=2.00,
        source="gitm/benchmarks/edge/optimize.py — frame-batching A/B, per-frame "
        "detection-equivalence gated against the single-frame baseline.",
        applicability=Applicability(workloads=["edge", "kitti", "nuscenes"]),
        safety=SafetyGate(
            tier="moderate",
            requires_rollback_window_s=0,
            forbid_if_oom_history=True,
            notes="Batching raises peak memory; gated on detection-equivalence + "
            "speedup, and forbidden after an OOM since larger batches can OOM.",
        ),
        review=None,
    )


class EdgeBatchingApplicator:
    """Apply frame batching through the standard rollback gate.

    The 'live state' is whether inference is serial or batched. :meth:`measure`
    runs the real serial-vs-batched A/B (:func:`optimize_edge` with
    ``serial``/``batched`` modes): it raises :class:`DetectionDivergenceError`
    when batched detections diverge (forcing a rollback), otherwise returns the
    signed throughput delta so the gate keeps batching only when it is faster.
    The full :class:`EdgeABResult` is stashed on :attr:`last_result`.

    ``run_mode(mode)`` runs the frames ``"serial"`` or ``"batched"`` and returns
    a detection summary — injected so this is testable without a GPU. Carries its
    own :attr:`spec` so the generalized loop can read it.
    """

    def __init__(
        self,
        run_mode: RunMode,
        *,
        batch_size: int = 4,
        reps: int = 2,
        sync: Callable[[], None] | None = None,
        score_atol: float = 0.02,
        spec: InterventionSpec | None = None,
    ):
        self._run_mode = run_mode
        self._batch_size = batch_size
        self._reps = reps
        self._sync = sync
        self._score_atol = score_atol
        self.active = "serial"
        self.last_result: EdgeABResult | None = None
        # Default spec records the actual batch size, so provenance ≠ a constant.
        self.spec = spec or edge_batching_spec(batch_size=batch_size)

    def snapshot(self) -> str:
        return self.active

    def apply(self, spec: InterventionSpec) -> None:
        self.active = "batched"

    def restore(self, snapshot: str) -> None:
        self.active = snapshot

    def measure(self, spec: InterventionSpec) -> float:
        r = optimize_edge(
            self._run_mode,
            baseline_mode="serial",
            candidate_mode="batched",
            reps=self._reps,
            sync=self._sync,
            score_atol=self._score_atol,
        )
        self.last_result = r
        if not r.identical:
            raise DetectionDivergenceError(
                "batched detections differ from single-frame beyond tolerance — rolling back"
            )
        return r.speedup - 1.0
