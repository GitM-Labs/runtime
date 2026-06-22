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
            return "rolled back — fp16 detections differ from fp32 beyond tolerance"
        if self.kept == "candidate":
            return f"kept candidate — verified +{(self.speedup - 1) * 100:.1f}% faster, detections equivalent"
        return "kept baseline — fp16 not faster"


def optimize_edge(
    run_mode: RunMode,
    *,
    reps: int = 2,
    sync: Callable[[], None] | None = None,
    score_atol: float = 0.02,
) -> EdgeABResult:
    """Run the measure→apply→prove A/B and return a gated verdict.

    ``run_mode(mode)`` runs the frames in ``"fp32"`` or ``"fp16"`` — a single
    callable dispatched by mode (the two precisions are the same code path on a
    different autocast context, not two distinct functions). Each precision is
    run ``reps`` times; the best (lowest) wall time is used to reduce launch
    jitter. ``sync`` is invoked after each run so GPU timing is honest (pass a
    device sync; default no-op for CPU/fake).
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

    base_summary, base_eps = _timed("fp32")
    cand_summary, cand_eps = _timed("fp16")

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
