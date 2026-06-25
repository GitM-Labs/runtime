"""Validate the counterfactual replay engine against synthetic ground truth.

The replay engine (:func:`gitm.optimizer.replay.predict_delta`) predicts an
intervention's wall-clock delta as ``coverage × expected_delta_mean`` — an
*aggregate* estimate. To check it's calibrated we build synthetic traces with a
known hot kernel, then compute an **independent, per-kernel** ground-truth delta:
each applicable kernel is sped up by its own draw from the intervention's
``[lo, hi]`` band, and the realized wall-clock saving is summed.

Because the band is centered on the mean, the aggregate prediction is the
*expectation* of the per-kernel ground truth — so they should agree within
tolerance once a trace has enough applicable kernels (law of large numbers).
The harness reports the mean absolute relative error across many injections;
the engine passes at the ticket's ±20 % tolerance.

    python -m gitm.optimizer.replay_validation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.replay import _applies, predict_delta
from gitm.tracer.schema import KernelEvent, Trace

_GRAND_NS = 1_000_000.0  # arbitrary fixed wall-clock scale per synthetic trace


def _synthetic_spec() -> InterventionSpec:
    # mean = (lo+hi)/2 so the aggregate prediction is an unbiased estimate of
    # the per-kernel ground truth.
    return InterventionSpec(
        name="synthetic", summary="synthetic injection", knob="x", value=1,
        applies_to_kernels=["hot"],
        expected_delta_mean=0.06, expected_delta_lo=0.02, expected_delta_hi=0.10,
        source="https://example.test/synthetic",
    )


def _build_trace(rng: np.random.Generator, n_kernels: int, hot_fraction: float) -> Trace:
    """A trace where 'hot' kernels account for ``hot_fraction`` of wall-clock."""
    n_hot = max(2, n_kernels // 3)
    n_cold = max(1, n_kernels - n_hot)
    hot_durs = rng.dirichlet(np.ones(n_hot)) * (hot_fraction * _GRAND_NS)
    cold_durs = rng.dirichlet(np.ones(n_cold)) * ((1 - hot_fraction) * _GRAND_NS)

    items = [(d, "hot_kernel") for d in hot_durs] + [(d, "cold_kernel") for d in cold_durs]
    rng.shuffle(items)

    events: list[KernelEvent] = []
    t = 0
    for d, name in items:
        dur = max(1, int(d))
        events.append(KernelEvent(
            kind="kernel", start_ns=t, end_ns=t + dur, stream_id=7, device_id=0, name=name,
        ))
        t += dur
    return Trace(
        workload_id="synthetic", fingerprint="synthetic", run_id="synthetic",
        device_count=1, vendor="nvidia", captured_at_ns=0, duration_ns=t, events=events,
    )


def _ground_truth_delta(trace: Trace, spec: InterventionSpec, rng: np.random.Generator) -> float:
    """Per-kernel simulated wall-clock saving fraction (independent of predict_delta)."""
    total = max(trace.duration_ns, 1)
    saved = 0.0
    for k in trace.kernels():
        if _applies(spec, k.name):
            s = rng.uniform(spec.expected_delta_lo, spec.expected_delta_hi)
            saved += (k.end_ns - k.start_ns) * s
    return saved / total


@dataclass
class ValidationResult:
    n: int
    tolerance: float
    mean_abs_rel_err: float
    p90_abs_rel_err: float
    frac_within_tol: float

    @property
    def passed(self) -> bool:
        # The aggregate estimator is calibrated if the *mean* relative error is
        # within tolerance across injections.
        return self.mean_abs_rel_err <= self.tolerance


def validate(n: int = 300, *, seed: int = 0, tolerance: float = 0.20) -> ValidationResult:
    rng = np.random.default_rng(seed)
    spec = _synthetic_spec()
    errs: list[float] = []
    for _ in range(n):
        n_k = int(rng.integers(48, 128))
        hot_frac = float(rng.uniform(0.2, 0.6))
        trace = _build_trace(rng, n_k, hot_frac)
        truth = _ground_truth_delta(trace, spec, rng)
        pred = predict_delta(trace, spec)
        errs.append(abs(pred - truth) / abs(truth) if truth else 0.0)
    arr = np.asarray(errs)
    return ValidationResult(
        n=n,
        tolerance=tolerance,
        mean_abs_rel_err=float(arr.mean()),
        p90_abs_rel_err=float(np.percentile(arr, 90)),
        frac_within_tol=float((arr <= tolerance).mean()),
    )


def main() -> int:
    r = validate()
    print(f"replay validation over {r.n} synthetic injections:")
    print(f"  mean abs rel err: {r.mean_abs_rel_err:.1%}  (tolerance {r.tolerance:.0%})")
    print(f"  p90 abs rel err:  {r.p90_abs_rel_err:.1%}")
    print(f"  within tolerance: {r.frac_within_tol:.1%}")
    print("  PASS" if r.passed else "  FAIL")
    return 0 if r.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
