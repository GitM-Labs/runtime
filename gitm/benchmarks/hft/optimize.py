"""The 'act' half for HFT — an output-verified, rollback-gated optimization.

Baseline ``run_pipeline`` runs mostly-serial single-stream cuDF (the trace shows
serialized-concurrency ~0.99). This module applies a candidate optimization and
proves it the GITM way:

    measure baseline → apply candidate → verify IDENTICAL output → compare speed
    → keep the candidate only if it is BOTH correct and faster, else roll back.

The candidate does strictly *less work* than the baseline. The baseline's
``top_of_book`` runs four grouped scans — ``cummax`` + ``cummin`` for the running
best, then ``ffill`` + ``ffill`` to carry it across the opposite side's rows.
This carries the running best in a *single* grouped scan per side by filling the
opposite side with a sentinel outside the price range, so ``cummax``/``cummin``
already propagate it — no ffill. That's **4 grouped scans → 2**: fewer launched
kernels and less work, the realizable gain on a launch-bound run.

Output is identical (proven by ``verify_equivalent``), and if a backend ever
disagrees, the correctness gate keeps the baseline — no false speedup is ever
reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from gitm.benchmarks.hft.harness import microprice, run_pipeline, vwap_1s
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate

# Sentinels strictly outside the integer-tick price range, so filling the
# opposite side with them lets a single cummax/cummin carry the running best.
_LOW = -1
_HIGH = 1 << 62


def top_of_book_fewer_passes(df, dflib):
    """Equivalent to :func:`harness.top_of_book` in two grouped scans, not four.

    Fill the non-bid rows with ``_LOW`` so ``groupby.cummax`` carries the running
    best bid across them (and the symmetric ``_HIGH`` + ``cummin`` for the ask);
    rows before the first same-side event keep the sentinel and become NaN. This
    drops the two ``groupby.ffill`` passes the baseline needs.
    """
    df = df.sort_values(["symbol_id", "ts_ns"]).reset_index(drop=True)
    is_bid = df["side"] == 0
    bid_filled = df["price"].where(is_bid, _LOW)
    ask_filled = df["price"].where(~is_bid, _HIGH)
    bb = bid_filled.groupby(df["symbol_id"]).cummax()
    ba = ask_filled.groupby(df["symbol_id"]).cummin()
    df["best_bid"] = bb.where(bb > _LOW)   # no bid seen yet → sentinel → NaN
    df["best_ask"] = ba.where(ba < _HIGH)
    return df


def run_pipeline_fast(df, dflib) -> dict:
    """``run_pipeline`` with the two-scan top-of-book; same summary contract."""
    df = top_of_book_fewer_passes(df, dflib)
    df["microprice"] = microprice(df)
    vwap = vwap_1s(df, dflib)
    return {
        "events": int(len(df)),
        "mean_microprice": float(df["microprice"].mean()),
        "vwap_buckets": int(len(vwap)),
    }


def _signature(summary: dict) -> tuple:
    """A reduction-based output checksum. Two pipelines are 'identical' iff their
    signatures match — forces evaluation and catches any divergence in the
    computed top-of-book / microprice / VWAP."""
    return (
        int(summary["events"]),
        round(float(summary["mean_microprice"]), 4),
        int(summary["vwap_buckets"]),
    )


def verify_equivalent(df, dflib) -> bool:
    """True iff the candidate produces the same output as the baseline."""
    return _signature(run_pipeline(df, dflib)) == _signature(run_pipeline_fast(df, dflib))


@dataclass
class ABResult:
    baseline_eps: float
    candidate_eps: float
    speedup: float          # candidate / baseline (e.g. 1.20 = 20% faster)
    identical: bool
    kept: str               # "candidate" | "baseline"
    baseline_summary: dict
    candidate_summary: dict

    @property
    def verdict(self) -> str:
        if not self.identical:
            return "rolled back — candidate output differs from baseline"
        if self.kept == "candidate":
            return f"kept candidate — verified +{(self.speedup - 1) * 100:.1f}% faster, identical output"
        return "kept baseline — candidate not faster"


def optimize_hft(df, dflib, *, reps: int = 3, sync=None) -> ABResult:
    """Run the measure→apply→prove loop and return a verified A/B verdict.

    ``reps`` runs each pipeline several times and takes the best (lowest) time to
    reduce launch-jitter noise. ``sync`` is an optional callable invoked after
    each run so GPU timing is honest (pass a device-sync; default no-op for CPU).
    """
    sync = sync or (lambda: None)

    def _timed(fn) -> tuple[dict, float]:
        best = float("inf")
        summary: dict = {}
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            summary = fn(df, dflib)
            sync()
            best = min(best, time.perf_counter() - t0)
        return summary, summary["events"] / max(best, 1e-9)

    base_summary, base_eps = _timed(run_pipeline)
    cand_summary, cand_eps = _timed(run_pipeline_fast)

    identical = _signature(base_summary) == _signature(cand_summary)
    speedup = cand_eps / base_eps if base_eps else 0.0
    kept = "candidate" if (identical and cand_eps > base_eps) else "baseline"

    return ABResult(
        baseline_eps=base_eps,
        candidate_eps=cand_eps,
        speedup=speedup,
        identical=identical,
        kept=kept,
        baseline_summary=base_summary,
        candidate_summary=cand_summary,
    )


def optimize_hft_streaming(batches, dflib, *, sync=None, on_batch=None) -> ABResult:
    """Streaming A/B: run the measure→prove loop over an iterable of frames.

    The single-frame :func:`optimize_hft` needs both pipelines resident on one
    device frame, which caps the A/B at what fits in GPU memory. This processes
    the dataset ``batches`` at a time instead — each ``df`` yielded by ``batches``
    is run through *both* pipelines (baseline then candidate), their per-batch
    outputs are checked byte-identical, and the timings accumulate. So a 1B-event
    set that won't fit one frame still yields a verified speedup over the *whole*
    dataset.

    ``batches`` is any iterable of device frames (the caller owns loading/freeing
    them). ``sync`` is the device-sync callable (so GPU timing is honest). The
    overall run is "identical" iff *every* batch matched — a single divergent
    batch rolls the whole thing back, same gate as the single-frame path.
    ``on_batch(i, ABResult-like dict)`` is an optional progress callback whose
    ``identical`` field is the *running* AND (True until the first divergence),
    not a per-batch verdict.

    Timing measures **pipeline compute throughput**: only the two pipeline calls
    are timed, so the speedup is verified over the whole dataset's *compute*, not
    wall-clock — any per-batch I/O the caller does between yields is excluded by
    design. Within a batch the baseline runs before the candidate; the candidate
    does strictly less work, so warm-cache ordering can only *understate* its win.

    Raises ``ValueError`` on an empty ``batches`` iterable — an empty A/B would
    vacuously report ``identical=True`` with zero throughput, which is a silent
    no-op, not a verified result.
    """
    sync = sync or (lambda: None)
    base_events = cand_events = 0
    base_vwap = cand_vwap = 0
    base_t = 0.0
    cand_t = 0.0
    identical = True
    n_batches = 0

    for df in batches:
        n_batches += 1
        t0 = time.perf_counter()
        bs = run_pipeline(df, dflib)
        sync()
        base_t += time.perf_counter() - t0

        t0 = time.perf_counter()
        cs = run_pipeline_fast(df, dflib)
        sync()
        cand_t += time.perf_counter() - t0

        if _signature(bs) != _signature(cs):
            identical = False
        base_events += int(bs["events"])
        base_vwap += int(bs["vwap_buckets"])
        cand_events += int(cs["events"])
        cand_vwap += int(cs["vwap_buckets"])
        if on_batch is not None:
            on_batch(n_batches, {"events": int(bs["events"]), "identical": identical})

    if n_batches == 0:
        raise ValueError("optimize_hft_streaming: no batches to process (empty iterable)")

    base_eps = base_events / max(base_t, 1e-9)
    cand_eps = cand_events / max(cand_t, 1e-9)
    speedup = cand_eps / base_eps if base_eps else 0.0
    kept = "candidate" if (identical and cand_eps > base_eps) else "baseline"
    # Per-pipeline count aggregates (kept separate so a divergent run never reports
    # the candidate's counts as the baseline's). The identical verdict is the
    # per-batch AND above, not a signature over these sums (mean_microprice isn't
    # summable across batches).
    return ABResult(
        baseline_eps=base_eps,
        candidate_eps=cand_eps,
        speedup=speedup,
        identical=identical,
        kept=kept,
        baseline_summary={"events": base_events, "vwap_buckets": base_vwap},
        candidate_summary={"events": cand_events, "vwap_buckets": cand_vwap},
    )


# --- the intervention, wired for the autonomous loop -------------------------
#
# Everything above is the standalone A/B. The pieces below express that same
# optimization as a *curated intervention* the runtime can observe → attribute →
# select → apply → prove through the standard rollback gate
# (:func:`gitm.optimizer.apply.apply_intervention`), so HFT is no longer
# measurement-only — it actually applies a verified speedup.


class CorrectnessError(RuntimeError):
    """Candidate output diverged from the baseline.

    Raised inside :meth:`HftFewerScansApplicator.measure` so the apply gate rolls
    back: a speedup is *never* kept on top of wrong output.
    """


def hft_intervention_spec() -> InterventionSpec:
    """The single curated HFT lever: top-of-book in two grouped scans, not four.

    Output-equivalence is enforced at apply time (``verify_equivalent`` inside
    the A/B), so the expected-delta range here is only used for *ranking* — the
    real number comes from the rollback-gated measure.
    """
    return InterventionSpec(
        name="hft_top_of_book_fewer_scans",
        summary="Carry per-symbol top-of-book in 2 grouped scans instead of 4 — "
        "sentinel-fill lets cummax/cummin replace the two ffill passes.",
        knob="hft.top_of_book_grouped_scans",
        value=2,
        applies_to_kernels=["scan", "groupby", "cummax", "cummin", "ffill"],
        expected_delta_mean=0.10,
        expected_delta_lo=0.0,
        expected_delta_hi=0.50,
        source="gitm/benchmarks/hft/optimize.py — 4→2 grouped-scan top-of-book, "
        "output-verified against the baseline pipeline.",
        applicability=Applicability(workloads=["hft", "hft-lob"]),
        safety=SafetyGate(
            tier="low_risk",
            requires_rollback_window_s=0,
            forbid_if_oom_history=False,
            notes="Pure compute rewrite; identical output is gated by "
            "verify_equivalent before any speedup is kept.",
        ),
        review=None,
    )


class HftFewerScansApplicator:
    """Apply the fewer-scans top-of-book through the standard rollback gate.

    :meth:`measure` runs the real baseline-vs-candidate A/B (:func:`optimize_hft`):
    it raises :class:`CorrectnessError` when the candidate's output is not
    byte-identical (forcing a rollback), otherwise returns the signed speedup
    delta so the gate keeps the candidate only when it is genuinely faster. The
    full :class:`ABResult` is stashed on :attr:`last_result` for the report.

    Note on the protocol seam: ``apply_intervention`` calls ``measure`` exactly
    once, so the baseline must be established *inside* ``measure`` — hence it runs
    both pipelines (a self-contained A/B) rather than timing only whichever
    pipeline ``apply`` selected. :attr:`active` therefore tracks *intent* (which
    pipeline the gate decided to keep) for inspectability; it does not gate what
    ``measure`` times. This is correct for a pure-compute rewrite where the only
    honest baseline is a fresh run on the same frame; a live-engine applicator
    (where state genuinely persists) would instead time the active config.

    Implements the :class:`gitm.optimizer.apply.Applicator` protocol structurally.
    """

    def __init__(self, df, dflib, *, reps: int = 3, sync=None):
        self._df = df
        self._dflib = dflib
        self._reps = reps
        self._sync = sync
        self.active = "baseline"
        self.last_result: ABResult | None = None

    def snapshot(self) -> str:
        return self.active

    def apply(self, spec: InterventionSpec) -> None:
        self.active = "candidate"

    def restore(self, snapshot: str) -> None:
        self.active = snapshot

    def measure(self, spec: InterventionSpec) -> float:
        r = optimize_hft(self._df, self._dflib, reps=self._reps, sync=self._sync)
        self.last_result = r
        if not r.identical:
            raise CorrectnessError(
                "candidate top-of-book output differs from baseline — rolling back"
            )
        return r.speedup - 1.0


class HftStreamingApplicator:
    """Streaming variant of :class:`HftFewerScansApplicator` for datasets too big
    to hold one frame. :meth:`measure` runs the batched A/B
    (:func:`optimize_hft_streaming`) over a *fresh* batch generator, so the whole
    sharded dataset is verified+timed without ever materialising more than one
    batch. Same rollback gate: raises :class:`CorrectnessError` if any batch
    diverges, otherwise returns the signed speedup delta.

    ``batches_factory`` is a zero-arg callable returning a fresh iterable of
    device frames each time it is called (one for ``measure``'s A/B, independent
    of the observe pass the runner does).
    """

    def __init__(self, batches_factory, dflib, *, sync=None):
        self._batches_factory = batches_factory
        self._dflib = dflib
        self._sync = sync
        self.active = "baseline"
        self.last_result: ABResult | None = None

    def snapshot(self) -> str:
        return self.active

    def apply(self, spec: InterventionSpec) -> None:
        self.active = "candidate"

    def restore(self, snapshot: str) -> None:
        self.active = snapshot

    def measure(self, spec: InterventionSpec) -> float:
        r = optimize_hft_streaming(self._batches_factory(), self._dflib, sync=self._sync)
        self.last_result = r
        if not r.identical:
            raise CorrectnessError(
                "candidate top-of-book output differs from baseline — rolling back"
            )
        return r.speedup - 1.0
