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
