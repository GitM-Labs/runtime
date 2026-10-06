"""Causal attribution on the residual subgraph.

Granger-causality test ranks candidate causes by Granger F p-value. The MLP
that contended for cache shows up as a Granger-cause of attention's residual,
not the symptom (attention's own residual).

A doubly-robust estimator runs alongside Granger (see gitm/optimizer/dr.py).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from gitm.optimizer.monitor import Residuals
from gitm.planner.graph import Graph

#: Two series are the same launch cardinality — and so alignable by position —
#: when the shorter is at least this fraction of the longer. Per-layer against
#: per-step ops differ by the layer count (tens), so any value well above
#: 1/n_layers separates them; 0.9 leaves room for a window's partial edge steps
#: and refused replays.
CARDINALITY_TOLERANCE = 0.9


def comparable(n_a: int, n_b: int, tolerance: float = CARDINALITY_TOLERANCE) -> bool:
    """True when series of these lengths can be aligned by position."""
    lo, hi = sorted((n_a, n_b))
    return hi > 0 and lo / hi >= tolerance


@dataclass
class Hypothesis:
    cause_op: str
    effect_op: str
    p_value: float
    direction: str  # "+ slower", "- faster"
    notes: str = ""


@dataclass
class RankedHypotheses:
    hypotheses: list[Hypothesis]

    def top(self, n: int = 5) -> list[Hypothesis]:
        return self.hypotheses[:n]


def attribute(
    residuals: Residuals,
    graph: Graph,
    max_lag: int = 2,
    *,
    stratify: tuple[str, ...] = (),
) -> RankedHypotheses:
    """Granger-causality on the residual subgraph.

    For each ordered pair (cause, effect) of distinct ops, fit a VAR-style
    Granger F-test on the residual time series. Rank by p-value ascending.

    residuals → ranked hypotheses → candidate intervention from library
      → predict_delta on captured trace (offline)
      → if Δ > threshold, attempt live (rollback-gated via gitm/optimizer/apply.py)
      → if not, drop or escalate

    ``stratify`` splits each op's series by side-table attributes
    (:mod:`gitm.tracer.kernel_attributes`; residuals must be computed
    ``with_attributes``), so a cause or effect confined to one archetype or
    wave is a series of its own instead of a fraction of the op's. Hypothesis
    ops are then stratum labels (``"mlp_down[wave=0]"``). Default: op only.
    """
    from gitm.tracer.kernel_attributes import stratum

    try:
        from statsmodels.tsa.stattools import (
            grangercausalitytests,  # type: ignore[import-not-found]
        )
    except Exception:
        return RankedHypotheses(hypotheses=[])

    # Group residuals by op into ordered time series (per layer-position step)
    series: dict[str, list[float]] = {}
    for kr in residuals.per_kernel:
        series.setdefault(stratum(kr.op, kr.attrs, stratify), []).append(kr.r_kt)

    ops = [op for op, vals in series.items() if len(vals) >= max_lag + 2]
    if len(ops) < 2:
        return RankedHypotheses(hypotheses=[])

    # Series are aligned by position, which pairs like with like only when two
    # series have the same launch cardinality: two per-layer ops put layer l of
    # step s at the same index, a per-layer op against lm_head (once a step)
    # does not. Truncating everything to the shortest series — the old
    # behaviour — let one once-per-step op set the length of every per-layer
    # pair, which on a short window is too few points for any lag model, so
    # attribution silently returned nothing. Cardinality is matched with a
    # tolerance (:func:`comparable`): a real window starts and ends mid-step,
    # and a refused graph replay drops some of an op's kernels, so equal-class
    # series routinely differ by a few launches.
    hypotheses: list[Hypothesis] = []
    for cause in ops:
        for effect in ops:
            if cause == effect or not comparable(len(series[cause]), len(series[effect])):
                continue
            n = min(len(series[cause]), len(series[effect]))
            arr = np.column_stack([np.asarray(series[effect][:n]), np.asarray(series[cause][:n])])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # deprecated-arg + convergence chatter
                    result = grangercausalitytests(arr, maxlag=max_lag, verbose=False)
                pvals = [result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
                p = float(min(pvals))
            except Exception:
                continue
            direction = "+ slower" if np.mean(series[cause]) > 0 else "- faster"
            hypotheses.append(
                Hypothesis(cause_op=cause, effect_op=effect, p_value=p, direction=direction)
            )

    hypotheses.sort(key=lambda h: h.p_value)
    return RankedHypotheses(hypotheses=hypotheses)
