"""Causal attribution on the residual subgraph.

Granger-causality test ranks candidate causes by Granger F p-value. The MLP
that contended for cache shows up as a Granger-cause of attention's residual,
not the symptom (attention's own residual).

A doubly-robust estimator runs alongside Granger (see gitm/optimizer/dr.py).
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass

import numpy as np

from gitm.optimizer.monitor import Residuals
from gitm.planner.graph import Graph

#: Series are aligned by position, which only pairs like with like at the same
#: launch cardinality: two per-layer ops yes, a per-layer op and lm_head (once a
#: step) no. The tolerance absorbs a window's partial edge steps and refused
#: replays; truncating every series to the shortest — the old behaviour — let a
#: once-per-step op starve every per-layer pair.
CARDINALITY_TOLERANCE = 0.9


def comparable(n_a: int, n_b: int, tolerance: float = CARDINALITY_TOLERANCE) -> bool:
    lo, hi = sorted((n_a, n_b))
    return hi > 0 and lo / hi >= tolerance


def aligned_pairs(series: dict[str, list[float]], ops: list[str]):
    """``(cause, effect, n)`` for every comparable ordered pair, n = shorter length."""
    for cause in ops:
        for effect in ops:
            if cause != effect and comparable(len(series[cause]), len(series[effect])):
                yield cause, effect, min(len(series[cause]), len(series[effect]))


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

    ``stratify`` splits each op's series by side-table attributes (residuals
    computed ``with_attributes``); hypothesis ops are then stratum labels.
    """
    from gitm.optimizer.monitor import residual_series

    try:
        from statsmodels.tsa.stattools import (
            grangercausalitytests,  # type: ignore[import-not-found]
        )
    except Exception:
        return RankedHypotheses(hypotheses=[])

    # Group residuals by op into ordered time series (per layer-position step)
    series = residual_series(residuals, stratify)

    ops = [op for op, vals in series.items() if len(vals) >= max_lag + 2]
    if len(ops) < 2:
        return RankedHypotheses(hypotheses=[])

    hypotheses: list[Hypothesis] = []
    for cause, effect, n in aligned_pairs(series, ops):
        arr = np.column_stack([np.asarray(series[effect][:n]), np.asarray(series[cause][:n])])
        try:
            # No `verbose=`: statsmodels 0.15 removed it, and passing it raised a
            # TypeError on every pair that the except below swallowed, so Granger
            # silently returned nothing. 0.14 prints without it, hence the redirect.
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")  # convergence chatter
                result = grangercausalitytests(arr, maxlag=max_lag)
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
