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
) -> RankedHypotheses:
    """Granger-causality on the residual subgraph.

    For each ordered pair (cause, effect) of distinct ops, fit a VAR-style
    Granger F-test on the residual time series. Rank by p-value ascending.

    residuals → ranked hypotheses → candidate intervention from library
      → predict_delta on captured trace (offline)
      → if Δ > threshold, attempt live (rollback-gated via gitm/optimizer/apply.py)
      → if not, drop or escalate
    """
    try:
        from statsmodels.tsa.stattools import (
            grangercausalitytests,  # type: ignore[import-not-found]
        )
    except Exception:
        return RankedHypotheses(hypotheses=[])

    # Group residuals by op into ordered time series (per layer-position step)
    series: dict[str, list[float]] = {}
    for kr in residuals.per_kernel:
        series.setdefault(kr.op, []).append(kr.r_kt)

    ops = [op for op, vals in series.items() if len(vals) >= max_lag + 2]
    if len(ops) < 2:
        return RankedHypotheses(hypotheses=[])

    n = min(len(series[op]) for op in ops)
    hypotheses: list[Hypothesis] = []
    for cause in ops:
        for effect in ops:
            if cause == effect:
                continue
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
