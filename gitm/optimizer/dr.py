"""Doubly-robust causal attribution, alongside Granger.

Granger ranks *temporal precedence* — does op A's residual help predict op B's?
That's necessary but not sufficient for a causal effect. The doubly-robust
(AIPW) estimator answers the complementary question: *how much* does op A being
anomalous move op B's residual, with an estimate that stays consistent if
**either** the outcome model **or** the propensity model is right (hence
"doubly robust"). Running both and agreeing is the bar before we act on a cause.

For each ordered pair (cause, effect):

* **treatment** ``T`` — 1 at steps where the cause op's residual is out of band,
* **outcome** ``Y`` — the effect op's residual at the same step,
* **covariates** ``X`` — step position (a simple confounder proxy; extend with
  more features as the graph grows).

AIPW estimate of the average treatment effect:

    ATE = mean[ T(Y-m1)/e + m1 ] - mean[ (1-T)(Y-m0)/(1-e) + m0 ]

where ``e = P(T=1|X)`` (propensity) and ``m_t = E[Y|T=t,X]`` (outcome models).
Nuisance models are fit with statsmodels; on degenerate inputs (no treated or no
control units, separable propensity) we fall back to unadjusted means so the
estimator always returns a number rather than throwing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from gitm.optimizer.attribution import Hypothesis, RankedHypotheses
from gitm.optimizer.invariants import INVARIANTS
from gitm.optimizer.monitor import Residuals
from gitm.planner.graph import Graph

_KT_BAND = next(i.band_width for i in INVARIANTS if i.id == "kernel_time")
#: Minimum treated and control units before the doubly-robust estimate is
#: trustworthy; below this the propensity model separates and we abstain.
_MIN_GROUP = 3


@dataclass
class DREffect:
    cause_op: str
    effect_op: str
    ate: float          # average treatment effect (signed, residual units)
    se: float           # standard error of the ATE
    z: float            # ate / se
    n_treated: int


def doubly_robust_ate(y: np.ndarray, t: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """AIPW estimate of the ATE of ``t`` on ``y`` given covariates ``X``.

    Returns ``(ate, se)``. Robust to a misspecified outcome *or* propensity model.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = y.size

    # Degenerate: with fewer than _MIN_GROUP treated *or* control units the
    # logistic propensity model perfectly separates and any estimate is
    # spurious. Refuse rather than emit a meaningless number (this is the
    # honest answer for a clean workload with almost no anomalies).
    n_t = int(t.sum())
    if n == 0 or n_t < _MIN_GROUP or (n - n_t) < _MIN_GROUP:
        return 0.0, float("inf")

    import statsmodels.api as sm

    Xc = sm.add_constant(X, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nuisance-model convergence chatter
        # Propensity e = P(T=1|X); clip away from 0/1 to bound the IPW weights.
        try:
            e = sm.Logit(t, Xc).fit(disp=0).predict(Xc)
        except Exception:
            e = np.full(n, t.mean())
        e = np.clip(e, 0.05, 0.95)

        # Outcome models m1 = E[Y|T=1,X], m0 = E[Y|T=0,X].
        def _outcome(mask: np.ndarray) -> np.ndarray:
            if mask.sum() <= Xc.shape[1]:  # too few rows to fit -> group mean
                return np.full(n, float(y[mask].mean()) if mask.any() else 0.0)
            try:
                return sm.OLS(y[mask], Xc[mask]).fit().predict(Xc)
            except Exception:
                return np.full(n, float(y[mask].mean()))

        m1 = _outcome(t == 1)
        m0 = _outcome(t == 0)

    psi1 = t * (y - m1) / e + m1
    psi0 = (1 - t) * (y - m0) / (1 - e) + m0
    contrast = psi1 - psi0
    ate = float(np.mean(contrast))
    se = float(np.std(contrast, ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return ate, se


def attribute_dr(residuals: Residuals, graph: Graph, *, band: float = _KT_BAND) -> RankedHypotheses:
    """Doubly-robust ranking of cause→effect pairs, as ``RankedHypotheses``.

    Mirrors :func:`gitm.optimizer.attribution.attribute` so the loop can run both
    and compare. p_value is a 2-sided normal approximation from the ATE z-score;
    notes carry the signed ATE for the report.
    """
    series: dict[str, list[float]] = {}
    for kr in residuals.per_kernel:
        series.setdefault(kr.op, []).append(kr.r_kt)

    ops = [op for op, v in series.items() if len(v) >= 4]
    if len(ops) < 2:
        return RankedHypotheses(hypotheses=[])
    n = min(len(series[op]) for op in ops)
    pos = np.arange(n, dtype=float)

    effects: list[DREffect] = []
    for cause in ops:
        t = (np.abs(np.asarray(series[cause][:n])) > band).astype(float)
        n_t = int(t.sum())
        if n_t < _MIN_GROUP or (n - n_t) < _MIN_GROUP:
            continue  # too few anomalies to support a doubly-robust estimate
        for effect in ops:
            if effect == cause:
                continue
            y = np.asarray(series[effect][:n], dtype=float)
            ate, se = doubly_robust_ate(y, t, pos)
            z = ate / se if se not in (0.0, float("inf")) else 0.0
            effects.append(DREffect(cause, effect, ate, se, z, int(t.sum())))

    # Rank by |z| (effect size over uncertainty) descending.
    effects.sort(key=lambda d: abs(d.z), reverse=True)

    from math import erfc, sqrt

    hyps = [
        Hypothesis(
            cause_op=d.cause_op,
            effect_op=d.effect_op,
            p_value=float(erfc(abs(d.z) / sqrt(2))),  # 2-sided normal approx
            direction="+ slower" if d.ate > 0 else "- faster",
            notes=f"doubly-robust ATE={d.ate:+.3f} (se={d.se:.3f}, n_treated={d.n_treated})",
        )
        for d in effects
    ]
    return RankedHypotheses(hypotheses=hyps)
