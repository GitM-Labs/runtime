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
estimator always returns a number rather than throwing. Every fallback and every
warning the fits raise is recorded: with step position as the only covariate,
anomalies that cluster in time separate the propensity model, which then fails
to converge and is clipped to [0.05, 0.95]. Such a number is still returned, but
the pair is counted as degraded, not completed.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from gitm.optimizer.attribution import AnalysisStatus, Hypothesis, RankedHypotheses, degrading
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


def doubly_robust_ate(
    y: np.ndarray, t: np.ndarray, X: np.ndarray, *, diag: dict | None = None
) -> tuple[float, float]:
    """AIPW estimate of the ATE of ``t`` on ``y`` given covariates ``X``.

    Returns ``(ate, se)``. Robust to a misspecified outcome *or* propensity model.
    ``diag``, when given, receives ``fallbacks`` (``(kind, message)`` for each
    nuisance fit replaced by a mean) and ``warnings`` (``(category, message)``
    for each warning the fits and the estimate raised; category is the class).
    """
    fallbacks: list[tuple[str, object]] = []
    if diag is not None:  # filled before any early return, so callers can always read it
        diag["fallbacks"] = fallbacks
        diag["warnings"] = []
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

    # Record, don't silence: a non-converged propensity is exactly the case
    # the estimate should not be trusted in (see module docstring).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Propensity e = P(T=1|X); clip away from 0/1 to bound the IPW weights.
        try:
            e = sm.Logit(t, Xc).fit(disp=0).predict(Xc)
        except Exception as exc:  # statsmodels raises several types; recorded by name
            fallbacks.append((f"propensity:{type(exc).__name__}", exc))
            e = np.full(n, t.mean())
        e = np.clip(e, 0.05, 0.95)

        # Outcome models m1 = E[Y|T=1,X], m0 = E[Y|T=0,X].
        def _outcome(mask: np.ndarray, label: str) -> np.ndarray:
            if mask.sum() <= Xc.shape[1]:  # too few rows to fit -> group mean
                fallbacks.append((f"{label}:too_few_rows", f"{int(mask.sum())} rows"))
                return np.full(n, float(y[mask].mean()) if mask.any() else 0.0)
            try:
                return sm.OLS(y[mask], Xc[mask]).fit().predict(Xc)
            except Exception as exc:  # recorded by name, as above
                fallbacks.append((f"{label}:{type(exc).__name__}", exc))
                return np.full(n, float(y[mask].mean()))

        m1 = _outcome(t == 1, "outcome_treated")
        m0 = _outcome(t == 0, "outcome_control")

        # Inside the block too: overflow or invalid values here are recorded.
        psi1 = t * (y - m1) / e + m1
        psi0 = (1 - t) * (y - m0) / (1 - e) + m0
        contrast = psi1 - psi0
        ate = float(np.mean(contrast))
        se = float(np.std(contrast, ddof=1) / np.sqrt(n)) if n > 1 else float("inf")

    if diag is not None:
        # Messages stay objects; RankedHypotheses cleans only the first of each kind.
        diag["warnings"] = [(w.category, w.message) for w in caught]
    return ate, se


def attribute_dr(residuals: Residuals, graph: Graph, *, band: float = _KT_BAND) -> RankedHypotheses:
    """Doubly-robust ranking of cause→effect pairs, as ``RankedHypotheses``.

    Mirrors :func:`gitm.optimizer.attribution.attribute` so the loop can run both
    and compare. p_value is a 2-sided normal approximation from the ATE z-score;
    notes carry the signed ATE for the report.

    The result carries the same record as Granger's: ``status``, pairs attempted,
    completed (an estimate exists) and degraded (the estimate needed a fallback or
    raised a warning; ``degraded`` on the hypothesis and in its note), fallbacks
    and errors under ``failures`` and fit warnings under ``warnings``.
    """
    min_obs = 4
    series: dict[str, list[float]] = {}
    for kr in residuals.per_kernel:
        series.setdefault(kr.op, []).append(kr.r_kt)
    out = RankedHypotheses(hypotheses=[], min_obs=min_obs)
    try:
        import statsmodels.api  # noqa: F401  (used by doubly_robust_ate)
    except Exception as exc:  # ImportError, or e.g. a ValueError from a binary mismatch
        out.status = AnalysisStatus.UNAVAILABLE
        out.record_failure(type(exc).__name__, exc)
        return out
    out.series_lengths = {op: len(v) for op, v in series.items()}
    out.excluded_ops = {op: k for op, k in out.series_lengths.items() if k < min_obs}
    # A NaN cause sample would read as in-band (abs(nan) > band is False) and a
    # NaN effect poisons the estimate; such ops are left out, on record.
    for op, vals in series.items():
        bad = int(np.count_nonzero(~np.isfinite(np.asarray(vals, dtype=float))))
        if bad and op not in out.excluded_ops:
            out.excluded_ops[op] = len(vals)
            out.record_failure("nonfinite_series", f"{op}: {bad} non-finite residual(s)")

    ops = [op for op in series if op not in out.excluded_ops]
    if len(ops) < 2:
        out.status = AnalysisStatus.INSUFFICIENT_DATA
        out.reason = f"fewer than 2 ops with at least {min_obs} samples"
        return out
    n = min(len(series[op]) for op in ops)
    out.n_obs_used = n
    pos = np.arange(n, dtype=float)

    effects: list[tuple[DREffect, bool]] = []
    for cause in ops:
        t = (np.abs(np.asarray(series[cause][:n])) > band).astype(float)
        n_t = int(t.sum())
        if n_t < _MIN_GROUP or (n - n_t) < _MIN_GROUP:
            # Too few anomalies (or too few normal samples) to estimate from;
            # on record so "ok, N/N pairs" is read against what was left out.
            out.skipped_causes[cause] = n_t
            continue
        for effect in ops:
            if effect == cause:
                continue
            out.pairs_attempted += 1
            y = np.asarray(series[effect][:n], dtype=float)
            diag: dict = {}
            try:
                ate, se = doubly_robust_ate(y, t, pos, diag=diag)
            except Exception as exc:  # one pair's failure keeps the others' results
                out.record_failure(type(exc).__name__, exc)
                continue
            # Recorded before the estimate is judged, so a rejected pair keeps
            # the fit problems that explain it.
            for kind, message in diag["fallbacks"]:
                out.record_failure(kind, message)
            for category, message in diag["warnings"]:
                out.record_warning(category.__name__, message)
            # A zero-variance fit gives an infinite or NaN estimate; it is a
            # failed pair, never a ranked one.
            if not (math.isfinite(ate) and math.isfinite(se) and se > 0):
                out.record_failure("nonfinite_estimate", f"ate={ate}, se={se}")
                continue
            clean = not diag["fallbacks"] and not any(degrading(c) for c, _ in diag["warnings"])
            out.pairs_completed += 1
            out.pairs_degraded += not clean
            z = ate / se
            effects.append((DREffect(cause, effect, ate, se, z, int(t.sum())), clean))

    if out.pairs_attempted == 0:
        out.status = AnalysisStatus.INSUFFICIENT_DATA
        out.reason = (
            f"no op had at least {_MIN_GROUP} out-of-band and {_MIN_GROUP} in-band samples"
        )
        return out

    from math import erfc, sqrt

    out.hypotheses = [
        Hypothesis(
            cause_op=d.cause_op,
            effect_op=d.effect_op,
            p_value=float(erfc(abs(d.z) / sqrt(2))),  # 2-sided normal approx
            direction="+ slower" if d.ate > 0 else "- faster",
            notes=(f"doubly-robust ATE={d.ate:+.3f} (se={d.se:.3f}, n_treated={d.n_treated})"
                   + ("" if clean else "; degraded: fit fallback or warning, see dr record")),
            degraded=not clean,
        )
        for d, clean in effects
    ]
    # By p-value (|z| descending), clean estimates first.
    out.rank(lambda h: h.p_value)
    out.finalize_status()
    return out
