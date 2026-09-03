"""Match semantics — which past result applies to the workload in front of you.

The whole deliverable turns on this module, because "close enough" is where a
wrong row gets applied. The design is a **split**, not a single score:

    exact equality   model + revision, GPU SKU, environment, source_kind,
                     concurrency policy          -> a gate: pass or reject
    distance         the numeric regime axes     -> a ranking among survivors

Categorical fields are gated because "nearly an H100" is not a thing, and a
distance that mixed a GPU mismatch into the same number as a token-count
mismatch would let a large enough workload similarity outvote running on
different silicon.

**The numeric axes are compared as log2 ratios**, which is the natural metric for
token counts and rates — what matters is the *factor*, not the difference:

    1,024 vs 2,048 tokens  -> 1.0     (a 2x change)
    1,024 vs 1,536 tokens  -> 0.58
    1,024 vs 1,024 tokens  -> 0.0

and combined with **L-infinity** (the max across axes), not a mean or a Euclidean
norm. A mean lets a close match on four axes hide a 4x mismatch on the fifth, and
the fifth is the one that breaks the row. L-inf says: a row is as far away as its
worst axis.

**The threshold is not known, and this module says so rather than picking one.**
A number like ``max_distance = 1.0`` reads as calibrated and is not: under log2
it means "accept up to a 2x mismatch on every axis at once", which may well be
safe for output p50 and is certainly not safe for long-context input p95, for
prefix-cache reuse, or for a queue-sensitive scheduling policy. So
:class:`AxisTolerance` **refuses to hold a number without the experiment that
produced it**, the shipped :data:`UNCALIBRATED_POLICY` has no numbers at all, and
a lookup that would need one returns :attr:`MatchStatus.UNCALIBRATED` and routes
to conservative discovery.

Calibrating an axis, which is what removes that status:

1. Run the same knob across nearby regimes, varying **one** axis at a time.
2. Find where the effect changes sign, or where the latency percentile criterion
   from deliverable 2 flips from pass to fail.
3. Set that axis's tolerance strictly inside the distance at which it flipped.
4. The L-inf limit is then the strictest relevant per-axis tolerance, by
   construction — there is no separate global number to choose.

Until step 1 has data, the honest state is "not yet calibrated", and the cost of
that state is a discovery run, which is the cheap failure.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gitm.playbook.schema import Playbook, PlaybookRow, RowIdentity
from gitm.traffic.regime import Regime

#: Distance returned when an axis cannot be compared at all — one side is zero,
#: the other is not. Not 0.0 and not "skip": an incomparable axis is a reason to
#: reject a row, and infinity is what makes L-inf say so without a special case.
INCOMPARABLE = math.inf


def log2_ratio(a: float, b: float) -> float:
    """``|log2(a / b)|`` — the distance between two positive magnitudes.

    Symmetric, zero at equality, 1.0 at a factor of two, and scale-free: 100 vs
    200 tokens is the same distance as 10,000 vs 20,000, which is the property a
    token-count axis needs. Two exact zeros are equal; one zero is
    :data:`INCOMPARABLE`, because "no output tokens at all" is not a small
    version of "some output tokens".
    """
    if a == b:
        return 0.0
    if a <= 0 or b <= 0:
        return INCOMPARABLE
    return abs(math.log2(a / b))


def dispersion_distance(a: float, b: float) -> float:
    """Distance between two index-of-dispersion values, ``|log2((1+a)/(1+b))|``.

    Burstiness cannot use :func:`log2_ratio` directly: a perfectly paced trace
    has ``D = 0`` and a bare ratio would make it incomparable to everything,
    including another paced trace. The ``1 +`` shift anchors the axis so that the
    Poisson reference ``D = 1`` sits one unit from flat ``D = 0``, and the two
    real traces deliverable 1 measured land where intuition puts them:

        flat (0.0)  vs poisson (1.0)   -> 1.00
        burstgpt (1.01) vs mooncake (6.74) -> 1.95   (far apart, correctly)
        moderate (5.0) vs mooncake (6.74)  -> 0.37   (near, correctly)
    """
    return log2_ratio(1.0 + a, 1.0 + b)


#: The numeric axes, and how each is compared. Names match
#: :class:`~gitm.traffic.regime.Regime` fields exactly — an axis that cannot be
#: read off a Regime by name is an axis that will silently stop being computed.
AXIS_METRICS = {
    "input_p50": log2_ratio,
    "input_p95": log2_ratio,
    "output_p50": log2_ratio,
    "output_p95": log2_ratio,
    "io_ratio": log2_ratio,
    "burstiness": dispersion_distance,
    "rate_rps": log2_ratio,
}

#: Axes on by default. **``rate_rps`` is deliberately absent.** It exists on
#: ``Regime`` and adding it to the distance because it is there would be exactly
#: the mistake this module is written to avoid: offered rate is largely captured
#: by burstiness plus the concurrency gate, and a knob that is insensitive to
#: rate would then be rejected for a workload it fits. Turning it on is a
#: decision with evidence behind it — see :data:`RATE_AXIS_DECISION`.
DEFAULT_AXES = ("input_p50", "input_p95", "output_p50", "output_p95", "io_ratio", "burstiness")

RATE_AXIS_DECISION = """\
rate_rps is not in the distance by default. Include it when knob outcomes are
shown to depend materially on offered load *after* concurrency and burstiness are
accounted for; omit it when they are not. Either way the decision is recorded
with the experiment that settled it, not inferred from the field existing."""


class AxisTolerance(BaseModel):
    """How far this axis may differ, and the experiment that says so.

    ``max_distance=None`` means **uncalibrated** — the axis is compared and
    reported, but no nonzero distance on it can be accepted automatically.

    A number without ``calibration`` is rejected at construction. That is the
    enforcement behind "the threshold is currently unknown": the only way to get
    a tolerance into a policy is to name the run that produced it, so a
    placeholder can never quietly become a production constant.
    """

    model_config = ConfigDict(extra="forbid")

    max_distance: float | None = None
    #: What measured it. Free text pointing at a run or a spec section, e.g.
    #: "prereg_rank1 E4, 2026-09-14: sign flip at 1.4 on input_p95".
    calibration: str | None = None

    @model_validator(mode="after")
    def _a_number_needs_a_reason(self) -> AxisTolerance:
        if self.max_distance is not None and not self.calibration:
            raise ValueError(
                "max_distance without calibration: a tolerance is a measured "
                "quantity, not a default. Run the knob across nearby regimes, "
                "find where the effect flips, and cite it here."
            )
        if self.max_distance is not None and self.max_distance < 0:
            raise ValueError("max_distance must be >= 0")
        return self

    @property
    def calibrated(self) -> bool:
        return self.max_distance is not None


class MatchPolicy(BaseModel):
    """Which axes count, how far each may stray, and what is gated exactly."""

    model_config = ConfigDict(extra="forbid")

    name: str
    axes: tuple[str, ...] = DEFAULT_AXES
    tolerances: dict[str, AxisTolerance] = Field(default_factory=dict)
    #: Exact-match gates. Each is a policy choice, listed so that loosening one
    #: is an edit to a named field rather than an accident in a comparison.
    match_source_kind: bool = True
    match_concurrency: bool = True
    match_env: bool = True

    @model_validator(mode="after")
    def _axes_are_real(self) -> MatchPolicy:
        unknown = [a for a in self.axes if a not in AXIS_METRICS]
        if unknown:
            raise ValueError(f"unknown regime axes {unknown}; known: {sorted(AXIS_METRICS)}")
        stray = [a for a in self.tolerances if a not in self.axes]
        if stray:
            raise ValueError(f"tolerance set for axes not in the policy: {stray}")
        return self

    def tolerance(self, axis: str) -> AxisTolerance:
        return self.tolerances.get(axis, AxisTolerance())

    @property
    def uncalibrated_axes(self) -> tuple[str, ...]:
        return tuple(a for a in self.axes if not self.tolerance(a).calibrated)


#: The policy that ships. Every axis uncalibrated, so the only automatic match is
#: an **exact** regime match and everything else routes to discovery. This is not
#: a placeholder to be edited in passing — replacing it means calibrating the
#: axes, and :class:`AxisTolerance` will not let a number in without the run.
UNCALIBRATED_POLICY = MatchPolicy(name="uncalibrated")


class MatchStatus(str, Enum):
    """The outcome of a lookup. Four states, and three of them are not a row."""

    EXACT_REGIME = "exact_regime"  # distance 0 on every axis; safe to apply
    NEAR_REGIME = "near_regime"  # within calibrated tolerances
    UNCALIBRATED = "uncalibrated"  # candidates exist, but no axis is calibrated
    NO_MATCH = "no_match"  # nothing passed the exact gates


class RegimeDistance(BaseModel):
    """Per-axis distances and the L-inf that summarizes them.

    Both halves are kept. The L-inf is what a threshold compares against; the
    per-axis dict is what tells a human *which* axis put the row out of range,
    which is the only actionable half when a lookup misses.
    """

    model_config = ConfigDict(extra="forbid")

    per_axis: dict[str, float]
    linf: float
    limiting_axis: str | None

    @property
    def exact(self) -> bool:
        return self.linf == 0.0

    def render(self) -> str:
        parts = " ".join(
            f"{a}={'inf' if math.isinf(d) else format(d, '.3f')}"
            for a, d in sorted(self.per_axis.items())
        )
        lim = f" (limited by {self.limiting_axis})" if self.limiting_axis else ""
        return f"L-inf {self.linf:.3f}{lim}  [{parts}]"


def regime_distance(a: Regime, b: Regime, policy: MatchPolicy = UNCALIBRATED_POLICY) -> RegimeDistance:
    """Distance between two regimes on the policy's axes, combined with L-inf."""
    per_axis = {axis: AXIS_METRICS[axis](getattr(a, axis), getattr(b, axis)) for axis in policy.axes}
    if not per_axis:
        return RegimeDistance(per_axis={}, linf=0.0, limiting_axis=None)
    limiting = max(per_axis, key=lambda k: per_axis[k])
    # An exact match has no limiting axis. Naming one would read as "this is the
    # axis that nearly failed", which is the opposite of what a 0.0 means.
    return RegimeDistance(
        per_axis=per_axis,
        linf=per_axis[limiting],
        limiting_axis=limiting if per_axis[limiting] > 0 else None,
    )


def _gate(query: RowIdentity, row: RowIdentity, policy: MatchPolicy) -> str:
    """Exact-match gate. Returns ``""`` when the row passes, else why it did not."""
    if query.model != row.model:
        return f"model {row.model!r} != {query.model!r}"
    if query.model_revision != row.model_revision:
        return f"model_revision {row.model_revision} != {query.model_revision}"
    if query.gpu_sku != row.gpu_sku:
        return f"gpu_sku {row.gpu_sku!r} != {query.gpu_sku!r}"
    if policy.match_env:
        ok, why = query.env.compatible_with(row.env)
        if not ok:
            return why
    if policy.match_source_kind and query.regime.source_kind is not row.regime.source_kind:
        return (
            f"source_kind {row.regime.source_kind.value} != {query.regime.source_kind.value}"
            " — a scoreboard result is not evidence about production traffic"
        )
    if policy.match_concurrency and query.regime.concurrency != row.regime.concurrency:
        return f"concurrency {row.regime.concurrency} != {query.regime.concurrency}"
    if query.knobs.keys() != row.knobs.keys():
        return f"knob set {sorted(row.knobs)} != {sorted(query.knobs)}"
    return ""


class Candidate(BaseModel):
    """A row that passed the gates, with how far its regime is from the query."""

    model_config = ConfigDict(extra="forbid")

    row: PlaybookRow
    distance: RegimeDistance


class MatchResult(BaseModel):
    """What a lookup found, and — when it found nothing — exactly why.

    ``rejected`` is not diagnostics-for-later. A miss that cannot say which gate
    it failed sends someone to read the whole playbook by hand, and a miss is the
    common case for a schema this young.
    """

    model_config = ConfigDict(extra="forbid")

    status: MatchStatus
    row: PlaybookRow | None = None
    distance: RegimeDistance | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    rejected: dict[str, str] = Field(default_factory=dict)  # row_id -> why
    policy: str = UNCALIBRATED_POLICY.name
    reason: str = ""

    @property
    def route_to_discovery(self) -> bool:
        """Whether the caller must fall back to conservative discovery mode.

        True for everything that is not a returned row. Deliverable 4 defines
        this handoff and **not** discovery itself: what a caller needs from the
        schema is an unambiguous "I have nothing for you", and a status that is
        sometimes a row and sometimes a suggestion is how a wrong row gets
        applied in a 72-hour window.
        """
        return self.row is None

    def render(self) -> str:
        head = f"{self.status.value}  (policy: {self.policy})"
        if self.row is not None and self.distance is not None:
            return f"{head}\n  {self.row.summary()}\n  {self.distance.render()}"
        lines = [head, f"  {self.reason}"] if self.reason else [head]
        for c in self.candidates:
            lines.append(f"  candidate {c.row.row_id}: {c.distance.render()}")
        for row_id, why in sorted(self.rejected.items()):
            lines.append(f"  rejected {row_id}: {why}")
        return "\n".join(lines)


def _precedence(c: Candidate) -> tuple[float, int, float, float]:
    """Sort key: nearest, then measured-in-envelope, then freshest, then smallest.

    Four terms, in this order and for these reasons:

    1. **Distance.** The workload question comes first; everything else is a
       tie-break among rows that answer it equally well.
    2. **In envelope.** A row measured at a point deliberately sampled *beyond*
       any observed trace (D1 marks these ``/xenv``) is weaker evidence than one
       measured inside it. This sits below distance rather than above it —
       ``todo.md`` had it above — because a nearby extrapolated point was still
       genuinely run, and preferring a 4x-away in-envelope row over it answers
       the wrong question.
    3. **Recency.** Among equals, the most recently verified.
    4. **Smallest claim.** The conservative tie-break from the plan: prefer the
       row claiming **less**, because a wrong row applied inside the live window
       costs more than a missed opportunity.
    """
    verified = c.row.provenance.verified_at
    recency = -(verified or datetime.min.replace(tzinfo=timezone.utc)).timestamp()
    extrapolated = 0 if c.row.identity.regime.in_envelope else 1
    return (c.distance.linf, extrapolated, recency, abs(c.row.delta.throughput_pct))


def lookup(
    playbook: Playbook,
    query: RowIdentity,
    policy: MatchPolicy = UNCALIBRATED_POLICY,
) -> MatchResult:
    """Find the row that applies to ``query``, or say why none does.

    Order of operations, and each step can only ever *reject*:

    1. **Gate** on the exact fields. Model, revision, GPU, environment,
       source_kind, concurrency, and the knob set being asked about.
    2. **Measure** the regime distance for the survivors.
    3. **Decide.** Distance 0 on every axis is an exact regime match and is
       returned. A nonzero distance needs a calibrated tolerance on every axis it
       is nonzero along; without one the result is
       :attr:`MatchStatus.UNCALIBRATED` and the caller goes to discovery.
    4. **Break ties** by :func:`_precedence`.

    Unselectable rows — invalidated, or worked examples — never reach step 2.
    """
    result_rejected: dict[str, str] = {}
    candidates: list[Candidate] = []

    for row in playbook.rows:
        if not row.selectable:
            result_rejected[row.row_id] = (
                "invalidated: " + row.invalidated.reason
                if row.invalidated
                else f"evidence={row.evidence.value}; never selectable"
            )
            continue
        why = _gate(query, row.identity, policy)
        if why:
            result_rejected[row.row_id] = why
            continue
        candidates.append(Candidate(row=row, distance=regime_distance(query.regime, row.identity.regime, policy)))

    if not candidates:
        return MatchResult(
            status=MatchStatus.NO_MATCH,
            rejected=result_rejected,
            policy=policy.name,
            reason="no row passed the exact-match gates",
        )

    candidates.sort(key=_precedence)
    best = candidates[0]

    if best.distance.exact:
        return MatchResult(
            status=MatchStatus.EXACT_REGIME,
            row=best.row,
            distance=best.distance,
            candidates=candidates,
            rejected=result_rejected,
            policy=policy.name,
        )

    # Nonzero distance: every axis it is nonzero along must have a calibrated
    # tolerance, and must be inside it. An uncalibrated axis is not "probably
    # fine" — it is an axis nobody has measured the knob across.
    uncalibrated = [
        a for a, d in best.distance.per_axis.items() if d > 0 and not policy.tolerance(a).calibrated
    ]
    if uncalibrated:
        return MatchResult(
            status=MatchStatus.UNCALIBRATED,
            candidates=candidates,
            rejected=result_rejected,
            policy=policy.name,
            reason=(
                f"nearest row {best.row.row_id} is {best.distance.linf:.3f} away, limited by "
                f"{best.distance.limiting_axis}; no calibrated tolerance for {sorted(uncalibrated)}. "
                "Automatic regime matching is not yet calibrated — routing to discovery."
            ),
        )

    over = [
        (a, d) for a, d in best.distance.per_axis.items() if d > (policy.tolerance(a).max_distance or 0.0)
    ]
    if over:
        axis, dist = max(over, key=lambda t: t[1])
        return MatchResult(
            status=MatchStatus.NO_MATCH,
            candidates=candidates,
            rejected=result_rejected,
            policy=policy.name,
            reason=(
                f"nearest row {best.row.row_id} exceeds its tolerance on {axis}: "
                f"{dist:.3f} > {policy.tolerance(axis).max_distance}"
            ),
        )

    return MatchResult(
        status=MatchStatus.NEAR_REGIME,
        row=best.row,
        distance=best.distance,
        candidates=candidates,
        rejected=result_rejected,
        policy=policy.name,
    )
