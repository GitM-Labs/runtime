"""Playbook row contracts — what a promoted tuning result *is*.

Deliverable 4. One row says:

    (model+revision, GPU SKU, workload regime, knob set, environment)
        -> measured delta + the provenance to re-verify it from scratch

The schema is the contract between Adit's detection and Seojun's apply runtime,
so it ships as **types**, not as a doc that two implementations read differently.

Three rules the types enforce rather than describe:

* **A row cannot exist without provenance.** :class:`Provenance` is required and
  ``extra="forbid"``, and its trace fields are the ones deliverable 1 already
  emits in :class:`~gitm.traffic.schema.TraceMeta`. A tuning claim without the
  raw trace checksum, the drop counts and the repeat data is not defensible, and
  the type is where that stops being a convention.
* **A row is retired by a field, never by a deletion.** :class:`Invalidation`
  carries a reason. A row deleted from a file leaves no record that the claim was
  ever made, which is exactly what a reviewer asks for.
* **Regime is imported, never re-declared.** :class:`~gitm.traffic.regime.Regime`
  is deliverable 1's type. A second copy here would drift within a week, and the
  distance metric in :mod:`gitm.playbook.match` would be measuring two different
  coordinate systems.

**R1, stated in the types:** the shared config-capture schema does not exist yet.
:class:`EnvCapture` below is the *named subset* this deliverable needs, marked
``pending-adit``. When Adit's types land they are **imported verbatim** and this
class is deleted — there is no translation layer, per the brief, because two
schemas that translate into each other are two schemas that drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gitm.traffic.regime import Regime
from gitm.traffic.results import BenchRun

#: Schema identity, in the style of ``gitm.bench.manifest.SCHEMA``. Bump on any
#: field change that is not purely additive.
SCHEMA = "gitm.playbook.row/v1"

#: Marker for every field that is waiting on the shared config-capture schema
#: (risk R1). Grep-able on purpose: when Adit's types land, this string is the
#: work list.
PENDING_ADIT = "pending-adit"

#: Knob-name fragments that make a row's delta depend on prefix-cache reuse.
#: Substring match, deliberately: engines rename these flags between versions,
#: and a list of exact names would go stale silently while a substring stays
#: right for ``enable_prefix_caching``, ``prefix_caching``, and the next
#: spelling after that.
PREFIX_CACHE_KNOB_MARKERS = ("prefix_cach", "prefix-cach", "kv_reuse")


class Evidence(str, Enum):
    """Whether a row's *delta* was measured or is an illustration.

    The worked examples ship in the same file format as real rows, so a field
    has to separate them. Without it, an example row copied into a live playbook
    is indistinguishable from a promoted one — and the whole point of the schema
    is that a row's standing is readable from the row.
    """

    MEASURED = "measured"  # produced by a real A/B under the promotion rule
    ILLUSTRATIVE = "illustrative"  # a worked example; never selectable


class EnvCapture(BaseModel):
    """The environment fields a playbook row must pin. ``pending-adit`` (R1).

    Deliberately thin. These are the values that, if they differ between the run
    that produced a row and the box about to apply it, make the row's number
    meaningless. Everything else Adit's capture records is welcome and arrives by
    *import*, not by being re-typed here.

    ``extra="allow"`` is the one place in this module that permits unknown keys:
    a capture record from a newer engine must round-trip through a playbook file
    without being silently truncated. Comparison uses :meth:`compatible_with`,
    which reads the named fields only — an unknown extra key never changes a
    match decision, it just survives the trip.
    """

    model_config = ConfigDict(extra="allow")

    schema_id: str = f"{PENDING_ADIT}/env-capture"
    engine: str  # e.g. "vllm"
    engine_version: str  # exact, e.g. "0.11.0"
    driver_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None

    def compatible_with(self, other: EnvCapture) -> tuple[bool, str]:
        """Exact on engine and engine version. Returns ``(ok, why_not)``.

        An engine version bump is the single most common way a knob's effect
        changes without anything in the workload changing — scheduler rewrites
        ship in point releases. So the default policy is **exact**, and anything
        looser has to be written down as a policy rather than assumed by a
        comparison that used ``startswith``.
        """
        if self.engine != other.engine:
            return False, f"engine {self.engine!r} != {other.engine!r}"
        if self.engine_version != other.engine_version:
            return False, f"engine_version {self.engine_version} != {other.engine_version}"
        return True, ""


class RowIdentity(BaseModel):
    """The lookup key. Everything here is matched, nothing here is a result.

    Split deliberately into fields matched **exactly** and one field matched by
    **distance** (:attr:`regime`). The split is the design: model, GPU and
    environment are categorical — "nearly an H100" is not a thing — while the
    workload is continuous and live traffic never lands on a measured point.
    :mod:`gitm.playbook.match` implements exactly that split and nothing else.
    """

    # ``protected_namespaces=()`` because the fields really are called ``model``
    # and ``model_revision`` — that is the vocabulary everyone else uses, and
    # renaming them to dodge a pydantic warning would make the schema wrong in
    # the one place it is read by hand.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str  # e.g. "Qwen/Qwen3.6-35B-A3B-FP8"
    model_revision: str  # exact commit/revision, e.g. "95a723d0"
    gpu_sku: str  # e.g. "NVIDIA H100 80GB"
    env: EnvCapture
    regime: Regime  # deliverable 1's type, imported
    #: ``bool`` first in the union on purpose: pydantic coerces left to right,
    #: and ``True`` arriving as ``1.0`` turns a boolean knob into a number in
    #: every rendering of the row.
    knobs: dict[str, bool | int | float | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _knobs_are_not_empty(self) -> RowIdentity:
        if not self.knobs:
            raise ValueError("a playbook row with no knob set says nothing; knobs is required")
        return self

    def exact_key(self) -> tuple[str, str, str]:
        """The part of the key compared by equality, for grouping and printing."""
        return (self.model, self.model_revision, self.gpu_sku)


class MeasuredDelta(BaseModel):
    """What the knob did, relative to the baseline arm of the same experiment.

    Throughput **and** latency percentiles, always both. Deliverable 2's
    criterion 3 is that a knob which gains throughput while blowing out latency
    percentiles fails promotion — a schema that let a row carry throughput alone
    would make that criterion unenforceable at the point it matters.

    Signs are stated once, here, because a sign error in a playbook is a knob
    applied backwards: **throughput is percent, higher is better; the latency
    fields are milliseconds of change, negative is better.**
    """

    model_config = ConfigDict(extra="forbid")

    throughput_pct: float  # % change vs baseline; + is faster
    ttft_p99_ms: float  # ms change vs baseline; - is better
    itl_p99_ms: float  # ms change vs baseline; - is better
    repeats: int = Field(ge=1)  # per arm, interleaved A/B/A/B per D2
    #: Bootstrap 95 % CI on the median difference in throughput, per D2-1. The
    #: promotion rule owns the predicate; the row carries the numbers it needs.
    throughput_ci95_pct: tuple[float, float] | None = None
    #: D2 criterion 3's verdict, **stored rather than recomputed**: did this knob
    #: blow out a latency percentile? The promotion gate and the live-window
    #: revert trigger must read the same value, and two callers each re-deriving
    #: "blowout" from the raw percentiles is two predicates that drift. D2 owns
    #: the predicate and writes the answer here; ``None`` means D2 has not run,
    #: which is every row today because D2 does not exist yet.
    latency_blowout: bool | None = None

    @model_validator(mode="after")
    def _repeats_are_plural(self) -> MeasuredDelta:
        if self.repeats < 2:
            raise ValueError(
                f"repeats={self.repeats}: a single run has no variance and cannot be "
                "promoted under D2. Record it as evidence=illustrative if it is an example."
            )
        return self


class Provenance(BaseModel):
    """Everything needed to re-run this row from scratch and get it again.

    The trace fields are deliverable 1's :class:`~gitm.traffic.schema.TraceMeta`
    verbatim in meaning: ``trace_sha256`` pins the raw bytes, ``trace_drops``
    says what was rejected getting to them. A row whose trace no longer hashes to
    ``trace_sha256`` is not the same experiment, and the field is what lets
    anyone find that out.
    """

    model_config = ConfigDict(extra="forbid")

    trace_source: str  # adapter name, e.g. "mooncake"
    trace_sha256: str  # of the raw file, via gitm.bench.manifest
    trace_drops: dict[str, int] = Field(default_factory=dict)
    regime_label: str  # Regime.label() as it stood when measured
    #: Where the per-repeat raw numbers live. D2 requires them; a summarized
    #: delta with the repeats thrown away cannot be re-analyzed under a different
    #: variance rule, and D2-1's thresholds are explicitly not settled yet.
    repeat_raw_data: list[str] = Field(default_factory=list)
    promotion_rule: str = f"{PENDING_ADIT}/promotion-rule"  # D2 doc + version
    config_capture: str = PENDING_ADIT  # R1
    verified_at: datetime | None = None  # last time the row was re-measured

    #: The conditions the replay actually ran under, from D1's ``ReplayPlan``.
    #: ``replay_chunk_hash_size`` is here because it is the deliverable-1 finding
    #: with the worst failure mode: at vLLM's default of 16 against Mooncake's
    #: 512-token blocks every prompt is 32x short while every count in the result
    #: still reads correctly. A row that does not record it cannot be checked.
    replay_chunk_hash_size: int | None = None
    replay_self_timed: bool | None = None
    #: True when D1 *synthesized* prefix blocks because the source had none —
    #: BurstGPT. See :attr:`PlaybookRow.delta_is_floor`: a prefix-cache knob
    #: measured on a synthesized-prefix trace saw no sharing the source never
    #: had, so its delta is a lower bound and may not be quoted as a gain.
    prefix_synthesized: bool = False

    @model_validator(mode="after")
    def _label_matches_nothing_yet(self) -> Provenance:
        if not self.trace_sha256:
            raise ValueError("trace_sha256 is required — a row that cannot be traced to bytes")
        return self

    @classmethod
    def from_bench_run(cls, run: BenchRun, **overrides) -> Provenance:
        """Build the trace half of a row's provenance from a seam-3 record.

        Every field here already exists on :class:`~gitm.traffic.results.BenchRun`
        because that is what seam 3 was for; copying them by hand at each call
        site is how the checksum and the label that ran drift apart. What is
        *not* here is the config-capture half — ``config_capture`` comes across
        still marked ``pending-adit`` (R1), which is the truthful value.

        ``overrides`` covers the fields no single run knows: ``repeat_raw_data``,
        ``promotion_rule``, ``verified_at``.
        """
        return cls(
            trace_source=run.source.source,
            trace_sha256=run.source.sha256,
            trace_drops=dict(run.source.drops),
            regime_label=run.regime_label,
            replay_chunk_hash_size=run.chunk_hash_size,
            replay_self_timed=run.self_timed,
            prefix_synthesized=run.prefix_synthesized,
            config_capture=run.config_capture,
            **overrides,
        )


class Invalidation(BaseModel):
    """Why a row stopped being usable. A field, never a deletion."""

    model_config = ConfigDict(extra="forbid")

    reason: str  # free text, e.g. "vLLM 0.11 -> 0.12 scheduler rewrite"
    at: datetime
    by: str | None = None


class PlaybookRow(BaseModel):
    """One promoted tuning result.

    Rows enter **only** through deliverable 2's promotion rule. That is not
    enforceable in a type — a type cannot see how a number was produced — so what
    the type does instead is refuse to hold a row that *could not* have come
    through it: no knobs, no provenance, a single repeat, or a delta missing its
    latency percentiles all fail construction.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: str = SCHEMA
    row_id: str
    identity: RowIdentity
    delta: MeasuredDelta
    provenance: Provenance
    evidence: Evidence = Evidence.MEASURED
    invalidated: Invalidation | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def delta_is_floor(self) -> bool:
        """True when this row's delta is a **lower bound**, not a measurement.

        D1 synthesizes prefix blocks for a source that has none (BurstGPT), with
        every request's ids unique so that lengths hold and **no prefix sharing
        is invented**. A prefix-cache knob measured on such a trace therefore saw
        the *least* reuse the real traffic could have had. The number is a floor:
        useful as "at least this much", never quotable as the gain.

        A property rather than a stored flag because it is a function of two
        fields already on the row, and a stored copy is a third place to get it
        wrong.
        """
        if not self.provenance.prefix_synthesized:
            return False
        return any(
            m in k.lower() for k in self.identity.knobs for m in PREFIX_CACHE_KNOB_MARKERS
        )

    @property
    def selectable(self) -> bool:
        """Whether a lookup may return this row at all.

        Two ways to be unselectable, and both are states rather than absences:
        the row was invalidated, or it is a worked example that was never
        measured. An example row in a live file is the failure mode this guards.
        """
        return self.invalidated is None and self.evidence is Evidence.MEASURED

    def summary(self) -> str:
        k = ", ".join(f"{k}={v}" for k, v in sorted(self.identity.knobs.items()))
        mark = "" if self.selectable else f"  [{'invalid' if self.invalidated else 'example'}]"
        mark += "  [FLOOR: prefixes synthesized]" if self.delta_is_floor else ""
        return (
            f"{self.row_id}: {k} on {self.identity.gpu_sku} / "
            f"{self.identity.regime.label()} -> "
            f"tput {self.delta.throughput_pct:+.1f}%, "
            f"ttft p99 {self.delta.ttft_p99_ms:+.1f}ms, "
            f"itl p99 {self.delta.itl_p99_ms:+.1f}ms{mark}"
        )


class Playbook(BaseModel):
    """A file of rows. Thin on purpose — the matching lives in ``match.py``."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str = SCHEMA
    rows: list[PlaybookRow] = Field(default_factory=list)

    def selectable(self) -> list[PlaybookRow]:
        return [r for r in self.rows if r.selectable]


#: The throughput a row's ``throughput_pct`` is a percentage of. ``bench serve``
#: reports three (request, output-token, total-token); a row that does not say
#: which one it means is a row two readers compare differently. Output tokens per
#: second is the serving number, and it is the one deliverable 2's criterion 3
#: pairs against the latency percentiles.
THROUGHPUT_METRIC = "output_throughput"

#: Metrics both arms must carry before a delta can be computed. TTFT and ITL are
#: here because deliverable 2's criterion 3 is unenforceable without them — the
#: schema already refuses a throughput-only :class:`MeasuredDelta`, and this
#: refuses to *build* one rather than failing later with a pydantic error.
REQUIRED_METRICS = (THROUGHPUT_METRIC, "p99_ttft_ms", "p99_itl_ms")


def row_from_runs(
    row_id: str,
    baseline: Sequence[BenchRun],
    treatment: Sequence[BenchRun],
    *,
    model: str,
    model_revision: str,
    gpu_sku: str,
    env: EnvCapture,
    knobs: dict[str, bool | int | float | str],
    repeat_raw_data: Sequence[str] = (),
    verified_at: datetime | None = None,
    notes: Sequence[str] = (),
) -> PlaybookRow:
    """Build one row from the two arms of a real experiment.

    This is the last mile deliverable 4 was missing: seam 3 turns a
    ``bench serve`` result into a :class:`~gitm.traffic.results.BenchRun`, and
    this turns a *pair* of arms into a row. A ``BenchRun`` is one arm; a row is a
    **difference between two**, which is why nothing before this could populate a
    row end to end no matter how complete the join was.

    What it enforces, because these are deliverable 4's business:

    * **Both arms ran the same workload** — same trace bytes, same regime label.
      A delta across two different traces measures the traces.
    * **Every run is** :attr:`~gitm.traffic.results.BenchRun.promotable` — it
      reconciles against its trace and had no failures. Below that bar there is
      nothing to take a difference of.
    * **Equal repeat counts**, because deliverable 2 interleaves A/B/A/B. Unequal
      arms mean the interleave broke, and the shorter arm is the one that was
      cut short.
    * **Medians, never means**, per deliverable 2's criterion 2.

    What it does **not** enforce, and why:

    * **"Same config minus exactly one knob."** That is deliverable 2's, and it
      is enforced by diffing two config-capture records — which do not exist
      (R1). Until they do, the caller asserts it and the row says so in a note
      that disappears on its own the moment ``config_capture`` is real.
    * ``throughput_ci95_pct`` and ``latency_blowout`` stay ``None``. Deliverable
      2 owns the variance rule and the blowout predicate; inventing either here
      would be the same mistake :class:`~gitm.playbook.match.AxisTolerance`
      refuses to make with a distance threshold.
    """
    arms = list(baseline) + list(treatment)
    if not baseline or not treatment:
        raise ValueError("both arms are required — a row is a difference between two")
    if len(baseline) != len(treatment):
        raise ValueError(
            f"unequal arms: {len(baseline)} baseline vs {len(treatment)} treatment. "
            "D2 interleaves A/B/A/B, so unequal counts mean the interleave broke."
        )

    unpromotable = [r.summary() for r in arms if not r.promotable]
    if unpromotable:
        raise ValueError(
            "a row cannot be built from runs that are not promotable "
            "(reconciled, no failures): " + "; ".join(unpromotable)
        )

    traces = {r.source.sha256 for r in arms}
    labels = {r.regime_label for r in arms}
    if len(traces) != 1 or len(labels) != 1:
        raise ValueError(
            f"the arms did not run the same workload: {len(traces)} distinct trace "
            f"checksum(s), regime labels {sorted(labels)} — a delta across two "
            "workloads measures the workloads"
        )

    for r in arms:
        missing = [m for m in REQUIRED_METRICS if r.metrics.get(m) is None]
        if missing:
            raise ValueError(f"{r.summary()}: result is missing {missing}")

    def med(runs: Sequence[BenchRun], metric: str) -> float:
        return median(float(r.metrics[metric]) for r in runs)

    base_tput = med(baseline, THROUGHPUT_METRIC)
    if base_tput == 0:
        raise ValueError(f"baseline {THROUGHPUT_METRIC} is 0 — no percentage to take")

    notes = list(notes)
    prov = Provenance.from_bench_run(
        treatment[0],
        repeat_raw_data=list(repeat_raw_data),
        verified_at=verified_at,
    )
    if prov.config_capture == PENDING_ADIT:
        notes.append(
            "R1: no config-capture record exists, so 'same config minus exactly one knob' "
            "is asserted by the caller here, not diffed. D2 owns that check."
        )

    return PlaybookRow(
        row_id=row_id,
        identity=RowIdentity(
            model=model, model_revision=model_revision, gpu_sku=gpu_sku, env=env,
            regime=treatment[0].regime, knobs=knobs,
        ),
        delta=MeasuredDelta(
            throughput_pct=(med(treatment, THROUGHPUT_METRIC) / base_tput - 1.0) * 100.0,
            ttft_p99_ms=med(treatment, "p99_ttft_ms") - med(baseline, "p99_ttft_ms"),
            itl_p99_ms=med(treatment, "p99_itl_ms") - med(baseline, "p99_itl_ms"),
            repeats=len(treatment),
        ),
        provenance=prov,
        evidence=Evidence.MEASURED,
        notes=notes,
    )
