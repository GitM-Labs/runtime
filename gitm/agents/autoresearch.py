"""Autoresearch — propose non-catalog levers within the attributed bottleneck class.

The curated library (``library.yaml``) is finite. Autoresearch is the agentic
half of the README's "select from a library of known optimizations *and* run
agentic search for novel ones": it proposes real vLLM config knobs *outside*
that catalog, constrained to the bottleneck class the attribution layer
identified (idle / memory / compute).

Every proposal is then routed through the exact same path as a catalog lever:

1. the selection gate — :func:`gitm.agents.policy.select_interventions` — which
   pre-filters on the safety tier and qualification commit, then ranks the
   survivors by counterfactual replay (:func:`gitm.optimizer.replay.predict_delta`);
2. the rollback-gated live apply — :func:`gitm.optimizer.apply.apply_intervention` —
   which snapshots, applies, measures, and keeps only on a measured win.

A proposal the gate rejects is recorded and dropped; one that applies but does
not measurably help is rolled back. Autoresearch is a *candidate source*, not a
new trust path — nothing it proposes can bypass the gate or be kept without a
measured win.

The proposed knobs are real, current vLLM arguments (verified against
docs.vllm.ai); their expected deltas, however, are unproven estimates. The
``source`` field says so, and only the measured A/B keeps or discards them.

v0 classifies the bottleneck from coarse trace telemetry (:func:`classify_bottleneck`)
and repoints the search at the largest-residual op. Candidates come from one of
two sources behind the :class:`Proposer` seam: the static per-class table
(:func:`propose`) or :class:`GenerativeProposer`, which searches a workload's knob
surface (supplied by a :class:`KnobSource` — vLLM's ``EngineArgs`` by default) at
a small value grid per knob. The seam is workload-agnostic: a new workload plugs
in a KnobSource rather than a per-workload table. The loop runs the generative
proposer with the table as a fallback. Later versions learn an effect model from
realized deltas and sample the knob space stochastically.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.library import load_library
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.apply import Applicator, apply_intervention
from gitm.optimizer.monitor import Residuals, _serialized_fraction
from gitm.tracer.schema import Trace

if TYPE_CHECKING:
    from gitm.optimizer.preconditions import GateContext

#: The module's public surface: the bottleneck vocabulary, the ``KnobSource`` and
#: ``Proposer`` seams (so any workload can plug in), and the entry points. Helpers
#: prefixed with ``_`` (the fallback table, value grids, EngineArgs introspection)
#: are internal.
__all__ = [
    "BOTTLENECK_CLASSES",
    "IDLE_STALL",
    "MEMORY_BOUND",
    "COMPUTE_BOUND",
    "classify_bottleneck",
    "ResidualTarget",
    "largest_residual",
    "Knob",
    "KnobSource",
    "VLLMKnobSource",
    "Proposer",
    "TableProposer",
    "GenerativeProposer",
    "EngineArgsProposer",
    "StochasticProposer",
    "FallbackProposer",
    "propose",
    "AutoresearchResult",
    "AutoresearchRun",
    "autoresearch",
    "autoresearch_v0",
]

# --- bottleneck classification ----------------------------------------------
#
# The attribution vocabulary autoresearch searches within. Nothing upstream
# emits these labels yet, so v0 derives them from coarse trace telemetry. These
# are deliberately simple heuristics, not a tuned model — the thresholds only
# have to route the search into the right candidate table; the rollback gate is
# what actually protects a wrong route (a bad proposal is measured and reverted).

#: The bottleneck classes autoresearch searches within — the single source of
#: truth shared by classify_bottleneck (the producer), the keyword-affinity map,
#: and the fallback table, so the three can't drift (guarded by a test). These are
#: workload-agnostic GPU-execution categories, not a per-workload vocabulary.
IDLE_STALL = "idle_stall"
MEMORY_BOUND = "memory_bound"
COMPUTE_BOUND = "compute_bound"
BOTTLENECK_CLASSES = (IDLE_STALL, MEMORY_BOUND, COMPUTE_BOUND)

#: Serialized-concurrency fraction above this ⇒ kernels ran back-to-back on one
#: stream instead of overlapping: scheduling gaps / launch-bound idle time.
_SC_THRESHOLD = 0.5
#: memcpy share of GPU operations above this ⇒ data movement dominates.
_MEMCPY_THRESHOLD = 0.25


def classify_bottleneck(trace: Trace) -> str:
    """Map a captured trace to one of ``idle_stall`` / ``memory_bound`` / ``compute_bound``.

    v0 heuristic on two signals read straight from the trace: the
    serialized-concurrency fraction (poor kernel overlap ⇒ idle/scheduling gaps)
    and the memcpy fraction (data movement dominating ⇒ memory bound). Each is
    scored against its threshold; the stronger signal wins, and if neither
    crosses its threshold the workload is treated as compute bound. An empty
    trace has no stall/movement signal, so it defaults to ``compute_bound``.
    """
    kernels = trace.kernels()
    if not kernels:
        return COMPUTE_BOUND

    memcpys = [e for e in trace.events if e.kind == "memcpy"]
    sc = _serialized_fraction(kernels)
    kernel_ns = sum(max(0, k.end_ns - k.start_ns) for k in kernels)
    memcpy_ns = sum(max(0, e.end_ns - e.start_ns) for e in memcpys)
    gpu_op_ns = kernel_ns + memcpy_ns
    memcpy_frac = memcpy_ns / gpu_op_ns if gpu_op_ns else 0.0

    sc_score = sc / _SC_THRESHOLD
    mem_score = memcpy_frac / _MEMCPY_THRESHOLD
    if max(sc_score, mem_score) < 1.0:
        return COMPUTE_BOUND
    return IDLE_STALL if sc_score >= mem_score else MEMORY_BOUND


# --- residual targeting -----------------------------------------------------
#
# "Repoint at the largest residual": instead of aiming the search at the whole
# trace, aim it at the single op whose kernels run furthest *over* the predicted
# ceiling — the biggest gap the attribution layer found. This is the gap residual
# ``r_kt = (t_obs - t_pred)/t_pred`` from gitm.optimizer.monitor.residuals(), NOT
# the within-kernel-type jitter that measure_trace() computes, and NOT the Granger
# p-value rank (significance, not magnitude).


@dataclass
class ResidualTarget:
    """The op with the largest kernel-time gap vs its predicted ceiling."""

    op: str
    residual: float  # mean r_kt over the op's kernels (fraction over the ceiling)
    n_kernels: int


def largest_residual(res: Residuals) -> ResidualTarget | None:
    """The op whose kernels run furthest over the predicted ceiling.

    Aggregates the per-kernel gap residual by op (mean ``r_kt``) and returns the
    op with the largest *positive* mean — the biggest bottleneck, not the
    jitteriest op. Returns ``None`` when there is no residual data or nothing runs
    over its ceiling (all means ≤ 0).
    """
    if not res.per_kernel:
        return None
    by_op: dict[str, list[float]] = {}
    for kr in res.per_kernel:
        by_op.setdefault(kr.op, []).append(kr.r_kt)
    op, values = max(by_op.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    return ResidualTarget(op=op, residual=mean, n_kernels=len(values))


def _op_present(trace: Trace, op: str) -> bool:
    """True if ``op`` names (a substring of) a real kernel in the trace.

    Guards the ``applies_to_kernels`` tagging: the residual op label comes from
    the predicted graph, so only tag a proposal with it when it actually matches
    a captured kernel name — otherwise ``predict_delta`` coverage would be 0 and
    the proposal would rank as worthless rather than untargeted.
    """
    return any(op in k.name for k in trace.kernels())


# --- candidate table --------------------------------------------------------
#
# Per-bottleneck candidate perturbations: (knob, value, one-line rationale).
# Every knob is a real, current vLLM argument (docs.vllm.ai) that is NOT in the
# curated library.yaml — autoresearch proposes *outside* the catalog. The
# rationales are plausibility arguments, not measured claims.
_RULES: dict[str, list[tuple[str, object, str]]] = {
    "idle_stall": [
        ("max_num_partial_prefills", 4,
         "raise partial-prefill concurrency so prefill overlaps decode instead of stalling it"),
        ("long_prefill_token_threshold", 2048,
         "lower the long-prefill threshold so big prompts chunk and interleave, closing decode gaps"),
    ],
    "memory_bound": [
        ("cpu_offload_gb", 4,
         "offload cold weights to host RAM to free HBM for a larger KV cache"),
        ("preemption_mode", "swap",
         "swap preempted KV blocks to host instead of recomputing them under memory pressure"),
    ],
    "compute_bound": [
        ("compilation_config", 3,
         "raise torch.compile to level 3 for kernel fusion + piecewise CUDA graphs"),
    ],
}


@dataclass
class AutoresearchResult:
    spec: InterventionSpec
    bottleneck_class: str
    predicted_delta: float
    applicable: bool
    rejected_reason: str | None
    measured_delta: float | None
    rolled_back: bool
    target_op: str | None = None  # the largest-residual op this proposal aimed at


@dataclass
class AutoresearchRun:
    """One end-to-end autoresearch pass: the classified bottleneck + its results."""

    bottleneck_class: str
    results: list[AutoresearchResult] = field(default_factory=list)
    target: ResidualTarget | None = None  # the largest-residual op the search aimed at


#: The honest, unproven delta band every candidate carries until the measured A/B
#: replaces it. One place to tune what "proposed, not measured" means.
_DELTA_MEAN, _DELTA_LO, _DELTA_HI = 0.05, 0.0, 0.15


def _candidate_spec(
    *,
    name: str,
    summary: str,
    knob: str,
    value: object,
    applies_to_kernels: list[str],
    bottleneck_class: str,
    workload: str,
    source: str,
) -> InterventionSpec:
    """Build a candidate spec with the fields every proposer forces.

    The single place for the honest-but-unproven delta band, the workload
    applicability, and the moderate/rollback-gated safety posture — so the table
    and generative proposers can't drift apart on what a *candidate* is. Only the
    caller-varying parts (name, summary, knob/value, source) are parameters.
    """
    return InterventionSpec(
        name=name,
        summary=summary,
        knob=knob,
        value=value,  # int | float | str | bool
        applies_to_kernels=applies_to_kernels,
        # Proposed, not measured: an honest, modest range. The measured A/B is
        # what turns this into a real number.
        expected_delta_mean=_DELTA_MEAN,
        expected_delta_lo=_DELTA_LO,
        expected_delta_hi=_DELTA_HI,
        source=source,
        applicability=Applicability(workloads=[workload], other=f"targets {bottleneck_class}"),
        # Unproven ⇒ never high-risk (topology/weights changes stay in the
        # reviewed catalog). Moderate + the rollback gate is the whole safety
        # story for a candidate.
        safety=SafetyGate(
            tier="moderate",
            notes="autoresearch candidate — kept only on a measured, rollback-gated win.",
        ),
    )


def propose(bottleneck_class: str, *, target_op: str | None = None) -> list[InterventionSpec]:
    """Emit candidate specs for a bottleneck class (empty if the class is unknown).

    The static ``_RULES`` table — the offline fallback. When ``target_op`` is
    given, each proposal is scoped to that op via ``applies_to_kernels`` so the
    ranking gate (``predict_delta``) weights it by that op's share of trace time —
    this is how "repoint at the largest residual" reaches the selection. The op is
    the caller's job to validate against the trace (see :func:`_op_present`); an
    off-trace op would zero the coverage.
    """
    applies = [target_op] if target_op else []
    aim = f" (targeting {target_op})" if target_op else ""
    return [
        _candidate_spec(
            name=f"autoresearch:{bottleneck_class}:{knob}",
            summary=why + aim,
            knob=knob,
            value=value,
            applies_to_kernels=applies,
            bottleneck_class=bottleneck_class,
            workload="vllm-decode",
            source="autoresearch-v0 (proposed knob, not catalog; verified real vLLM arg)",
        )
        for knob, value, why in _RULES.get(bottleneck_class, [])
    ]


# --- proposal sources (the "Proposer" seam) ---------------------------------
#
# ``propose`` above is the static per-class table. ``GenerativeProposer`` is the
# generative counterpart: instead of a frozen list it searches a workload's knob
# surface — supplied by a ``KnobSource`` — keeping the knobs affine to the
# attributed bottleneck class and trying a small value grid per knob.
# ``VLLMKnobSource`` (introspect ``EngineArgs``) is one source; another workload
# plugs in by yielding its own knobs, so the mechanism is workload-agnostic with
# no ``{workload: knobs}`` table. Both proposers return ``list[InterventionSpec]``
# and feed the exact same selection + rollback gate — a Proposer is a candidate
# *source*, not a new trust path. Nothing here can propose a knob outside the
# workload's real surface, or one that duplicates the curated library.


class Proposer(Protocol):
    """A source of candidate specs for a bottleneck class. The gate is source-agnostic."""

    def propose(
        self, bottleneck_class: str, *, target_op: str | None = None
    ) -> list[InterventionSpec]: ...


class TableProposer:
    """The static ``_RULES`` table (see :func:`propose`) as a Proposer.

    The offline default and the fallback under :class:`FallbackProposer` when the
    generative proposer has nothing to offer for a class.
    """

    def propose(
        self, bottleneck_class: str, *, target_op: str | None = None
    ) -> list[InterventionSpec]:
        return propose(bottleneck_class, target_op=target_op)


@dataclass(frozen=True)
class Knob:
    """A workload config knob and how to search its value.

    ``grid`` is an explicit set of search points, used where a derived grid would
    be nonsensical (e.g. token thresholds); when empty, the grid is derived from
    ``kind``/``default``. ``classes`` optionally tags which bottleneck classes the
    knob is affine to — a :class:`KnobSource` can declare this for any class
    vocabulary, so affinity need not rely on the vLLM-flavoured keyword heuristic.
    """

    name: str
    kind: str  # "int" | "float" | "bool" | "enum" | "str"
    default: object = None
    choices: tuple = ()
    grid: tuple = ()
    classes: tuple = ()  # bottleneck classes this knob is affine to (optional)


#: Frozen fallback catalog: real, current vLLM EngineArgs (docs.vllm.ai) that are
#: NOT in library.yaml. Used when vLLM can't be imported (air-gapped operator, no
#: GPU stack) so the generative path still runs offline and deterministically.
_FALLBACK_KNOBS: tuple[Knob, ...] = (
    Knob("max_num_partial_prefills", "int", default=1),
    Knob("long_prefill_token_threshold", "int", default=0, grid=(2048, 4096)),
    Knob("max_long_partial_prefills", "int", default=1),
    Knob("cpu_offload_gb", "int", default=0),
    Knob("preemption_mode", "enum", default="recompute", choices=("recompute", "swap")),
    Knob("compilation_config", "int", default=0, grid=(2, 3)),
)

#: Which knobs to search for each bottleneck class, matched against the knob NAME
#: (substring). Deliberately a keyword heuristic, and honest about being one: it
#: biases the search toward the attributed class; the gate does the proving.
_CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "idle_stall": ("prefill", "partial", "schedul", "chunk", "overlap"),
    "memory_bound": ("cache", "swap", "offload", "block", "gpu_memory", "kv", "preempt", "cpu"),
    "compute_bound": ("compil", "cudagraph", "cuda_graph", "graph", "quant", "fus", "eager"),
}


def _affine(knob: Knob, bottleneck_class: str, keywords: tuple[str, ...]) -> bool:
    """Is ``knob`` worth searching for this class?

    An explicit ``Knob.classes`` tag wins (a source can declare affinity for any
    class vocabulary); otherwise fall back to a keyword-substring match on the
    name — the honest, vLLM-flavoured heuristic for sources that don't self-tag.
    """
    if knob.classes:
        return bottleneck_class in knob.classes
    lname = knob.name.lower()
    return any(k in lname for k in keywords)


#: Multipliers applied to a numeric knob's default to derive its search points.
_GRID_MULTIPLIERS = (0.5, 2.0, 4.0)


def _value_grid(knob: Knob) -> list[object]:
    """A small set of candidate values to search for ``knob`` (excludes the default).

    An explicit ``grid`` wins; otherwise the grid is derived from the kind: flip a
    bool, the other members of an enum, or a ½×/2×/4× ladder for a number. This is
    what turns the search from a single-value lookup into an actual value search.
    """
    if knob.grid:
        return [v for v in knob.grid if v != knob.default]
    if knob.kind == "bool":
        return [not bool(knob.default)]
    if knob.kind == "enum":
        return [c for c in knob.choices if c != knob.default]
    if knob.kind in ("int", "float"):
        d = knob.default
        base = d if isinstance(d, int | float) and d not in (0, False) else 1
        raw = [base * m for m in _GRID_MULTIPLIERS]
        if knob.kind == "int":
            vals = sorted({int(round(x)) for x in raw if round(x) >= 1})
        else:
            vals = sorted({round(x, 3) for x in raw if x > 0})
        return [v for v in vals if v != d]
    return []  # unknown / free-form (str): nothing safe to search


def _annotation_kind(annotation: object) -> str:
    """Coarsely map a dataclass-field annotation to a value-grid kind (best-effort)."""
    text = str(annotation).lower()
    if "bool" in text:
        return "bool"
    if "int" in text:
        return "int"
    if "float" in text:
        return "float"
    return "str"


def _field_kind_and_choices(annotation: object) -> tuple[str, tuple]:
    """(kind, choices) for a field annotation.

    A ``Literal[...]`` becomes an enum with its members as choices, so those knobs
    become searchable (a value grid = the other members). Anything else falls back
    to the coarse string match with no choices. Best-effort: vLLM's stringised
    annotations (``from __future__ import annotations``) won't resolve here, so
    Literal extraction only fires when the annotation is a real typing object.
    """
    try:
        import typing

        if typing.get_origin(annotation) is typing.Literal:
            return "enum", tuple(typing.get_args(annotation))
    except Exception:
        pass
    return _annotation_kind(annotation), ()


#: EngineArgs field-name fragments that are never runtime *performance* knobs —
#: model identity, I/O paths, logging, RNG. Mutating them wouldn't close a
#: bottleneck (and could be nonsensical), so the introspected surface excludes
#: them even though they're typed int/bool. vLLM-specific, so applied only here;
#: the curated fallback catalog is authoritative and left untouched.
_NON_TUNABLE_HINTS = (
    "model",
    "tokenizer",
    "seed",
    "log",
    "name",
    "path",
    "dir",
    "revision",
    "trust_remote_code",
    "config_format",
    "download",
)


def _is_tunable(field_name: str) -> bool:
    """False for EngineArgs fields that aren't runtime performance knobs."""
    lname = field_name.lower()
    return not any(h in lname for h in _NON_TUNABLE_HINTS)


def _engine_arg_knobs() -> list[Knob]:
    """Enumerate real vLLM EngineArgs, or fall back to the frozen catalog.

    Best-effort introspection: when vLLM is importable, each tunable dataclass
    field is a candidate knob (typed as best we can from its annotation, with
    ``Literal`` fields exposed as searchable enums). Non-performance fields (model
    identity, paths, logging, RNG — see :data:`_NON_TUNABLE_HINTS`) are skipped.
    When vLLM isn't importable (no GPU stack / air-gapped), the frozen catalog
    keeps the generative path working offline.
    """
    try:
        import dataclasses

        from vllm import EngineArgs  # type: ignore
    except Exception:
        return list(_FALLBACK_KNOBS)

    knobs: list[Knob] = []
    for f in dataclasses.fields(EngineArgs):
        if not _is_tunable(f.name):
            continue
        default = None if f.default is dataclasses.MISSING else f.default
        kind, choices = _field_kind_and_choices(f.type)
        knobs.append(Knob(name=f.name, kind=kind, default=default, choices=choices))
    return knobs


class KnobSource(Protocol):
    """Yields the knob surface to search — a workload's config namespace.

    vLLM's is :class:`VLLMKnobSource` (introspect ``EngineArgs``). Another workload
    plugs in by yielding its own ``Knob`` list; there is deliberately no
    ``{workload: knobs}`` table — versatility comes from the source, not a map.
    """

    def knobs(self) -> list[Knob]: ...


class VLLMKnobSource:
    """The real vLLM ``EngineArgs`` surface (frozen fallback when vLLM is absent)."""

    def knobs(self) -> list[Knob]:
        return _engine_arg_knobs()


@dataclass(frozen=True)
class _ListKnobSource:
    """A fixed knob list as a KnobSource (the ``knobs=`` convenience and tests)."""

    _knobs: tuple[Knob, ...]

    def knobs(self) -> list[Knob]:
        return list(self._knobs)


class _ProposerBase:
    """Shared setup + eligibility for knob-surface proposers.

    Holds the source, workload label, catalog exclusion, and affinity map, and
    exposes the two pieces every knob-surface proposer needs: the searchable knob
    set (outside the catalog, with something to search) and the per-candidate spec
    builder. Subclasses decide *how* to pick from the searchable knobs — exhaustive
    value grid (:class:`GenerativeProposer`) or weighted sampling
    (:class:`StochasticProposer`).
    """

    def __init__(
        self,
        knob_source: KnobSource,
        *,
        workload: str = "vllm-decode",
        catalog_knobs: set[str] | None = None,
        affinity_keywords: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._source = knob_source
        self._workload = workload
        # Keyword affinity is the fallback for knobs that don't self-tag via
        # ``Knob.classes``. Default is the vLLM-flavoured vocabulary; a workload
        # with its own knob naming can supply its own (still no per-workload table).
        self._affinity = _CLASS_KEYWORDS if affinity_keywords is None else affinity_keywords
        self._catalog = (
            set(catalog_knobs)
            if catalog_knobs is not None
            else {s.knob for s in load_library()}
        )

    def _searchable(self) -> list[Knob]:
        """Knobs worth searching: outside the catalog and with a non-empty grid."""
        return [
            k for k in self._source.knobs() if k.name not in self._catalog and _value_grid(k)
        ]

    def _spec(
        self,
        bottleneck_class: str,
        knob: Knob,
        value: object,
        *,
        target_op: str | None,
        verb: str,
        source: str,
    ) -> InterventionSpec:
        aim = f" (targeting {target_op})" if target_op else ""
        return _candidate_spec(
            name=f"autoresearch:{bottleneck_class}:{knob.name}={value}",
            summary=f"{verb} {knob.name}={value} for {bottleneck_class}{aim}",
            knob=knob.name,
            value=value,
            applies_to_kernels=[target_op] if target_op else [],
            bottleneck_class=bottleneck_class,
            workload=self._workload,
            source=source,
        )


class GenerativeProposer(_ProposerBase):
    """Search a workload's knob surface exhaustively, gated like any spec.

    Pulls knobs from ``knob_source`` (any workload's config namespace), drops any
    that duplicate the curated library, keeps the ones affine to the bottleneck
    class (an explicit ``Knob.classes`` tag, else a keyword heuristic on the name),
    and emits one candidate per value-grid point. Forced to ``moderate`` tier with
    an honest, unproven delta band — it can only *widen* the candidate set; the
    selection + rollback gate is what keeps anything. ``workload`` labels the
    candidates' applicability, so one mechanism serves any workload without a table.
    """

    def __init__(
        self,
        knob_source: KnobSource,
        *,
        workload: str = "vllm-decode",
        catalog_knobs: set[str] | None = None,
        max_candidates: int | None = None,
        affinity_keywords: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        super().__init__(
            knob_source,
            workload=workload,
            catalog_knobs=catalog_knobs,
            affinity_keywords=affinity_keywords,
        )
        self._max = max_candidates

    def propose(
        self, bottleneck_class: str, *, target_op: str | None = None
    ) -> list[InterventionSpec]:
        keywords = self._affinity.get(bottleneck_class, ())
        out = [
            self._spec(
                bottleneck_class,
                knob,
                value,
                target_op=target_op,
                verb="search",
                source="autoresearch-v0 (generated candidate; real workload knob, unproven delta)",
            )
            for knob in self._searchable()
            if _affine(knob, bottleneck_class, keywords)
            for value in _value_grid(knob)
        ]
        # Bound the per-class candidate count so a large config surface can't
        # flood the gate; the rollback gate still ranks and proves what survives.
        return out if self._max is None else out[: self._max]


class EngineArgsProposer(GenerativeProposer):
    """vLLM binding of :class:`GenerativeProposer`: the EngineArgs surface + vllm-decode.

    The loop's convenience entry point. ``knobs=`` overrides the surface (used by
    tests); otherwise it introspects EngineArgs, falling back to the frozen catalog
    offline. Defaults to a candidate cap since the real ``EngineArgs`` surface is
    large — the offline fallback catalog is well under it, so counts are unchanged.
    """

    def __init__(
        self,
        *,
        knobs: list[Knob] | None = None,
        catalog_knobs: set[str] | None = None,
        max_candidates: int | None = 24,
    ) -> None:
        source: KnobSource = (
            _ListKnobSource(tuple(knobs)) if knobs is not None else VLLMKnobSource()
        )
        super().__init__(
            source,
            workload="vllm-decode",
            catalog_knobs=catalog_knobs,
            max_candidates=max_candidates,
        )


class FallbackProposer:
    """Try ``primary``; use ``secondary`` only when primary yields nothing.

    Wires the generative proposer as the active source with the static table as
    the genuine fallback — a class the EngineArgs surface can't populate (or an
    unknown class) still gets the reviewed catalog's levers.
    """

    def __init__(self, primary: Proposer, secondary: Proposer) -> None:
        self._primary = primary
        self._secondary = secondary

    def propose(
        self, bottleneck_class: str, *, target_op: str | None = None
    ) -> list[InterventionSpec]:
        specs = self._primary.propose(bottleneck_class, target_op=target_op)
        return specs or self._secondary.propose(bottleneck_class, target_op=target_op)


class StochasticProposer(_ProposerBase):
    """Entropy-guided sampling of a workload's knob surface (reproducible by seed).

    The heuristic weights the dice: knobs affine to the bottleneck class carry most
    of the mass, but every eligible knob keeps a nonzero floor (``epsilon``), so the
    search can wander off-class and surface a lever the keyword heuristic would
    never pick. A seeded RNG draws the actual (knob, value) candidates —
    reproducible for a given seed, varied by changing it — and the rollback gate
    makes unbounded entropy safe. Same seam as the others: a candidate *source*, not
    a trust path. ``epsilon=0`` collapses to pure heuristic (affine knobs only);
    higher ``epsilon`` explores more widely.
    """

    def __init__(
        self,
        knob_source: KnobSource,
        *,
        workload: str = "vllm-decode",
        catalog_knobs: set[str] | None = None,
        n_samples: int = 6,
        seed: int = 0,
        epsilon: float = 0.15,
        affinity_keywords: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        super().__init__(
            knob_source,
            workload=workload,
            catalog_knobs=catalog_knobs,
            affinity_keywords=affinity_keywords,
        )
        self._n = n_samples
        self._seed = seed
        self._epsilon = epsilon

    def propose(
        self, bottleneck_class: str, *, target_op: str | None = None
    ) -> list[InterventionSpec]:
        keywords = self._affinity.get(bottleneck_class, ())
        eligible = [(k, _value_grid(k)) for k in self._searchable()]
        eligible = [(k, grid) for k, grid in eligible if grid]
        # Bias the dice toward the attributed class; the floor keeps off-class knobs
        # reachable — that's the entropy. All-zero weight (epsilon=0, nothing affine)
        # means no heuristic signal *and* no entropy, so nothing to sample.
        weights = [
            1.0 if _affine(k, bottleneck_class, keywords) else self._epsilon
            for k, _grid in eligible
        ]
        if not any(weights):
            return []

        rng = random.Random(self._seed)  # noqa: S311 - reproducible search, not security
        seen: set[tuple[str, object]] = set()
        out: list[InterventionSpec] = []
        max_attempts = max(self._n * 4, len(eligible) * 4)
        for _ in range(max_attempts):
            if len(out) >= self._n:
                break
            knob, grid = rng.choices(eligible, weights=weights, k=1)[0]
            value = rng.choice(grid)
            if (knob.name, value) in seen:  # don't gate the same candidate twice
                continue
            seen.add((knob.name, value))
            out.append(
                self._spec(
                    bottleneck_class,
                    knob,
                    value,
                    target_op=target_op,
                    verb="sample",
                    source="autoresearch-v0 (stochastic sample; real workload knob, unproven delta)",
                )
            )
        return out


def autoresearch_v0(
    trace: Trace,
    bottleneck_class: str,
    *,
    applicator: Applicator,
    policy: Policy | None = None,
    min_keep_delta: float = 0.0,
    target: ResidualTarget | None = None,
    proposer: Proposer | None = None,
    ctx: GateContext | None = None,
    reject: Callable[[InterventionSpec], str | None] | None = None,
) -> list[AutoresearchResult]:
    """Propose → gate → (apply + measure + rollback) for one bottleneck class.

    Proposals are ranked and pre-filtered by :func:`select_interventions` (the
    same gate the catalog goes through), then each survivor is applied behind the
    rollback gate so a proposal that doesn't clear ``min_keep_delta`` is reverted.

    ``proposer`` chooses the candidate source. Default (``None``) is the static
    :func:`propose` table; pass an :class:`EngineArgsProposer` (or a
    :class:`FallbackProposer`) to generate candidates from the real EngineArgs
    surface. Either way the ranking + rollback gate below is identical.

    ``ctx`` is the precondition gate context, forwarded to
    :func:`select_interventions` so candidates face the *same* applicability gate
    as the catalog. ``reject`` is an optional per-candidate veto applied after the
    gate but before apply (the loop uses it for the live structural-knob-needs-
    restart guard) — it keeps engine-specific policy out of this workload-agnostic
    core.

    A ``target`` (the largest-residual op) repoints the search: when that op is
    present in the trace, proposals are scoped to it so the gate prioritizes
    levers hitting the biggest gap, and every result records the op it aimed at.
    """
    # Only tag with the op when it matches a real kernel — otherwise coverage is 0.
    target_op = target.op if (target is not None and _op_present(trace, target.op)) else None
    if proposer is None:
        proposals = propose(bottleneck_class, target_op=target_op)
    else:
        proposals = proposer.propose(bottleneck_class, target_op=target_op)
    if not proposals:
        return []

    ranked = select_interventions(
        trace, proposals, policy or Policy(), top_n=len(proposals), ctx=ctx
    )
    aimed_at = target.op if target is not None else None

    results: list[AutoresearchResult] = []
    for c in ranked:
        # Gate rejection wins; else the caller's veto (e.g. a live structural knob
        # with no restart hook) can reject before we touch the engine. Rejected
        # candidates are recorded but never applied; survivors go through the
        # rollback-gated apply. Both land in one result shape.
        reason = c.rejected_reason
        if reason is None and reject is not None:
            reason = reject(c.spec)
        applied = (
            apply_intervention(c.spec, applicator, min_keep_delta=min_keep_delta)
            if reason is None
            else None
        )
        results.append(
            AutoresearchResult(
                spec=c.spec,
                bottleneck_class=bottleneck_class,
                predicted_delta=c.predicted_delta,
                applicable=applied is not None,
                rejected_reason=reason,
                measured_delta=applied.measured_delta if applied else None,
                rolled_back=applied.rolled_back if applied else False,
                target_op=aimed_at,
            )
        )
    return results


def autoresearch(
    trace: Trace,
    *,
    applicator: Applicator,
    policy: Policy | None = None,
    min_keep_delta: float = 0.0,
    residuals: Residuals | None = None,
    proposer: Proposer | None = None,
    ctx: GateContext | None = None,
    reject: Callable[[InterventionSpec], str | None] | None = None,
) -> AutoresearchRun:
    """Classify the trace's bottleneck, then run the full propose→gate→apply pass.

    This is the end-to-end entry point: hand it a captured trace and a live
    applicator and it decides which class to search, proposes non-catalog levers
    for that class, and routes each through the selection + rollback gates.

    ``proposer`` selects the candidate source (default: the static table); the
    loop passes an :class:`EngineArgsProposer` so the search generates candidates
    from the real EngineArgs surface rather than a frozen list. ``ctx`` forwards
    the precondition gate context (same applicability gate as the catalog);
    ``reject`` is an optional per-candidate veto (the loop's structural-knob
    guard). See :func:`autoresearch_v0`.

    When ``residuals`` (from :func:`gitm.optimizer.monitor.residuals`) are passed,
    the search is repointed at the largest-residual op — the biggest gap vs the
    predicted ceiling — rather than the whole trace. The loop already computes
    these in its attribution phase, so it passes them straight through.
    """
    bottleneck_class = classify_bottleneck(trace)
    target = largest_residual(residuals) if residuals is not None else None
    return AutoresearchRun(
        bottleneck_class=bottleneck_class,
        target=target,
        results=autoresearch_v0(
            trace,
            bottleneck_class,
            applicator=applicator,
            policy=policy,
            min_keep_delta=min_keep_delta,
            target=target,
            proposer=proposer,
            ctx=ctx,
            reject=reject,
        ),
    )
