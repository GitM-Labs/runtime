"""Which levers aim at the region that is actually losing time.

    table = from_trace("kimi.jsonl", graphs, steps=100)
    for t in targets(table, load_library(workload="vllm-decode")):
        print(t.row.region, t.row.recoverable_ms, [s.name for s in t.levers])

:mod:`gitm.optimizer.deviation_table` answers "where is time recoverable". The
intervention library answers "what can be changed". Nothing joined them: ranking
asked only how much of the trace a lever touches, so a lever with wide coverage
over a region already running at its predicted floor outranked one aimed at the
region that was actually over.

The join is by op, and the two vocabularies already agree — ``applies_to_kernels``
names the same canonical ops ``classify_op`` produces, which is what the library
header requires of every vLLM entry.

**A row with no op-scoped lever is a result, not an empty list to skip.** It
says nothing in the catalog is *aimed* at the place the time is going, which is
the most useful thing this module can report. It does not say nothing can affect
the region: whole-step levers apply to every row and are reported separately, so
an uncovered row still has those.

``moe_routed`` was the motivating case — half the predicted step on a sparse-MoE
checkpoint with no entry naming it. #124 scoped the MoE levers to it, so that
particular gap is closed; the query is what remains useful, and the test that
pinned the gap now pins its closure.

Deliberately not here: a score combining the two sides. ``recoverable_ms`` is a
duration and ``expected_delta_mean`` is a fraction of the whole step, so a product
of them is not a quantity anything measures. Picking the regions and listing what
aims at them is well defined on its own; how to order candidates *within* a region
is a separate decision with a separate answer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.deviation_table import (
    UNMODELED,
    DeviationRow,
    DeviationTable,
    rank_by_recoverable,
)

__all__ = ["Target", "levers_for", "levers_naming", "ops_aimed_at", "targets",
           "targets_from_recoverable", "render_targets"]


@dataclass(frozen=True)
class Target:
    """One region worth an experiment, and the levers that aim at it."""

    row: DeviationRow
    #: Levers naming this row's op in ``applies_to_kernels``.
    levers: tuple[InterventionSpec, ...] = ()
    #: Levers that reshape the whole step — batch shape, admission order, graph
    #: capture. They are not aimed at this region and are listed apart from the
    #: ones that are, because "something applies here" and "something targets
    #: this" are different claims and only the second justifies the row.
    whole_step: tuple[InterventionSpec, ...] = ()

    @property
    def uncovered(self) -> bool:
        """No lever names this op. The catalog cannot address this region."""
        return not self.levers


def ops_aimed_at(spec: InterventionSpec) -> tuple[str, ...]:
    """The ops ``spec`` aims at: the canonical ops it names in its declared scope.

    The one definition of the lever↔op join. It is made from three sides: from a
    deviation row (:func:`levers_for`), from the per-kernel residuals in the
    ranking's floor gate and its ordering
    (:func:`gitm.agents.policy.select_interventions`), and in the run's
    ``targets.json`` (:func:`targets_from_recoverable`). Each used to spell it
    out for itself, and the docstring here claimed they shared it when the gate
    did not.

    Op identity only — no substring fallback. A lever earns an op by naming the
    one the graph and the classifier agree on, and a coincidental substring is
    how an untargeted lever gets tagged as targeted. A whole-step lever aims at
    no op: it reshapes the step, and naming ops would not make it targeted.
    """
    if spec.whole_step:
        return ()
    return tuple(spec.applies_to_kernels)


def levers_naming(op: str, library: Iterable[InterventionSpec]) -> list[InterventionSpec]:
    """Levers that name ``op`` in their declared scope (:func:`ops_aimed_at`)."""
    return [s for s in library if op in ops_aimed_at(s)]


def levers_for(row: DeviationRow, library: Iterable[InterventionSpec]) -> list[InterventionSpec]:
    """Levers that name ``row.op`` in their declared scope."""
    if row.op == UNMODELED:
        return []
    return levers_naming(row.op, library)


def targets(
    table: DeviationTable | list[DeviationRow],
    library: Iterable[InterventionSpec],
    *,
    top: int = 5,
    phase: str | None = None,
    bound: str | None = None,
    min_share: float = 0.0,
) -> list[Target]:
    """The regions with the most recoverable time, each with what aims at it.

    Rows come from :func:`rank_by_recoverable`, so the filters are its filters and
    unmodeled work never ranks. Rows with no levers are kept: they are the
    question this module exists to raise.
    """
    library = list(library)
    rows = table.rows if isinstance(table, DeviationTable) else list(table)
    whole = tuple(s for s in library if s.whole_step)
    return [
        Target(row=r, levers=tuple(levers_for(r, library)), whole_step=whole)
        for r in rank_by_recoverable(rows, top=top, phase=phase, bound=bound,
                                     min_share=min_share)
    ]


def targets_from_recoverable(
    recoverable: Mapping[str, float | None],
    library: Iterable[InterventionSpec],
    *,
    device_s: float,
    top: int = 10,
) -> dict[str, Any]:
    """Where this run's time is recoverable, and what aims at it, for ``targets.json``.

    Built from :func:`gitm.optimizer.monitor.recoverable_by_op`, the same map the
    ranking gates and orders on, rather than from a deviation table. The table
    reaches the same quantity by scaling a one-step floor by a step count that a
    trace cannot supply, and two derivations of one fact are two answers that can
    disagree. This is the one the ranking used, so it is the one written down.

    Per op, most recoverable first:

    * ``levers`` — levers whose gain comes from making this op faster
      (``recovers_kernel_time``). These are the ones the ranking orders by it.
    * ``also_named_by`` — levers that name the op but work some other way, e.g.
      through cache capacity. Touching the op is not a claim on its time.

    ``uncovered`` lists every op with recoverable time that no lever names, past
    the ``top`` cutoff as well: time
    the catalogue has nothing aimed at. Whole-step levers still apply to them,
    which is why it says "aimed at" and not "can affect".

    An op judged unjudgeable (``None``) is listed apart, not ranked: it may be over
    its floor, and nothing says by how much.
    """
    library = list(library)
    losing = sorted(((op, gap) for op, gap in recoverable.items() if gap),
                    key=lambda kv: (-kv[1], kv[0]))
    rows = losing[:top]
    out = []
    for op, gap in rows:
        naming = levers_naming(op, library)
        out.append({
            "op": op,
            "recoverable_s": gap,
            "share_of_device": (gap / device_s) if device_s > 0 else None,
            "levers": [s.name for s in naming if s.recovers_kernel_time],
            "also_named_by": [s.name for s in naming if not s.recovers_kernel_time],
        })
    return {
        "basis": "per-kernel residuals against this model's graph",
        "device_s": device_s,
        "regions": out,
        # Over every op losing time, not only the ``top`` listed above: an op past
        # the cutoff with nothing aimed at it is still a gap in the catalogue.
        "uncovered": [op for op, _ in losing if not levers_naming(op, library)],
        "unjudgeable": sorted(op for op, gap in recoverable.items() if gap is None),
        "whole_step_levers": sorted(s.name for s in library if s.whole_step),
    }


def render_targets(found: list[Target]) -> str:
    """The table as text, most recoverable first."""
    if not found:
        return "no recoverable time found"
    # Widened to the content rather than cut to a fixed 24. The layer suffix is
    # the last thing on a region name, so a fixed cut takes exactly the part that
    # distinguishes the rows: `attn_qnorm_rope_insert@L10` and `@L11` both land
    # on the same string, and the column stops saying which layer is over.
    w = max(len("region"), *(len(t.row.region) for t in found))
    out = [f"  {'region':{w}s} {'phase':8s} {'bound':14s} {'recover_ms':>11s}  levers"]
    for t in found:
        names = ", ".join(s.name for s in t.levers[:3]) if t.levers else "— none target this"
        more = f" (+{len(t.levers) - 3})" if len(t.levers) > 3 else ""
        out.append(
            f"  {t.row.region:{w}s} {t.row.phase:8s} {(t.row.bound or '-')[:14]:14s} "
            f"{t.row.recoverable_ms:11.2f}  {names}{more}"
        )
    blind = [t for t in found if t.uncovered]
    if blind:
        ms = sum(t.row.recoverable_ms for t in blind)
        out.append("")
        # "no lever names them", not "the catalog cannot address them": whole-step
        # levers apply to every row, so the second claim is wider than the query
        # supports and would read as a catalog gap that is not there.
        whole = " (whole-step levers still apply)" if any(t.whole_step for t in blind) else ""
        out.append(f"  {len(blind)} of {len(found)} region(s) have no lever naming them — "
                   f"{ms:.1f} ms recoverable with nothing aimed at it{whole}.")
    return "\n".join(out)
