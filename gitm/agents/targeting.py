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

**A row with no levers is a result, not an empty list to skip.** It says the
catalog cannot address the place the time is going, and that is the most useful
thing this module can report: on a sparse-MoE checkpoint ``moe_routed`` is half
the predicted step and no entry names it, so the loop has nothing to try there no
matter how it ranks.

Deliberately not here: a score combining the two sides. ``recoverable_ms`` is a
duration and ``expected_delta_mean`` is a fraction of the whole step, so a product
of them is not a quantity anything measures. Picking the regions and listing what
aims at them is well defined on its own; how to order candidates *within* a region
is a separate decision with a separate answer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.deviation_table import (
    UNMODELED,
    DeviationRow,
    DeviationTable,
    rank_by_recoverable,
)

__all__ = ["Target", "levers_for", "targets", "render_targets"]


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


def levers_for(row: DeviationRow, library: Iterable[InterventionSpec]) -> list[InterventionSpec]:
    """Levers that name ``row.op`` in their declared scope.

    Op identity only — no substring fallback. A lever earns a row by naming the
    op the graph and the classifier agree on, and a coincidental substring is how
    an untargeted lever gets tagged as targeted.
    """
    if row.op == UNMODELED:
        return []
    return [s for s in library if row.op in s.applies_to_kernels]


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


def render_targets(found: list[Target]) -> str:
    """The table as text, most recoverable first."""
    if not found:
        return "no recoverable time found"
    out = [f"  {'region':24s} {'phase':8s} {'bound':14s} {'recover_ms':>11s}  levers"]
    for t in found:
        names = ", ".join(s.name for s in t.levers[:3]) if t.levers else "— none target this"
        more = f" (+{len(t.levers) - 3})" if len(t.levers) > 3 else ""
        out.append(
            f"  {t.row.region[:24]:24s} {t.row.phase:8s} {(t.row.bound or '-')[:14]:14s} "
            f"{t.row.recoverable_ms:11.2f}  {names}{more}"
        )
    blind = [t for t in found if t.uncovered]
    if blind:
        ms = sum(t.row.recoverable_ms for t in blind)
        out.append("")
        out.append(f"  {len(blind)} of {len(found)} region(s) have no lever naming them — "
                   f"{ms:.1f} ms recoverable the catalog cannot address.")
    return "\n".join(out)
