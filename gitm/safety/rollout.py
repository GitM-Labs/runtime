"""Gated rollout: shadow first, manual promote. (crude is fine — day one)

Day-one posture is not confidence-gated auto-act. A change is staged in shadow
(recorded + validated, never applied live), and only a manual ``promote(confirm=
True)`` moves it live. ``abort`` discards it. Every transition is audited, so the
trail shows what was staged, what was promoted (and by whom/why), and what was
dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gitm.safety.audit import AuditLog
from gitm.safety.failopen import FailOpenGuard


@dataclass
class StagedChange:
    name: str
    knob: str
    value: Any
    state: str = "shadow"  # "shadow" -> "promoted" | "aborted"
    reason: str = ""


class GatedRollout:
    """Stage changes in shadow; a manual, confirmed ``promote`` moves them live.

    When a ``guard`` is supplied and a staged change carries an ``apply_fn`` (and
    optionally a ``revert_fn``), ``promote`` genuinely applies it and registers
    the revert with the :class:`FailOpenGuard` — so a promoted change is both
    live and fail-open protected. Without those, ``GatedRollout`` is bookkeeping
    only (state + audit), which is all the shadow-accounting callers need.
    """

    def __init__(
        self, *, audit: AuditLog | None = None, guard: FailOpenGuard | None = None
    ) -> None:
        self._audit = audit
        self._guard = guard
        self._staged: dict[str, StagedChange] = {}
        # name -> (apply_fn, revert_fn|None); only for changes that mutate a live target.
        self._appliers: dict[str, tuple[Callable[[], None], Callable[[], None] | None]] = {}

    def stage(
        self,
        name: str,
        *,
        knob: str,
        value: Any,
        reason: str = "",
        apply_fn: Callable[[], None] | None = None,
        revert_fn: Callable[[], None] | None = None,
    ) -> StagedChange:
        change = StagedChange(name=name, knob=knob, value=value, reason=reason)
        self._staged[name] = change
        if apply_fn is not None:
            self._appliers[name] = (apply_fn, revert_fn)
        if self._audit is not None:
            self._audit.record("stage", name, reason or "shadow", knob=knob, value=value)
        return change

    def promote(self, name: str, *, confirm: bool, cause: str = "manual promote") -> bool:
        """Promote a staged change to live. Requires ``confirm=True`` (manual).

        If the change was staged with an ``apply_fn`` it is applied *before* the
        state flips, so a failing apply leaves the change in shadow (and raises)
        rather than falsely reporting it live. A ``revert_fn`` + guard means the
        change is auto-reverted on fail-open.
        """
        change = self._staged.get(name)
        if change is None or change.state != "shadow":
            return False
        if not confirm:
            self.abort(name, reason="promote not confirmed")
            return False

        applier = self._appliers.get(name)
        if applier is not None:
            apply_fn, revert_fn = applier
            apply_fn()  # may raise -> propagate; change stays in shadow, not promoted
            if self._guard is not None and revert_fn is not None:
                self._guard.register(name, revert_fn, cause=cause)

        change.state = "promoted"
        if self._audit is not None:
            self._audit.record_apply(name, knob=change.knob, value=change.value, cause=cause)
        return True

    def abort(self, name: str, *, reason: str = "aborted") -> None:
        change = self._staged.get(name)
        if change is not None and change.state == "shadow":
            change.state = "aborted"
            if self._audit is not None:
                self._audit.record("abort", name, reason)

    def is_live(self, name: str) -> bool:
        c = self._staged.get(name)
        return c is not None and c.state == "promoted"

    def live_changes(self) -> list[StagedChange]:
        return [c for c in self._staged.values() if c.state == "promoted"]

    def shadow_changes(self) -> list[StagedChange]:
        return [c for c in self._staged.values() if c.state == "shadow"]
