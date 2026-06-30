"""Gated rollout: shadow first, manual promote. (crude is fine — day one)

Day-one posture is not confidence-gated auto-act. A change is staged in shadow
(recorded + validated, never applied live), and only a manual ``promote(confirm=
True)`` moves it live. ``abort`` discards it. Every transition is audited, so the
trail shows what was staged, what was promoted (and by whom/why), and what was
dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gitm.safety.audit import AuditLog


@dataclass
class StagedChange:
    name: str
    knob: str
    value: Any
    state: str = "shadow"  # "shadow" -> "promoted" | "aborted"
    reason: str = ""


class GatedRollout:
    def __init__(self, *, audit: AuditLog | None = None) -> None:
        self._audit = audit
        self._staged: dict[str, StagedChange] = {}

    def stage(self, name: str, *, knob: str, value: Any, reason: str = "") -> StagedChange:
        change = StagedChange(name=name, knob=knob, value=value, reason=reason)
        self._staged[name] = change
        if self._audit is not None:
            self._audit.record("stage", name, reason or "shadow", knob=knob, value=value)
        return change

    def promote(self, name: str, *, confirm: bool, cause: str = "manual promote") -> bool:
        """Promote a staged change to live. Requires ``confirm=True`` (manual)."""
        change = self._staged.get(name)
        if change is None or change.state != "shadow":
            return False
        if not confirm:
            self.abort(name, reason="promote not confirmed")
            return False
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