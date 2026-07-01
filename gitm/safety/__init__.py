"""
Safety primitives - the guardrails every live apply runs behind
Day-one posture is detect-revert-page, not confidence-gated auto-act.
These are the mechanisms that make that posture real:
- an append-only audit log of every applied change and revert (with cause),
- a fail-open guard that reverts every live mutation on any exit,
- an auto-revert that detects a regression vs baseline over a window, and
- a gated rollout that stages changes in shadow until a manual, confirmed promote.
"""

from __future__ import annotations

from gitm.safety.audit import AuditEvent, AuditLog
from gitm.safety.autorevert import AutoRevert, AutoRevertDecision
from gitm.safety.failopen import FailOpenGuard
from gitm.safety.rollout import GatedRollout, StagedChange

__all__ = [
    "AuditEvent", "AuditLog", "AutoRevert", "AutoRevertDecision", "FailOpenGuard", "GatedRollout", "StagedChange"
]
