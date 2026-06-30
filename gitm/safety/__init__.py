"""
Safety primitives - the guardrails every live apply runs behind
Day-one posture is detect-revert-page, not confidence-gated auto-act. 
These are the mechanisms that make that posture real: 
an append-only audit log of every applied change and revert (with cause), 
and (landing alongside) fail-open and auto-revert.
"""

from __future__ import annotations

from gitm.safety.audit import AuditEvent, AuditLog
from gitm.safety.autorevert import AutoRevert, AutoRevertDecision
from gitm.safety.failopen import FailOpenGuard
from gitm.safety.rollout import GatedRollout, StagedChange

__all__ = [
    "AuditEvent", "AuditLog", "AutoRevert", "AutoRevertDecision", "FailOpenGuard", "GatedRollout", "StagedChange"
]
