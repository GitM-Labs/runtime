"""Append-only audit log: every applied change and revert, with its cause.

A safety requirement, not telemetry: when something runs unattended on a real
box, we must be able to answer "what did gitm change, when, why, and did it
revert?" from a durable record. The log is newline-delimited JSON (one event per
line, fsync'd on write) so a crash mid-run still leaves a readable, append-only
trail — entries are never rewritten.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    ts_ns: int
    event: str  # "apply" | "revert" | str
    intervention: str
    cause: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """Durable append-only audit trail at ``path`` (JSONL).

    ``clock`` is injectable for deterministic tests; it defaults to wall-clock
    nanoseconds.
    """

    def __init__(self, path: str | Path, *, clock: Callable[[], int] = time.time_ns) -> None:
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, intervention: str, cause: str, **detail: Any) -> AuditEvent:
        """Append one event and flush it to disk before returning."""
        ev = AuditEvent(
            ts_ns=self._clock(), event=event, intervention=intervention,
            cause=cause, detail=detail,
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev.to_dict(), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return ev

    def record_apply(
        self, intervention: str, *, knob: str, value: Any, cause: str
    ) -> AuditEvent:
        return self.record("apply", intervention, cause, knob=knob, value=value)

    def record_revert(self, intervention: str, *, reason: str, cause: str) -> AuditEvent:
        return self.record("revert", intervention, cause, reason=reason)

    def entries(self) -> list[AuditEvent]:
        """Read the full trail back, in write order."""
        if not self.path.exists():
            return []
        out: list[AuditEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(AuditEvent(**json.loads(line)))
        return out