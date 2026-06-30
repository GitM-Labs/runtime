"""Fail-open guard: gitm's death must leave the workload untouched.

Day-one safety invariant. Every live mutation registers a revert with the guard;
on any exit — normal, exception, or a catchable signal (SIGTERM/SIGINT) — the
guard runs all reverts in LIFO order, best-effort, each logged to the audit
trail. ``disarm`` marks a change as intentionally kept (it cleared the gate) so
it is not rolled back.

SIGKILL / power loss can't run code; for those the guarantee must come from the
mutations themselves being non-persistent (in-memory engine knobs, removable
hooks) — which is why the live applicator only hot-swaps reversible knobs.
"""

from __future__ import annotations

import signal
from collections.abc import Callable
from typing import Any

from gitm.safety.audit import AuditLog


class FailOpenGuard:
    def __init__(
        self,
        *,
        audit: AuditLog | None = None,
        install_signal_handlers: bool = True,
    ) -> None:
        self._reverts: list[tuple[str, Callable[[], None], str]] = []
        self._audit = audit
        self._fired = False
        self._install = install_signal_handlers
        self._prev_handlers: dict[int, Any] = {}

    def register(self, name: str, revert_fn: Callable[[], None], *, cause: str = "") -> None:
        """Register a revert to run if we exit before it is disarmed."""
        self._reverts.append((name, revert_fn, cause))

    def disarm(self, name: str) -> None:
        """Mark a change as intentionally kept — it will not be reverted."""
        self._reverts = [r for r in self._reverts if r[0] != name]

    def fire(self) -> list[str]:
        """Run all pending reverts in LIFO order; return the names reverted."""
        if self._fired:
            return []
        self._fired = True
        done: list[str] = []
        for name, fn, cause in reversed(self._reverts):
            try:
                fn()
                done.append(name)
                if self._audit is not None:
                    self._audit.record_revert(name, reason="fail-open", cause=cause or "guard exit")
            except Exception:
                # Best-effort: one revert failing must not block the others.
                continue
        self._reverts.clear()
        return done

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.fire()
        prev = self._prev_handlers.get(signum, signal.SIG_DFL)
        if callable(prev):
            prev(signum, frame)

    def __enter__(self) -> FailOpenGuard:
        self._fired = False
        if self._install:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    self._prev_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, self._signal_handler)
                except (ValueError, OSError):
                    # not in the main thread (e.g. tests) — context exit still covers us
                    pass
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.fire()
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev_handlers.clear()