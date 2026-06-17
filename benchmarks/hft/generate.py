"""Back-compat shim — the HFT generator moved into the installable package.

Canonical location is now :mod:`gitm.benchmarks.hft.generate` (it ships in the
wheel so the loop can auto-stage a smoke dataset from a pip install). Existing
imports of ``benchmarks.hft.generate`` keep working via the re-exports below.
"""

from __future__ import annotations

from gitm.benchmarks.hft.generate import *  # noqa: F401,F403
from gitm.benchmarks.hft.generate import main  # noqa: F401  (used in __main__ below)

if __name__ == "__main__":
    raise SystemExit(main())
