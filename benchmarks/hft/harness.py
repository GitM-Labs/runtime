"""Back-compat shim — the HFT harness moved into the installable package.

Canonical location is now :mod:`gitm.benchmarks.hft.harness` (it ships in the
wheel; this top-level ``benchmarks/`` tree does not). Existing imports of
``benchmarks.hft.harness`` keep working via the re-exports below.
"""

from __future__ import annotations

from gitm.benchmarks.hft.harness import *  # noqa: F401,F403

# `import *` skips underscore names; forward the private helpers explicitly so
# the old import path stays fully compatible. `main` is also used below.
from gitm.benchmarks.hft.harness import _gpu_name, _seed_dir, main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
