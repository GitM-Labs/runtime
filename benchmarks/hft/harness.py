"""Back-compat shim — the HFT harness moved into the installable package.

Canonical location is now :mod:`gitm.benchmarks.hft.harness` (it ships in the
wheel; this top-level ``benchmarks/`` tree does not). Existing imports of
``benchmarks.hft.harness`` keep working via the re-exports below.
"""

from __future__ import annotations

from gitm.benchmarks.hft.harness import *  # noqa: F401,F403
from gitm.benchmarks.hft.harness import (  # noqa: F401
    _gpu_name,
    _seed_dir,
    load_events,
    main,
    microprice,
    run_pipeline,
    select_backend,
    top_of_book,
    vwap_1s,
)

if __name__ == "__main__":
    raise SystemExit(main())
