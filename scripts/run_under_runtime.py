#!/usr/bin/env python
"""Back-compat shim — the driver moved into the installable package.

Canonical location is now :mod:`gitm.runtime_driver`, also exposed as the
``gitm-run-workload`` console command (so ``pip install gitm-labs`` ships it).
This script keeps the old ``python scripts/run_under_runtime.py ...`` invocation
working from a repo checkout.
"""

from gitm.runtime_driver import main

if __name__ == "__main__":
    raise SystemExit(main())
