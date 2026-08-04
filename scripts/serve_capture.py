#!/usr/bin/env python3
"""Back-compat entry point for the traced ``vllm serve`` run.

The implementation moved into the package (:mod:`gitm.serve.vllm`) so that both
capture paths ship in the wheel and are reachable as ``gitm capture serve`` /
``gitm capture attach``. This shim stays because it is baked into pod runbooks and
shell history:

    python scripts/serve_capture.py -- vllm serve MODEL ...

is identical to

    gitm capture serve -- vllm serve MODEL ...
"""

from __future__ import annotations

import sys

from gitm.serve.vllm import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
