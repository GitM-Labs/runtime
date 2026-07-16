"""Spawn-safe entrypoint for the full vllm-decode optimizer loop.

    python scripts/run_vllm_optimize.py [budget] [target]
    python scripts/run_vllm_optimize.py 30m 0.15

A thin wrapper around ``gitm.optimize`` that exists as a real importable file with
an ``if __name__ == "__main__"`` guard — required because a structural-restart
candidate (fp8 KV, quantization) may build its engine under the ``spawn`` start
method, which re-imports ``__main__`` in the child. ``python -c "..."`` and stdin
heredocs have no importable ``__main__`` and fail; this file does not.

Reads the same GITM_VLLM_* environment as the workload factory. Saturate the KV
cache first or structural levers measure noise — see docs/vllm_decode_runbook.md.

Known issue: a scheduler-knob candidate can deadlock vLLM's scheduler under heavy
oversubscription (GPU util drops to 0%, generate() never returns). The A/B probe
has no wall-clock timeout yet, so the loop can hang. Until that lands, scripts/
fp8_ab.py is the safer path to the fp8 result.
"""

from __future__ import annotations

import sys

from gitm import optimize
from gitm.workloads import set_decode_run_defaults


def main(argv: list[str]) -> None:
    budget = argv[0] if len(argv) > 0 else "30m"
    target = float(argv[1]) if len(argv) > 1 else 0.15

    # Injection + KV-pressure env, set in-process before any CUDA init; exports win.
    set_decode_run_defaults()

    result = optimize(workload="vllm-decode", budget=budget, target=target)
    print(result["summary"]["report_path"])


if __name__ == "__main__":
    main(sys.argv[1:])
