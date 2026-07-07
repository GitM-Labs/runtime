"""Public embedded API.

    from gitm import optimize
    optimize(engine, budget="24h", target=0.15)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gitm.scheduler import LoopConfig, run_loop


def optimize(
    engine: Any | None = None,
    *,
    workload: str | None = None,
    budget: str = "24h",
    target: float = 0.15,
    scratch: str | None = None,
    workload_runner: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the autonomous 24-hour optimization loop and return a report.

    Either pass an ``engine`` (e.g. a running vLLM engine handle) for the
    embedded path, or pass ``workload`` (e.g. ``"vllm-decode"``) for the CLI
    path. ``budget`` and ``target`` follow the SKU contract: a verified floor
    of ``target`` fraction improvement within ``budget`` wall time, or a
    qualification-gate diagnostic explaining why the floor was not committed.

    ``workload_runner`` optionally supplies an explicit zero-arg callable that
    launches the workload's GPU work; it runs inside the capture window. When
    omitted, the loop resolves ``workload`` against the registry in
    :mod:`gitm.workloads`.
    """
    cfg = LoopConfig(
        engine=engine,
        workload=workload,
        budget=budget,
        target=target,
        scratch=scratch,
        workload_runner=workload_runner,
    )
    return run_loop(cfg)
